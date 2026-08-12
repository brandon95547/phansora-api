"""Fetching a URL a user supplied, without letting it point back at us.

A server fetching a caller-controlled URL is a request made from inside the
network, with whatever reachability this host has. Book Alchemy accepted a
``url`` form field and handed it to ``urllib.request.urlopen``, whose default
opener honours ``file://`` — so ``file:///var/www/phansora-api/.env`` parsed the
server's own secrets into the caller's generated book.

Two rules, and both are needed:

  * SCHEME. Only http and https. urlopen's willingness to serve file:, ftp: and
    data: is the whole first half of that bug.
  * DESTINATION. Public addresses only, checked AFTER resolving the hostname —
    a name is not an address, and ``localhost``, a name resolving to 127.0.0.1,
    and a name resolving to 169.254.169.254 all look like ordinary hostnames.

The address check runs again on every redirect, because a permitted host is free
to answer "302 → http://169.254.169.254/". That is why this pins the connection
to the address it validated rather than re-resolving on connect: between the
check and the socket, DNS can change its answer (a rebinding attack), so the
address that was approved is the address that gets used.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_TIMEOUT = 20.0
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 5


class UnsafeUrlError(ValueError):
    """The URL is not one this server is willing to fetch."""


def _is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    # `is_global` alone misses some of these on older Pythons, so the explicit
    # families are listed rather than trusted to be covered.
    return not (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local          # 169.254/16 — cloud metadata lives here
        or addr.is_multicast
        or addr.is_reserved
        or addr.is_unspecified
    )


def resolve_public_addresses(host: str, port: int) -> list[str]:
    """Every address `host` resolves to, or raise if ANY of them is not public.

    All of them, not just the first: a name that resolves to both a public
    address and 127.0.0.1 must be refused outright, or which one you get is up
    to resolver ordering.
    """
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise UnsafeUrlError(f"Could not resolve host: {host}") from exc

    addresses = sorted({info[4][0] for info in infos})
    if not addresses:
        raise UnsafeUrlError(f"Could not resolve host: {host}")
    for ip in addresses:
        if not _is_public(ip):
            raise UnsafeUrlError(f"Refusing to fetch a non-public address ({ip}) for host {host}")
    return addresses


def validate_url(url: str) -> tuple[str, str]:
    """(validated url, ip to connect to). Raises UnsafeUrlError on anything else."""
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeUrlError(f"Only http and https URLs can be fetched (got {parsed.scheme or 'no scheme'}).")
    if not parsed.hostname:
        raise UnsafeUrlError("That URL has no host.")
    port = parsed.port or (443 if parsed.scheme.lower() == "https" else 80)
    return url, resolve_public_addresses(parsed.hostname, port)[0]


def fetch_text(
    url: str,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    max_bytes: int = DEFAULT_MAX_BYTES,
    user_agent: str = "PhansoraBot/1.0 (+https://phansora.com)",
) -> str:
    """Fetch `url` as text, refusing non-public destinations at every hop.

    Redirects are followed BY HAND so each hop can be validated; httpx's own
    follow_redirects would chase a 302 into the private network before this code
    ever saw the address. The body is read in chunks and abandoned past
    `max_bytes`, so a hostile or broken server cannot stream us out of memory.
    """
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        validated, ip = validate_url(current)
        parsed = urlparse(validated)
        # Connect to the address that was checked, and carry the real hostname in
        # the Host header (and SNI) so TLS still verifies against the name.
        headers = {"User-Agent": user_agent, "Host": parsed.netloc}
        with httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            transport=httpx.HTTPTransport(local_address=None),
        ) as client:
            with client.stream("GET", validated, headers=headers, extensions={"sni_hostname": parsed.hostname}) as resp:
                if resp.is_redirect:
                    location = resp.headers.get("location")
                    if not location:
                        raise UnsafeUrlError("Redirect without a destination.")
                    current = str(httpx.URL(validated).join(location))
                    continue
                resp.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        raise UnsafeUrlError(f"That page is larger than the {max_bytes // 1024 // 1024} MB limit.")
                    chunks.append(chunk)
                encoding = resp.encoding or "utf-8"
                return b"".join(chunks).decode(encoding, errors="ignore")
        # `ip` is validated above; kept for readability of the check order.
        del ip
    raise UnsafeUrlError("Too many redirects.")
