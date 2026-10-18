"""Outbound email over SMTP: the contact form's fixed recipient, and admin mail to a user.

The SMTP settings are read when a message is sent, not when this module is imported, so
they never depend on which module happened to load .env first (phansora.config loads it
once, at startup).
"""
import os
from email.message import EmailMessage

from aiosmtplib import send


async def send_email(data):
    # Always send to EMAIL_TO from env. This is what the public /contact route calls,
    # so a submission can never choose where the mail goes.
    return await send_message(
        os.getenv("EMAIL_TO"),
        data.get("subject", "No Subject"),
        data.get("message", ""),
        data.get("reply_to", ""),
    )


async def send_message(to_address, subject, message_body, reply_to=""):
    """Send one plain-text email to ``to_address``.

    The recipient is the caller's to choose, so only key-gated routes may call this with
    anything but EMAIL_TO (shared/admin/email.py checks X-Admin-Key first).
    """
    if not message_body:
        raise ValueError("Missing 'message'")

    smtp_user = os.getenv("SMTP_USER")
    msg = EmailMessage()
    msg["From"] = os.getenv("DEFAULT_FROM", smtp_user)
    msg["To"] = to_address
    msg["Subject"] = subject

    # Optional Reply-To so replies go straight to the person who submitted the
    # form. Guard against header injection and obvious non-addresses.
    reply_to = str(reply_to or "").strip()
    if reply_to and "\n" not in reply_to and "\r" not in reply_to and "@" in reply_to:
        msg["Reply-To"] = reply_to

    msg.set_content(message_body)

    # Let SMTP/connection errors propagate so the caller can return a
    # non-2xx status instead of reporting a false success.
    await send(
        msg,
        hostname=os.getenv("SMTP_HOST"),
        port=int(os.getenv("SMTP_PORT", "587")),
        username=smtp_user,
        password=os.getenv("SMTP_PASS"),
        start_tls=True,
    )
    return "Email sent"
