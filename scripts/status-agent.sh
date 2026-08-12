#!/usr/bin/env bash
#
# status-agent — lightweight production health watchdog for the Phansora stack.
#
# Runs from cron alongside `snapshot`. Each run it:
#   1. Confirms the core services are active   (nginx, phansora-api, frontend, postgres)
#   2. Validates the nginx config              (nginx -t)
#   3. Probes the HTTP health of API + frontend (are users getting served?)
#      /health is public and answers 200 off a bare process, so it ALSO probes a
#      non-public route and the shared secrets behind it — see check_api_auth.
#   4. Scans the API + frontend journald logs over the last window for errors
#      (500s, tracebacks, worker failures) — i.e. errors users are hitting live
#   5. Runs standard host checks               (disk, load, postgres connectivity)
#      Disk is two-tier: a heads-up once a mount passes $DISK_WARN_PCT (50%) of
#      capacity, and an urgent alert past $DISK_CRIT_PCT (90%).
#
# If anything is wrong it emails you via the phansora-api email endpoint
# (POST /contact -> delivers to $EMAIL_TO), falling back to direct SMTP when that
# fails. Both paths matter: /contact is a non-public route, so the watchdog must
# present the internal key to use it, and the API is one of the things being
# watched — an API outage must not be able to silence its own alert. (On
# 2026-08-12 it did exactly that: /contact answered 503 for 2.5h while the
# frontend was down, and every alert went nowhere but the local log.)
#
# It de-dupes: the same set of problems won't re-email more than once per
# $ALERT_COOLDOWN_SECONDS, so an ongoing outage nudges you at most a few times a
# day instead of every run. Runs that find nothing but capacity warnings use the
# longer $DISK_WARN_COOLDOWN_SECONDS (24h). An alert that could not be DELIVERED
# retries after the much shorter $DELIVERY_RETRY_SECONDS — a send that failed is
# not a notification, and must not sit out the full cooldown as though it were.
#
# Exit codes: 0 = all clean, 1 = issues found (alert sent or suppressed by cooldown).
#
# Requires: bash, curl, systemctl, journalctl, awk. Uses python3 (already present
# for the API) to JSON-encode the email safely. Should run as root (or a user in
# the systemd-journal group with sudo-less `nginx -t`) so it can read unit logs
# and test the nginx config.
#
# Install into cron (every 10 minutes) — see the block at the bottom of this file.

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Config — override any of these via the environment (e.g. in the cron line).
# ⚠ VERIFY the *_SERVICE names below match your server's actual systemd units
#   before enabling. `systemctl list-units --type=service | grep -Ei
#   'phansora|nginx|postgre'` will show them.
# ─────────────────────────────────────────────────────────────────────────────
API_SERVICE="${API_SERVICE:-phansora-api.service}"     # FastAPI/uvicorn unit
FRONTEND_SERVICE="${FRONTEND_SERVICE:-phansora.service}" # Node/Express unit — VERIFY
NGINX_SERVICE="${NGINX_SERVICE:-nginx.service}"

# Postgres runs in Docker (compose service "postgres" -> container "phansora_postgres"),
# not under systemd, and the host has no libpq client tools — so both the unit check and
# a host-side pg_isready are wrong here. When PG_CONTAINER is set we check the container
# and run pg_isready *inside* it (the postgres image always ships it). Set PG_CONTAINER=""
# on a host where Postgres really is a systemd unit, and PG_SERVICE takes over.
PG_CONTAINER="${PG_CONTAINER:-phansora_postgres}"        # docker container name, or "" for systemd
PG_SERVICE="${PG_SERVICE:-postgresql.service}"           # only used when PG_CONTAINER is empty

API_URL="${API_URL:-http://127.0.0.1:8000}"              # phansora-api base (uvicorn)
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:3000}"    # Express base

# Both apps' env files. Read for two reasons: the internal key the watchdog needs to
# POST /contact, and the SMTP credentials it falls back to. Values read from these are
# secrets — they are compared and used, never printed into the report or an alert.
FRONTEND_ENV_FILE="${FRONTEND_ENV_FILE:-/var/www/phansora/.env}"
API_ENV_FILE="${API_ENV_FILE:-/var/www/phansora-api/.env}"

# The API's auth gate (phansora.shared.auth): every route outside a small public
# allowlist needs this header. A missing PHANSORA_INTERNAL_KEY doesn't take the process
# down — /health keeps answering 200 while every real route answers 503 — so liveness
# checks alone cannot see it. Probe a non-public route to find out.
INTERNAL_KEY_HEADER="x-phansora-internal-key"
API_AUTH_PROBE_PATH="${API_AUTH_PROBE_PATH:-/spokenverse/voices}"   # must be a NON-public route

# How far back to scan logs. Keep >= your cron interval (+ a small overlap) so no
# window is ever skipped; the de-dupe below prevents the overlap from spamming.
LOG_WINDOW="${LOG_WINDOW:-11 min ago}"

# Host thresholds.
DISK_WARN_PCT="${DISK_WARN_PCT:-50}"                     # early heads-up: mount has passed half capacity
DISK_CRIT_PCT="${DISK_CRIT_PCT:-90}"                     # urgent: mount is nearly full
DISK_MOUNTS="${DISK_MOUNTS:-/ /var/lib/phansora}"        # space-separated mounts to watch
LOAD_WARN_PER_CORE="${LOAD_WARN_PER_CORE:-3.0}"          # 1-min loadavg per CPU core

# Alerting.
EMAIL_SUBJECT_PREFIX="${EMAIL_SUBJECT_PREFIX:-[status-agent] phansora.com}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-21600}" # 6h: suppress identical repeat alerts
# An alert that failed to send was never seen, so it doesn't earn the full cooldown —
# retry it this often instead (still slow enough not to hammer a down mail path).
DELIVERY_RETRY_SECONDS="${DELIVERY_RETRY_SECONDS:-1800}"  # 30m
# Disk sitting above DISK_WARN_PCT is a standing condition, not a new incident, so
# a run that finds *only* capacity warnings re-emails at most this often (24h).
DISK_WARN_COOLDOWN_SECONDS="${DISK_WARN_COOLDOWN_SECONDS:-86400}"
STATE_DIR="${STATE_DIR:-/var/lib/status-agent}"           # persists last-alert fingerprint
ALERT_LOG="${ALERT_LOG:-/var/log/status-agent.log}"       # local fallback if the API email can't send

# Error signatures to look for in logs (case-insensitive, extended regex).
ERR_PATTERNS="${ERR_PATTERNS:-traceback (most recent call last)|unhandled|unhandledrejection|referenceerror|typeerror|\" 5[0-9][0-9] |http 5[0-9][0-9]|internal server error|worker failed|econnrefused|etimedout|out of memory|oomkill|fatal}"
# Lines matching this are ignored even if they matched above (tune to your noise).
EXCLUDE_PATTERNS="${EXCLUDE_PATTERNS:-favicon|GET /health|GET /robots.txt|DeprecationWarning}"

MAX_SAMPLE_LINES="${MAX_SAMPLE_LINES:-40}"                # log excerpt lines per source in the email

VERBOSE=0
TEST_MODE=0
for arg in "$@"; do
  case "$arg" in
    --verbose|-v) VERBOSE=1 ;;
    --test)       TEST_MODE=1 ;;
    --help|-h)    grep -E '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Internals
# ─────────────────────────────────────────────────────────────────────────────
REPORT=""            # human-readable body accumulated across checks
FINGERPRINT=""       # stable one-line-per-issue key set, for de-dupe (no timestamps)
ISSUES=0
CAPACITY_WARNINGS=0  # subset of ISSUES that are early disk-capacity heads-ups
DELIVERED_VIA=""     # which transport actually got the last alert out

note() { [ "$VERBOSE" -eq 1 ] && echo "$*" >&2 || true; }

# add_issue <short-key> <detail-block>
# short-key: stable identifier used for de-dupe (no volatile data)
# detail-block: full text shown in the email
add_issue() {
  local key="$1"; shift
  local detail="$*"
  ISSUES=$((ISSUES + 1))
  FINGERPRINT+="${key}"$'\n'
  REPORT+="• ${detail}"$'\n\n'
  note "ISSUE: ${detail%%$'\n'*}"
}

have() { command -v "$1" >/dev/null 2>&1; }

# env_value <file> <KEY> — last assignment wins, surrounding quotes stripped.
# Returns non-zero when the file is unreadable or the key is absent/empty. Callers
# treat the result as a secret: compare it, send it, never report it.
env_value() {
  local file="$1" key="$2" line
  [ -r "$file" ] || return 1
  line="$(grep -E "^[[:space:]]*${key}=" "$file" 2>/dev/null | tail -n1 || true)"
  [ -n "$line" ] || return 1
  line="${line#*=}"
  line="${line%\"}"; line="${line#\"}"
  line="${line%\'}"; line="${line#\'}"
  [ -n "$line" ] || return 1
  printf '%s' "$line"
}

# ── 1. Service liveness ──────────────────────────────────────────────────────
check_service() {
  local unit="$1" label="$2"
  local state
  state="$(systemctl is-active "$unit" 2>/dev/null || true)"
  if [ "$state" != "active" ]; then
    # Distinguish "unit doesn't exist" (likely a misconfigured *_SERVICE var)
    # from "unit exists but is down".
    if ! systemctl cat "$unit" >/dev/null 2>&1; then
      add_issue "svc-missing:${unit}" "${label} (${unit}) not found — check the *_SERVICE config var."
    else
      local since
      since="$(systemctl show -p ActiveEnterTimestamp --value "$unit" 2>/dev/null || true)"
      add_issue "svc-down:${unit}" "${label} is ${state:-unknown} (${unit}). Last active: ${since:-n/a}."
    fi
  else
    note "ok: ${label} active"
  fi
}

# ── 2. nginx config validity ─────────────────────────────────────────────────
check_nginx_config() {
  have nginx || { note "nginx binary not on PATH; skipping config test"; return; }
  local out
  if ! out="$(nginx -t 2>&1)"; then
    add_issue "nginx-config-invalid" "nginx config test FAILED:"$'\n'"$(echo "$out" | sed 's/^/    /')"
  else
    note "ok: nginx -t"
  fi
}

# ── 3. HTTP health probes ────────────────────────────────────────────────────
# Sets a global http_code / http_err for the caller.
_probe() {
  local url="$1"
  http_err=""; http_code=""
  http_code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 10 "$url" 2>/dev/null)" || http_err="connection failed"
}

check_api_http() {
  local body
  if ! body="$(curl -s --max-time 10 "${API_URL}/health" 2>/dev/null)" || \
     ! printf '%s' "$body" | grep -q '"status"[[:space:]]*:[[:space:]]*"ok"'; then
    add_issue "api-health" "API health check failed at ${API_URL}/health (users can't be served). Response: ${body:-<none/unreachable>}"
  else
    note "ok: API /health"
  fi
}

# ── 3b. API auth gate ────────────────────────────────────────────────────────
# The blind spot this closes: on 2026-08-12 a deploy added PHANSORA_INTERNAL_KEY as a
# requirement and prod's .env never got it. The API stayed "active", /health stayed 200,
# and every non-public route answered 503 — TTS, Tomeweaver, Chrono, Book Alchemy, all
# dead, invisible to a liveness check. Probing WITHOUT credentials distinguishes the
# three states cleanly: 401 = gate configured and enforcing (healthy), 503 = key not
# configured (everything broken), 2xx = a route that should demand credentials isn't.
check_api_auth() {
  _probe "${API_URL}${API_AUTH_PROBE_PATH}"
  if [ -n "$http_err" ]; then
    add_issue "api-auth-unreachable" "API unreachable at ${API_URL}${API_AUTH_PROBE_PATH} (${http_err})."
  elif [ "${http_code:-0}" = "503" ]; then
    add_issue "api-auth-unconfigured" \
      "API auth is NOT configured: ${API_AUTH_PROBE_PATH} answers 503 with no credentials."$'\n'"    PHANSORA_INTERNAL_KEY is missing from ${API_ENV_FILE}, so EVERY non-public route is"$'\n'"    refusing traffic — all product features are down even though /health returns 200."
  elif [ "${http_code:-0}" -ge 200 ] && [ "${http_code:-0}" -lt 400 ]; then
    add_issue "api-auth-open" \
      "API auth is NOT enforcing: ${API_AUTH_PROBE_PATH} returned HTTP ${http_code} with no credentials."$'\n'"    A non-public route is serving anonymous callers — check the auth middleware and"$'\n'"    PHANSORA_AUTH_DISABLED (which must never be set in production)."
  elif [ "${http_code:-0}" -ge 500 ]; then
    add_issue "api-auth-5xx" "API returned HTTP ${http_code} on ${API_AUTH_PROBE_PATH} — server-side error behind the auth gate."
  else
    note "ok: API auth gate enforcing (HTTP ${http_code})"
  fi
}

# ── 3c. Shared secrets both apps must agree on ───────────────────────────────
# File-level companion to check_api_auth: the probe above can only see the API's side of
# the key. A key that is present but DIFFERENT in the two apps looks perfectly healthy
# from outside and fails every real call, so compare them directly. SESSION_SECRET is
# here for a different reason — its absence hard-throws at server.js boot, and naming it
# turns "frontend unreachable" into a one-line diagnosis.
check_shared_secrets() {
  local fe_key api_key
  if [ -r "$FRONTEND_ENV_FILE" ]; then
    if ! env_value "$FRONTEND_ENV_FILE" SESSION_SECRET >/dev/null; then
      add_issue "session-secret-missing" \
        "SESSION_SECRET is missing from ${FRONTEND_ENV_FILE} — the frontend hard-throws at boot"$'\n'"    in production (server.js) and will crash-loop until it is set."
    fi
  else
    note "frontend env file ${FRONTEND_ENV_FILE} unreadable; skipping secret checks"
    return
  fi

  fe_key="$(env_value "$FRONTEND_ENV_FILE" PHANSORA_INTERNAL_KEY || true)"
  api_key="$(env_value "$API_ENV_FILE" PHANSORA_INTERNAL_KEY || true)"
  if [ -z "$fe_key" ] && [ -z "$api_key" ]; then
    add_issue "internal-key-missing" \
      "PHANSORA_INTERNAL_KEY is set in neither ${FRONTEND_ENV_FILE} nor ${API_ENV_FILE} —"$'\n'"    the frontend cannot authenticate to the API and every product call will fail."
  elif [ -z "$fe_key" ]; then
    add_issue "internal-key-missing-frontend" "PHANSORA_INTERNAL_KEY is missing from ${FRONTEND_ENV_FILE} — frontend calls to the API will be rejected."
  elif [ -z "$api_key" ]; then
    add_issue "internal-key-missing-api" "PHANSORA_INTERNAL_KEY is missing from ${API_ENV_FILE} — the API will 503 every non-public route."
  elif [ "$fe_key" != "$api_key" ]; then
    add_issue "internal-key-mismatch" \
      "PHANSORA_INTERNAL_KEY differs between ${FRONTEND_ENV_FILE} and ${API_ENV_FILE} —"$'\n'"    the two apps disagree on the shared secret, so every frontend→API call gets 401."
  else
    note "ok: shared secrets present and matching"
  fi
}

check_frontend_http() {
  _probe "${FRONTEND_URL}/"
  if [ -n "$http_err" ]; then
    add_issue "frontend-down" "Frontend unreachable at ${FRONTEND_URL}/ (${http_err})."
  elif [ "${http_code:-0}" -ge 500 ]; then
    add_issue "frontend-5xx" "Frontend returned HTTP ${http_code} on the homepage — server-side error users will see."
  else
    note "ok: frontend HTTP ${http_code}"
  fi
}

# ── 4. Log error scan (the core ask) ─────────────────────────────────────────
scan_logs() {
  local unit="$1" label="$2"
  have journalctl || { note "journalctl unavailable; skipping log scan for ${label}"; return; }
  systemctl cat "$unit" >/dev/null 2>&1 || { note "no unit ${unit}; skipping log scan"; return; }

  local raw hits count sample
  # -p 0..4 = emerg..warning: catches everything Node/uvicorn writes to stderr
  # (console.error / tracebacks) AND explicit warnings. We then also keyword-match
  # so info-level access lines carrying a 5xx are caught too.
  raw="$(journalctl -u "$unit" --since "$LOG_WINDOW" --no-pager -o cat 2>/dev/null || true)"
  [ -z "$raw" ] && { note "no recent logs for ${label}"; return; }

  hits="$(printf '%s\n' "$raw" \
            | grep -Ei "$ERR_PATTERNS" 2>/dev/null \
            | grep -Eiv "$EXCLUDE_PATTERNS" 2>/dev/null || true)"
  count="$(printf '%s' "$hits" | grep -c . || true)"

  if [ "${count:-0}" -gt 0 ]; then
    sample="$(printf '%s\n' "$hits" | tail -n "$MAX_SAMPLE_LINES" | cut -c1-500 | sed 's/^/    /')"
    add_issue "log-errors:${unit}:${count}" \
      "${label}: ${count} error line(s) in the last window (${LOG_WINDOW}). Recent examples:"$'\n'"${sample}"
  else
    note "ok: no error lines for ${label}"
  fi
}

# ── 5. Standard host checks ──────────────────────────────────────────────────
check_disk() {
  local m used avail
  for m in $DISK_MOUNTS; do
    [ -d "$m" ] || continue
    used="$(df -P "$m" 2>/dev/null | awk 'NR==2{gsub("%","",$5); print $5}')"
    [ -z "$used" ] && continue
    avail="$(df -Ph "$m" 2>/dev/null | awk 'NR==2{print $4}')"
    # Two tiers, distinct fingerprint keys — crossing from warn into crit changes
    # the fingerprint, so the urgent one emails immediately instead of waiting out
    # the warning's cooldown.
    if [ "$used" -ge "$DISK_CRIT_PCT" ]; then
      add_issue "disk-crit:${m}" \
        "Disk ${m} is ${used}% full — only ${avail:-?} free (critical threshold ${DISK_CRIT_PCT}%). Free space now."
    elif [ "$used" -ge "$DISK_WARN_PCT" ]; then
      CAPACITY_WARNINGS=$((CAPACITY_WARNINGS + 1))
      add_issue "disk-warn:${m}" \
        "Disk ${m} has passed ${DISK_WARN_PCT}% capacity — ${used}% used, ${avail:-?} free. Heads-up only; nothing is failing yet."
    else
      note "ok: disk ${m} ${used}%"
    fi
  done
}

check_load() {
  local cores load1 limit over
  cores="$(nproc 2>/dev/null || echo 1)"
  load1="$(awk '{print $1}' /proc/loadavg 2>/dev/null || echo 0)"
  limit="$(awk -v c="$cores" -v p="$LOAD_WARN_PER_CORE" 'BEGIN{printf "%.2f", c*p}')"
  over="$(awk -v l="$load1" -v lim="$limit" 'BEGIN{print (l>lim)?1:0}')"
  if [ "$over" = "1" ]; then
    add_issue "load-high" "1-min load average ${load1} exceeds ${limit} (${cores} cores × ${LOAD_WARN_PER_CORE})."
  else
    note "ok: load ${load1} / ${limit}"
  fi
}

# Liveness of the Postgres *process*: a Docker container here, a systemd unit elsewhere.
check_postgres_service() {
  if [ -n "$PG_CONTAINER" ]; then
    have docker || { add_issue "pg-docker-missing" "PG_CONTAINER is set (${PG_CONTAINER}) but docker is not on PATH — postgres cannot be checked."; return; }
    local state
    state="$(docker inspect -f '{{.State.Status}}' "$PG_CONTAINER" 2>/dev/null || true)"
    if [ -z "$state" ]; then
      add_issue "pg-container-missing" "postgres container '${PG_CONTAINER}' not found — check the PG_CONTAINER config var."
    elif [ "$state" != "running" ]; then
      add_issue "pg-container-down" "postgres container '${PG_CONTAINER}' is ${state} — DB-backed features will error."
    else
      note "ok: postgres container ${PG_CONTAINER} running"
    fi
    return
  fi
  check_service "$PG_SERVICE" "postgresql"
}

check_postgres() {
  if [ -n "$PG_CONTAINER" ]; then
    have docker || return   # already reported by check_postgres_service
    # Skip when the container isn't up — check_postgres_service has said so already, and
    # a second "unreachable" line for the same root cause is just noise.
    [ "$(docker inspect -f '{{.State.Status}}' "$PG_CONTAINER" 2>/dev/null || true)" = "running" ] || return
    if docker exec "$PG_CONTAINER" pg_isready -q >/dev/null 2>&1; then
      note "ok: postgres accepting connections"
    else
      add_issue "postgres-unreachable" "PostgreSQL is not accepting connections (pg_isready inside ${PG_CONTAINER} failed) — DB-backed features will error."
    fi
    return
  fi
  have pg_isready || { note "pg_isready not installed; relying on service check"; return; }
  if ! pg_isready -q >/dev/null 2>&1; then
    add_issue "postgres-unreachable" "PostgreSQL is not accepting connections (pg_isready failed) — DB-backed features will error."
  else
    note "ok: postgres accepting connections"
  fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Alert delivery — via the phansora-api email endpoint (POST /contact -> EMAIL_TO)
# ─────────────────────────────────────────────────────────────────────────────
json_payload() {
  # $1 subject, $2 body -> compact JSON, safely escaped.
  SUBJECT="$1" BODY="$2" python3 - <<'PY'
import json, os
print(json.dumps({"subject": os.environ["SUBJECT"], "message": os.environ["BODY"]}))
PY
}

send_via_api() {
  local subject="$1" body="$2" code key tmp rc=0
  # /contact is NOT on the API's public allowlist, so an unauthenticated POST gets 401
  # (or 503 when the key is unset server-side). The watchdog must present the same
  # internal key the frontend uses — without it this transport is simply closed.
  key="$(env_value "$API_ENV_FILE" PHANSORA_INTERNAL_KEY || true)"
  tmp="$(mktemp)" || return 1
  chmod 600 "$tmp" 2>/dev/null || true
  json_payload "$subject" "$body" > "$tmp" || { rm -f "$tmp"; return 1; }
  # The key goes in through a -K config on stdin rather than -H, so it never appears
  # in `ps` output for every other user on the box to read.
  code="$(printf 'header = "%s: %s"\n' "$INTERNAL_KEY_HEADER" "$key" \
    | curl -s -o /dev/null -w '%{http_code}' --max-time 20 -K - \
        -X POST "${API_URL}/contact" \
        -H 'Content-Type: application/json' --data-binary "@${tmp}" 2>/dev/null)" || rc=1
  rm -f "$tmp"
  [ "$rc" -eq 0 ] && [ "$code" = "200" ]
}

# Fallback transport: talk to SMTP directly with the API's own credentials. The whole
# point is that this path shares nothing with the API process — when uvicorn is down,
# misconfigured, or 503-ing, the alert still has to get out.
send_via_smtp() {
  local subject="$1" body="$2" host port user pass to from
  host="$(env_value "$API_ENV_FILE" SMTP_HOST || true)"
  [ -n "$host" ] || return 1
  port="$(env_value "$API_ENV_FILE" SMTP_PORT || echo 587)"
  user="$(env_value "$API_ENV_FILE" SMTP_USER || true)"
  pass="$(env_value "$API_ENV_FILE" SMTP_PASS || true)"
  to="$(env_value "$API_ENV_FILE" EMAIL_TO || true)"
  from="$(env_value "$API_ENV_FILE" DEFAULT_FROM || printf '%s' "$user")"
  [ -n "$to" ] || return 1
  have python3 || return 1
  # Secrets travel in the environment, not argv, for the same `ps` reason as above.
  SUBJECT="$subject" BODY="$body" MAIL_TO="$to" MAIL_FROM="${from:-$user}" \
  S_HOST="$host" S_PORT="$port" S_USER="$user" S_PASS="$pass" \
  python3 - <<'PY' 2>/dev/null
import os, smtplib, ssl, sys
from email.message import EmailMessage

msg = EmailMessage()
msg["Subject"] = os.environ["SUBJECT"]
msg["From"] = os.environ["MAIL_FROM"]
msg["To"] = os.environ["MAIL_TO"]
msg.set_content(os.environ["BODY"])

host, port = os.environ["S_HOST"], int(os.environ.get("S_PORT") or 587)
user, pw = os.environ.get("S_USER") or "", os.environ.get("S_PASS") or ""
ctx = ssl.create_default_context()
try:
    if port == 465:
        srv = smtplib.SMTP_SSL(host, port, timeout=20, context=ctx)
    else:
        srv = smtplib.SMTP(host, port, timeout=20)
        try:
            srv.starttls(context=ctx)
        except smtplib.SMTPNotSupportedError:
            pass          # plain relay on the local network
    with srv:
        if user:
            srv.login(user, pw)
        srv.send_message(msg)
except Exception as exc:
    print(f"smtp fallback failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

# Try the normal path, then the independent one. Echoes which transport worked so
# --test and the log can say so.
deliver() {
  local subject="$1" body="$2"
  if send_via_api "$subject" "$body"; then
    DELIVERED_VIA="API ${API_URL}/contact"
    return 0
  fi
  if send_via_smtp "$subject" "$body"; then
    DELIVERED_VIA="direct SMTP (API path failed)"
    return 0
  fi
  DELIVERED_VIA=""
  return 1
}

# record_alert <fingerprint> [delivered]
# `delivered` is "yes" when the mail actually went out. A failed attempt is still
# recorded (so a dead transport isn't retried every single run) but marked, so the
# retry window below is minutes rather than the full cooldown.
record_alert() {
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  printf '%s' "$1" > "${STATE_DIR}/last_fingerprint" 2>/dev/null || true
  date +%s > "${STATE_DIR}/last_alert_epoch" 2>/dev/null || true
  if [ "${2:-no}" = "yes" ]; then
    rm -f "${STATE_DIR}/last_delivery_failed" 2>/dev/null || true
  else
    : > "${STATE_DIR}/last_delivery_failed" 2>/dev/null || true
  fi
}

# should_alert <fingerprint> [cooldown-seconds]
# Returns 0 if we should send (new problem OR cooldown elapsed), 1 to suppress.
should_alert() {
  local fp="$1" cooldown="${2:-$ALERT_COOLDOWN_SECONDS}" last_fp="" last_ts=0 now
  now="$(date +%s)"
  [ -f "${STATE_DIR}/last_fingerprint" ] && last_fp="$(cat "${STATE_DIR}/last_fingerprint" 2>/dev/null || true)"
  [ -f "${STATE_DIR}/last_alert_epoch" ] && last_ts="$(cat "${STATE_DIR}/last_alert_epoch" 2>/dev/null || echo 0)"
  if [ "$fp" != "$last_fp" ]; then return 0; fi
  # Last time round the alert was composed but never delivered — you haven't actually
  # been told, so don't let it serve the full cooldown in silence.
  if [ -f "${STATE_DIR}/last_delivery_failed" ] && [ "$DELIVERY_RETRY_SECONDS" -lt "$cooldown" ]; then
    cooldown="$DELIVERY_RETRY_SECONDS"
  fi
  [ $(( now - last_ts )) -ge "$cooldown" ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if [ "$TEST_MODE" -eq 1 ]; then
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  # Report the transports separately: "it worked" via the fallback alone still means
  # the primary path is broken and worth fixing before you need it.
  if send_via_api "${EMAIL_SUBJECT_PREFIX} test alert" "status-agent test email sent at ${ts} from $(hostname -f 2>/dev/null || hostname) via the API."; then
    echo "Test email dispatched via ${API_URL}/contact."
  else
    echo "PRIMARY path FAILED: POST ${API_URL}/contact." >&2
    echo "  Check that ${API_SERVICE} is up and that PHANSORA_INTERNAL_KEY in ${API_ENV_FILE}" >&2
    echo "  is readable — /contact is not a public route and 401/503 both mean no email." >&2
  fi
  if send_via_smtp "${EMAIL_SUBJECT_PREFIX} test alert (SMTP fallback)" "status-agent test email sent at ${ts} from $(hostname -f 2>/dev/null || hostname) via direct SMTP."; then
    echo "Fallback email dispatched via direct SMTP."
    exit 0
  fi
  echo "FALLBACK path FAILED — check SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_TO in ${API_ENV_FILE}." >&2
  exit 1
fi

check_service "$NGINX_SERVICE"    "nginx"
check_service "$API_SERVICE"      "phansora-api"
check_service "$FRONTEND_SERVICE" "frontend"
check_postgres_service
check_nginx_config
check_api_http
check_api_auth
check_shared_secrets
check_frontend_http
scan_logs "$API_SERVICE"      "phansora-api logs"
scan_logs "$FRONTEND_SERVICE" "frontend logs"
check_disk
check_load
check_postgres

HOST="$(hostname -f 2>/dev/null || hostname)"
TS="$(date '+%Y-%m-%d %H:%M:%S %Z')"

if [ "$ISSUES" -eq 0 ]; then
  note "All clean at ${TS}."
  exit 0
fi

BODY="status-agent found ${ISSUES} issue(s) on ${HOST} at ${TS}:

${REPORT}
— Automated watchdog. Log window scanned: ${LOG_WINDOW}."

# A run whose only findings are capacity heads-ups isn't an incident — label it as
# one and re-send it far less often than a real failure.
if [ "$CAPACITY_WARNINGS" -eq "$ISSUES" ]; then
  SUBJECT="${EMAIL_SUBJECT_PREFIX} disk capacity warning"
  COOLDOWN="$DISK_WARN_COOLDOWN_SECONDS"
else
  SUBJECT="${EMAIL_SUBJECT_PREFIX} ${ISSUES} issue(s) detected"
  COOLDOWN="$ALERT_COOLDOWN_SECONDS"
fi
FP="$(printf '%s' "$FINGERPRINT" | sort | (sha256sum 2>/dev/null || shasum -a 256) | awk '{print $1}')"

# Always keep a local record so nothing is lost even if email delivery fails.
{ echo "===== ${TS} ${HOST} (${ISSUES} issues, fp=${FP}) ====="; printf '%s\n' "$BODY"; } \
  >> "$ALERT_LOG" 2>/dev/null || true

if should_alert "$FP" "$COOLDOWN"; then
  if deliver "$SUBJECT" "$BODY"; then
    record_alert "$FP" yes
    note "Alert emailed via ${DELIVERED_VIA}."
  else
    # Both transports are down — surface loudly to cron/syslog so it's noticed, and
    # mark the attempt so should_alert retries in minutes rather than hours.
    have logger && logger -t status-agent "ALERT delivery FAILED on every transport; ${ISSUES} issues on ${HOST} (see ${ALERT_LOG})" || true
    echo "status-agent: ${ISSUES} issue(s) but BOTH the API and SMTP paths FAILED — see ${ALERT_LOG}" >&2
    record_alert "$FP" no
  fi
else
  note "Same issues as last alert and within cooldown (${COOLDOWN}s) — not re-emailing."
fi

exit 1

# ─────────────────────────────────────────────────────────────────────────────
# CRON INSTALL (run on prod as root, alongside your `snapshot` job)
#
#   sudo install -m 0755 /var/www/phansora-api/scripts/status-agent.sh \
#        /usr/local/bin/status-agent
#   sudo install -d -m 0755 /var/lib/status-agent
#
#   # then `sudo crontab -e` and add (every 10 min; LOG_WINDOW default 11m overlaps safely):
#   */10 * * * * FRONTEND_SERVICE=phansora.service /usr/local/bin/status-agent
#
#   Do NOT pass PG_SERVICE on this host: Postgres is the phansora_postgres container, and
#   the defaults already point at it. An old cron line carrying PG_SERVICE=postgresql.service
#   is harmless (PG_CONTAINER wins) but misleading — drop it when you next edit the crontab.
#
# The watchdog reads both apps' .env files (FRONTEND_ENV_FILE, API_ENV_FILE) for the
# internal key and the SMTP fallback credentials, so it must run as a user that can read
# them — root, as above. Defaults assume /var/www/phansora/.env and
# /var/www/phansora-api/.env; override in the cron line if your layout differs.
#
# Validate BOTH delivery paths before trusting them — --test now exercises each in turn
# and reports them separately, because the API path silently closed once already:
#   sudo /usr/local/bin/status-agent --test         # sends via the API, then via SMTP
#   sudo /usr/local/bin/status-agent --verbose      # runs all checks, prints results, no email unless issues
#
# Expect --test to print BOTH "Test email dispatched via .../contact" and "Fallback email
# dispatched via direct SMTP", and two emails to arrive. Only the fallback arriving means
# the primary path is broken (usually PHANSORA_INTERNAL_KEY unreadable or absent) — fix it
# then, not during the outage when you need it.
# ─────────────────────────────────────────────────────────────────────────────
