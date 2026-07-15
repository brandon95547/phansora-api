#!/usr/bin/env bash
#
# status-agent — lightweight production health watchdog for the Phansora stack.
#
# Runs from cron alongside `snapshot`. Each run it:
#   1. Confirms the core services are active   (nginx, phansora-api, frontend, postgres)
#   2. Validates the nginx config              (nginx -t)
#   3. Probes the HTTP health of API + frontend (are users getting served?)
#   4. Scans the API + frontend journald logs over the last window for errors
#      (500s, tracebacks, worker failures) — i.e. errors users are hitting live
#   5. Runs standard host checks               (disk, load, postgres connectivity)
#
# If anything is wrong it emails you via the phansora-api email endpoint
# (POST /contact -> delivers to $EMAIL_TO). It de-dupes: the same set of problems
# won't re-email more than once per $ALERT_COOLDOWN_SECONDS, so an ongoing
# outage nudges you at most a few times a day instead of every run.
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
# ⚠ VERIFY the three *_SERVICE names below match your server's actual systemd
#   units before enabling. `systemctl list-units --type=service | grep -Ei
#   'phansora|nginx|postgre'` will show them.
# ─────────────────────────────────────────────────────────────────────────────
API_SERVICE="${API_SERVICE:-phansora-api.service}"     # FastAPI/uvicorn unit
FRONTEND_SERVICE="${FRONTEND_SERVICE:-phansora.service}" # Node/Express unit — VERIFY
NGINX_SERVICE="${NGINX_SERVICE:-nginx.service}"
PG_SERVICE="${PG_SERVICE:-postgresql.service}"           # e.g. postgresql-16.service on RHEL — VERIFY

API_URL="${API_URL:-http://127.0.0.1:8000}"              # phansora-api base (uvicorn)
FRONTEND_URL="${FRONTEND_URL:-http://127.0.0.1:3000}"    # Express base

# How far back to scan logs. Keep >= your cron interval (+ a small overlap) so no
# window is ever skipped; the de-dupe below prevents the overlap from spamming.
LOG_WINDOW="${LOG_WINDOW:-11 min ago}"

# Host thresholds.
DISK_WARN_PCT="${DISK_WARN_PCT:-90}"                     # alert when any watched mount is fuller than this
DISK_MOUNTS="${DISK_MOUNTS:-/ /var/lib/phansora}"        # space-separated mounts to watch
LOAD_WARN_PER_CORE="${LOAD_WARN_PER_CORE:-3.0}"          # 1-min loadavg per CPU core

# Alerting.
EMAIL_SUBJECT_PREFIX="${EMAIL_SUBJECT_PREFIX:-[status-agent] phansora.com}"
ALERT_COOLDOWN_SECONDS="${ALERT_COOLDOWN_SECONDS:-21600}" # 6h: suppress identical repeat alerts
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
REPORT=""          # human-readable body accumulated across checks
FINGERPRINT=""     # stable one-line-per-issue key set, for de-dupe (no timestamps)
ISSUES=0

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
  local m used
  for m in $DISK_MOUNTS; do
    [ -d "$m" ] || continue
    used="$(df -P "$m" 2>/dev/null | awk 'NR==2{gsub("%","",$5); print $5}')"
    [ -z "$used" ] && continue
    if [ "$used" -ge "$DISK_WARN_PCT" ]; then
      add_issue "disk:${m}" "Disk ${m} is ${used}% full (threshold ${DISK_WARN_PCT}%)."
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

check_postgres() {
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
  local subject="$1" body="$2" code
  code="$(json_payload "$subject" "$body" \
    | curl -s -o /dev/null -w '%{http_code}' --max-time 20 \
        -X POST "${API_URL}/contact" \
        -H 'Content-Type: application/json' --data-binary @- 2>/dev/null)" || return 1
  [ "$code" = "200" ]
}

record_alert() {
  mkdir -p "$STATE_DIR" 2>/dev/null || true
  printf '%s' "$1" > "${STATE_DIR}/last_fingerprint" 2>/dev/null || true
  date +%s > "${STATE_DIR}/last_alert_epoch" 2>/dev/null || true
}

# Returns 0 if we should send (new problem OR cooldown elapsed), 1 to suppress.
should_alert() {
  local fp="$1" last_fp="" last_ts=0 now
  now="$(date +%s)"
  [ -f "${STATE_DIR}/last_fingerprint" ] && last_fp="$(cat "${STATE_DIR}/last_fingerprint" 2>/dev/null || true)"
  [ -f "${STATE_DIR}/last_alert_epoch" ] && last_ts="$(cat "${STATE_DIR}/last_alert_epoch" 2>/dev/null || echo 0)"
  if [ "$fp" != "$last_fp" ]; then return 0; fi
  [ $(( now - last_ts )) -ge "$ALERT_COOLDOWN_SECONDS" ]
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if [ "$TEST_MODE" -eq 1 ]; then
  ts="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  if send_via_api "${EMAIL_SUBJECT_PREFIX} test alert" "status-agent test email sent at ${ts} from $(hostname -f 2>/dev/null || hostname)."; then
    echo "Test email dispatched via ${API_URL}/contact."
    exit 0
  else
    echo "Test email FAILED — is ${API_SERVICE} up and EMAIL_TO/SMTP configured in the API .env?" >&2
    exit 1
  fi
fi

check_service "$NGINX_SERVICE"    "nginx"
check_service "$API_SERVICE"      "phansora-api"
check_service "$FRONTEND_SERVICE" "frontend"
check_service "$PG_SERVICE"       "postgresql"
check_nginx_config
check_api_http
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

SUBJECT="${EMAIL_SUBJECT_PREFIX} ${ISSUES} issue(s) detected"
FP="$(printf '%s' "$FINGERPRINT" | sort | (sha256sum 2>/dev/null || shasum -a 256) | awk '{print $1}')"

# Always keep a local record so nothing is lost even if email delivery fails.
{ echo "===== ${TS} ${HOST} (${ISSUES} issues, fp=${FP}) ====="; printf '%s\n' "$BODY"; } \
  >> "$ALERT_LOG" 2>/dev/null || true

if should_alert "$FP"; then
  if send_via_api "$SUBJECT" "$BODY"; then
    record_alert "$FP"
    note "Alert emailed via API."
  else
    # API/email path is itself down — surface loudly to cron/syslog so it's noticed.
    have logger && logger -t status-agent "ALERT email via API FAILED; ${ISSUES} issues on ${HOST} (see ${ALERT_LOG})" || true
    echo "status-agent: ${ISSUES} issue(s) but email via ${API_URL}/contact FAILED — see ${ALERT_LOG}" >&2
    record_alert "$FP"   # avoid hammering a down API every run; local log still has details
  fi
else
  note "Same issues as last alert and within cooldown (${ALERT_COOLDOWN_SECONDS}s) — not re-emailing."
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
#   */10 * * * * FRONTEND_SERVICE=phansora.service PG_SERVICE=postgresql.service /usr/local/bin/status-agent
#
# Validate the pipe once before trusting it:
#   sudo /usr/local/bin/status-agent --test        # sends a test email via the API
#   sudo /usr/local/bin/status-agent --verbose      # runs all checks, prints results, no email unless issues
# ─────────────────────────────────────────────────────────────────────────────
