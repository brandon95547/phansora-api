#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# net-watchdog — catch the recurring 5–13 minute network blackouts in the act.
#
# Prod goes totally unreachable from the internet for 5–13 minutes at a time,
# many times a week, while the guest stays up and healthy. The provider asked
# for live diagnostics captured DURING an event; the events are short, arrive
# without warning and often land overnight, so a human cannot be at a terminal
# for one. This daemon is that human.
#
# It samples every SAMPLE_SECS and keeps a rolling heartbeat. When it decides an
# event has started it captures a snapshot immediately, keeps sampling through
# the event, captures another on recovery, and emails the report once the
# network comes back (it cannot mail out while the link is dark — that is the
# whole point of the outage).
#
# THE MEASUREMENT THAT MATTERS
# Everything the guest can see is downstream of its own NIC, so the sharpest
# instrument available is the NIC packet counters. Per cycle we record the TX
# and RX deltas:
#
#   TX climbing, RX flat  → we are transmitting and nothing is coming back.
#                           Traffic is being discarded upstream of the VM.
#   TX flat,     RX flat  → the guest stopped transmitting too: a link, driver
#                           or hypervisor-level fault rather than a filter.
#
# That distinction is the one thing the provider's own logs cannot tell them and
# guest logs alone have never settled. Ping cannot answer it: an echo reply has
# to come back through the same dropped return path, so a failed ping is
# ambiguous between "nothing left" and "nothing returned". The counters are not.
#
# DETECTION
# Inbound silence is the primary signal. This box absorbs constant SSH
# brute-force traffic from hundreds of source IPs worldwide, which makes the
# sshd journal a free always-on inbound traffic monitor: minutes with zero sshd
# entries mean nothing on the internet reached the interface. Normal quiet gaps
# run to about a minute, so SILENCE_SECS is set well clear of them.
#
# Requires: bash, ping, ip, journalctl, python3. Uses tracepath if present
# (prod has no mtr or traceroute). Everything else degrades to a noted absence
# rather than a failure — a diagnostic tool must never die mid-incident.
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail          # deliberately no -e: a failed probe must not kill the daemon

VERSION="1.0.0"

# ── Config (all overridable from the systemd unit) ───────────────────────────
SAMPLE_SECS="${SAMPLE_SECS:-15}"          # how often we take a reading
SILENCE_SECS="${SILENCE_SECS:-180}"       # sshd quiet for this long ⇒ inbound is dark
RECOVER_SECS="${RECOVER_SECS:-60}"        # sshd talking again within this ⇒ event over
MIN_EVENT_SECS="${MIN_EVENT_SECS:-60}"    # shorter blips are logged, not emailed
MAX_MAILS_PER_HOUR="${MAX_MAILS_PER_HOUR:-6}"

IFACE="${IFACE:-}"                        # autodetected from the default route
PING_TARGETS="${PING_TARGETS:-8.8.8.8 1.1.1.1 9.9.9.9}"
EDGE_ROUTER="${EDGE_ROUTER:-173.208.136.130}"   # the provider hop named in the ticket
TRACE_TARGET="${TRACE_TARGET:-8.8.8.8}"

API_ENV_FILE="${API_ENV_FILE:-/var/www/phansora-api/.env}"
NGINX_ACCESS_GLOB="${NGINX_ACCESS_GLOB:-/var/log/nginx/*access*log}"
SSHD_UNIT="${SSHD_UNIT:-sshd.service}"

STATE_DIR="${STATE_DIR:-/var/lib/net-watchdog}"
LOG_FILE="${LOG_FILE:-/var/log/net-watchdog.log}"
EVENT_DIR="${EVENT_DIR:-${STATE_DIR}/events}"
HEARTBEAT="${STATE_DIR}/heartbeat"
MAIL_STAMPS="${STATE_DIR}/mail-stamps"
SUBJECT_PREFIX="${SUBJECT_PREFIX:-[phansora net-watchdog]}"

mkdir -p "$STATE_DIR" "$EVENT_DIR" 2>/dev/null

have() { command -v "$1" >/dev/null 2>&1; }
now()  { date +%s; }
iso()  { date -u -d "@${1:-$(date +%s)}" +%Y-%m-%dT%H:%M:%SZ; }
local_ts() { date -d "@${1:-$(date +%s)}" +'%Y-%m-%d %H:%M:%S %Z'; }

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >> "$LOG_FILE" 2>/dev/null; }

# ── Interface discovery ──────────────────────────────────────────────────────
# Whatever carries the default route is the interface that matters. Docker
# bridges and veths on this box would otherwise be tempting and wrong.
detect_iface() {
  [ -n "$IFACE" ] && { printf '%s' "$IFACE"; return; }
  ip route show default 2>/dev/null | awk '/^default/ {for(i=1;i<=NF;i++) if($i=="dev") {print $(i+1); exit}}'
}

# ── NIC counters — the instrument this whole tool is built around ────────────
# Emits "rx_packets tx_packets rx_bytes tx_bytes rx_errors tx_errors rx_dropped".
nic_counters() {
  local d="/sys/class/net/$1/statistics"
  if [ -d "$d" ]; then
    printf '%s %s %s %s %s %s %s' \
      "$(cat "$d/rx_packets" 2>/dev/null || echo 0)" \
      "$(cat "$d/tx_packets" 2>/dev/null || echo 0)" \
      "$(cat "$d/rx_bytes"   2>/dev/null || echo 0)" \
      "$(cat "$d/tx_bytes"   2>/dev/null || echo 0)" \
      "$(cat "$d/rx_errors"  2>/dev/null || echo 0)" \
      "$(cat "$d/tx_errors"  2>/dev/null || echo 0)" \
      "$(cat "$d/rx_dropped" 2>/dev/null || echo 0)"
  else
    printf '0 0 0 0 0 0 0'
  fi
}

# ── Inbound traffic freshness ────────────────────────────────────────────────
# Seconds since the last sshd journal entry. The brute-force flood makes this a
# reliable proxy for "a packet from the internet reached us".
sshd_silence_secs() {
  local last
  last="$(journalctl -u "$SSHD_UNIT" -n 1 --no-pager -q -o short-unix 2>/dev/null | awk '{print int($1)}')"
  [ -n "$last" ] || { printf '%s' "-1"; return; }
  printf '%s' "$(( $(now) - last ))"
}

# nginx is an independent second opinion; its access log mtime is enough.
nginx_silence_secs() {
  local newest=0 m f
  for f in $NGINX_ACCESS_GLOB; do
    [ -f "$f" ] || continue
    m="$(stat -c %Y "$f" 2>/dev/null || echo 0)"
    [ "$m" -gt "$newest" ] && newest="$m"
  done
  [ "$newest" -gt 0 ] || { printf '%s' "-1"; return; }
  printf '%s' "$(( $(now) - newest ))"
}

# ── Outbound reachability ────────────────────────────────────────────────────
# Reported for completeness, but read it with the caveat in the header: a failed
# ping does not prove our packets never left.
ping_result() {
  if ping -c 2 -W 2 -n "$1" >/dev/null 2>&1; then printf 'up'; else printf 'DOWN'; fi
}

outbound_up_count() {
  local t n=0
  for t in $PING_TARGETS; do
    ping -c 1 -W 2 -n "$t" >/dev/null 2>&1 && n=$((n+1))
  done
  printf '%s' "$n"
}

# ── Outbound path, hop by hop ────────────────────────────────────────────────
# This box has no mtr or traceroute, and its tracepath build exits 0 printing
# nothing, so we walk the TTL by hand with plain ping. Each hop that expires a
# packet answers with "From <ip> ... Time to live exceeded", which is all a
# traceroute really is.
#
# The first two hops are the ones to watch: 173.208.138.1 is our gateway and
# 173.208.136.129/173.208.136.130 is the provider edge named in the ticket. If
# the path dies at hop 1 during an event the fault is on our own segment; if it
# dies at hop 2 or 3 it is inside their network.
trace_out() {
  local target="$1" max="${2:-12}" ttl out hop rtt
  for ttl in $(seq 1 "$max"); do
    out="$(timeout 5 ping -c 1 -W 2 -n -t "$ttl" "$target" 2>&1)"
    hop="$(printf '%s' "$out" | grep -oE 'From [0-9.]+' | head -1 | awk '{print $2}')"
    rtt="$(printf '%s' "$out" | grep -oE 'time=[0-9.]+ ms' | head -1)"
    if printf '%s' "$out" | grep -qE '^[0-9]+ bytes from '; then
      # The echo reply came back: this is the destination, so we are done.
      hop="$(printf '%s' "$out" | grep -oE '^[0-9]+ bytes from [0-9.]+' | head -1 | awk '{print $4}')"
      printf '    %2d  %-16s %-12s (destination)\n' "$ttl" "${hop:-$target}" "${rtt:-}"
      return 0
    fi
    printf '    %2d  %-16s %s\n' "$ttl" "${hop:-*}" "${rtt:-}"
  done
  return 0
}

# ── Snapshot ─────────────────────────────────────────────────────────────────
snapshot() {
  local label="$1" iface="$2" ts; ts="$(now)"
  echo "────────────────────────────────────────────────────────────────────"
  echo "SNAPSHOT: ${label}"
  echo "  UTC:   $(iso "$ts")"
  echo "  local: $(local_ts "$ts")"
  echo
  echo "  Guest is executing (uptime / load):"
  echo "    $(uptime 2>/dev/null | sed 's/^ *//')"
  echo
  echo "  Interface ${iface}:"
  ip -s link show "$iface" 2>/dev/null | sed 's/^/    /'
  echo
  echo "  Carrier / operstate:"
  echo "    carrier=$(cat /sys/class/net/$iface/carrier 2>/dev/null || echo '?')" \
       "operstate=$(cat /sys/class/net/$iface/operstate 2>/dev/null || echo '?')"
  echo
  echo "  Default route:"
  ip route show default 2>/dev/null | sed 's/^/    /'
  echo
  # A gateway ARP entry going FAILED/STALE mid-event would be a strong clue that
  # the fault is on the local segment rather than further upstream.
  echo "  Neighbour / ARP table (gateway reachability):"
  ip neigh show 2>/dev/null | head -20 | sed 's/^/    /'
  echo
  echo "  Outbound probes:"
  local t
  for t in $PING_TARGETS $EDGE_ROUTER; do
    printf '    %-18s %s\n' "$t" "$(ping_result "$t")"
  done
  echo
  echo "  Inbound freshness:"
  echo "    seconds since last sshd entry:  $(sshd_silence_secs)"
  echo "    seconds since last nginx write: $(nginx_silence_secs)"
  echo
  echo "  Outbound path to ${TRACE_TARGET} (ping TTL walk):"
  trace_out "$TRACE_TARGET" 12
  echo
  echo "  Outbound path to provider edge ${EDGE_ROUTER}:"
  trace_out "$EDGE_ROUTER" 5
  echo
  echo "  Socket summary:"
  ss -s 2>/dev/null | head -6 | sed 's/^/    /'
  echo "────────────────────────────────────────────────────────────────────"
}

# Per-minute inbound rates around a window — the table the provider responds to.
# Takes epoch bounds. nginx stamps its log as "21/Aug/2026:18:03", which is not
# sortable as text, so rather than parse it we generate the exact minute keys the
# window covers and match those literally. Windows here are minutes long, so the
# key list stays small.
inbound_rate_table() {
  local from_ts="$1" to_ts="$2" t keys
  echo "  sshd entries per minute ($(date -d "@$from_ts" +'%H:%M') → $(date -d "@$to_ts" +'%H:%M') local):"
  journalctl -u "$SSHD_UNIT" \
      --since "$(date -d "@$from_ts" +'%Y-%m-%d %H:%M:%S')" \
      --until "$(date -d "@$to_ts" +'%Y-%m-%d %H:%M:%S')" \
      --no-pager -q -o short-iso 2>/dev/null \
    | awk '{print substr($1,12,5)}' | uniq -c \
    | awk '{printf "    %s ... %s\n", $2, $1}'
  echo
  echo "  nginx requests per minute (same window):"
  keys="$(mktemp)"
  for (( t=from_ts; t<=to_ts; t+=60 )); do
    date -d "@$t" +'%d/%b/%Y:%H:%M' >> "$keys"
  done
  cat $NGINX_ACCESS_GLOB 2>/dev/null \
    | grep -oE '[0-9]{2}/[A-Za-z]{3}/[0-9]{4}:[0-9]{2}:[0-9]{2}' \
    | grep -Ff "$keys" 2>/dev/null | sort | uniq -c \
    | awk '{printf "    %s ... %s\n", substr($2,13), $1}'
  rm -f "$keys"
  echo
  echo "  A minute absent from the sshd list had zero entries. This box takes"
  echo "  constant SSH brute-force traffic from hundreds of source IPs, so a run"
  echo "  of zero-entry minutes means nothing on the internet reached the NIC."
}

# ── Mail (mirrors status-agent.sh: same creds, independent of the API) ───────
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
  SUBJECT="$subject" BODY="$body" MAIL_TO="$to" MAIL_FROM="${from:-$user}" \
  S_HOST="$host" S_PORT="$port" S_USER="$user" S_PASS="$pass" \
  python3 - <<'PY' 2>>"$LOG_FILE"
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
            pass
    with srv:
        if user:
            srv.login(user, pw)
        srv.send_message(msg)
except Exception as exc:
    print(f"net-watchdog smtp failed: {exc}", file=sys.stderr)
    sys.exit(1)
PY
}

# A flapping detector must not turn into a mail flood.
mail_allowed() {
  local cutoff recent
  cutoff=$(( $(now) - 3600 ))
  touch "$MAIL_STAMPS" 2>/dev/null
  recent="$(awk -v c="$cutoff" '$1>c' "$MAIL_STAMPS" 2>/dev/null | wc -l)"
  awk -v c="$cutoff" '$1>c' "$MAIL_STAMPS" 2>/dev/null > "${MAIL_STAMPS}.tmp" && mv "${MAIL_STAMPS}.tmp" "$MAIL_STAMPS"
  [ "$recent" -lt "$MAX_MAILS_PER_HOUR" ]
}

# ── Modes ────────────────────────────────────────────────────────────────────
# --snapshot prints one capture and exits; --test also proves the mail path,
# which is the part most likely to be quietly broken when it is finally needed.
case "${1:-}" in
  --snapshot)
    _if="$(detect_iface)"; snapshot "MANUAL SNAPSHOT" "${_if:-lo}"; exit 0 ;;
  --test)
    _if="$(detect_iface)"
    echo "net-watchdog ${VERSION} self-test"
    echo "  interface:      ${_if:-NONE FOUND}"
    echo "  sshd quiet for: $(sshd_silence_secs)s (event threshold ${SILENCE_SECS}s)"
    echo "  nginx quiet:    $(nginx_silence_secs)s"
    echo "  outbound path:  $(trace_out "$TRACE_TARGET" 4 | tr -s ' ' | tr '\n' ' ')"
    echo "  API env file:   ${API_ENV_FILE} $([ -r "$API_ENV_FILE" ] && echo '(readable)' || echo '(NOT READABLE)')"
    echo -n "  sending test email... "
    if send_via_smtp "${SUBJECT_PREFIX} self-test" \
        "net-watchdog ${VERSION} self-test from $(hostname 2>/dev/null) at $(local_ts).\n\nInterface ${_if}. If you are reading this, the blackout report will reach you too."; then
      echo "OK"; exit 0
    else
      echo "FAILED — check SMTP_HOST/SMTP_USER/SMTP_PASS/EMAIL_TO in ${API_ENV_FILE}"; exit 1
    fi ;;
  --help|-h)
    echo "usage: net-watchdog [--snapshot|--test|--help]"; exit 0 ;;
esac

# ── Main loop ────────────────────────────────────────────────────────────────
IFACE="$(detect_iface)"
[ -n "$IFACE" ] || { log "FATAL: no default-route interface found"; exit 1; }
log "net-watchdog ${VERSION} starting: iface=${IFACE} sample=${SAMPLE_SECS}s silence=${SILENCE_SECS}s"

in_event=0
event_start=0
event_file=""
prev_counters="$(nic_counters "$IFACE")"
prev_sample_ts="$(now)"

while true; do
  ts="$(now)"
  printf '%s\n' "$ts" > "$HEARTBEAT" 2>/dev/null   # proof the guest kept executing

  cur_counters="$(nic_counters "$IFACE")"
  read -r c_rxp c_txp c_rxb c_txb c_rxe c_txe c_rxd <<< "$cur_counters"
  read -r p_rxp p_txp p_rxb p_txb p_rxe p_txe p_rxd <<< "$prev_counters"
  d_secs=$(( ts - prev_sample_ts )); [ "$d_secs" -gt 0 ] || d_secs=1
  d_rxp=$(( c_rxp - p_rxp )); d_txp=$(( c_txp - p_txp ))
  d_rxb=$(( c_rxb - p_rxb )); d_txb=$(( c_txb - p_txb ))

  sshd_q="$(sshd_silence_secs)"
  nginx_q="$(nginx_silence_secs)"

  # Compact per-cycle line. During an event these are the readings that show
  # whether we were transmitting into a void.
  sample_line="$(iso "$ts") rx_pkts=+${d_rxp} tx_pkts=+${d_txp} rx_bytes=+${d_rxb} tx_bytes=+${d_txb} over=${d_secs}s sshd_quiet=${sshd_q}s nginx_quiet=${nginx_q}s"

  inbound_dark=0
  [ "$sshd_q" -ge "$SILENCE_SECS" ] 2>/dev/null && inbound_dark=1

  if [ "$in_event" -eq 0 ]; then
    if [ "$inbound_dark" -eq 1 ]; then
      in_event=1
      event_start="$ts"
      event_file="${EVENT_DIR}/event-$(date -u -d "@$ts" +%Y%m%dT%H%M%SZ).txt"
      ob="$(outbound_up_count)"
      {
        echo "NETWORK BLACKOUT DETECTED"
        echo "  detected at:  $(iso "$ts") UTC / $(local_ts "$ts")"
        echo "  host:         $(hostname 2>/dev/null)"
        echo "  public IP:    $(ip -4 route get 8.8.8.8 2>/dev/null | grep -oE 'src [0-9.]+' | awk '{print $2}')"
        echo "  interface:    ${IFACE}"
        echo "  trigger:      no sshd entry for ${sshd_q}s (threshold ${SILENCE_SECS}s)"
        echo "  outbound:     ${ob}/$(echo $PING_TARGETS | wc -w) ping targets responding at detection"
        echo
        snapshot "AT DETECTION" "$IFACE"
        echo
        echo "PER-CYCLE READINGS (every ${SAMPLE_SECS}s from detection):"
      } > "$event_file" 2>/dev/null
      log "EVENT START sshd_quiet=${sshd_q}s outbound=${ob} → ${event_file}"
    fi
  else
    echo "  $sample_line" >> "$event_file" 2>/dev/null
    # Recovery: the internet is reaching us again.
    if [ "$sshd_q" -ge 0 ] && [ "$sshd_q" -lt "$RECOVER_SECS" ] 2>/dev/null; then
      dur=$(( ts - event_start ))
      {
        echo
        echo "RECOVERED"
        echo "  recovered at: $(iso "$ts") UTC / $(local_ts "$ts")"
        echo "  duration:     ${dur}s (~$(( (dur + 30) / 60 )) min)"
        echo
        snapshot "AFTER RECOVERY" "$IFACE"
        echo
        inbound_rate_table "$(( event_start - 300 ))" "$(( ts + 120 ))"
        echo
        echo "HOW TO READ THE PER-CYCLE READINGS"
        echo "  tx_pkts climbing while rx_pkts stays flat means the guest was"
        echo "  transmitting normally and nothing was coming back — traffic"
        echo "  discarded upstream of this VM, not a fault on the VM."
        echo "  Both flat would instead point at the link, driver or hypervisor."
      } >> "$event_file" 2>/dev/null
      log "EVENT END duration=${dur}s"

      if [ "$dur" -ge "$MIN_EVENT_SECS" ]; then
        if mail_allowed; then
          body="$(cat "$event_file" 2>/dev/null)"
          subj="${SUBJECT_PREFIX} blackout $(( (dur + 30) / 60 )) min — $(local_ts "$event_start")"
          if send_via_smtp "$subj" "$body"; then
            printf '%s\n' "$ts" >> "$MAIL_STAMPS"
            log "report emailed (${dur}s event)"
          else
            log "EMAIL FAILED — report retained at ${event_file}"
          fi
        else
          log "mail rate limit hit (${MAX_MAILS_PER_HOUR}/h) — report retained at ${event_file}"
        fi
      else
        log "event too short to email (${dur}s < ${MIN_EVENT_SECS}s) — retained at ${event_file}"
      fi
      in_event=0
      event_file=""
    fi
  fi

  prev_counters="$cur_counters"
  prev_sample_ts="$ts"
  sleep "$SAMPLE_SECS"
done
