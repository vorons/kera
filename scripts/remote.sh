#!/bin/bash
# Reaches the owner's Windows machine from a cloud session, over the tailnet (docs/DECISIONS.md D-005).
#
#   scripts/remote.sh up                  join the tailnet: Tailscale in userspace mode with its state in memory only,
#                                         using $TS_AUTHKEY, or else printing a login link for the owner to approve
#   scripts/remote.sh run COMMAND         run a PowerShell command there
#   scripts/remote.sh ps SCRIPT [ARGS]    copy a PowerShell script to the Kera folder's "compare" folder and run it
#   scripts/remote.sh put FILE [DIR]      copy a file there (DIR is relative to the user's home folder)
#   scripts/remote.sh get PATH [DIR]      copy a file back
#
# Needs $KERA_SSH_TARGET (user@address, the machine's Tailscale name or address) and a login key, $KERA_SSH_KEY (a
# file) or $KERA_SSH_KEY_B64. Tailscale is downloaded to $KERA_TS_DIR (default ~/.cache/kera-tailscale).
set -euo pipefail
TSD=${KERA_TS_DIR:-$HOME/.cache/kera-tailscale}
REMOTE_DIR="AppData/Local/Kera/compare"   # the Kera folder's compare folder, from the user's home folder
mkdir -p "$TSD" && chmod 700 "$TSD"

ts() { "$(ls -d "$TSD"/tailscale_*/ | sort -V | tail -1)tailscale" --socket="$TSD/tsd.sock" "$@"; }

if [ "${1:-}" = up ]; then
    if ! ls -d "$TSD"/tailscale_*/ >/dev/null 2>&1; then
        curl -sSfL "https://pkgs.tailscale.com/stable/tailscale_latest_amd64.tgz" | tar xz -C "$TSD"
    fi
    st=$(ts status 2>&1 || true)   # (not a pipe: with pipefail, status's own failure would hide grep's match)
    if [[ $st == *"failed to connect"* ]]; then
        "$(ls -d "$TSD"/tailscale_*/ | sort -V | tail -1)tailscaled" --tun=userspace-networking --state=mem: \
            --socket="$TSD/tsd.sock" --statedir="$TSD/state" >"$TSD/tailscaled.log" 2>&1 &
        sleep 4
    fi
    if [ -n "${TS_AUTHKEY:-}" ]; then
        ts up --auth-key="$TS_AUTHKEY" --hostname=kera-cloud-agent --accept-dns=false
    else
        # waits in the background for the owner to open the link
        nohup "$(ls -d "$TSD"/tailscale_*/ | sort -V | tail -1)tailscale" --socket="$TSD/tsd.sock" up \
            --hostname=kera-cloud-agent --accept-dns=false >"$TSD/up.log" 2>&1 &
        sleep 8
        grep -o "https://login.tailscale.com/[^ ]*" "$TSD/up.log" | head -1 || true
    fi
    ts status | head -20 || true
    exit 0
fi

: "${KERA_SSH_TARGET:?set KERA_SSH_TARGET to user@machine}"
KEY=${KERA_SSH_KEY:-$TSD/ssh-key}
if [ -z "${KERA_SSH_KEY:-}" ]; then
    : "${KERA_SSH_KEY_B64:?set KERA_SSH_KEY or KERA_SSH_KEY_B64}"
    (umask 077 && echo "$KERA_SSH_KEY_B64" | base64 -d >"$KEY")
fi
O=(-i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=90 -o ServerAliveInterval=30
   -o "ProxyCommand=$(ls -d "$TSD"/tailscale_*/ | sort -V | tail -1)tailscale --socket=$TSD/tsd.sock nc %h %p"
   -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=$TSD/known_hosts")

# Windows' SSH server starts cmd.exe or PowerShell, depending on its settings. A command sent as
# powershell -EncodedCommand (UTF-16, base64) works under either, and no shell on the way changes its quotes.
psrun() {
    local e
    e=$(printf '%s' "\$ProgressPreference = 'SilentlyContinue'; $1" | iconv -f UTF-8 -t UTF-16LE | base64 -w0)
    # PowerShell adds its progress records to stderr as a "#< CLIXML" block: leave them out
    ssh "${O[@]}" "$KERA_SSH_TARGET" "powershell -NoProfile -ExecutionPolicy Bypass -EncodedCommand $e" \
        2> >(grep -v -e '^#< CLIXML' -e '^<Objs' >&2)
}

case "${1:-}" in
    run) shift; psrun "$*" ;;
    ps)
        shift; f=$1; shift
        psrun "New-Item -ItemType Directory -Force \"\$env:LOCALAPPDATA\\Kera\\compare\" | Out-Null"
        scp -q "${O[@]}" "$f" "$KERA_SSH_TARGET:$REMOTE_DIR/$(basename "$f")"
        c="& \"\$env:LOCALAPPDATA\\Kera\\compare\\$(basename "$f")\""
        for a in "$@"; do
            if [[ $a =~ ^-[A-Za-z]+$ ]]; then c+=" $a"; else c+=" '${a//\'/\'\'}'"; fi
        done
        psrun "$c"
        ;;
    put) shift; scp "${O[@]}" "$1" "$KERA_SSH_TARGET:${2:-$REMOTE_DIR/}" ;;
    get) shift; scp "${O[@]}" "$KERA_SSH_TARGET:$1" "${2:-.}" ;;
    *) sed -n '2,13p' "$0"; exit 2 ;;
esac
