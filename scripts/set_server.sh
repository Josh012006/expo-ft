#!/usr/bin/env bash

# Create a reverse SSH tunnel
#
# Usage:
#   bash scripts/set_server.sh --node-ip IP --port PORT --username USER
#     --node-ip   address of the node to tunnel to (e.g. .XXX)
#     --port      port forwarded from the node to localhost
#     --username  ssh user on that node

usage() {
    cat <<'EOF'
Usage: bash scripts/set_server.sh --node-ip IP --port PORT --username USER
  --node-ip   address of the node to tunnel to (e.g. .XXX)
  --port      port forwarded from the node to localhost
  --username  ssh user on that node
EOF
}
need_value() {   # need_value <flag> <number of remaining args> <next arg>
    if [[ $2 -lt 2 || "$3" == --* ]]; then echo "ERROR: $1 needs a value" >&2; exit 1; fi
}

NODE1_IP=""
PORT=""
USERNAME=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --node-ip) need_value "$1" "$#" "${2-}"; NODE1_IP="$2"; shift 2 ;;
        --port) need_value "$1" "$#" "${2-}"; PORT="$2"; shift 2 ;;
        --username) need_value "$1" "$#" "${2-}"; USERNAME="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done
if [[ -z "$NODE1_IP" ]]; then echo "ERROR: --node-ip is required" >&2; usage >&2; exit 1; fi
if [[ -z "$PORT" ]]; then echo "ERROR: --port is required" >&2; usage >&2; exit 1; fi
if [[ -z "$USERNAME" ]]; then echo "ERROR: --username is required" >&2; usage >&2; exit 1; fi

ssh -N -R "${PORT}:localhost:${PORT}" "${USERNAME}@${NODE1_IP}"
