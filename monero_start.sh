#!/usr/bin/env bash
# Start monero-wallet-rpc for Boonie's wallet.
# Wallet dir is always relative to this script, regardless of cwd.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALLET_DIR="$SCRIPT_DIR/wallet"
WALLET_FILE="$WALLET_DIR/boonie"
RPC_PORT=18082

# Daemon and proxy are env-configurable so .secrets can override per-host.
# Set MONERO_DAEMON to a .onion and MONERO_PROXY=127.0.0.1:9050 (or 'tor')
# to route the daemon connection entirely through Tor.
DAEMON="${MONERO_DAEMON:-node.community.rino.io:18081}"
PROXY="${MONERO_PROXY:-}"

PROXY_ARG=()
if [ -n "$PROXY" ]; then
    PROXY_ARG=(--proxy "$PROXY")
fi

mkdir -p "$WALLET_DIR"

if [ ! -f "$WALLET_FILE" ]; then
    echo "[monero] Creating new wallet at $WALLET_FILE ..."
    monero-wallet-cli \
        --generate-new-wallet "$WALLET_FILE" \
        --mnemonic-language English \
        --password "" \
        --daemon-address "$DAEMON" \
        "${PROXY_ARG[@]}" \
        --command "exit"
    # Save address to a plain text file so it's readable without a daemon
    monero-wallet-cli \
        --wallet-file "$WALLET_FILE" \
        --password "" \
        --daemon-address "$DAEMON" \
        "${PROXY_ARG[@]}" \
        --command "address" 2>/dev/null \
        | grep -oP '(?<=\d: )[0-9A-Za-z]{95}' \
        > "$WALLET_DIR/address.txt"
fi

exec monero-wallet-rpc \
    --wallet-file "$WALLET_FILE" \
    --password "" \
    --rpc-bind-port "$RPC_PORT" \
    --rpc-bind-ip 127.0.0.1 \
    --disable-rpc-login \
    --daemon-address "$DAEMON" \
    "${PROXY_ARG[@]}" \
    --log-level 0 \
    2>/dev/null
