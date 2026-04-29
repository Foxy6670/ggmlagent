"""
Monero wallet integration.

Primary: monero-wallet-rpc on localhost:18082 (start manually with monero_start.sh).
Fallback for address: wallet/address.txt (written on wallet creation).
Fallback for balance: xmrchain.net API via Tor (output count only — amounts need daemon).

Web traffic goes through Tor (socks5h://127.0.0.1:9050).
"""

from pathlib import Path
import requests

RPC_URL       = "http://127.0.0.1:18082/json_rpc"
PICONERO      = 1e12
_ADDRESS_FILE = Path(__file__).parent / "wallet" / "address.txt"
_HEIGHT_FILE  = Path(__file__).parent / "wallet" / "creation_height.txt"
_TOR          = {"http": "socks5h://127.0.0.1:9050", "https": "socks5h://127.0.0.1:9050"}
_EXPLORER     = "https://xmrchain.net"


def _rpc(method: str, params: dict | None = None) -> dict:
    try:
        resp = requests.post(
            RPC_URL,
            json={"jsonrpc": "2.0", "id": "0", "method": method, "params": params or {}},
            timeout=10,
        )
        data = resp.json()
    except Exception as e:
        raise ConnectionError(f"RPC unavailable: {e}")
    if "error" in data:
        raise RuntimeError(data["error"]["message"])
    return data["result"]


def _saved_address() -> str:
    if _ADDRESS_FILE.exists():
        return _ADDRESS_FILE.read_text().strip()
    return ""


def address() -> str:
    # Try live RPC first
    try:
        result = _rpc("get_address", {"account_index": 0})
        return f"[wallet] Address: {result['address']}"
    except Exception:
        pass
    # Fall back to saved file
    addr = _saved_address()
    if addr:
        return f"[wallet] Address: {addr}\n[wallet] (daemon offline — address read from wallet file)"
    return "[wallet] Address unavailable — run monero_start.sh to create wallet."


def balance() -> str:
    # Try live RPC
    try:
        result = _rpc("get_balance", {"account_index": 0})
        total    = result["balance"] / PICONERO
        unlocked = result["unlocked_balance"] / PICONERO
        return f"[wallet] Balance: {total:.6f} XMR ({unlocked:.6f} unlocked)"
    except Exception:
        pass

    # Fallback: count received outputs via xmrchain.net
    addr = _saved_address()
    if not addr:
        return "[wallet] Balance unavailable — daemon offline and no saved address."

    try:
        start_height = int(_HEIGHT_FILE.read_text().strip()) if _HEIGHT_FILE.exists() else 0
        current = requests.get(
            f"{_EXPLORER}/api/emission",
            proxies=_TOR, timeout=15,
        ).json()["data"]["blk_no"]

        # Scan up to 500 recent blocks for outputs belonging to this address
        # (amounts need daemon — we can only count outputs here)
        output_count = 0
        scan_from = max(start_height, current - 500)
        for height in range(scan_from, current, 5):
            block = requests.get(
                f"{_EXPLORER}/api/block/{height}",
                proxies=_TOR, timeout=10,
            ).json().get("data", {})
            for tx in block.get("txs", []):
                tx_data = requests.get(
                    f"{_EXPLORER}/api/transaction/{tx['tx_hash']}",
                    proxies=_TOR, timeout=10,
                ).json().get("data", {})
                # Can only check output count; amounts require daemon for ecdhInfo decryption
                for out in tx_data.get("outputs", []):
                    if out.get("amount", 0) > 0:  # only visible for non-RingCT txs
                        output_count += 1

        note = f"({output_count} transparent outputs found in last 500 blocks)" if output_count else "(no transparent outputs found — daemon needed for RingCT amounts)"
        return (
            f"[wallet] Balance: unknown — daemon offline {note}\n"
            f"[wallet] View on explorer: {_EXPLORER}/search?value={addr}"
        )
    except Exception as e:
        addr = _saved_address()
        return (
            f"[wallet] Balance: unknown — daemon offline, explorer unreachable ({e})\n"
            f"[wallet] Address: {addr}"
        )


def send(dest_address: str, amount_xmr: float) -> str:
    try:
        piconero = int(amount_xmr * PICONERO)
        result = _rpc("transfer", {
            "destinations": [{"amount": piconero, "address": dest_address}],
            "account_index": 0,
            "priority": 1,
            "ring_size": 16,
            "get_tx_key": True,
        })
        txid = result.get("tx_hash", "?")
        fee  = result.get("fee", 0) / PICONERO
        return f"[wallet] Sent {amount_xmr} XMR. TX: {txid} (fee: {fee:.6f} XMR)"
    except ConnectionError:
        return "[wallet] Send failed — daemon offline. Start monero_start.sh first."
    except RuntimeError as e:
        return f"[wallet] Send failed: {e}"
