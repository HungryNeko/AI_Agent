"""Keep agent code away from the backend's own HTTP API.

The API has no login of its own (in the Docker deployment the lease gateway authenticates people before
they reach it). Code the agent runs lives in the same container, so it could call the API directly and
make the privileged backend do things the code itself may not: change provider settings, add MCP servers,
change the security level. Every request whose TCP connection belongs to the unprivileged `agentcode`
user is therefore refused. Requests from the gateway or nginx come from another container and are not
affected, and neither are the agent code's calls to third-party services: only the backend's own port is
guarded.

The connection's owner is found in the kernel's socket table (/proc/net/tcp and tcp6), which lists the
user id of both ends of a connection inside this container.
"""

from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from typing import Any

PROC_NET = Path("/proc/net")
_LOCAL_IPS: set[str] | None = None


def local_ips() -> set[str]:
    global _LOCAL_IPS
    if _LOCAL_IPS is None:
        found = {"127.0.0.1", "::1"}
        try:
            found.update(socket.gethostbyname_ex(socket.gethostname())[2])
        except OSError:
            pass
        _LOCAL_IPS = found
    return _LOCAL_IPS


def _proc_hex(ip: str, port: int) -> list[str]:
    """How /proc/net/tcp(6) spells an address: 32-bit words in host (little-endian) byte order."""

    address = ipaddress.ip_address(ip)
    candidates: list[str] = []
    forms = [address.packed]
    if address.version == 4:
        forms.append(b"\x00" * 10 + b"\xff\xff" + address.packed)  # the same peer as seen on an IPv6 socket
    for packed in forms:
        words = [packed[i : i + 4][::-1].hex().upper() for i in range(0, len(packed), 4)]
        candidates.append("".join(words) + f":{port:04X}")
    return candidates


def _rows(path: Path) -> list[list[str]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()[1:]
    except OSError:
        return []
    return [line.split() for line in lines if line.strip()]


def peer_uid(client: tuple[str, int] | None, server: tuple[str, int] | None, proc_root: Path = PROC_NET) -> int | None:
    """User id that owns the client end of this connection, or None when it is not a local connection."""

    if not client or not server:
        return None
    try:
        local = _proc_hex(client[0], int(client[1]))
        remote = _proc_hex(server[0], int(server[1]))
    except (ValueError, TypeError):
        return None
    for name in ("tcp", "tcp6"):
        for row in _rows(proc_root / name):
            # sl local_address rem_address st tx:rx tr:when retrnsmt uid ...
            if len(row) > 7 and row[1] in local and row[2] in remote:
                try:
                    return int(row[7])
                except ValueError:
                    return None
    return None


def should_refuse(client: tuple[str, int] | None, server: tuple[str, int] | None, low_uid: int | None, proc_root: Path = PROC_NET) -> bool:
    if low_uid is None or not client:
        return False
    if client[0] not in local_ips():
        return False  # another container (the gateway): nothing to look up
    return peer_uid(client, server, proc_root) == low_uid


def refusal(scope: dict[str, Any]) -> bool:
    from tools import security

    info = security.low_user()
    if info is None:
        return False
    return should_refuse(scope.get("client"), scope.get("server"), info[0])
