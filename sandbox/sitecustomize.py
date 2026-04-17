"""Network guard installed at Python startup inside the sandbox image.

Container network isolation gives us a fresh netns so ``localhost`` is the
container itself, but the link-local cloud-metadata address 169.254.169.254
is still reachable on cloud hosts. We cannot drop NET_ADMIN-gated iptables
rules (we run with ``--cap-drop=ALL``), so we filter at the Python socket
layer instead. Anything using ``socket.getaddrinfo`` or ``socket.create_connection``
(which is everything: requests, urllib, httpx, etc.) is covered.

A sufficiently creative payload can bypass this by calling libc ``connect(2)``
directly through ctypes. Defense in depth, not a boundary.
"""

from __future__ import annotations

import ipaddress
import socket

_BLOCKED_NETS = (
    ipaddress.ip_network("169.254.0.0/16"),        # link-local (cloud metadata)
    ipaddress.ip_network("fe80::/10"),              # IPv6 link-local
    ipaddress.ip_network("::ffff:169.254.0.0/112"),  # v4-mapped-in-v6 link-local
)


def _is_blocked(ip_str: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(addr in net for net in _BLOCKED_NETS)


_orig_getaddrinfo = socket.getaddrinfo
_orig_create_connection = socket.create_connection


def _guarded_getaddrinfo(host, *args, **kwargs):
    results = _orig_getaddrinfo(host, *args, **kwargs)
    for family, _type, _proto, _canon, sockaddr in results:
        ip = sockaddr[0]
        if _is_blocked(ip):
            raise OSError(f"sandbox: blocked connect to metadata/link-local address {ip}")
    return results


def _guarded_create_connection(address, *args, **kwargs):
    host = address[0]
    if _is_blocked(host):
        raise OSError(f"sandbox: blocked connect to metadata/link-local address {host}")
    return _orig_create_connection(address, *args, **kwargs)


socket.getaddrinfo = _guarded_getaddrinfo
socket.create_connection = _guarded_create_connection
