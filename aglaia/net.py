# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""Outbound HTTP that does not stall on a dead IPv6 route.

Measured on a normal home network, fetching 15 KB from
`raw.githubusercontent.com`::

    DNS -> 8 addresses: four IPv6, then four IPv4
      IPv6 2606:50c0:8001::154 : TimeoutError after 10.00s
      IPv6 2606:50c0:8000::154 : TimeoutError after 10.00s
      IPv6 2606:50c0:8002::154 : TimeoutError after 10.00s
      IPv6 2606:50c0:8003::154 : TimeoutError after 10.00s
      IPv4 185.199.110.133     : connect 0.05s  TLS 0.05s

The router advertises IPv6 and does not route it. `curl` is fast on the same
machine because it does **Happy Eyeballs** — it starts an IPv4 attempt about
200 ms into the IPv6 one and takes whichever answers first. Neither `httpx`
nor `urllib` does anything of the kind: they walk `getaddrinfo`'s list in
order and spend the full connect timeout on every dead address before
reaching a working one. That turned a 15 KB download into 24 seconds, and a
plugin install into a minute of apparent hang.

The fix is one line in two dialects: bind the local socket to an IPv4 address,
which makes the resolver hand back A records only.

**Both entry points fall back.** A machine really can be IPv6-only, and it is
not this module's place to decide otherwise — so an IPv4 attempt that fails is
retried on the default stack. What we refuse to do is spend 24 seconds
discovering that IPv4 was there all along.

## What this is NOT for

Anything on **localhost**. The local VLM server and the phone bridge may
legitimately be reached over `::1`, and forcing IPv4 there would break a
working setup to fix a problem it does not have. Those callers use plain
`urllib` on purpose.
"""

from __future__ import annotations

import http.client
import urllib.request
from typing import Any, Optional

#: Binding the local end to this address makes the stack resolve and connect
#: over IPv4 only. `0.0.0.0` means "any IPv4 interface" — it does not pin a
#: route, it only excludes IPv6.
IPV4_ANY = "0.0.0.0"

#: How long to wait for a *connection* before giving up on one address. Far
#: more than a reachable host needs, far less than a black hole takes to
#: admit itself.
CONNECT_TIMEOUT_S = 6.0


# ── httpx ─────────────────────────────────────────────────────────────

def http_client(timeout: float = 20.0, *, ipv4: bool = True, **kwargs) -> Any:
    """An `httpx.Client` that prefers IPv4. Use it in a `with` block.

    Callers that want the plain behaviour pass ``ipv4=False``; that is also
    what the retry path uses when the IPv4 attempt fails."""
    import httpx
    kwargs.setdefault("follow_redirects", True)
    limits = httpx.Timeout(timeout, connect=CONNECT_TIMEOUT_S)
    transport = None
    if ipv4:
        try:
            transport = httpx.HTTPTransport(local_address=IPV4_ANY)
        except Exception:
            transport = None
    return httpx.Client(timeout=limits, transport=transport, **kwargs)


def http_get(url: str, *, timeout: float = 20.0, **kwargs) -> Any:
    """One GET, IPv4 first, falling back to the default stack.

    For the callers that want a response and not a session."""
    import httpx
    last: Optional[Exception] = None
    for ipv4 in (True, False):
        try:
            with http_client(timeout, ipv4=ipv4, **kwargs) as c:
                return c.get(url)
        except Exception as e:  # noqa: BLE001
            last = e
    raise last if last is not None else RuntimeError("no attempt was made")


# ── urllib ────────────────────────────────────────────────────────────

class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, *a, **kw):
        kw.setdefault("source_address", (IPV4_ANY, 0))
        super().__init__(*a, **kw)


class _IPv4HTTPConnection(http.client.HTTPConnection):
    def __init__(self, *a, **kw):
        kw.setdefault("source_address", (IPV4_ANY, 0))
        super().__init__(*a, **kw)


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req,
                            context=self._context)


class _IPv4HTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(_IPv4HTTPConnection, req)


_opener: Optional[urllib.request.OpenerDirector] = None


def ipv4_opener() -> urllib.request.OpenerDirector:
    global _opener
    if _opener is None:
        _opener = urllib.request.build_opener(_IPv4HTTPSHandler(),
                                              _IPv4HTTPHandler())
    return _opener


def urlopen(req, *, timeout: float = 30.0):
    """`urllib.request.urlopen`, IPv4 first, falling back to the default.

    Same contract as the stdlib call — returns the response object, so a
    caller can keep using it in a `with` block and read `.status`, `.read()`,
    `.headers` exactly as before."""
    try:
        return ipv4_opener().open(req, timeout=timeout)
    except OSError:
        # Includes URLError. An IPv6-only machine lands here, and gets the
        # ordinary path rather than a failure we invented.
        return urllib.request.urlopen(req, timeout=timeout)  # noqa: S310
