"""Reachability checks for remote catalogues.

This is internet connectivity, separate from ``geolattice.spatial_analysis`` routing.
"""

from __future__ import annotations


def is_online(url: str | None = None, timeout: float = 8) -> bool:
    """Return True when ``url`` answers within ``timeout`` seconds.

    With no URL, the check uses the Planetary Computer STAC catalogue. A failure
    (timeout, DNS, HTTP error) returns False.
    """
    import urllib.request

    target = url or "https://planetarycomputer.microsoft.com/api/stac/v1"
    try:
        urllib.request.urlopen(target, timeout=timeout)
        return True
    except Exception:
        return False


__all__ = ["is_online"]
