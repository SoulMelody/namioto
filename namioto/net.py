# SPDX-License-Identifier: AGPL-3.0-only
"""Outbound requests: the GitHub mirror and the proxy, read from the settings when a call is made.

Every address the program fetches on its own - a release download - goes through here, so the two
settings have one meaning wherever a request is built. `mirrored` puts the
GitHub mirror in front of a github.com address, and `opener` is the `urlopen`-shaped callable the
proxy setting asks for. Reading the settings at call time, rather than once at import, is what lets
a change take effect without a restart. Qt-free on purpose.
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from collections.abc import Callable
from typing import Any

from namioto import settings as store

GITHUB_HOST = "github.com"


def github_mirror() -> str:
    """The configured mirror prefix, empty when releases are fetched from GitHub directly."""
    return store.load().network.github_mirror.strip()


def proxy() -> str:
    """The configured proxy, empty when the environment's own proxies are to apply."""
    return store.load().network.proxy.strip()


def mirrored(url: str, prefix: str | None = None) -> str:
    """`url` through the GitHub mirror, which is a prefix put in front of the original address.

    Only a github.com address is rewritten: the mirror is a GitHub one, and what it serves redirects
    to wherever the release asset itself lives on its own.
    """
    base = github_mirror() if prefix is None else prefix.strip()
    if not base or urllib.parse.urlsplit(url).hostname != GITHUB_HOST:
        return url
    return f"{base.rstrip('/')}/{url}"


def opener(address: str | None = None) -> Callable[..., Any]:
    """A `urlopen`-shaped callable that sends its requests through the proxy setting.

    An empty setting leaves the standard `urlopen`, which already honours the environment's own
    `http_proxy` / `https_proxy`; a proxy is given to urllib as an HTTP one, with the scheme added
    when a bare host and port were typed.
    """
    value = proxy() if address is None else address.strip()
    if not value:
        return urllib.request.urlopen
    if "://" not in value:
        value = f"http://{value}"
    handlers = urllib.request.ProxyHandler({"http": value, "https": value})
    return urllib.request.build_opener(handlers).open
