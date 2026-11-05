# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the network settings: the GitHub mirror and the proxy opener."""

from __future__ import annotations

import urllib.request

import pytest

from namioto import net
from namioto import settings as store


@pytest.fixture
def settings_file(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(path))
    return path


def test_the_github_mirror_defaults_to_gh_proxy(settings_file) -> None:
    assert net.github_mirror() == "https://gh-proxy.org"
    assert net.proxy() == ""


def test_a_github_address_is_prefixed_with_the_mirror() -> None:
    assert net.mirrored("https://github.com/a/b.zip", "https://gh-proxy.org") == (
        "https://gh-proxy.org/https://github.com/a/b.zip"
    )


def test_a_mirror_with_a_trailing_slash_is_still_a_prefix() -> None:
    assert net.mirrored("https://github.com/a/b.zip", "https://gh-proxy.org/") == (
        "https://gh-proxy.org/https://github.com/a/b.zip"
    )


def test_only_a_github_address_goes_through_the_mirror() -> None:
    assert net.mirrored("https://example.com/a/b.zip", "https://gh-proxy.org") == "https://example.com/a/b.zip"


def test_an_empty_mirror_leaves_the_github_address_alone() -> None:
    assert net.mirrored("https://github.com/a/b.zip", "") == "https://github.com/a/b.zip"


def test_the_mirror_and_the_proxy_come_from_the_settings(settings_file) -> None:
    settings = store.Settings()
    store.set_value(settings, "network", "github_mirror", "https://mirror.example/")
    store.set_value(settings, "network", "proxy", "127.0.0.1:7890")
    store.save(settings)

    assert net.github_mirror() == "https://mirror.example/"
    assert net.proxy() == "127.0.0.1:7890"
    assert net.mirrored("https://github.com/a/b.zip") == "https://mirror.example/https://github.com/a/b.zip"


def test_the_hugging_face_endpoint_defaults_to_the_mirror(settings_file, monkeypatch) -> None:
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    assert net.hf_endpoint() == "https://hf-mirror.com"


def test_an_empty_endpoint_falls_back_to_the_environment_then_the_site(settings_file, monkeypatch) -> None:
    settings = store.Settings()
    store.set_value(settings, "network", "hf_endpoint", "")
    store.save(settings)
    monkeypatch.setenv("HF_ENDPOINT", "https://mirror.example/hf/")

    assert net.hf_endpoint() == "https://mirror.example/hf"
    assert net.hf_url("org/model", "rev", "model.safetensors") == (
        "https://mirror.example/hf/org/model/resolve/rev/model.safetensors"
    )

    monkeypatch.delenv("HF_ENDPOINT")
    assert net.hf_endpoint() == "https://huggingface.co"
    assert net.opener("") is urllib.request.urlopen


def test_a_proxy_scheme_is_added_when_it_was_left_out(monkeypatch) -> None:
    seen = []

    class Opener:
        def open(self, *args, **kwargs):
            return None

    monkeypatch.setattr(urllib.request, "build_opener", lambda handler: seen.append(handler.proxies) or Opener())
    net.opener("127.0.0.1:7890")

    assert seen == [{"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"}]


def test_a_proxy_that_names_its_scheme_is_left_as_it_is(monkeypatch) -> None:
    seen = []

    class Opener:
        def open(self, *args, **kwargs):
            return None

    monkeypatch.setattr(urllib.request, "build_opener", lambda handler: seen.append(handler.proxies) or Opener())
    net.opener("https://proxy.example:8080")

    assert seen == [{"http": "https://proxy.example:8080", "https": "https://proxy.example:8080"}]
