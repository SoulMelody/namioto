# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the model registry: where a variant is filed, and the session opened on it."""

from __future__ import annotations

import subprocess
import sys
import zipfile

import pytest

from namioto.analysis import model_store


def test_each_family_keeps_the_directory_it_already_uses(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    assert model_store.path("game", "small") == tmp_path / "namioto" / "models" / "game" / "small"
    assert model_store.path("tempocnn") == tmp_path / "namioto" / "models" / "tempocnn"


def test_the_providers_a_name_stands_for() -> None:
    assert model_store.providers("cpu") == ("CPUExecutionProvider",)
    assert model_store.providers("cuda") == ("CUDAExecutionProvider", "CPUExecutionProvider")
    assert model_store.providers("webgpu") == ("WebGpuExecutionProvider", "CPUExecutionProvider")
    with pytest.raises(ValueError, match="unknown provider"):
        model_store.providers("gpu")


def test_a_session_is_built_with_the_providers_that_were_asked_for(monkeypatch) -> None:
    made: list[tuple[str, list[str]]] = []

    class Session:
        def __init__(self, path, providers=None, **options):
            made.append((path, providers))

    monkeypatch.setattr("onnxruntime.InferenceSession", Session)
    model_store.session("/models/x.onnx", "cuda")
    model_store.session("/models/x.onnx")

    assert made == [
        ("/models/x.onnx", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
        ("/models/x.onnx", ["CPUExecutionProvider"]),
    ]


def test_a_plugin_device_binds_a_session_to_its_device(monkeypatch) -> None:
    class Device:
        def __init__(self, ep_name):
            self.ep_name = ep_name

    class Options:
        def __init__(self):
            self.bound = []

        def add_provider_for_devices(self, devices, options):
            self.bound.append(([device.ep_name for device in devices], options))

    class Session:
        def __init__(self, path, options=None):
            self.options = options

    monkeypatch.setattr(model_store.devices, "installed", lambda: ("WebGpuExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(
        model_store.devices,
        "provider_options",
        lambda name: {"powerPreference": "low-power"} if name == "WebGpuExecutionProvider" else {},
    )
    monkeypatch.setattr(
        "onnxruntime.get_ep_devices",
        lambda: [Device("CPUExecutionProvider"), Device("WebGpuExecutionProvider")],
    )
    monkeypatch.setattr("onnxruntime.SessionOptions", Options)
    monkeypatch.setattr("onnxruntime.InferenceSession", Session)

    session = model_store.session("/models/x.onnx", "webgpu")

    # the CPU goes in as the fallback, and the plugin's device first, carrying its power preference
    assert session.options.bound == [
        (["WebGpuExecutionProvider"], {"powerPreference": "low-power"}),
        (["CPUExecutionProvider"], {}),
    ]


def test_a_session_carries_its_own_options_on_top_of_the_devices(monkeypatch) -> None:
    class Device:
        def __init__(self, ep_name):
            self.ep_name = ep_name

    class Options:
        def __init__(self):
            self.bound = []

        def add_provider_for_devices(self, devices, options):
            self.bound.append(options)

    class Session:
        def __init__(self, path, options=None):
            self.options = options

    monkeypatch.setattr(model_store.devices, "installed", lambda: ("WebGpuExecutionProvider", "CPUExecutionProvider"))
    monkeypatch.setattr(
        model_store.devices,
        "provider_options",
        lambda name: {"powerPreference": "low-power"} if name == "WebGpuExecutionProvider" else {},
    )
    monkeypatch.setattr(
        "onnxruntime.get_ep_devices",
        lambda: [Device("WebGpuExecutionProvider"), Device("CPUExecutionProvider")],
    )
    monkeypatch.setattr("onnxruntime.SessionOptions", Options)
    monkeypatch.setattr("onnxruntime.InferenceSession", Session)

    session = model_store.session("/models/x.onnx", "webgpu", {"forceCpuNodeNames": "/ReduceMean"})

    assert session.options.bound == [
        {"powerPreference": "low-power", "forceCpuNodeNames": "/ReduceMean"},
        {},
    ]


def test_a_plugin_device_that_found_no_gpu_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(model_store.devices, "installed", lambda: ())
    monkeypatch.setattr("onnxruntime.get_ep_devices", lambda: [])
    with pytest.raises(RuntimeError, match="WebGPU"):
        model_store.session("/models/x.onnx", "webgpu")


def test_unpacking_takes_the_folder_holding_the_first_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    archive = tmp_path / "release.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("config.json", "{}")

    target = model_store.unpack("game", archive, ("small",))

    assert target == model_store.path("game", "small")
    assert (target / "config.json").read_text() == "{}"
    assert model_store.installed("game", "small")


def test_nothing_installed_and_no_download_says_what_to_do(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    hinted = model_store.Model(
        name="hinted",
        env="NAMIOTO_HINTED_MODEL",
        files=("model.onnx",),
        hint="convert one with a script",
    )
    monkeypatch.setitem(model_store.MODELS, "hinted", hinted)
    monkeypatch.delenv(hinted.env, raising=False)

    with pytest.raises(FileNotFoundError) as error:
        model_store.resolve("hinted", parts=("mms", "ja"), download=False)

    message = str(error.value)
    assert "no mms/ja model" in message
    assert hinted.env in message
    assert hinted.hint in message


def test_a_model_with_nothing_published_says_what_to_do(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    unpublished = model_store.Model(name="unpublished", env="NAMIOTO_UNPUBLISHED_MODEL", files=("model.onnx",))
    monkeypatch.setitem(model_store.MODELS, "unpublished", unpublished)

    with pytest.raises(FileNotFoundError) as error:
        model_store.resolve("unpublished")

    message = str(error.value)
    assert "no unpublished model" in message
    assert unpublished.env in message


def test_importing_the_store_does_not_reach_for_the_runtime() -> None:
    """The runtime is optional, so the store's own import must not pull it in; a run reaches it."""
    probe = "import sys, namioto.analysis.model_store; assert 'onnxruntime' not in sys.modules"
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
