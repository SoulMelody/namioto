# SPDX-License-Identifier: AGPL-3.0-only
"""Where the ONNX models come from: one registry, one order of search, one downloader.

The program ships no model: every one of them is fetched into the data directory on demand. A family
keeps its own keys and its own files - GAME a size, TempoCNN none - while the order they are looked
up in, the download that puts them there and the session opened on them are the same for all of
them. A family is looked up as an explicit path, its environment variable, an installed copy under
`models/`, then the release it names; the first file of a family is what a downloaded package is
recognized by.

This module does not import ONNX Runtime of its own: the runtime is optional and is reached only
when a session is opened, so importing this module never pulls it into the GUI's startup path. The
address a package comes from goes through `namioto.net`, so the GitHub mirror and the proxy settings
apply to every download.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from namioto import net
from namioto.analysis import devices
from namioto.utils import data_dir

if TYPE_CHECKING:
    import onnxruntime as ort

# every device the program can offer, with the provider list each one runs with
PROVIDERS = devices.PROVIDERS


@dataclass(frozen=True)
class Model:
    """One family of models: where its copies are filed, what a whole copy holds, where it is published.

    The first name in `files` is also the one a downloaded package is recognized by: the folder
    holding it is what gets moved into place. `directory` is the segment the family is filed under,
    empty for the one whose models are the whole of `models/` - a copy already there has to keep
    being found where it is.
    """

    name: str
    env: str
    files: tuple[str, ...]
    directory: str = ""
    asset: str = ""  # the release URL of one variant, once one is published; `{model}` is its key
    hint: str = ""  # what to tell the user instead, while nothing is published to fetch

    def url(self, parts: tuple[str, ...]) -> str:
        """Where one variant is published: the entry's URL with the first key filled in."""
        return self.asset.format(model=parts[0] if parts else "")


MODELS: dict[str, Model] = {
    "game": Model(
        name="game",
        env="NAMIOTO_GAME_MODEL",
        files=("config.json",),
        directory="game",
        asset="https://github.com/openvpi/GAME/releases/download/v1.0.3/GAME-1.0.3-{model}-onnx.zip",
    ),
    "tempocnn": Model(
        name="tempocnn",
        env="NAMIOTO_TEMPOCNN_MODEL",
        files=("deeptemp-k16-3.onnx",),
        directory="tempocnn",
        asset="https://github.com/sitiyou/namioto/releases/download/models/tempocnn-onnx.zip",
    ),
}


def path(name: str, *parts: str) -> pathlib.Path:
    """Where one variant of a model is filed, whether or not it is installed there."""
    model = MODELS[name]
    return data_dir("models", *((model.directory, *parts) if model.directory else parts))


def installed(name: str, *parts: str) -> bool:
    """Whether the copy filed there is whole."""
    target = path(name, *parts)
    return all((target / file).is_file() for file in MODELS[name].files)


def resolve(
    name: str,
    explicit: str | pathlib.Path | None = None,
    *,
    parts: tuple[str, ...] = (),
    download: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> pathlib.Path:
    """The directory holding one variant of a model, which must already hold it.

    An explicit path or the family's own environment variable wins, then an installed copy, and only
    then the release it is published in; a path naming a file is read as living in the directory
    beside the rest. Nothing here trusts a directory: whatever is settled on has to hold the model,
    so a typo is refused with what is missing rather than left to fail inside ONNX Runtime.
    """
    model = MODELS[name]
    for candidate in (explicit, os.environ.get(model.env) or None):
        if candidate is not None:
            return _whole(model, pathlib.Path(candidate))
    target = path(name, *parts)
    if installed(name, *parts):
        return target
    if download and model.asset:
        return install(name, parts, progress)
    raise _missing(model, parts, target)


def install(
    name: str,
    parts: tuple[str, ...] = (),
    progress: Callable[[int, int], None] | None = None,
    *,
    opener: Callable[..., object] | None = None,
) -> pathlib.Path:
    """Fetch one variant into the data directory and unpack it there.

    The zip is downloaded to a temporary file first, so a broken connection leaves no half-installed
    model behind, and the package is moved into place only once it is whole.
    """
    model = MODELS[name]
    target = path(name, *parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    open_url = opener or net.opener()
    with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".zip", delete=False) as handle:
        archive = pathlib.Path(handle.name)
    try:
        with open_url(net.mirrored(model.url(parts))) as response, archive.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                block = response.read(1 << 20)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if progress is not None:
                    progress(done, total)
        return unpack(name, archive, parts)
    finally:
        archive.unlink(missing_ok=True)


def unpack(name: str, archive: pathlib.Path, parts: tuple[str, ...] = ()) -> pathlib.Path:
    """Unpack a downloaded release into its place, top-level folder and all."""
    model = MODELS[name]
    target = path(name, *parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target.parent) as staging:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(staging)
        source = next((path.parent for path in pathlib.Path(staging).rglob(model.files[0])), None)
        if source is None:
            raise ValueError(f"{archive.name} holds no {model.files[0]}: not a {model.name} package")
        target.mkdir(parents=True, exist_ok=True)
        for entry in source.iterdir():
            shutil.move(str(entry), str(target / entry.name))
    return target


def providers(key: str) -> tuple[str, ...]:
    """The ONNX Runtime providers a name stands for."""
    if key not in PROVIDERS:
        raise ValueError(f"unknown provider {key!r}, expected one of {', '.join(PROVIDERS)}")
    return PROVIDERS[key]


def session(
    source: str | pathlib.Path, provider: str = "cpu", options: dict[str, str] | None = None
) -> ort.InferenceSession:
    """A session over one ONNX file, on the device a name stands for.

    `options` are provider options this one session opens with, on top of the device's own; a device
    that runs through a built-in provider takes none.
    """
    import onnxruntime as ort  # the runtime is optional; reach it only where a session is opened

    device = devices.get(provider)
    if device.plugin:
        return _plugin_session(ort, device, source, options)
    return ort.InferenceSession(str(source), providers=list(device.providers))


def _plugin_session(
    ort, device: devices.Device, source: str | pathlib.Path, options: dict[str, str] | None = None
) -> ort.InferenceSession:
    """A session on a plugin execution provider, which ONNX Runtime reaches only through a device.

    The plugin's library has to be registered first, which `installed` does, and the session is bound
    to one of the devices the runtime found for it; naming the provider instead would leave the work
    on the CPU without saying so.
    """
    devices.installed()
    found: dict[str, object] = {}
    for entry in ort.get_ep_devices():
        found.setdefault(entry.ep_name, entry)
    if device.providers[0] not in found:
        raise RuntimeError(f"no {device.label} device found; check that its driver is installed")
    session_options = ort.SessionOptions()
    for index, name in enumerate(device.providers):
        if name in found:
            opened = devices.provider_options(name)
            if index == 0:
                opened.update(options or {})
            session_options.add_provider_for_devices([found[name]], opened)
    return ort.InferenceSession(str(source), session_options)


def _whole(model: Model, candidate: pathlib.Path) -> pathlib.Path:
    """`candidate` as the directory holding the model, once it holds all of it."""
    directory = candidate if candidate.is_dir() else candidate.parent
    missing = [file for file in model.files if not (directory / file).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} does not hold {', '.join(missing)}")
    return directory


def _missing(model: Model, parts: tuple[str, ...], target: pathlib.Path) -> FileNotFoundError:
    """Nothing is installed and nothing is published: say what to do about it."""
    label = "/".join(parts) or model.name
    hint = f"; {model.hint}" if model.hint else ""
    return FileNotFoundError(
        f"no {label} model in {target}{hint}; set {model.env} to a directory holding {', '.join(model.files)}"
    )
