# SPDX-License-Identifier: AGPL-3.0-only
"""A Qwen checkpoint in the data directory: where it is filed, and how a first use fetches it.

Both models this program asks Qwen for are one pinned snapshot of a few files, fetched on demand
from the Hugging Face endpoint the network settings name. What differs is only the repository, the
revision and the file names, so those are a `Checkpoint` and the rest is here once.

Qt-free on purpose, and free of `huggingface_hub` until `install` runs: importing this module never
pulls a downloader into the editor's startup path.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from dataclasses import dataclass

from namioto import net
from namioto.utils import data_dir

# both models come from the same package, so one sentence names what to install for either
MISSING = "qwen-asr is not installed; run `pip install namioto[pytorch-cpu]` (or namioto[ort-cuda])"


@dataclass(frozen=True)
class Checkpoint:
    """One Hugging Face work: the repository to ask, the revision to pin, and the files read.

    `directory` is the name it is filed under inside the data root's `models/`, so two works never
    share a folder; `files` is what `from_pretrained` reads, and a directory is a whole copy only
    when every one of them is there, because a half-fetched snapshot is the usual mistake.
    """

    repo: str
    revision: str
    files: tuple[str, ...]
    directory: str

    @property
    def label(self) -> str:
        """The model's own name, for the sentence that refuses a directory."""
        return self.repo.rsplit("/", 1)[-1]

    def model_dir(self) -> pathlib.Path:
        """Where the weights are filed, whether or not they have been fetched yet."""
        return data_dir("models", self.directory)

    def is_installed(self, directory: pathlib.Path | None = None) -> bool:
        target = self.model_dir() if directory is None else pathlib.Path(directory)
        return all((target / name).is_file() for name in self.files)

    def resolve(
        self,
        explicit: str | pathlib.Path | None = None,
        *,
        download: bool = True,
        progress: Callable[[int, int], None] | None = None,
    ) -> pathlib.Path:
        """The directory to load from: an explicit one, the one here, or a fetch of the revision."""
        if explicit:
            return self.whole(pathlib.Path(explicit))
        target = self.model_dir()
        if self.is_installed(target):
            return target
        if not download:
            raise FileNotFoundError(f"no {self.label} weights in {target}")
        return self.install(progress)

    def install(self, progress: Callable[[int, int], None] | None = None) -> pathlib.Path:
        """Fetch the pinned revision through the endpoint in force, one snapshot into `model_dir()`.

        `huggingface_hub` reports no byte progress for a snapshot, so `progress` is told nothing
        rather than given a bar that would not move.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as error:  # the window turns this into `MISSING`
            raise ModuleNotFoundError(MISSING) from error
        target = self.model_dir()
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=self.repo,
            revision=self.revision,
            local_dir=str(target),
            allow_patterns=list(self.files),
            endpoint=net.hf_endpoint(),
        )
        if progress is not None:
            progress(len(self.files), len(self.files))
        return self.whole(target)

    def whole(self, directory: pathlib.Path) -> pathlib.Path:
        """`directory` if it holds every file, refused by name otherwise."""
        missing = [name for name in self.files if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"{directory} is not a {self.label} checkpoint: missing {', '.join(missing)}")
        return directory
