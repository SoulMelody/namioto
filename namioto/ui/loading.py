# SPDX-License-Identifier: AGPL-3.0-only
"""A one-shot background job: run a function off the GUI thread and report it or its failure."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QThread, Signal


class LoadingThread(QThread):
    """The shape every loader shares: a file path, a `load()` that produces the result, and one
    `failed` signal for whatever the work throws - a broken file must not take the editor down."""

    failed = Signal(str)

    def __init__(self, path: str | Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)

    def run(self) -> None:
        try:
            self.load()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")

    def load(self) -> None:
        raise NotImplementedError
