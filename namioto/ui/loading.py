# SPDX-License-Identifier: AGPL-3.0-only
"""A one-shot background job, and the only thread the interface starts itself.

The work runs off the GUI thread with the cyclic collector's existing heap frozen. A collection in a
worker thread is free to destroy anything whose last reference the GUI has dropped, and a Qt object
may only be destroyed on the thread that owns it: freeing a window from here leaves its timers
pointing into a timer list the GUI thread is still walking, which is a segmentation fault, not an
exception. Freezing makes everything that already exists off-limits for the length of the job, so a
collection here can only reclaim what the job itself allocated.
"""

from __future__ import annotations

import gc
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class BackgroundThread(QThread):
    """The shape every background job shares: a `job()` off the GUI thread, and one `failed` signal
    for whatever it throws - a broken file must not take the editor down."""

    failed = Signal(str)

    def run(self) -> None:
        gc.freeze()
        try:
            self.job()
        except Exception as error:  # noqa: BLE001 - the worker reports whatever the job raised
            self.failed.emit(f"{type(error).__name__}: {error}")
        finally:
            gc.unfreeze()

    def job(self) -> None:
        raise NotImplementedError


class LoadingThread(BackgroundThread):
    """A `BackgroundThread` that reads one file, which is every loader but the lyric mapper."""

    def __init__(self, path: str | Path, parent=None):
        super().__init__(parent)
        self.path = Path(path)

    def job(self) -> None:
        self.load()

    def load(self) -> None:
        raise NotImplementedError
