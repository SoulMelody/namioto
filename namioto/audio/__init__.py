# SPDX-License-Identifier: AGPL-3.0-only
"""The audio backend: the song's transport, the phase vocoder under it, and the device on top.

Qt-free like `project.py`, `midi.py` and `settings.py`, and device-free until an `Output` is opened:
`Stretcher` and `Engine` are pure numpy and are driven frame by frame, so the whole realtime path can
be judged deterministically without a sound card.
"""

from namioto.audio.engine import Engine
from namioto.audio.output import FrameRing, Output
from namioto.audio.stretch import Stretcher

__all__ = ["Engine", "FrameRing", "Output", "Stretcher"]
