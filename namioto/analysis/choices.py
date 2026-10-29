# SPDX-License-Identifier: AGPL-3.0-only
"""The values a setting may take, kept apart from the modules that run them.

`namioto.settings` and `namioto.project` read these to build their spec tables, and building one must
not import the analysis stack: `namioto.project` reads the settings, and reading a project should not
pull librosa and scipy in. The module that runs an option imports its list from here and so still
offers the same name.
"""

from __future__ import annotations

ALGORITHMS = ("wavetone", "librosa", "tempocnn")
CHANNEL_MODES = ("mono", "left", "right", "sum", "side", "both")
# how the note table is built: WaveTone's correlation, noteDigger's STFT, or that STFT fused with
# a constant-Q transform, which reads the low bands at a resolution the FFT window cannot reach
BACKENDS = ("wavetone", "notedigger", "cqt")
