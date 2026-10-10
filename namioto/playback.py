# SPDX-License-Identifier: AGPL-3.0-only
"""Qt-free conversion from MIDI pitch to equal-tempered frequency."""


def note_frequency(pitch: int, a4: float = 440.0) -> float:
    """Equal-tempered frequency of a MIDI pitch, A4 as the reference."""
    return a4 * 2.0 ** ((pitch - 69) / 12.0)
