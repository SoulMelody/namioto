# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for MIDI pitch frequency conversion."""

import pytest

from namioto.playback import note_frequency


@pytest.mark.parametrize(("pitch", "frequency"), [(69, 440.0), (81, 880.0), (57, 220.0), (60, 261.63)])
def test_note_frequency_follows_equal_temperament(pitch, frequency):
    assert note_frequency(pitch) == pytest.approx(frequency, abs=0.01)
