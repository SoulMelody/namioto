# SPDX-License-Identifier: AGPL-3.0-only
"""Channel 0's edit mapping and the other tracks' literal KRC NOTE slots."""

import pytest

from namioto.karaoke.canonical import rebuild
from namioto.karaoke.operations import Match, SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds
from namioto.karaoke.writer import dumps
from namioto.ui.lyric_map import map_lyrics


def test_krc_tracks_are_zero_based_without_reinterpreting_explicit_numbers():
    text = "あ\n{0}い\n{1}う\n{15}え"
    assert [line.track for line in natural_sounds(text)] == [0, 0, 1, 15]
    assert dumps(parse(text)) == "あ\nい\n{1}う\n{15}え"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("{1}あ.2い", (((10, 20), (30,)),)),
        ("{1}あい.+", (((10,), (10,)),)),
        ("{1}あ.0い", (((), (10,)),)),
    ],
)
def test_nonzero_channels_read_literal_slots_even_in_edit_mode(text, expected):
    lines = natural_sounds(text)
    result = map_lyrics(
        text,
        lines,
        [[(None, None)] * len(line.sounds) for line in lines],
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (1.0, 2.0, 62, 20), (2.0, 3.0, 64, 30))},
    )
    assert result.mapped == expected
    assert result.operations == ()
    assert not result.error


def test_faithful_cursors_are_independent_per_channel():
    text = "{1}あ\n{2}い\n{1}う"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(None, None)]] * 3,
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (2.0, 3.0, 62, 20)), 2: ((0.0, 1.0, 64, 30),)},
    )
    assert result.mapped == (((10,),), ((30,),), ((20,),))
    assert result.spans == (((0.0, 1.0),), ((0.0, 1.0),), ((2.0, 3.0),))
    assert not result.error


def test_unmapped_faithful_sounds_keep_raw_times_without_inventing_matches():
    text = "{1}あい"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(5.0, 0.5), (6.0, 0.5)]],
        notes_by_channel={1: ((0.0, 1.0, 60, 10),)},
    )
    assert result.spans == (((0.0, 1.0), (6.0, 6.5)),)
    assert result.mapped == (((10,), ()),)
    assert result.error
    assert result.operations == ()


def test_rebuilding_selected_lines_preserves_other_tracks_literal_structure():
    text = "あ\n{1}い.2うえ.+\nえ"
    operations = (Match(SoundRef(0, 0), (10, 20)), Match(SoundRef(2, 0), (30,)))
    assert rebuild(text, operations, line_indices=(0, 2)) == "あ.2\n{1}い.2うえ.+\nえ"
