# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the one export gate every lyric export reads."""

from __future__ import annotations

from namioto.document import Note
from namioto.karaoke.operations import Match, Merge, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyricmap.raw import Raw
from namioto.lyricmap.solver import solve
from namioto.lyricmap.verify import verify


def _notes(*spans):
    return [Note(60, start, end - start, id=index + 1) for index, (start, end) in enumerate(spans)]


def _gate(text, raw, notes, anchors=(), **extra):
    lines = natural_sounds(text)
    operations = solve(lines, raw, notes, anchors)
    return lines, operations, verify(text, lines, raw, notes, operations, **extra)


def test_a_good_mapping_opens_and_hands_back_the_canonical_krc():
    _lines, _operations, gate = _gate("あ", [[Raw(0.0, 1.0, 0.9)]], _notes((0.0, 1.0)))
    assert gate.open()
    assert gate.canonical == "あ"


def test_no_sounds_or_notes_closes_the_gate():
    lines = natural_sounds("！？")
    assert verify("！？", lines, [[]], _notes((0.0, 1.0)), []).counts() == {"no_lyric_sounds": 1}
    lines = natural_sounds("あ")
    assert verify("あ", lines, [[Raw(0.0, 1.0, 0.9)]], [], []).counts() == {"no_target_notes": 1}


def test_a_filtered_note_closes_the_gate():
    _lines, _operations, gate = _gate(
        "あ", [[Raw(0.0, 1.0, 0.9)]], _notes((0.0, 1.0)), filtered=[_notes((0.0, 1.0))[0]]
    )
    assert not gate.open()
    assert gate.counts() == {"filtered_note": 1}


def test_low_alignment_scores_do_not_require_confirmation():
    lines = natural_sounds("あ")
    raw = [[Raw(0.0, 1.0, 0.1)]]
    notes = _notes((0.0, 1.0))
    operations = solve(lines, raw, notes)
    assert verify("あ", lines, raw, notes, operations).open()
    confirmed = [Match(SoundRef(0, 0), (1,), confirmed=True)]
    assert verify("あ", lines, raw, notes, confirmed).open()


def test_a_cross_container_merge_opens_the_gate():
    text = "胡椒[こ,(しょう)]"
    lines = natural_sounds(text)
    raw = [[Raw(0.0, 0.5, 0.9), Raw(0.5, 0.5, 0.9), Raw(1.0, 1.0, 0.9)]]
    notes = _notes((0.0, 1.0), (1.0, 2.0))
    operations = [Merge((SoundRef(0, 0), SoundRef(0, 1)), 1), Match(SoundRef(0, 2), (2,))]
    gate = verify(text, lines, raw, notes, operations)
    assert gate.open()
    assert gate.canonical == "胡椒[こ,(しょ).+う]"


def test_a_mapping_that_would_move_a_sound_closes_the_gate():
    # `あ.21` would read back as one sound with a twenty-one-note override
    text = "あ1"
    lines = natural_sounds(text)
    raw = [[Raw(0.0, 1.0, 0.9), Raw(1.0, 0.1, 0.9)]]
    notes = _notes((0.0, 1.0), (1.0, 2.0), (2.0, 3.0))
    operations = [
        Match(SoundRef(0, 0), (1, 2), confirmed=True),
        Match(SoundRef(0, 1), (3,), confirmed=True),
    ]
    assert verify(text, lines, raw, notes, operations).counts() == {"round_trip_mismatch": 1}


def test_broken_lyrics_close_the_gate():
    assert verify("世界", [], [], [], [], broken=True).counts() == {"unnormalizable_krc": 1}
