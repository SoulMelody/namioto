# SPDX-License-Identifier: AGPL-3.0-only
"""Independent channel mappings and literal KRC NOTE slots in read mode."""

import pytest

from namioto.karaoke.canonical import rebuild
from namioto.karaoke.operations import Match, SoundRef
from namioto.karaoke.parser import parse
from namioto.karaoke.sounds import natural_sounds
from namioto.karaoke.writer import dumps
from namioto.ui.lyric_map import map_lyrics


@pytest.mark.parametrize("changed", ["notes", "raw", "scores", "flagged", "text"])
def test_faithful_mapping_reuses_only_unchanged_channels(monkeypatch, changed):
    from namioto.ui import lyric_map

    text = "あ\n@1{い}"
    request = {
        "text": text,
        "lines": natural_sounds(text),
        "raw": [[(0.0, 1.0)], [(2.0, 1.0)]],
        "scores": [[0.9], [0.9]],
        "flagged": [False, False],
        "notes_by_channel": {0: ((0.0, 1.0, 60, 10),), 1: ((2.0, 3.0, 60, 20),)},
    }
    baseline = map_lyrics(**request)
    if changed == "notes":
        request["notes_by_channel"] = {0: ((0.1, 1.1, 60, 10),), 1: request["notes_by_channel"][1]}
    elif changed == "raw":
        request["raw"] = [[(0.2, 1.0)], request["raw"][1]]
    elif changed == "scores":
        request["scores"] = [[0.1], request["scores"][1]]
    elif changed == "flagged":
        request["flagged"] = [True, False]
    else:
        request["text"] = "あ.0\n@1{い}"
        request["lines"] = natural_sounds(request["text"])
    expected = map_lyrics(**request)
    calls = []
    original = lyric_map._read_result

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lyric_map, "_read_result", counted)
    result = map_lyrics(**request, previous=baseline)
    assert result == expected
    assert calls == [True]
    assert dict(result.channel_results)[1] is dict(baseline.channel_results)[1]


def test_preview_and_finishing_reuse_unchanged_channels():
    text = "あ\n@1{い}"
    notes = {0: ((0.0, 1.0, 60, 10),), 1: ((2.0, 3.0, 60, 20),)}
    raw = [[(0.0, 1.0)], [(2.0, 1.0)]]
    baseline = map_lyrics(text, natural_sounds(text), raw, notes_by_channel=notes)
    changed = "あ.0\n@1{い}"
    preview = map_lyrics(
        changed,
        natural_sounds(changed),
        raw,
        notes_by_channel=notes,
        previous=baseline,
    )
    assert dict(preview.channel_results)[1] is dict(baseline.channel_results)[1]
    complete = map_lyrics(changed, natural_sounds(changed), raw, notes_by_channel=notes, previous=preview)
    assert complete == map_lyrics(changed, natural_sounds(changed), raw, notes_by_channel=notes)
    assert dict(complete.channel_results)[0] is dict(preview.channel_results)[0]


@pytest.mark.parametrize("text", ["@c0{あい}", "@c2{あい}", "@c15{文字[あ,い]}"])
def test_color_tags_preserve_mapping_and_roundtrip(text):
    lines = natural_sounds(text)
    assert all(sound.channel == 0 for sound in lines[0].sounds)
    assert all(sound.color_channel == int(text[2 : text.index("{")]) for sound in lines[0].sounds)
    assert dumps(parse(text)) == text
    result = map_lyrics(
        text,
        lines,
        [[(None, None), (None, None)]],
        mode="read",
        notes_by_channel={0: ((0.0, 1.0, 60, 10), (1.0, 2.0, 62, 20)), 2: ((3.0, 4.0, 64, 30),)},
    )
    assert result.mapped == (((10,), (20,)),)
    assert not result.error
    assert rebuild(text, result.operations) == text


@pytest.mark.parametrize("text", ["@c16{あ}", "@c2{あ", "@c1{@2{あ}}"])
def test_invalid_color_tags(text):
    from namioto.karaoke.model import KrcError

    with pytest.raises(KrcError):
        parse(text)


def test_krc_tracks_are_zero_based_without_reinterpreting_explicit_numbers():
    text = "あ\n@0{い}\n@1{う}\n@15{え}"
    assert [line.track for line in natural_sounds(text)] == [0, 0, 1, 15]
    assert dumps(parse(text)) == text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("@1{あ.2い}", (((10, 20), (30,)),)),
        ("@1{あい.+}", (((10,), (10,)),)),
        ("@1{あ.0い}", (((), (10,)),)),
    ],
)
def test_nonzero_channels_read_literal_slots_in_read_mode(text, expected):
    lines = natural_sounds(text)
    result = map_lyrics(
        text,
        lines,
        [[(None, None)] * len(line.sounds) for line in lines],
        mode="read",
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (1.0, 2.0, 62, 20), (2.0, 3.0, 64, 30))},
    )
    assert result.mapped == expected
    rebuilt = rebuild(text, result.operations)
    reread = map_lyrics(
        rebuilt,
        natural_sounds(rebuilt),
        result.raw,
        mode="read",
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (1.0, 2.0, 62, 20), (2.0, 3.0, 64, 30))},
    )
    assert reread.mapped == expected
    assert not result.error


def test_faithful_cursors_are_independent_per_channel():
    text = "@1{あ}\n@2{い}\n@1{う}"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(None, None)]] * 3,
        mode="read",
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (2.0, 3.0, 62, 20)), 2: ((0.0, 1.0, 64, 30),)},
    )
    assert result.mapped == (((10,),), ((30,),), ((20,),))
    assert result.spans == (((0.0, 1.0),), ((0.0, 1.0),), ((2.0, 3.0),))
    assert not result.error


def test_unmapped_faithful_sounds_do_not_use_alignment_as_a_mapping():
    text = "@1{あい}"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(5.0, 0.5), (6.0, 0.5)]],
        mode="read",
        notes_by_channel={1: ((0.0, 1.0, 60, 10),)},
    )
    assert result.spans == (((0.0, 1.0), (None, None)),)
    assert result.raw == (((5.0, 0.5), (6.0, 0.5)),)
    assert result.mapped == (((10,), ()),)
    assert not result.error
    assert not result.channel_errors
    assert result.zero == ((False, True),)
    assert result.operations == (Match(SoundRef(0, 0), (10,), confirmed=True),)


@pytest.mark.parametrize(
    "text,notes,spans,mapped",
    [
        ("あ", ((0.0, 1.0, 60, 10), (1.0, 2.0, 62, 20)), (((0.0, 1.0),),), (((10,),),)),
        ("あ.2い", ((0.0, 1.0, 60, 10),), (((0.0, 1.0), (None, None)),), (((10,), ()),)),
    ],
)
@pytest.mark.parametrize("channel", [0, 1])
def test_faithful_count_mismatches_keep_available_note_spans_without_errors(text, notes, spans, mapped, channel):
    text = f"@{channel}{{{text}}}"
    lines = natural_sounds(text)
    result = map_lyrics(
        text,
        lines,
        [[(None, None)] * len(line.sounds) for line in lines],
        notes_by_channel={channel: notes},
    )
    assert not result.error
    assert not result.channel_errors
    assert result.spans == spans
    assert result.mapped == mapped


def test_rebuilding_selected_lines_preserves_other_tracks_literal_structure():
    text = "あ\n@1{い.2うえ.+}\nえ"
    operations = (Match(SoundRef(0, 0), (10, 20)), Match(SoundRef(2, 0), (30,)))
    assert rebuild(text, operations, line_indices=(0, 2)) == "あ.2\n@1{い.2うえ.+}\nえ"


@pytest.mark.parametrize("mode", ["read", "edit"])
def test_channel_cursors_are_independent_within_one_display_line(mode):
    text = "@1{あ}@2{い}@1{う}"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(0.0, 1.0), (0.0, 1.0), (2.0, 1.0)]],
        mode=mode,
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (2.0, 3.0, 62, 20)), 2: ((0.0, 1.0, 64, 30),)},
    )
    assert not result.error
    assert result.mapped == (((10,), (30,), (20,)),)
    assert result.spans == (((0.0, 1.0), (0.0, 1.0), (2.0, 3.0)),)
    if mode == "edit":
        assert result.operations == (
            Match(SoundRef(0, 0), (10,)),
            Match(SoundRef(0, 2), (20,)),
            Match(SoundRef(0, 1), (30,)),
        )
        assert {channel for channel, _result in result.channel_results} == {1, 2}


def test_an_invalid_channel_does_not_discard_another_channels_edit_mapping():
    text = "@1{あ}@2{い}"
    result = map_lyrics(
        text, natural_sounds(text), [[(0.0, 1.0), (0.0, 1.0)]], mode="edit", notes_by_channel={1: ((0.0, 1.0, 60, 10),)}
    )
    assert result.operations == (Match(SoundRef(0, 0), (10,)),)
    assert result.mapped == (((10,), ()),)
    assert dict(result.channel_errors).keys() == {2}


@pytest.mark.parametrize("text", ["(@1{あ}@2{い}).1", "@1{あ}@2{い.+}", "@1{し}@2{ょ}"])
def test_mapping_rejects_indivisible_sounds_or_note_groups_with_multiple_channels(text):
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(0.0, 1.0)] * len(natural_sounds(text)[0].sounds)],
        notes_by_channel={1: ((0.0, 1.0, 60, 10),), 2: ((0.0, 1.0, 64, 20),)},
    )
    assert result.error
    assert not result.operations


def test_dropped_group_can_span_channels_without_sharing_a_note():
    text = "(@1{あ}@2{い}).0@1{う}@2{え}"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(None, None)] * 4],
        mode="read",
        notes_by_channel={1: ((0.0, 1.0, 60, 10),), 2: ((0.0, 1.0, 64, 20),)},
    )
    assert not result.error
    assert result.mapped == (((), (), (10,), (20,)),)
    assert result.zero == ((True, True, False, False),)


def test_edit_constraints_keep_explicit_note_counts_on_nonzero_channels():
    text = "@1{あ.2}@2{い}"
    result = map_lyrics(
        text,
        natural_sounds(text),
        [[(0.0, 1.0), (0.0, 1.0)]],
        mode="edit",
        use_constraints=True,
        notes_by_channel={1: ((0.0, 1.0, 60, 10), (2.0, 3.0, 62, 20)), 2: ((0.0, 1.0, 64, 30),)},
    )
    assert not result.error
    assert result.mapped == (((10, 20), (30,)),)


def test_invalid_alignment_evidence_cannot_override_faithful_krc_slots():
    result = map_lyrics(
        "あ",
        natural_sounds("あ"),
        [[(4.0, -1.0)]],
        notes=((0.0, 1.0, 60, 10),),
    )
    assert not result.error
    assert result.spans == (((0.0, 1.0),),)
    assert result.mapped == (((10,),),)
    assert result.red == ((False,),)
