# SPDX-License-Identifier: AGPL-3.0-only
"""Independent voices, punctuation ownership and final ASS colours."""

import re

import pytest
from kara_templater import Document

from namioto import ass_cli, midi, project, subtitle
from namioto.channels import Channel
from namioto.karaoke import AssSettings, generate_ass
from namioto.karaoke.ass import _convert
from namioto.karaoke.sounds import natural_sounds


def _base(result, layer=2):
    return [
        e
        for e in Document.parse(result).lines
        if e.layer == layer and e.style.startswith("K") and "furigana" not in e.style
    ]


def _surface(event):
    return re.sub(r"\{[^}]*\}", "", event.text)


def test_inline_overlapping_channels_keep_exact_animation_times_and_colours():
    text = "@1{あ}@2{い}@1{う}"
    notes = (project.Note(0, 2, 60, 1), project.Note(1, 2, 64, 2), project.Note(4, 1, 62, 1))
    data = midi.to_bytes((Channel(channel=1), Channel(channel=2)), notes, 60)
    result = subtitle.build_ass(text, data, settings=AssSettings(channel_colors={1: "FF0000", 2: "00FF00"}))
    assert result == generate_ass(
        text, [[(0, 2), (1, 3), (4, 5)]], settings=AssSettings(channel_colors="1=FF0000,2=00FF00")
    )
    events = _base(result, 3)
    assert [_surface(e) for e in events] == ["あ", "い", "う"]
    for event, start, end, colour in zip(
        events, (0, 1000, 4000), (2000, 3000, 5000), ("0000FF", "00FF00", "0000FF"), strict=True
    ):
        assert rf"\t({start},{end},0.5,\fscx110" in event.text
        assert rf"\1c&H{colour}&" in event.text
    assert {e.style for e in events} == {"K1"}
    assert "namioto_" not in result


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "text,surface", [("Show you guts（cool）!", "Show you guts（cool）!"), ("「青[あお]」！", "「青」！")]
)
def test_export_preserves_characters_without_sounds(text, surface, explicit):
    sounds = natural_sounds(text)[0].sounds
    settings = AssSettings(channel_colors={1: "FF0000"} if explicit else {})
    result = generate_ass(text, [[(i, i + 1) for i in range(len(sounds))]], settings=settings)
    source = _convert(natural_sounds(text)[0], [(i, i + 1) for i in range(len(sounds))], settings, 0)
    assert "".join(unit.text for unit in source.source.units) == surface
    assert "".join(_surface(e) for e in _base(result)) == surface.replace(" ", "")


@pytest.mark.parametrize("text", ["もっと@c1{（もっと）}うまく@c1{（うまく）}なりたいな.2！", "あ@c1{！}"])
def test_colour_only_tags_preserve_punctuation_colours(text):
    sounds = natural_sounds(text)[0].sounds
    result = generate_ass(
        text, [[(i, i + 1) for i in range(len(sounds))]], settings=AssSettings(channel_colors={1: "FF0000"})
    )
    line = _convert(natural_sounds(text)[0], [(i, i + 1) for i in range(len(sounds))], AssSettings(), 0)
    punctuation = [(c, u.text) for u, c in zip(line.source.units, line.colours, strict=True) if u.text in "（）！"]
    assert punctuation == (
        [(1, "（"), (1, "）"), (1, "（"), (1, "）"), (0, "！")] if text.startswith("もっと") else [(1, "！")]
    )
    assert "".join(_surface(e) for e in _base(result)).endswith("！")
    rendered = [e for e in _base(result, 3) if _surface(e) in "（）！"]
    assert [(int(r"\1c&H0000FF&" in e.text), _surface(e)) for e in rendered] == punctuation


def test_punctuation_borrows_time_only_from_its_own_channel():
    source = _convert(natural_sounds("@0{Rolling! }@1{Ah—}")[0], [(0, 1), (1, 2)], AssSettings(), 0)
    assert [(u.text, [(s.start_ms, s.end_ms) for s in u.segments]) for u in source.source.units] == [
        ("Rolling", [(0, 920)]),
        ("!", [(920, 1000)]),
        (" ", [(1000, 1000)]),
        ("Ah", [(1000, 1920)]),
        ("—", [(1920, 2000)]),
    ]
    assert source.colours == (0, 0, 0, 1, 1)


@pytest.mark.parametrize("start", [0.5, 2])
def test_new_channel_opening_punctuation_borrows_from_its_first_sound(start):
    text = "よ！@1{（(ハイ).1 い・く・よ）}"
    spans = [
        (0, 1),
        (start, start + 0.5),
        (start + 0.5, start + 1),
        (start + 1, start + 2),
        (start + 2, start + 3),
        (start + 3, start + 4),
    ]
    source = _convert(natural_sounds(text)[0], spans, AssSettings(), 0)
    unit = next(u for u in source.source.units if u.text == "ハ")
    assert unit.segments[0].start_ms == round(start * 1000) + 80
    punctuation = next(u for u in source.source.units if u.text == "（").segments[0]
    assert (punctuation.start_ms, punctuation.end_ms) == (round(start * 1000), round(start * 1000) + 80)


@pytest.mark.parametrize("start", [10, 20])
def test_repeated_channel_run_keeps_its_own_times(start):
    text = "おとなの@1{（(きゃー).1＞＿＜）}階段[かい,だん]@1{（コワイ！）}"
    sounds = natural_sounds(text)[0].sounds
    spans, channel_one = [], 0
    for index, sound in enumerate(sounds):
        if sound.channel == 1:
            onset = 4 + channel_one if channel_one < 2 else start + channel_one - 2
            channel_one += 1
        else:
            onset = index
        spans.append((onset, onset + 1))
    source = _convert(natural_sounds(text)[0], spans, AssSettings(), 0)
    assert next(u for u in source.source.units if u.text == "コ").segments[0].start_ms == start * 1000 + 80
    openings = [u.segments[0] for u in source.source.units if u.text == "（"]
    assert [(s.start_ms, s.end_ms) for s in openings] == [(4000, 4080), (start * 1000, start * 1000 + 80)]


def test_multiline_scope_does_not_share_note_cursors():
    text = "@1{\nあ\nう\n}\n@2{い}"
    notes = (project.Note(0, 1, 60, 1), project.Note(2, 1, 62, 1), project.Note(0, 3, 64, 2))
    data = midi.to_bytes((Channel(channel=1), Channel(channel=2)), notes, 60)
    assert subtitle.build_ass(text, data) == generate_ass(text, [[(0, 1)], [(2, 3)], [(0, 3)]])


def test_more_than_two_simultaneous_lines_have_distinct_positions():
    result = generate_ass("@1{あ}\n@2{い}\n@3{う}", [[(0, 3)], [(0, 3)], [(0, 3)]])
    events = _base(result)
    assert [e.style for e in events] == ["K1", "K2", "K3"]
    assert len({re.search(r"\\pos\(([^)]+)\)", e.text)[1] for e in events}) == 3


def test_ruby_gap_preserves_separate_times_without_continuous_sweep():
    result = generate_ass("@1{字[あい]}", [[(0, 1), (2, 3)]], settings=AssSettings(channel_colors={1: "FF0000"}))
    events = _base(result, 3)
    assert len(events) == 2
    assert all(_surface(e) == "字" for e in events)
    for event, start in zip(events, (0, 2000), strict=True):
        sweep = re.search(r"\\t\(([\d.]+),([\d.]+),0\.5,\\clip", event.text)
        assert float(sweep[1]) == start
        assert float(sweep[2]) == pytest.approx(start + 1000 * 3 / 17)
    assert len(_base(result)) == 1


def test_shared_base_can_change_colour_between_nonoverlapping_readings():
    result = generate_ass(
        "字[@c1{あ}@c2{い}]",
        [[(0, 1), (2, 3)]],
        settings=AssSettings(channel_colors={1: "FF0000", 2: "00FF00"}),
    )
    base = _base(result, 3)
    assert len(base) == 2
    assert r"\1c&H0000FF&" in base[0].text
    assert r"\1c&H00FF00&" in base[1].text
    ruby = [e for e in Document.parse(result).lines if e.layer == 3 and e.style == "K1-furigana"]
    assert [_surface(e) for e in ruby] == ["あ", "い"]
    assert r"\1c&H0000FF&" in ruby[0].text
    assert r"\1c&H00FF00&" in ruby[1].text


@pytest.mark.parametrize("colours", ["16=FF0000", "1=xyz", "a=FF0000", {True: "FF0000"}, {1: "GG0000"}])
def test_invalid_channel_colours_are_rejected(colours):
    with pytest.raises(ValueError):
        subtitle.validate_settings(AssSettings(channel_colors=colours))


def test_cli_channel_colours_without_project_metadata(monkeypatch):
    monkeypatch.setenv("NAMIOTO_ASS_CHANNEL_COLORS", "1=FF0000,2=00FF00")
    args = ass_cli.build_parser().parse_args(["--out", "out.ass"])
    assert ass_cli.resolve_settings(args).channel_colours() == {1: "FF0000", 2: "00FF00"}
    args = ass_cli.build_parser().parse_args(["--out", "out.ass", "--channel-colors", "3=ABCDEF"])
    assert ass_cli.resolve_settings(args).channel_colours() == {3: "ABCDEF"}
