# SPDX-License-Identifier: AGPL-3.0-only
"""MIDI and KRC alone retain overlapping voices, inline ownership and export colours."""

import pytest

from namioto import ass_cli, midi, project, subtitle
from namioto.channels import Channel
from namioto.karaoke import AssSettings, generate_ass


def test_inline_overlapping_channels_keep_explicit_syllable_times():
    text = "@1{あ}@2{い}@1{う}"
    notes = (project.Note(0, 2, 60, 1), project.Note(1, 2, 64, 2), project.Note(4, 1, 62, 1))
    data = midi.to_bytes((Channel(channel=1), Channel(channel=2)), notes, 60)
    result = subtitle.build_ass(text, data, settings=AssSettings(channel_colors={1: "FF0000", 2: "00FF00"}))
    assert result == generate_ass(
        text, [[(0, 2), (1, 3), (4, 5)]], settings=AssSettings(channel_colors="1=FF0000,2=00FF00")
    )
    assert "\\-namioto_1_0_2000_2000" in result
    assert "\\-namioto_2_1000_3000_3000" in result
    assert "\\-namioto_1_4000_5000_5000" in result
    assert "code syl furi" in result
    assert "[1]={'0000FF'" in result
    assert "[2]={'00FF00'" in result
    assert "syl.namioto_overlay" in result
    assert "tonumber" not in result
    assert "syl.start_time = first + 0" in result
    assert len([line for line in result.splitlines() if line.startswith("Dialogue:") and ",K1," in line]) == 1


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "text, surface", [("Show you guts（cool）!", "Show you guts（cool）!"), ("「青[あお]」！", "「青|<あ#|お」！")]
)
def test_export_preserves_characters_without_sounds(text, surface, explicit):
    import re

    from namioto.karaoke.sounds import natural_sounds

    sounds = natural_sounds(text)[0].sounds
    settings = AssSettings(channel_colors={1: "FF0000"} if explicit else {})
    result = generate_ass(text, [[(index, index + 1) for index in range(len(sounds))]], settings=settings)
    line = next(line for line in result.splitlines() if line.startswith("Dialogue:") and ",LEAD," not in line)
    assert re.sub(r"\{[^}]*\}", "", line.split(",", 9)[9]) == surface


def test_punctuation_at_channel_boundary_keeps_its_owner():
    result = generate_ass(
        "@0{Rolling! }@1{Ah—}", [[(0, 1), (1, 2)]], settings=AssSettings(channel_colors={1: "FF0000"})
    )
    assert "}g" in result
    assert r"{\k8\-namioto_0_920_1000_1000}!" in result
    assert "}A" in result
    assert "}! A" not in result
    assert "}h" in result
    assert "}—" in result


@pytest.mark.parametrize(
    "text, spans, expected",
    [
        ("Ah—", [(0, 1)], r"{\k46\-namioto_0_0_460_460}A{\k46\-namioto_0_460_920_920}h{\k8\-namioto_0_920_1000_1000}—"),
        ("！あ", [(0, 1)], r"{\k8\-namioto_0_0_80_80}！{\k92\-namioto_0_80_1000_1000}あ"),
        ("あい.0", [(0, 1), (1, 1)], r"{\k92\-namioto_0_0_920_920}あ{\k8\-namioto_0_920_1000_1000}い"),
    ],
)
def test_borrowed_durations_update_explicit_timing(text, spans, expected):
    result = generate_ass(text, [spans], settings=AssSettings(channel_colors={1: "FF0000"}))
    assert expected in result


@pytest.mark.parametrize("start", [0.5, 2])
def test_new_channel_opening_punctuation_anchors_to_its_first_sound(start):
    result = generate_ass(
        "よ！@1{（(ハイ).1 い・く・よ）}",
        [
            [
                (0, 1),
                (start, start + 0.5),
                (start + 0.5, start + 1),
                (start + 1, start + 2),
                (start + 2, start + 3),
                (start + 3, start + 4),
            ]
        ],
        settings=AssSettings(channel_colors={1: "FF0000"}),
    )
    first = round(start * 1000)
    assert rf"{{\k8\-namioto_1_{first}_{first + 80}_{first + 80}}}（" in result
    assert rf"\-namioto_1_{first + 80}_" in result


def test_multiline_scope_and_simultaneous_display_lines_do_not_share_note_cursors():
    text = "@1{\nあ\nう\n}\n@2{い}"
    notes = (project.Note(0, 1, 60, 1), project.Note(2, 1, 62, 1), project.Note(0, 3, 64, 2))
    data = midi.to_bytes((Channel(channel=1), Channel(channel=2)), notes, 60)
    assert subtitle.build_ass(text, data) == generate_ass(text, [[(0, 1)], [(2, 3)], [(0, 3)]])


def test_more_than_two_simultaneous_display_lines_get_separate_styles():
    result = generate_ass("@1{あ}\n@2{い}\n@3{う}", [[(0, 3)], [(0, 3)], [(0, 3)]])
    assert "Style: K3," in result
    dialogues = [line for line in result.splitlines() if line.startswith("Dialogue:") and ",LEAD," not in line]
    assert [line.split(",")[3] for line in dialogues] == ["K1", "K2", "K3"]


def test_ruby_timing_distinguishes_the_base_span_and_each_reading():
    result = generate_ass("@1{字[あい]}", [[(0, 1), (2, 3)]], settings=AssSettings(channel_colors={1: "FF0000"}))
    assert "\\-namioto_1_0_1000_3000" in result
    assert "\\-namioto_1_2000_3000_3000" in result
    assert "syl.isfuri and last or base_last" in result


@pytest.mark.parametrize("colours", ["16=FF0000", "1=xyz", "a=FF0000", {True: "FF0000"}, {1: "GG0000"}])
def test_invalid_channel_colours_are_rejected(colours):
    with pytest.raises(ValueError):
        subtitle.validate_settings(AssSettings(channel_colors=colours))


def test_cli_channel_colours_are_available_without_project_metadata(monkeypatch):
    monkeypatch.setenv("NAMIOTO_ASS_CHANNEL_COLORS", "1=FF0000,2=00FF00")
    args = ass_cli.build_parser().parse_args(["--out", "out.ass"])
    assert ass_cli.resolve_settings(args).channel_colours() == {1: "FF0000", 2: "00FF00"}
    args = ass_cli.build_parser().parse_args(["--out", "out.ass", "--channel-colors", "3=ABCDEF"])
    assert ass_cli.resolve_settings(args).channel_colours() == {3: "ABCDEF"}
