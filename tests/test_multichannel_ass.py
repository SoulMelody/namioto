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
    assert len([line for line in result.splitlines() if line.startswith("Dialogue:") and ",K1," in line]) == 1


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
