# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the ASS subtitle: its header, its `\\k` stream, its gaps, dots and settings."""

from __future__ import annotations

import pytest

from namioto.karaoke import AssSettings, generate_ass


def _dialogues(text, times, **kwargs):
    return [line for line in generate_ass(text, times, **kwargs).splitlines() if line.startswith("Dialogue:")]


def test_the_header_carries_the_styles_and_one_events_section():
    ass = generate_ass("あ", [[(0.0, 0.5)]])
    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert ass.count("[Events]") == 1
    for style in ("Style: K1,", "Style: K2,", "Style: LEAD,", "Style: H1,", "Style: H2,"):
        assert style in ass
    assert "template syl" in ass
    assert "template furi" in ass


def test_a_ruby_word_writes_its_base_and_readings():
    line = _dialogues("青[あお]", [[(0.0, 0.5), (0.5, 1.0)]])[0]
    assert "\\k50}青|<あ" in line
    assert "\\k50}#|お" in line


def test_a_rest_between_sounds_writes_a_gap():
    line = _dialogues("あい", [[(0.0, 0.5), (0.8, 1.0)]])[0]
    assert "\\k50}あ" in line
    assert "{\\k30}{\\k20}い" in line


def test_a_small_kana_stays_one_sound_with_the_sound_before_it():
    # 京[きょう] reads きょ and う - one span each, never き + ょ
    line = _dialogues("京[きょう]", [[(0.0, 1.0), (1.0, 2.0)]])[0]
    assert "\\k100}京|<きょ" in line
    assert "\\k100}#|う" in line


def test_the_times_are_written_as_the_format_counts_them():
    line = _dialogues("あ", [[(3661.23, 3661.5)]], settings=AssSettings(lead_time_ms=0))[0]
    assert "1:01:01.23" in line


def test_a_chapter_opens_with_three_guide_dots():
    dialogues = _dialogues("あ", [[(5.0, 5.5)]])
    assert dialogues[0].startswith("Dialogue: 0,")
    assert ",LEAD,,0,0,0,,{\\k100}●{\\k100}●{\\k100}●" in dialogues[0]
    assert dialogues[1].split(",", 4)[3] == "K1"


def test_a_first_word_too_soon_gets_no_dots():
    dialogues = _dialogues("あ", [[(1.0, 1.5)]])
    assert len(dialogues) == 1  # the line itself, no LEAD dot row


def test_an_input_override_does_not_change_the_natural_sounds():
    spans = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0), (1.0, 1.5)]
    line = _dialogues("(幾千)[いくせん].1あ", [spans])[0]
    assert "\\k25}幾千|<い" in line
    assert "\\k25}#|く" in line
    assert "\\k25}#|せ" in line
    assert "\\k25}#|ん" in line
    assert "\\k50}あ" in line


def test_a_ruby_of_even_morae_gets_one_glyph_per_sound():
    # 胡椒[こ,しょう] is three sounds, not four: the old mora walk split しょ
    line = _dialogues("胡椒[こ,しょう]", [[(0.0, 0.5), (0.5, 1.0), (1.0, 1.5)]])[0]
    assert "\\k50}胡|<こ" in line
    assert "\\k50}椒|しょ" in line
    assert "\\k50}#|う" in line


def test_a_merge_tiles_its_note_across_its_sounds():
    line = _dialogues("(あい).1", [[(0.0, 0.5), (0.5, 1.0)]])[0]
    assert "\\k50}あ" in line
    assert "\\k50}い" in line


def test_a_cross_container_merge_keeps_the_ruby_and_tiles_the_note():
    line = _dialogues("泣[な]い.+", [[(0.0, 0.5), (0.5, 1.0)]])[0]
    assert "\\k50}泣|<な" in line
    assert "\\k50}い" in line


def test_the_settings_name_the_font_and_the_overlay():
    ass = generate_ass("あ", [[(0.0, 0.5)]], settings=AssSettings(font="Test Font", overlay_color="FF0000"))
    assert "Style: K1,Test Font,96," in ass
    assert r"\1c&H0000FF&" in ass  # RGB red is written in the ASS order


def test_an_unknown_track_style_is_refused():
    with pytest.raises(ValueError, match="track style"):
        generate_ass("あ", [[(0.0, 0.5)]], settings=AssSettings(track_style={0: "nowhere"}))


def test_a_line_with_no_time_is_left_out():
    ass = generate_ass("あ", [[(None, None)]])
    assert "[Events]" in ass
    assert "Dialogue:" not in ass


def test_layout_and_effect_options_reach_styles_and_templates():
    settings = AssSettings(
        font="Test Font",
        font_size=72,
        border=7,
        border_furi=4,
        margin_h=32,
        margin_v=20,
        ruby_offset=-6,
        blur=8,
        blur_scale=1.5,
        base_outline_color="123456",
        overlay_outline_color="ABCDEF",
        overlay_blur_color="102030",
        base_blur_color="405060",
        clip_size=18,
        fade_in_ms=300,
        fade_out_ms=150,
    )
    ass = generate_ass("字[じ]", [[(5.0, 6.0)]], settings=settings)
    assert "Style: K1,Test Font,72,&H00FFFFFF,&H000000FF,&H00563412" in ass
    assert ",1,32,32,148,1" in ass
    assert ",3,32,32,20,1" in ass
    assert "Style: LEAD,sans-serif,72," in ass
    assert ",1,32,32,276,1" in ass
    for tag in (
        r"\bord7",
        r"\bord4",
        r"\bord10",
        r"\bord6",
        r"\blur8",
        r"\fad(300,150)",
        r"\3c&H563412&",
        r"\3c&HEFCDAB&",
        r"\3c&H302010&",
        r"\3c&H605040&",
        "!$middle-6!",
        "!$sleft-18!",
    ):
        assert tag in ass


def test_zero_borders_blur_and_ruby_offset_are_supported():
    ass = generate_ass(
        "字[じ]",
        [[(1.0, 2.0)]],
        settings=AssSettings(
            border=0,
            border_furi=0,
            blur=0,
            blur_scale=0,
            ruby_offset=0,
            clip_size=0,
        ),
    )
    assert r"\bord0" in ass
    assert r"\blur0" in ass
    assert "!$middle!" in ass
    assert "!$sleft-0!" in ass


def test_positive_offset_moves_dialogues_and_guide_dots_without_changing_spans():
    spans = [[(5.0, 5.5)]]
    dialogues = _dialogues("あ", spans, settings=AssSettings(offset_ms=2000))
    assert dialogues[0].startswith("Dialogue: 0,0:00:04.00,0:00:07.00,LEAD")
    assert dialogues[1].startswith("Dialogue: 0,0:00:02.00,0:00:07.50,K1")
    assert spans == [[(5.0, 5.5)]]


def test_negative_offset_trims_elapsed_karaoke_without_borrowing_time_back():
    dialogues = _dialogues("青[あお]い", [[(0.0, 1.0), (1.0, 2.0), (2.5, 3.0)]], settings=AssSettings(offset_ms=-1500))
    assert len(dialogues) == 1
    assert dialogues[0].startswith("Dialogue: 0,0:00:00.00,0:00:01.50,K1")
    assert r"{\k0}青|<あ{\k50}#|お{\k50}{\k50}い" in dialogues[0]
    assert r"\k-" not in dialogues[0]


def test_a_partial_syllable_keeps_only_its_time_after_zero():
    line = _dialogues("あい", [[(0.0, 1.0), (1.0, 2.0)]], settings=AssSettings(offset_ms=-250))[0]
    assert r"{\k0}{\k75}あ{\k100}い" in line
    assert ",0:00:00.00,0:00:01.75,K1" in line


def test_lines_fully_before_zero_are_omitted():
    dialogues = _dialogues("あ\nい", [[(0.0, 1.0)], [(3.0, 4.0)]], settings=AssSettings(offset_ms=-2000))
    assert len(dialogues) == 1
    assert "あ" not in dialogues[0]
    assert r"{\k100}い" in dialogues[0]


def test_auto_base_blur_follows_the_manual_overlay_blur():
    settings = AssSettings(overlay_blur_color="123456")
    overlay, base = settings.blur_colours()
    assert overlay == "123456"
    assert base != AssSettings().blur_colours()[1]
    ass = generate_ass("あ", [[(0.0, 1.0)]], settings=settings)
    assert r"\3c&H563412&" in ass
    assert rf"\3c&H{base[4:] + base[2:4] + base[:2]}&" in ass
