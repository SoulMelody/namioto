# SPDX-License-Identifier: AGPL-3.0-only
"""Final ASS events, structured timing, ruby layout and export settings."""

import re
from dataclasses import replace

import pytest
from kara_templater import Document

from namioto.karaoke import AssSettings, generate_ass
from namioto.karaoke.ass import _convert
from namioto.karaoke.sounds import natural_sounds


def _source(text, spans, settings=None):
    return _convert(natural_sounds(text)[0], spans, settings or AssSettings(), 0)


def _events(text, spans, settings=None):
    return Document.parse(generate_ass(text, spans, settings=settings)).lines


def _surface(event):
    return re.sub(r"\{[^}]*\}", "", event.text)


def test_trusted_templates_use_independent_times_and_style_overrides():
    from kara_templater import Line, Record, Style

    templates = Document(
        [
            Style(name="K1", fontname="sans-serif", fontsize=72, align=8, margin_v=100),
            Line(comment=True, style="K1", effect="template syl multi noblank", text='!retime("syl")!{$sleft,$stop}'),
            Line(comment=True, style="K1", effect="template furi", text='!retime("syl")!{$sleft,$stop}'),
            Line(text="not imported", style="K1"),
        ],
        Record(),
    )
    result = Document.parse(generate_ass("青[あお]", [[(0, 0.5), (1, 1.5)]], templates=templates))
    events = [line for line in result.lines if line.effect == "fx"]
    assert len(events) == 4
    assert [(line.start_time, line.end_time) for line in events] == [(0, 500), (1000, 1500)] * 2
    assert [line.style for line in events] == ["K1", "K1", "K1-furigana", "K1-furigana"]
    assert result.styles["K1"].fontsize == 72
    assert not any(line.text == "not imported" for line in result.lines)
    assert templates.lines[0].text == '!retime("syl")!{$sleft,$stop}'


def test_templates_keep_builtin_countdown_and_reject_custom_effect():
    from kara_templater import Line, Record

    templates = Document([Line(comment=True, style="K1", effect="template syl noblank", text="")], Record())
    result = Document.parse(generate_ass("あ", [[(6, 7)]], templates=templates))
    assert len([line for line in result.lines if line.effect == "guide"]) == 12
    assert len([line for line in result.lines if line.effect == "fx"]) == 1
    with pytest.raises(ValueError, match="cannot be combined"):
        generate_ass("あ", [[(0, 1)]], templates=templates, effect=lambda *args: [])


def test_custom_effect_receives_structured_layout_and_independent_times():
    from kara_templater import Line

    calls = []

    def effect(unit, style, show, hide, settings, colours, **options):
        calls.append((unit, style, show, hide, settings, colours, options))
        return [Line(text=unit.source.text, start_time=show, end_time=hide, style=style)]

    result = generate_ass("青[あお]", [[(0, 0.5), (1, 1.5)]], effect=effect)
    assert len(calls) == 1
    unit, style, show, hide, settings, colours, options = calls[0]
    assert [(s.source.start_ms, s.source.end_ms) for s in unit.segments] == [(0, 500), (1000, 1500)]
    assert all(s.ruby is not None for s in unit.segments)
    assert len(options["segment_colours"]) == 2
    events = Document.parse(result).lines
    assert len(events) == 1
    assert events[0].text == "青"


def test_final_output_has_styles_but_no_templates_or_timing_protocol():
    result = generate_ass("あ", [[(0, 0.5)]])
    assert result.count("[Events]") == 1
    for name in ("K1", "K2", "H1", "H2", "LEAD"):
        assert f"Style: {name}," in result
    for forbidden in ("template", "code syl", "namioto_", r"\k", "$sstart"):
        assert forbidden not in result
    assert len(Document.parse(result).lines) == 4


@pytest.mark.parametrize("spacing", [5, 0, -2.5, 7.5])
def test_font_spacing_reaches_every_style(spacing):
    document = Document.parse(generate_ass("青[あお]", [[(0, 0.5), (0.5, 1)]], settings=AssSettings(spacing=spacing)))
    assert all(style.spacing == spacing for style in document.styles.values())


@pytest.mark.parametrize("text,duration", [("abc", 1), ("abc", 0.02), ("きょ", 1.01), ("abc[あ]", 1)])
def test_multicharacter_sound_retains_its_exact_time(text, duration):
    unit = _source(text, [(0, duration)]).source.units[0]
    assert unit.text == ("abc" if text.startswith("abc") else "きょ")
    assert [(s.start_ms, s.end_ms) for s in unit.segments] == [(0, round(duration * 1000))]


@pytest.mark.parametrize("text,expected", [("青[あお]", ["あ", "お"]), ("京[きょう]", ["きょ", "う"])])
def test_one_base_has_separate_ruby_times(text, expected):
    source = _source(text, [(0, 1), (1.5, 2)])
    assert len(source.source.units) == 1
    assert [s.ruby for s in source.source.units[0].segments] == expected
    events = _events(text, [[(0, 1), (1.5, 2)]])
    base = [e for e in events if e.layer == 2 and e.style == "K1"]
    assert len(base) == 1
    assert [_surface(e) for e in events if e.layer == 2 and e.style.endswith("-furigana")] == expected


def test_shared_base_overlap_is_rejected_with_line_and_surface():
    with pytest.raises(ValueError, match="lyric line 1:.*青"):
        generate_ass("青[あお]", [[(0, 2), (1, 3)]])


def test_zero_duration_borrows_time_without_removing_the_rest():
    source = _source("あい", [(0, 0.5), (0.8, 0.8)])
    assert [(s.start_ms, s.end_ms) for u in source.source.units for s in u.segments] == [(0, 420), (720, 800)]
    result = generate_ass("あい", [[(0, 0.5), (0.8, 0.8)]])
    assert r"\t(720,800,0.5," in result
    assert r"\k" not in result


@pytest.mark.parametrize(
    "text,spans,expected",
    [
        ("あい.0", [(0, 1), (1, 1)], [(0, 920), (920, 1000)]),
        ("あ.0い", [(0, 0), (0, 1)], [(0, 80), (80, 1000)]),
        ("あ.0", [(1, 1)], [(1000, 1010)]),
        ("@0{あ}@1{い.0}", [(0, 1), (1, 1)], [(0, 1000), (1000, 1010)]),
        ("青[あお]", [(0, 1), (1, 1)], [(0, 920), (920, 1000)]),
        ("あ@c1{い.0}", [(0, 1), (1, 1)], [(0, 920), (920, 1000)]),
    ],
)
def test_borrowed_display_times(text, spans, expected):
    original = list(spans)
    source = _source(text, spans)
    assert [(s.start_ms, s.end_ms) for u in source.source.units for s in u.segments] == expected
    assert spans == original


def test_guide_dots_share_fades_and_never_scale():
    dots = [e for e in _events("あ", [[(5, 6)]]) if e.style == "LEAD"]
    assert len(dots) == 12
    assert all(r"\fad(800,200)" in e.text for e in dots)
    assert all(r"\fsc" not in e.text for e in dots)


def test_display_order_can_differ_from_time_order():
    events = _events("あい", [[(2, 3), (0, 1)]])
    overlays = [e for e in events if e.layer == 3]
    assert [_surface(e) for e in overlays] == ["あ", "い"]
    assert r"\t(2000,3000,0.5," in overlays[0].text
    assert r"\t(0,1000,0.5," in overlays[1].text
    assert {e.end_time for e in events} == {3200}


def test_excess_simultaneous_rows_fail_instead_of_rendering_offscreen():
    with pytest.raises(ValueError, match="does not fit vertically"):
        generate_ass("\n".join("あ" for _ in range(8)), [[(0, 3)]] * 8)


def test_times_use_ass_centiseconds():
    events = _events("あ", [[(3661.23, 3661.5)]], AssSettings(lead_time_ms=0))
    main = [e for e in events if e.style == "K1"]
    assert {e.start_time for e in main} == {3661230}
    assert "1:01:01.23" in main[0].serialize()


def test_guide_dots_are_positioned_independent_events():
    events = _events("あ", [[(5, 5.5)]])
    dots = [e for e in events if e.style == "LEAD" and e.layer == 3]
    assert len(dots) == 3
    assert all(_surface(e) == "●" and e.start_time == 2000 and e.end_time == 5000 for e in dots)
    for event, start in zip(dots, (0, 1000, 2000), strict=True):
        assert rf"\t({start},{start + 1000},1," in event.text


def test_first_word_too_soon_gets_no_dots():
    assert not any(e.style == "LEAD" for e in _events("あ", [[(1, 1.5)]]))


def test_mapping_override_does_not_change_natural_sounds():
    source = _source("(幾千)[いくせん].1あ", [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1), (1, 1.5)])
    assert [u.text for u in source.source.units] == ["幾千", "あ"]
    assert [s.ruby for s in source.source.units[0].segments] == ["い", "く", "せ", "ん"]


def test_ruby_parts_have_distinct_bases():
    source = _source("胡椒[こ,しょう]", [(0, 0.5), (0.5, 1), (1, 1.5)])
    assert [u.text for u in source.source.units] == ["胡", "椒"]
    assert [[s.ruby for s in u.segments] for u in source.source.units] == [["こ"], ["しょ", "う"]]


@pytest.mark.parametrize("text", ["(あい).1", "泣[な]い.+"])
def test_merge_keeps_sound_times(text):
    source = _source(text, [(0, 0.5), (0.5, 1)])
    assert [(s.start_ms, s.end_ms) for u in source.source.units for s in u.segments] == [(0, 500), (500, 1000)]


def test_font_and_rgb_colour_reach_final_output():
    result = generate_ass("あ", [[(0, 0.5)]], settings=AssSettings(font="Test Font", overlay_color="FF0000"))
    assert "Style: K1,Test Font,96," in result
    assert r"\1c&H0000FF&" in result


def test_unknown_track_style_is_refused():
    with pytest.raises(ValueError, match="track style"):
        generate_ass("あ", [[(0, 0.5)]], settings=AssSettings(track_style={0: "nowhere"}))


def test_no_time_or_mismatched_spans_omits_line():
    assert not _events("あ", [[(None, None)]])
    assert not _events("あ", [[]])


def test_layout_and_effect_options_reach_final_events():
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
    result = generate_ass("字[じ]", [[(5, 6)]], settings=settings)
    doc = Document.parse(result)
    assert doc.styles["K1"].margin_v == 148
    assert doc.styles["K2"].margin_v == 20
    assert doc.styles["LEAD"].margin_v == 276
    assert doc.styles["K1-furigana"].fontsize == 36
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
    ):
        assert tag in result
    shifted = _events("字[じ]", [[(5, 6)]], replace(settings, ruby_offset=0))
    ruby = next(e for e in doc.lines if e.style == "K1-furigana")
    unshifted_ruby = next(e for e in shifted if e.style == "K1-furigana")
    position = re.search(r"\\pos\(([^,]+),([^)]+)\)", ruby.text)
    unshifted_position = re.search(r"\\pos\(([^,]+),([^)]+)\)", unshifted_ruby.text)
    assert position[1] == unshifted_position[1]
    assert float(position[2]) == float(unshifted_position[2]) - 6


def test_zero_borders_and_blur_are_supported():
    result = generate_ass(
        "字[じ]",
        [[(1, 2)]],
        settings=AssSettings(
            border=0,
            border_furi=0,
            blur=0,
            blur_scale=0,
            ruby_offset=0,
            clip_size=0,
        ),
    )
    assert r"\bord0" in result
    assert r"\blur0" in result
    assert "nan" not in result.lower()


def test_positive_offset_moves_lyrics_and_dots_without_mutating_spans():
    spans = [[(5, 5.5)]]
    events = _events("あ", spans, AssSettings(offset_ms=2000))
    assert {(e.start_time, e.end_time) for e in events if e.style == "LEAD"} == {(4000, 7000)}
    assert {(e.start_time, e.end_time) for e in events if e.style == "K1"} == {(2000, 7700)}
    assert spans == [[(5, 5.5)]]


def test_negative_offset_clips_elapsed_segments_without_borrowing():
    source = _source("青[あお]い", [(0, 1), (1, 2), (2.5, 3)], AssSettings(offset_ms=-1500))
    assert [(s.start_ms, s.end_ms) for u in source.source.units for s in u.segments] == [(0, 0), (0, 500), (1000, 1500)]
    assert [s.ruby for s in source.source.units[0].segments] == ["あ", "お"]


def test_partial_syllable_keeps_only_time_after_zero():
    source = _source("あい", [(0, 1), (1, 2)], AssSettings(offset_ms=-250))
    assert [(s.start_ms, s.end_ms) for u in source.source.units for s in u.segments] == [(0, 750), (750, 1750)]


def test_lines_fully_before_zero_are_omitted():
    events = _events("あ\nい", [[(0, 1)], [(3, 4)]], AssSettings(offset_ms=-2000))
    assert {_surface(e) for e in events} == {"い"}


def test_auto_base_blur_follows_manual_overlay_blur():
    settings = AssSettings(overlay_blur_color="123456")
    overlay, base = settings.blur_colours()
    assert overlay == "123456"
    assert base != AssSettings().blur_colours()[1]
    result = generate_ass("あ", [[(0, 1)]], settings=settings)
    assert r"\3c&H563412&" in result
    assert rf"\3c&H{base[4:] + base[2:4] + base[:2]}&" in result
