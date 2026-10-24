# SPDX-License-Identifier: AGPL-3.0-only
"""Compile authoritative KRC and independent Sound times into playable ASS.

Namioto owns lyric semantics, row scheduling and its built-in effects. The
kara-templater library owns text measurement, ruby layout and ASS serialization.
No intermediate karaoke tags or executable templates are emitted. Qt-free.
"""

from __future__ import annotations

import colorsys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from kara_templater import Document, KaraokeLine, Record, Segment, Style, TextUnit, layout_line

from namioto.karaoke.ass_effects import render_unit
from namioto.karaoke.sounds import natural_sounds

FALLBACK_COLOR = "0000FF"
TRACK_STYLES = {"default": ("K1", "K2"), "top": ("H1", "H2")}


@dataclass(frozen=True)
class AssSettings:
    """Export preferences; colours are RGB hex, with automatic blur colours."""

    font: str = "sans-serif"
    overlay_color: str = FALLBACK_COLOR
    fade_in_ms: int = 800
    fade_out_ms: int = 200
    lead_time_ms: int = 5000
    guide_dot_duration_ms: int = 1000
    offset_ms: int = 0
    font_size: int = 96
    spacing: float = 5
    border: int = 5
    border_furi: int = 3
    margin_h: int = 64
    margin_v: int = 48
    ruby_offset: int = 10
    base_outline_color: str = "222222"
    overlay_outline_color: str = "EFEFEF"
    blur: int = 10
    blur_scale: float = 2.0
    overlay_blur_color: str = ""
    base_blur_color: str = ""
    clip_size: int = 24
    track_style: Mapping[int, str] = field(default_factory=dict)
    channel_colors: Mapping[int, str] | str = ""

    def channel_colours(self) -> dict[int, str]:
        if isinstance(self.channel_colors, str):
            colours = {}
            for entry in self.channel_colors.split(","):
                if not entry.strip():
                    continue
                number, separator, colour = entry.partition("=")
                if not separator or not number.strip().isdigit():
                    raise ValueError("channel colours must use channel=RRGGBB pairs")
                colours[int(number)] = colour.strip()
        else:
            colours = dict(self.channel_colors)
        for channel, colour in colours.items():
            if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel < 16:
                raise ValueError("a colour channel must be between 0 and 15")
            if not isinstance(colour, str):
                raise ValueError("channel colours must be six-digit RGB colours")
            colour = colour.lstrip("#")
            if len(colour) != 6 or any(char not in "0123456789abcdefABCDEF" for char in colour):
                raise ValueError("channel colours must be six-digit RGB colours")
            colours[channel] = colour.upper()
        return colours

    def blur_colours(self) -> tuple[str, str]:
        overlay = _to_ass(self.overlay_color)
        overlay_blur = _to_ass(self.overlay_blur_color) if self.overlay_blur_color else _shade(overlay, 0.6, 1)
        base = _to_ass(self.base_blur_color) if self.base_blur_color else _shade(overlay_blur, 0.618, 0.618, 0.5)
        return _to_ass(overlay_blur), _to_ass(base)


def _to_ass(rgb: str) -> str:
    text = str(rgb).strip().lstrip("#")
    if len(text) != 6 or any(char not in "0123456789abcdefABCDEF" for char in text):
        text = FALLBACK_COLOR
    return text[4:] + text[2:4] + text[:2]


def _shade(bgr: str, saturation_scale: float, value: float, hue_offset: float = 0) -> str:
    blue, green, red = (int(bgr[at : at + 2], 16) / 255 for at in (0, 2, 4))
    hue, saturation, _ = colorsys.rgb_to_hsv(red, green, blue)
    red, green, blue = (
        round(channel * 255)
        for channel in colorsys.hsv_to_rgb((hue + hue_offset) % 1, min(1, saturation * saturation_scale), value)
    )
    return f"{blue:02X}{green:02X}{red:02X}"


@dataclass
class _DisplayLine:
    source: KaraokeLine
    colours: tuple[int, ...]
    segment_colours: tuple[tuple[int, ...], ...]
    track: int
    chapter: int
    index: int
    first: int
    last: int
    style: str = ""
    show: int = 0
    hide: int = 0


def _convert(line, spans, settings, index):
    if len(spans) != len(line.sounds):
        return None
    units, colours, segment_colours = [], [], []
    first, last = None, 0
    for container_index, container in enumerate(line.containers):
        owned = [sound for sound in line.sounds if sound.container == container_index]
        cursor = 0
        for sound in owned:
            if container.key[0] == "top":
                for atom in range(cursor, sound.first_atom):
                    units.append(TextUnit(container.atoms[atom]))
                    colour = container.color_channels[atom]
                    colours.append(container.channels[atom] if colour is None else colour)
                    segment_colours.append(())
            span = spans[sound.index]
            segment = None
            if span is not None and span[0] is not None and span[1] is not None:
                start, end = (round(float(value) * 1000) + settings.offset_ms for value in span)
                if end < start:
                    raise ValueError(f"lyric line {index + 1}: reversed interval {start}..{end}")
                if end > 0 or settings.offset_ms >= 0:
                    segment = Segment(max(0, start), max(0, end), sound.reading if sound.rubied else "")
                    first = segment.start_ms if first is None else min(first, segment.start_ms)
                    last = max(last, segment.end_ms)
            if sound.rubied and segment is None:
                anchor = units[-1].segments[-1].end_ms if sound is not owned[0] and units[-1].segments else 0
                segment = Segment(anchor, anchor, sound.reading)
            colour = sound.channel if sound.color_channel is None else sound.color_channel
            if sound.rubied and sound is not owned[0]:
                unit = units[-1]
                segments = unit.segments + ((segment,) if segment is not None else ())
                try:
                    units[-1] = TextUnit(unit.text, segments)
                except ValueError as error:
                    raise ValueError(f"lyric line {index + 1}: {error}") from error
                segment_colours[-1] += (colour,)
            else:
                units.append(TextUnit(sound.base if sound.rubied else sound.reading, (segment,) if segment else ()))
                colours.append(colour)
                segment_colours.append((colour,) if segment else ())
            cursor = sound.last_atom + 1
        if container.key[0] == "top":
            for atom in range(cursor, len(container.atoms)):
                units.append(TextUnit(container.atoms[atom]))
                colour = container.color_channels[atom]
                colours.append(container.channels[atom] if colour is None else colour)
                segment_colours.append(())
    if first is None:
        return None
    preceding = []
    previous = None
    for unit, colour in zip(units, colours, strict=True):
        preceding.append(previous)
        if unit.segments:
            previous = (colour, unit.segments[-1].end_ms)
    following = None
    for position in range(len(units) - 1, -1, -1):
        unit, colour = units[position], colours[position]
        if unit.segments:
            following = (colour, unit.segments[0].start_ms)
            continue
        previous = preceding[position]
        if following is not None and (previous is None or (previous[0] != colour and colour == following[0])):
            anchor = following[1]
        else:
            anchor = previous[1] if previous is not None else first
        units[position] = TextUnit(unit.text, (Segment(anchor, anchor),))
        segment_colours[position] = (colour,)
    return _DisplayLine(
        KaraokeLine(tuple(units)), tuple(colours), tuple(segment_colours), line.track, line.chapter, index, first, last
    )


def _schedule(lines, settings):
    rows = {}
    counters = {}
    for line in lines:
        family = settings.track_style.get(line.track, "default")
        if family not in TRACK_STYLES:
            raise ValueError(f"unknown track style {family!r}")
        names = TRACK_STYLES[family]
        key = (line.chapter, family)
        preferred = names[counters.get(key, 0) % 2]
        candidates = [preferred, *(name for name in names if name != preferred)]
        candidates += [name for name in rows if name.startswith(names[0][0]) and name not in names]
        line.style = next((name for name in candidates if not rows.get(name) or rows[name][-1].last <= line.first), "")
        if not line.style:
            line.style = names[0][0] + str(1 + sum(name.startswith(names[0][0]) for name in rows))
        counters[key] = counters.get(key, 0) + 1
        rows.setdefault(line.style, []).append(line)
    for row in rows.values():
        for index, line in enumerate(row):
            line.show = max(0, line.first - settings.lead_time_ms)
            line.hide = line.last + settings.fade_out_ms
            if index:
                previous = row[index - 1]
                previous.hide = min(previous.hide, line.first)
                line.show = max(line.show, previous.hide)
    return rows


def _style(name, settings):
    upper_margin = settings.margin_v * 2 + settings.font_size // 2 * 3
    if name == "LEAD":
        align, margin = 1, settings.margin_v * 3 + settings.font_size * 3
    elif name in ("K1", "K2", "H1", "H2"):
        align = {"K1": 1, "K2": 3, "H1": 8, "H2": 8}[name]
        margin = upper_margin if name in ("K1", "H2") else settings.margin_v
    else:
        align = 8 if name.startswith("H") else 1
        margin = settings.margin_v + (int(name[1:]) - 1) * (settings.font_size * 2 + settings.margin_v)
    return Style(
        name=name,
        fontname="sans-serif" if name == "LEAD" else settings.font,
        fontsize=settings.font_size,
        spacing=settings.spacing,
        align=align,
        margin_l=settings.margin_h,
        margin_r=settings.margin_h,
        margin_v=margin,
        outline=settings.border,
        shadow=0,
    )


def generate_ass(
    lyrics_text: str, spans: Sequence[Sequence[tuple | None]], *, settings: AssSettings | None = None
) -> str:
    """Compile KRC with seconds-valued Sound spans into final, independently timed events."""
    settings = settings or AssSettings()
    palette = settings.channel_colours()
    lines = []
    for index, line in enumerate(natural_sounds(lyrics_text)):
        converted = _convert(line, spans[index] if index < len(spans) else (), settings, index)
        if converted is not None:
            lines.append(converted)
    rows = _schedule(lines, settings)
    styles = {name: _style(name, settings) for name in dict.fromkeys(("K1", "K2", "LEAD", "H1", "H2", *rows))}
    ruby_styles = {
        name: replace(style, name=name + "-furigana", fontsize=style.fontsize / 2, outline=settings.border_furi)
        for name, style in styles.items()
        if name != "LEAD"
    }
    events = []

    def resolved_colours(colour):
        owned = replace(settings, overlay_color=palette.get(colour, settings.overlay_color))
        overlay_blur, base_blur = owned.blur_colours()
        return tuple(
            _to_ass(value)
            for value in (
                owned.overlay_color,
                overlay_blur,
                base_blur,
                owned.base_outline_color,
                owned.overlay_outline_color,
            )
        )

    def render(source, colours, name, show, hide, *, guide=False, segment_colours=()):
        layout = layout_line(source, style=styles[name], ruby_style=ruby_styles.get(name))
        boxes = [unit.box for unit in layout.units]
        boxes += [
            replace(s.ruby, top=s.ruby.top + settings.ruby_offset)
            for unit in layout.units
            for s in unit.segments
            if s.ruby is not None
        ]
        if any(box.top < 0 or box.top + box.height > 1080 for box in boxes):
            raise ValueError(f"subtitle row {name} does not fit vertically; reduce font size or simultaneous lines")
        for index, (unit, colour) in enumerate(zip(layout.units, colours, strict=True)):
            per_segment = tuple(resolved_colours(c) for c in segment_colours[index]) if segment_colours else ()
            events.extend(
                render_unit(
                    unit,
                    name,
                    show,
                    hide,
                    settings,
                    resolved_colours(colour),
                    guide=guide,
                    segment_colours=per_segment,
                )
            )

    firsts = {}
    for line in lines:
        if not line.style.startswith("H"):
            firsts.setdefault(line.chapter, line)
    duration = settings.guide_dot_duration_ms
    if duration > 0:
        for line in firsts.values():
            if line.first >= duration * 3:
                start = line.first - duration * 3
                source = KaraokeLine(
                    tuple(TextUnit("●", (Segment(start + i * duration, start + (i + 1) * duration),)) for i in range(3))
                )
                render(source, (0, 0, 0), "LEAD", start, line.first, guide=True)
    for line in lines:
        render(line.source, line.colours, line.style, line.show, line.hide, segment_colours=line.segment_colours)
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
        "PlayResX: 1920",
        "PlayResY: 1080",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        *styles.values(),
        *ruby_styles.values(),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        *events,
    ]
    return Document(header, Record(res_x=1920, res_y=1080, video_x_correct_factor=1)).dumps()
