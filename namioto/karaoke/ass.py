# SPDX-License-Identifier: AGPL-3.0-only
"""The ASS karaoke subtitle: the header its templates need, and the dialogues the lyrics make.

`generate_ass` turns a `.krc` text and the per-sound times the strip maps it to into one ASS file.
The header is the two styles the lines alternate on (`K1`/`K2`, `H1`/`H2` for a second track), the
`LEAD` guide dots, and the `template syl`/`template furi` comment rows the Aegisub kara-templater
turns into per-syllable motion; each dialogue then carries one `\\k` per syllable, a ruby written
`base|<ruby` or `#|ruby` the way the template reads it.

`AssSettings` holds the export options: timing, font, layout, colours and blur. Unspecified blur
colours are derived from the overlay, while the sweep's motion stays fixed.

Qt-free: the module reads a `.krc` and times and returns text; the editor is what calls it.
"""

from __future__ import annotations

import colorsys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from namioto.karaoke.sounds import natural_sounds

# the sweep's middle stretch: the first SWEEP_FAST_FRAC of the distance at an average of
# SWEEP_SPEED_RATIO times the steady speed, easing down to it, the two meeting at t1_frac
SWEEP_FAST_FRAC = 0.3
SWEEP_SPEED_RATIO = 2

MIN_WORD_DURATION = 8  # centiseconds a borrowed-in word keeps
MIN_GAP_MS = 100  # least gap two rows of one style may leave
FALLBACK_COLOR = "0000FF"  # the overlay the module falls back to when the settings name no colour
SMALL_KANA = "ャュョァィゥェォゃゅょぁぃぅぇぉ"

TRACK_STYLES: dict[str, tuple[str, str]] = {"default": ("K1", "K2"), "top": ("H1", "H2")}
DEFAULT_TRACK_STYLE = "default"
LEAD_STYLE = "LEAD"

_SCRIPT_INFO = (
    "[Script Info]",
    "Title: Karaoke Subtitle",
    "ScriptType: v4.00+",
    "WrapStyle: 0",
    "ScaledBorderAndShadow: yes",
    "YCbCr Matrix: None",
    "PlayResX: 1920",
    "PlayResY: 1080",
)
_PROJECT_GARBAGE = (
    "[Aegisub Project Garbage]",
    "Audio File: origin.wav",
    "Video File: video.mp4",
    "Video AR Mode: 4",
    "Video AR Value: 1.777778",
    "Video Zoom Percent: 0.500000",
)
_STYLES_FORMAT = (
    "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
    "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
    "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENTS_FORMAT = "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"


@dataclass(frozen=True)
class AssSettings:
    """Export preferences; colours are RGB hex, with empty blur colours derived automatically."""

    font: str = "sans-serif"
    overlay_color: str = FALLBACK_COLOR  # RGB hex, converted to the ASS order
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
        """The resolved blur colours in RGB order, for the export form's automatic values."""
        config = _Config.from_settings(self)
        return _to_ass(config.overlay_blur), _to_ass(config.base_blur)


@dataclass(frozen=True)
class _Config:
    """The colours and spacings the templates are written with, derived once per file."""

    settings: AssSettings
    overlay: str
    overlay_blur: str
    base_blur: str
    top_overlay: str
    top_overlay_blur: str
    top_base_blur: str

    @classmethod
    def from_settings(cls, settings: AssSettings) -> _Config:
        overlay = _to_ass(settings.overlay_color)
        overlay_blur = (
            _to_ass(settings.overlay_blur_color) if settings.overlay_blur_color else _shade(overlay, 0.6, 1.0)
        )
        base_blur = _to_ass(settings.base_blur_color) if settings.base_blur_color else _base_shade(overlay_blur)
        return cls(
            settings=settings,
            overlay=overlay,
            overlay_blur=overlay_blur,
            base_blur=base_blur,
            top_overlay=overlay,
            top_overlay_blur=overlay_blur,
            top_base_blur=base_blur,
        )

    @property
    def border_blur(self) -> int:
        return round(self.settings.border * self.settings.blur_scale)

    @property
    def border_furi_blur(self) -> int:
        return round(self.settings.border_furi * self.settings.blur_scale)

    @property
    def margin_k1(self) -> int:
        return self.settings.margin_v * 2 + self.settings.font_size // 2 * 3

    @property
    def margin_lead(self) -> int:
        return self.settings.margin_v * 3 + self.settings.font_size * 3

    @property
    def ruby_offset(self) -> str:
        offset = self.settings.ruby_offset
        return f"{offset:+d}" if offset else ""


def _to_ass(rgb: str) -> str:
    """A six-digit RGB hex in the ASS order (BGR); anything else falls back to the default blue."""
    text = str(rgb).strip().lstrip("#")
    if len(text) != 6 or any(char not in "0123456789abcdefABCDEF" for char in text):
        text = FALLBACK_COLOR
    return text[4:] + text[2:4] + text[:2]


def _shade(bgr: str, saturation_scale: float, value: float) -> str:
    """The same hue, its saturation and value moved, written back in the ASS order."""
    blue, green, red = (int(bgr[at : at + 2], 16) for at in (0, 2, 4))
    hue, saturation, _ = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    saturation = max(0.0, min(1.0, saturation * saturation_scale))
    value = max(0.0, min(1.0, value))
    red, green, blue = (round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, value))
    return f"{blue:02X}{green:02X}{red:02X}"


def _base_shade(bgr: str) -> str:
    """The complementary hue, desaturated and dimmed: the base colour under a blurred overlay."""
    blue, green, red = (int(bgr[at : at + 2], 16) for at in (0, 2, 4))
    hue, saturation, _ = colorsys.rgb_to_hsv(red / 255.0, green / 255.0, blue / 255.0)
    hue = (hue + 0.5) % 1.0
    saturation *= 0.618
    red, green, blue = (round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, 0.618))
    return f"{blue:02X}{green:02X}{red:02X}"


def _pos(x, y) -> str:
    return rf"\pos({x},{y})"


def _an(v) -> str:
    return rf"\an{v}"


def _shad(v) -> str:
    return rf"\shad{v}"


def _blur(v) -> str:
    return rf"\blur{v}"


def _fad(t1, t2) -> str:
    return rf"\fad({t1},{t2})"


def _bord(v) -> str:
    return rf"\bord{v}"


def _fscx(v) -> str:
    return rf"\fscx{v}"


def _fscy(v) -> str:
    return rf"\fscy{v}"


def _alpha(v) -> str:
    if isinstance(v, int):
        v = f"{v:02X}"
    return rf"\alpha&H{v!s}&"


def _color(*, primary=None, secondary=None, outline=None, back=None) -> str:
    """The four colour tags: primary, secondary, outline and back."""
    tags = ((r"\1c", primary), (r"\2c", secondary), (r"\3c", outline), (r"\4c", back))
    return "".join(rf"{tag}&H{v}&" for tag, v in tags if v is not None)


def _t(*parts, accel=None) -> str:
    # the four-argument form is \t(t1,t2,accel,tags), accel before the tags, as libass and VSFilter read it
    if accel is not None:
        *head, tags = parts
        parts = (*head, accel, tags)
    return rf"\t({','.join(str(part) for part in parts)})"


def _clip(*parts) -> str:
    return rf"\clip({','.join(str(part) for part in parts)})"


def _size_pop() -> list[str]:
    """The syllable's size pulse: grow to 1.1x over its time, then back over half of it again."""
    return [
        _fscx(100),
        _fscy(100),
        _t("$sstart", "$send", _fscx(110) + _fscy(110), accel=0.5),
        _t("$send", "!$send+$sdur/2!", _fscx(100) + _fscy(100), accel=2),
    ]


def _style(name, *, alignment, margin_v, config: _Config, fontname=None, fontsize=None, outline=2, shadow=2) -> str:
    fontname = config.settings.font if fontname is None else fontname
    fontsize = config.settings.font_size if fontsize is None else fontsize
    return (
        f"Style: {name},{fontname},{fontsize},"
        f"&H00FFFFFF,&H000000FF,&H00{_to_ass(config.settings.base_outline_color)},&H00000000,"
        f"0,0,0,0,100,100,{config.settings.spacing:g},0,1,{outline},{shadow},{alignment},"
        f"{config.settings.margin_h},{config.settings.margin_h},{margin_v},1"
    )


def _template(effect: str, *, style: str, layer: int, commands: list[str], name: str = "") -> str:
    return f"Comment: {layer},0:00:00.00,0:00:00.00,{style},{name},0,0,0,{effect}," + "{" + "".join(commands) + "}"


def _assemble(styles: list[str], effects: list[str]) -> str:
    return (
        "\n".join(
            [
                *_SCRIPT_INFO,
                "",
                *_PROJECT_GARBAGE,
                "",
                "[V4+ Styles]",
                _STYLES_FORMAT,
                *styles,
                "",
                "[Events]",
                _EVENTS_FORMAT,
                *effects,
            ]
        )
        + "\n"
    )


def _effect_rows(
    style: str,
    *,
    config: _Config,
    overlay: str,
    overlay_blur: str,
    base_blur: str,
    base_outline: str,
    overlay_outline: str,
    scale: bool = True,
    furi: bool = True,
    sweep: bool = True,
    explicit: bool = False,
) -> list[str]:
    """The karaoke template rows of one style: the sweep's four layers, then the ruby's four."""

    def num(value: float) -> str:
        return format(value, "g")

    if explicit:
        overlay = f"!syl.namioto_overlay or '{overlay}'!"
        overlay_blur = f"!syl.namioto_overlay_blur or '{overlay_blur}'!"
        base_blur = f"!syl.namioto_base_blur or '{base_blur}'!"
    size = config.settings.clip_size
    closed_clip = _clip(f"!$sleft-{size}!", 0, f"!$sleft-{size}!", 1080)
    pop = _size_pop() if scale else []

    if sweep:
        fast = SWEEP_FAST_FRAC
        t1_frac = fast / (fast + SWEEP_SPEED_RATIO * (1 - fast))
        fast_end_x = f"!math.floor($sleft*{num(1 - fast)}+$sright*{num(fast)})!"
        mid_clip = [
            _t(
                "$sstart",
                f"!$sstart+$sdur*{num(t1_frac)}!",
                _clip(f"!$sleft-{size}!", 0, fast_end_x, 1080),
                accel=1 / SWEEP_SPEED_RATIO,
            ),
            _t(f"!$sstart+$sdur*{num(t1_frac)}!", "$send", _clip(f"!$sleft-{size}!", 0, "$sright", 1080)),
        ]
    else:
        mid_clip = [_t("$sstart", "$send", _clip(f"!$sleft-{size}!", 0, "$sright", 1080))]

    def quad(effect: str, overlay_effect: str, y: str, *, blur_bord: int, edge_bord: int) -> list[str]:
        return [
            _template(
                effect,
                style=style,
                layer=0,
                commands=[
                    _pos("$center", y),
                    _an(5),
                    _shad(0),
                    _blur(config.settings.blur),
                    _fad(config.settings.fade_in_ms, config.settings.fade_out_ms),
                    *pop,
                    _bord(blur_bord),
                    _color(outline=base_blur),
                    _alpha(0x33),
                ],
            ),
            _template(
                overlay_effect,
                style=style,
                layer=1,
                name="overlay",
                commands=[
                    _pos("$center", y),
                    _an(5),
                    _shad(0),
                    _blur(config.settings.blur),
                    _fad(config.settings.fade_in_ms, config.settings.fade_out_ms),
                    _color(primary=overlay, outline=overlay_blur),
                    _alpha(0xCC),
                    _t("$sstart", "$send", _alpha(0x33)),
                    *pop,
                    _bord(blur_bord),
                ],
            ),
            _template(
                effect,
                style=style,
                layer=2,
                commands=[
                    _pos("$center", y),
                    _an(5),
                    _shad(0),
                    _fad(config.settings.fade_in_ms, config.settings.fade_out_ms),
                    *pop,
                    _bord(edge_bord),
                    _color(outline=base_outline),
                ],
            ),
            _template(
                overlay_effect,
                style=style,
                layer=3,
                name="overlay",
                commands=[
                    _pos("$center", y),
                    _an(5),
                    _shad(0),
                    _fad(config.settings.fade_in_ms, config.settings.fade_out_ms),
                    _color(primary=overlay, outline=overlay_outline),
                    closed_clip,
                    _t(f"!$sstart-{size}!", "$sstart", _clip(f"!$sleft-{size}!", 0, "$sleft", 1080)),
                    *mid_clip,
                    _t("$send", f"!$send+{size}!", _clip(f"!$sleft-{size}!", 0, f"!$sright+{size}!", 1080)),
                    *pop,
                    _bord(edge_bord),
                ],
            ),
        ]

    rows = quad(
        "template syl",
        "template syl noblank",
        "$middle",
        blur_bord=config.border_blur,
        edge_bord=config.settings.border,
    )
    if furi:
        rows += quad(
            "template furi",
            "template furi",
            f"!$middle{config.ruby_offset}!",
            blur_bord=config.border_furi_blur,
            edge_bord=config.settings.border_furi,
        )
    return rows


def _timing_code(style: str, config: _Config) -> str:
    colours = []
    for channel, colour in config.settings.channel_colours().items():
        owned = _Config.from_settings(replace(config.settings, overlay_color=colour))
        colours.append(f"[{channel}]={{'{owned.overlay}','{owned.overlay_blur}','{owned.base_blur}'}}")
    code = (
        "local channel, first, last, base_last = (syl.inline_fx or ''):match('^namioto_(%d+)_(%d+)_(%d+)_(%d+)$'); "
        "if channel then syl.start_time = first + 0; "
        "syl.end_time = (syl.isfuri and last or base_last) + 0; "
        "syl.duration = syl.end_time - syl.start_time; "
        f"local colours = ({{{','.join(colours)}}})[channel + 0]; "
        "if colours then syl.namioto_overlay = colours[1]; syl.namioto_overlay_blur = colours[2]; "
        "syl.namioto_base_blur = colours[3] end end"
    )
    return f"Comment: 0,0:00:00.00,0:00:00.00,{style},,0,0,0,code syl furi,{code}"


def _header(config: _Config, *, explicit: bool = False, extra_rows: Sequence[str] = ()) -> str:
    styles = [
        _style("K1", alignment=1, margin_v=config.margin_k1, config=config),
        _style("K2", alignment=3, margin_v=config.settings.margin_v, config=config),
        _style(LEAD_STYLE, alignment=1, margin_v=config.margin_lead, fontname="sans-serif", config=config),
        _style("H1", alignment=8, margin_v=config.settings.margin_v, config=config),
        _style("H2", alignment=8, margin_v=config.margin_k1, config=config),
    ]
    for name in extra_rows:
        styles.append(
            _style(
                name,
                alignment=8 if name.startswith("H") else 1,
                margin_v=config.settings.margin_v
                + (int(name[1:]) - 1) * (config.settings.font_size * 2 + config.settings.margin_v),
                config=config,
            )
        )
    base_outline = _to_ass(config.settings.base_outline_color)
    overlay_outline = _to_ass(config.settings.overlay_outline_color)
    effects: list[str] = []
    for name in ("K1", "K2"):
        if explicit:
            effects.append(_timing_code(name, config))
        effects += _effect_rows(
            name,
            config=config,
            overlay=config.overlay,
            overlay_blur=config.overlay_blur,
            base_blur=config.base_blur,
            base_outline=base_outline,
            overlay_outline=overlay_outline,
            explicit=explicit,
        )
    effects += _effect_rows(
        LEAD_STYLE,
        config=config,
        overlay=config.overlay,
        overlay_blur=config.overlay_blur,
        base_blur=config.base_blur,
        base_outline=base_outline,
        overlay_outline=overlay_outline,
        scale=False,
        furi=False,
        sweep=False,
    )
    for name in ("H1", "H2"):
        if explicit:
            effects.append(_timing_code(name, config))
        effects += _effect_rows(
            name,
            config=config,
            overlay=config.top_overlay,
            overlay_blur=config.top_overlay_blur,
            base_blur=config.top_base_blur,
            base_outline=base_outline,
            overlay_outline=overlay_outline,
            explicit=explicit,
        )
    for name in extra_rows:
        if explicit:
            effects.append(_timing_code(name, config))
        effects += _effect_rows(
            name,
            config=config,
            overlay=config.overlay,
            overlay_blur=config.overlay_blur,
            base_blur=config.base_blur,
            base_outline=base_outline,
            overlay_outline=overlay_outline,
            explicit=explicit,
        )
    return _assemble(styles, effects)


@dataclass
class _Glyph:
    """One rendered syllable: the base line's text, its furigana, and what the reading is.

    `text` is what the base layer shows - the Sound's own surface, `#` when it continues the same
    base unit, or `#` plus the surface when a new ruby part shares the word. `reading` is the ruby
    the template gets.
    """

    text: str
    reading: str
    ruby: str | None = None
    small: bool = False
    continuation: bool = False
    channel: int = 0
    container: int = -1


@dataclass
class _GlyphTiming:
    glyph: _Glyph
    start_ms: int
    end_ms: int
    clipped: bool = False

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


@dataclass
class _LineTimings:
    glyph_timings: list[_GlyphTiming]


@dataclass
class _PlannedLine:
    track: int
    timing: _LineTimings


@dataclass
class _LineInfo:
    chapter_idx: int
    line_idx: int
    style: str
    timing: _LineTimings
    line_start_ms: int = -1
    line_end_ms: int = 0

    @property
    def first_word_start(self) -> int:
        return min(item.start_ms for item in self.timing.glyph_timings if not item.clipped)


def _ms(seconds: float) -> int:
    return int(round(float(seconds) * 1000))


def _glyph(sound, continuation: bool) -> _Glyph:
    """What one Sound renders as: its own surface without a ruby, its base and reading with one."""
    if not sound.rubied:
        return _Glyph(
            text=sound.reading, reading=sound.reading, small=sound.reading in SMALL_KANA, channel=sound.channel
        )
    if continuation:
        text = "#"
    elif sound.first:
        text = sound.base
    else:
        text = "#" + sound.base
    return _Glyph(
        text=text,
        reading=sound.reading,
        ruby=sound.reading,
        small=sound.reading in SMALL_KANA,
        continuation=continuation,
        channel=sound.channel,
        container=sound.container,
    )


def _line_timings(line, spans: Sequence[tuple | None], offset_ms: int = 0) -> _LineTimings | None:
    """One glyph per Sound, its own derived span; a dropped Sound has no time of its own.

    The spans are one per Sound, so nothing is counted or divided here - a Sound of two characters
    is one syllable, and a ruby of two morae is one Sound per mora. A line none of whose spans has a
    time is dropped, since there is nothing to draw it on.
    """
    if len(spans) != len(line.sounds):
        return None
    timings: list[_GlyphTiming] = []
    seen = False
    previous: int | None = None
    last = 0
    for sound, span in zip(line.sounds, spans, strict=True):
        continuation = sound.rubied and previous == sound.container
        clipped = False
        if span is None or span[0] is None or span[1] is None:
            start = end = last
            clipped = bool(offset_ms < 0 and not seen)
        else:
            start, end = _ms(span[0]) + offset_ms, _ms(span[1]) + offset_ms
            clipped = offset_ms < 0 and end <= 0
            seen = seen or not clipped
            start, end = max(0, start), max(0, end)
        last = end
        timings.append(_GlyphTiming(_glyph(sound, continuation), start, end, clipped))
        previous = sound.container
    rendered = []
    for container_index, container in enumerate(line.containers):
        owned = [
            (sound, item)
            for sound, item in zip(line.sounds, timings, strict=True)
            if sound.container == container_index
        ]
        if container.key[0] != "top":
            rendered.extend(item for _, item in owned)
            continue
        cursor = 0
        for sound, item in owned:
            for index in range(cursor, sound.first_atom):
                channel = container.channels[index]
                preceding = next((entry for entry in reversed(rendered) if entry.glyph.channel == channel), None)
                following = next((entry for entry in timings[sound.index :] if entry.glyph.channel == channel), item)
                time = preceding.end_ms if preceding else following.start_ms
                rendered.append(
                    _GlyphTiming(
                        _Glyph(container.atoms[index], container.atoms[index], channel=channel),
                        time,
                        time,
                        (preceding or following).clipped,
                    )
                )
            rendered.append(item)
            cursor = sound.last_atom + 1
        for index in range(cursor, len(container.atoms)):
            channel = container.channels[index]
            preceding = next((entry for entry in reversed(rendered) if entry.glyph.channel == channel), None)
            following = next(
                (
                    entry
                    for sound, entry in zip(line.sounds, timings, strict=True)
                    if sound.container > container_index and entry.glyph.channel == channel
                ),
                None,
            )
            anchor = preceding or following
            time = preceding.end_ms if preceding else following.start_ms if following else 0
            rendered.append(
                _GlyphTiming(
                    _Glyph(container.atoms[index], container.atoms[index], channel=channel),
                    time,
                    time,
                    anchor.clipped if anchor else False,
                )
            )
    return _LineTimings(rendered) if seen else None


def _plan(lines: Sequence, times: Sequence[Sequence[tuple | None]], offset_ms: int = 0) -> list[list[_PlannedLine]]:
    """The glyphs of every line, grouped by chapter, with the mapping's spans in reading order."""
    planned: list[list[_PlannedLine]] = []
    rows: list[_PlannedLine] = []
    chapter = -1
    for index, line in enumerate(lines):
        if line.chapter != chapter:
            if chapter != -1:
                planned.append(rows)
            rows = []
            chapter = line.chapter
        timing = _line_timings(line, times[index] if index < len(times) else (), offset_ms)
        if timing is not None:
            rows.append(_PlannedLine(line.track, timing))
    if chapter != -1:
        planned.append(rows)
    return planned


def _assign_row_styles(planned: list[list[_PlannedLine]], track_style: Mapping[int, str]) -> dict[str, list[_LineInfo]]:
    """Alternate every track's lines over the two rows of its style, and group them by style."""
    style_rows: dict[str, list[_LineInfo]] = {}
    for chapter_idx, chapter_rows in enumerate(planned):
        counters: dict[str, int] = {}
        for line_idx, line in enumerate(chapter_rows):
            style_name = track_style.get(line.track, DEFAULT_TRACK_STYLE)
            rows = TRACK_STYLES.get(style_name)
            if rows is None:
                raise ValueError(f"unknown track style {style_name!r} (choose from {', '.join(TRACK_STYLES)})")
            preferred = rows[counters.get(style_name, 0) % 2]
            first = min(item.start_ms for item in line.timing.glyph_timings if not item.clipped)
            candidates = [
                preferred,
                *(row for row in rows if row != preferred),
                *(row for row in style_rows if row.startswith(rows[0][0]) and row not in rows),
            ]
            style = next(
                (row for row in candidates if not style_rows.get(row) or style_rows[row][-1].line_end_ms <= first), None
            )
            if style is None:
                style = rows[0][0] + str(3 + sum(row.startswith(rows[0][0]) and row not in rows for row in style_rows))
            counters[style_name] = counters.get(style_name, 0) + 1
            info = _LineInfo(chapter_idx, line_idx, style, line.timing)
            info.line_end_ms = max(item.end_ms for item in line.timing.glyph_timings)
            style_rows.setdefault(style, []).append(info)
    return style_rows


def _calculate_lead_times(lines: list[_LineInfo], config: _Config, lead_time_ms: int) -> None:
    """Give each row of one style a start early enough to fade in before its first word."""
    for index, line in enumerate(lines):
        ideal = line.first_word_start - lead_time_ms
        if index == 0:
            line.line_start_ms = max(0, ideal)
            continue
        previous = lines[index - 1]
        previous_end = previous.line_end_ms + config.settings.fade_out_ms
        required = line.first_word_start - config.settings.fade_in_ms
        if ideal - previous_end > MIN_GAP_MS:
            previous.line_end_ms += config.settings.fade_out_ms
            line.line_start_ms = ideal
        elif required - previous_end > MIN_GAP_MS:
            previous.line_end_ms += config.settings.fade_out_ms
            line.line_start_ms = previous_end + MIN_GAP_MS
        else:
            # the two rows are too close to fade apart cleanly; let the fade overlap the karaoke
            previous.line_end_ms += config.settings.fade_out_ms
            line.line_start_ms = max(0, line.first_word_start - config.settings.fade_in_ms)


def _format_ass_time(ms: int) -> str:
    """Milliseconds as `H:MM:SS.CC`, the hundredth the format counts in."""
    centiseconds = int(ms) // 10
    seconds, remainder = divmod(centiseconds, 100)
    minutes, second = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours}:{minute:02d}:{second:02d}.{remainder:02d}"


def _render_guide_dots(style_rows: dict[str, list[_LineInfo]], duration_ms: int) -> list[str]:
    """Three dots before the first line of each chapter, on the main track's bottom row."""
    firsts: dict[int, _LineInfo] = {}
    for row in TRACK_STYLES[DEFAULT_TRACK_STYLE]:
        for line in style_rows.get(row, []):
            if line.chapter_idx not in firsts or line.line_idx < firsts[line.chapter_idx].line_idx:
                firsts[line.chapter_idx] = line
    dialogues = []
    for chapter_idx in sorted(firsts):
        start = firsts[chapter_idx].first_word_start
        total = duration_ms * 3
        if start < total:
            continue
        dot = duration_ms // 10
        text = "".join(f"{{\\k{dot}}}●" for _ in range(3))
        dialogues.append(
            f"Dialogue: 0,{_format_ass_time(start - total)},{_format_ass_time(start)},{LEAD_STYLE},,0,0,0,,{text}"
        )
    return dialogues


def _render_karaoke_dialogues(
    style_rows: dict[str, list[_LineInfo]], lead_time_ms: int, *, explicit: bool = False
) -> list[str]:
    """Every line as one dialogue, in (chapter, line) order."""
    lines = [line for row in style_rows.values() for line in row]
    lines.sort(key=lambda line: (line.chapter_idx, line.line_idx))
    dialogues = []
    for line in lines:
        same = style_rows[line.style]
        current = next(index for index, other in enumerate(same) if other is line)
        next_start = same[current + 1].line_start_ms if current + 1 < len(same) else None
        actual_lead = line.first_word_start - line.line_start_ms
        text = _generate_karaoke_text(line.timing, actual_lead, next_start, line.line_end_ms, explicit=explicit)
        dialogues.append(
            f"Dialogue: 0,{_format_ass_time(max(0, line.line_start_ms))},"
            f"{_format_ass_time(line.line_end_ms)},{line.style},,0,0,0,,{text}"
        )
    return dialogues


def _generate_karaoke_text(
    timing: _LineTimings, lead_time_ms: int, next_start_ms: int | None, line_end_ms: int, *, explicit: bool = False
) -> str:
    """One line's `\\k` stream, the lead and every Sound's ruby written the way the template reads."""
    parts = [f"{{\\k{lead_time_ms // 10}}}"]
    glyphs = timing.glyph_timings
    line_start = min(item.start_ms for item in glyphs if not item.clipped) - lead_time_ms
    base_ends = {}

    def tag(item, duration, start=None, end=None):
        marker = ""
        if explicit:
            start = item.start_ms if start is None else start
            end = item.end_ms if end is None else end
            base_end = base_ends.get(item.glyph.container, end) if item.glyph.ruby is not None else end
            marker = f"\\-namioto_{item.glyph.channel}_{start - line_start}_{end - line_start}_{base_end - line_start}"
        return f"{{\\k{duration}{marker}}}"

    durations = [max(0, item.duration_ms) // 10 for item in glyphs]

    for index, item in enumerate(glyphs):
        glyph = item.glyph
        if item.clipped:
            continue
        if durations[index] == 0 and glyph.small:
            for earlier in range(index - 1, -1, -1):
                if glyphs[earlier].glyph.channel == glyph.channel and durations[earlier] >= 2 * MIN_WORD_DURATION:
                    durations[index] = durations[earlier] // 2
                    durations[earlier] -= durations[index]
                    break
        elif durations[index] == 0 and glyph.text.strip(" 　"):
            borrowed = False
            for earlier in range(index - 1, -1, -1):
                if glyphs[earlier].glyph.channel == glyph.channel and durations[earlier] >= 2 * MIN_WORD_DURATION:
                    durations[earlier] -= MIN_WORD_DURATION
                    durations[index] = MIN_WORD_DURATION
                    borrowed = True
                    break
            if not borrowed:
                durations[index] = -1

    for index in range(len(glyphs)):
        if durations[index] == -1:
            for later in range(index + 1, len(durations)):
                if (
                    glyphs[later].glyph.channel == glyphs[index].glyph.channel
                    and durations[later] > 2 * MIN_WORD_DURATION
                ):
                    durations[later] -= MIN_WORD_DURATION
                    durations[index] = MIN_WORD_DURATION
                    break
            if durations[index] == -1:
                durations[index] = 1

    shifts: dict[int, int] = {}
    adjusted = []
    for item, duration in zip(glyphs, durations, strict=True):
        channel = item.glyph.channel
        shift = shifts.get(channel, 0)
        delta = duration * 10 - max(0, item.duration_ms) // 10 * 10
        adjusted.append(_GlyphTiming(item.glyph, item.start_ms + shift, item.end_ms + shift + delta, item.clipped))
        if not item.clipped:
            shifts[channel] = shift + delta

    base_ends.clear()
    for item in adjusted:
        if item.glyph.ruby is not None:
            base_ends[item.glyph.container] = max(base_ends.get(item.glyph.container, 0), item.end_ms)

    for index, (item, duration) in enumerate(zip(adjusted, durations, strict=True)):
        glyph = item.glyph
        if index > 0:
            gap = glyphs[index].start_ms - glyphs[index - 1].end_ms
            if gap > 0:
                gap_cs = gap // 10
                if "|" in parts[-1] and glyph.continuation:
                    parts.append(f"{{\\k{gap_cs}}}#|")
                else:
                    parts.append(f"{{\\k{gap_cs}}}")
        text = glyph.text
        if glyph.ruby is not None:
            if text == "#":
                split = "|"
            elif text.startswith("#"):
                split = "|"
                text = text[1:]
            else:
                split = "|<"
            text = f"{text}{split}{glyph.ruby}"
        if glyph.ruby is None and len(text) > 1 and duration >= len(text):
            base, extra = divmod(duration, len(text))
            for position, char in enumerate(text):
                start = item.start_ms + item.duration_ms * position // len(text)
                end = item.start_ms + item.duration_ms * (position + 1) // len(text)
                parts.append(tag(item, base + (1 if position < extra else 0), start, end) + char)
        else:
            parts.append(tag(item, duration) + text)
    if next_start_ms is not None:
        extension = next_start_ms - line_end_ms
        if extension >= lead_time_ms:
            parts.append(f"{{\\k{extension // 10}}}")
    return "".join(parts)


def generate_ass(
    lyrics_text: str, spans: Sequence[Sequence[tuple | None]], *, settings: AssSettings | None = None
) -> str:
    """The ASS karaoke subtitle of a canonical `.krc` and the mapping's derived spans.

    `spans` holds one `(start, end)` per Sound of every line, or `None` where the Sound is dropped -
    the same shape `sound_spans` derives from the operations. A line with no time anywhere is left
    out, so a `.krc` longer than its notes still writes.
    """
    settings = settings or AssSettings()
    config = _Config.from_settings(settings)
    planned = _plan(natural_sounds(lyrics_text), spans, settings.offset_ms)
    style_rows = _assign_row_styles(planned, settings.track_style)
    for rows in style_rows.values():
        if rows:
            _calculate_lead_times(rows, config, settings.lead_time_ms)
    dialogues = _render_guide_dots(style_rows, settings.guide_dot_duration_ms)
    explicit = bool(settings.channel_colours()) or any(
        right.start_ms < left.end_ms
        for chapter in planned
        for line in chapter
        for left, right in zip(line.timing.glyph_timings, line.timing.glyph_timings[1:], strict=False)
    )
    dialogues += _render_karaoke_dialogues(style_rows, settings.lead_time_ms, explicit=explicit)
    return (
        _header(
            config,
            explicit=explicit,
            extra_rows=tuple(
                row for row in style_rows if row not in {name for rows in TRACK_STYLES.values() for name in rows}
            ),
        )
        + "\n".join(dialogues)
        + "\n"
    )
