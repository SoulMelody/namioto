# SPDX-License-Identifier: AGPL-3.0-only
"""Built-in karaoke effects compiled directly to positioned ASS events, without templates."""

from kara_templater import Box, Line

SWEEP_FAST_FRAC = 0.3
SWEEP_SPEED_RATIO = 2


def _number(value):
    return format(value, ".6f").rstrip("0").rstrip(".")


def _clip(left, right):
    return rf"\clip({_number(left)},0,{_number(right)},1080)"


def _transform(start, end, tags, accel=1):
    return rf"\t({_number(start)},{_number(end)},{_number(accel)},{tags})"


def _escape(text):
    return text.replace("\\", "\\\u2060").replace("{", "\\{").replace("}", "\\}").replace("\n", r"\N")


def _sweep(clip_left, left, right, start, end, *, guide=False):
    if end <= start:
        return _transform(start, start, _clip(clip_left, right))
    if guide:
        return _transform(start, end, _clip(clip_left, right))
    fraction = SWEEP_FAST_FRAC / (SWEEP_FAST_FRAC + SWEEP_SPEED_RATIO * (1 - SWEEP_FAST_FRAC))
    middle = start + (end - start) * fraction
    edge = left + (right - left) * SWEEP_FAST_FRAC
    return _transform(start, middle, _clip(clip_left, edge), 1 / SWEEP_SPEED_RATIO) + _transform(
        middle, end, _clip(clip_left, right)
    )


def _paint(
    text,
    box,
    style,
    show,
    hide,
    segments,
    settings,
    colours,
    *,
    ruby=False,
    guide=False,
    segment_colours=(),
):
    if not text or not text.strip() or hide <= show:
        return []
    overlay, overlay_blur, base_blur, base_outline, overlay_outline = colours
    border = settings.border_furi if ruby else settings.border
    blur_border = round(border * settings.blur_scale)
    position = rf"\an5\pos({_number(box.center)},{_number(box.middle)})\shad0"
    first = min((s[0] for s in segments), default=show + settings.fade_in_ms)
    last = max((s[1] for s in segments), default=hide - settings.fade_out_ms)
    fade_in = min(settings.fade_in_ms, max(0, first - show))
    fade_out = min(settings.fade_out_ms, max(0, hide - last))
    common = position + rf"\fad({fade_in},{fade_out})"
    if segments and not guide:
        start = min(s[0] for s in segments) - show
        end = max(s[1] for s in segments) - show
        common += r"\fscx100\fscy100"
        common += _transform(start, end, r"\fscx110\fscy110", 0.5)
        common += _transform(end, end + (end - start) / 2, r"\fscx100\fscy100", 2)
    events = []

    def emit(layer, tags):
        events.append(
            Line(
                text="{" + common + tags + "}" + _escape(text),
                start_time=show,
                end_time=hide,
                style=style,
                layer=layer,
            )
        )

    emit(0, rf"\bord{blur_border}\blur{settings.blur}\3c&H{base_blur}&\alpha&H33&")
    emit(2, rf"\bord{border}\1c&HFFFFFF&\3c&H{base_outline}&")
    for index, (absolute_start, absolute_end, left, right) in enumerate(segments):
        if segment_colours:
            overlay, overlay_blur, _, _, overlay_outline = segment_colours[index]
        start, end = absolute_start - show, absolute_end - show
        padding = settings.clip_size
        clip_left = left - padding if index == 0 else left
        clip_right = right + padding if index == len(segments) - 1 else right
        glow_clip = _clip(clip_left, clip_right)
        glow = rf"\bord{blur_border}\blur{settings.blur}\1c&H{overlay}&\3c&H{overlay_blur}&\alpha&HCC&"
        glow += glow_clip + _transform(start, end, r"\alpha&H33&")
        emit(1, glow)
        sweep = _clip(clip_left, clip_left)
        sweep += _transform(max(0, start - padding), start, _clip(clip_left, left))
        sweep += _sweep(clip_left, left, right, start, end, guide=guide)
        sweep += _transform(end, end + padding, _clip(clip_left, clip_right))
        emit(3, rf"\bord{border}\1c&H{overlay}&\3c&H{overlay_outline}&" + sweep)
    return events


def render_unit(unit, style, show, hide, settings, colours, *, guide=False, segment_colours=()):
    """Render one positioned base and its ruby, preserving every independent interval."""
    segments = [(s.source.start_ms, s.source.end_ms, s.left, s.right) for s in unit.segments]
    events = _paint(
        unit.source.text,
        unit.box,
        style,
        show,
        hide,
        segments,
        settings,
        colours,
        guide=guide,
        segment_colours=segment_colours,
    )
    for index, segment in enumerate(unit.segments):
        box = segment.ruby
        if box is None:
            continue
        box = Box(box.left, box.top + settings.ruby_offset, box.width, box.height)
        timing = [(segment.source.start_ms, segment.source.end_ms, box.left, box.right)]
        events.extend(
            _paint(
                segment.source.ruby,
                box,
                style + "-furigana",
                show,
                hide,
                timing,
                settings,
                segment_colours[index] if segment_colours else colours,
                ruby=True,
            )
        )
    return events
