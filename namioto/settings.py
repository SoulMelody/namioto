# SPDX-License-Identifier: AGPL-3.0-only
"""The program's own preferences: what it remembers between runs, and the file it keeps them in.

Qt-free on purpose. The table below is the whole set: a value the user can change about the program
itself, never about a song. A document's values belong to `namioto.project`, and the window's own
state to `namioto.state`, so nothing here has to say whether it is remembered or for one file.

`namioto.ui.settings_dialog` builds its pages from this table without repeating anything, and the
`Section`s that get a page are named in that window; `editor` has no page, so its two switches are
kept and never typed in. A value read from the file is checked by `namioto.params.coerce`, which
brings a bad one back in line.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

from namioto import params
from namioto.analysis import devices
from namioto.analysis.choices import ALGORITHMS
from namioto.i18n import LANGUAGE_CODES, LANGUAGE_LABELS, SYSTEM
from namioto.params import Field, Section
from namioto.utils import config_dir, write_json

VERSION = 1

SECTIONS: tuple[Section, ...] = (
    Section(
        "general",
        "General",
        (
            Field(
                "language",
                "choice",
                SYSTEM,
                "Language",
                "Which language the interface speaks; a change takes effect the next time it starts",
                choices=LANGUAGE_CODES,
                labels=LANGUAGE_LABELS,
            ),
            Field(
                "auto_save",
                "bool",
                False,
                "Auto-save",
                "Save the open project once editing stops, and when the window loses focus",
            ),
            Field(
                "style",
                "style",
                "",
                "Style",
                "Which widget style draws the window; the desktop's own unless another one is picked",
            ),
        ),
    ),
    Section(
        "devices",
        "Devices",
        (
            Field(
                "gpu",
                "device",
                devices.AUTO,
                "GPU",
                "Which GPU backend a run that asks for the GPU uses; the line under it says whether "
                "its runtime is installed",
                choices=(devices.AUTO, *devices.GPU_KEYS),
                labels=("Automatic", *devices.GPU_LABELS),
            ),
            Field(
                "power",
                "choice",
                devices.POWER_PREFERENCES[0],
                "GPU power",
                "Which of the machine's GPUs the WebGPU backend runs on: the discrete one, or the "
                "integrated one to save power",
                choices=devices.POWER_PREFERENCES,
                labels=devices.POWER_LABELS,
            ),
        ),
    ),
    Section(
        "tempo",
        "Tempo",
        (
            Field(
                "estimator",
                "choice",
                ALGORITHMS[0],
                "Algorithm",
                "Which algorithm estimates the tempo from the audio",
                choices=ALGORITHMS,
            ),
            Field(
                "window_seconds",
                "float",
                12.0,
                "Window (s)",
                "Length of the windows the local tempo is fitted to",
                low=2,
                high=120,
                step=1,
            ),
            Field(
                "window_hop_seconds",
                "float",
                6.0,
                "Hop (s)",
                "Distance between those windows",
                low=1,
                high=60,
                step=1,
            ),
        ),
    ),
    Section(
        "lyrics",
        "Lyrics",
        (
            Field(
                "api_base",
                "text",
                "",
                "API base",
                "OpenAI-compatible endpoint up to its /v1, such as https://api.deepseek.com/v1",
            ),
            Field(
                "api_key",
                "secret",
                "",
                "API key",
                "Bearer token sent to that endpoint; the settings file keeps it in plain text",
            ),
            Field(
                "model",
                "text",
                "",
                "Model",
                "Model name the endpoint serves, such as deepseek-chat",
            ),
            Field(
                "temperature",
                "float",
                0.2,
                "Temperature",
                "How far the model may wander; the annotation wants it low",
                low=0.0,
                high=2.0,
                step=0.1,
                decimals=1,
            ),
            Field(
                "timeout",
                "float",
                120.0,
                "Timeout (s)",
                "How long one request may take before it is given up on",
                low=1.0,
                high=600.0,
                step=1.0,
            ),
            Field(
                "editor",
                "text",
                "",
                "External editor",
                "Command that opens a .krc, such as code; empty picks the platform's own",
            ),
        ),
    ),
    Section(
        "network",
        "Network",
        (
            Field(
                "github_mirror",
                "text",
                "https://gh-proxy.org",
                "GitHub mirror",
                "Downloads of GitHub releases are fetched through this mirror; empty goes straight to github.com",
            ),
            Field(
                "proxy",
                "text",
                "",
                "Proxy",
                "HTTP(S) proxy for downloads and the lyrics API, such as http://127.0.0.1:7890; "
                "empty follows the environment",
            ),
        ),
    ),
    Section(
        "midi",
        "MIDI",
        (
            Field(
                "wavetone",
                "bool",
                True,
                "WaveTone compatibility",
                "WaveTone's MIDI export starts every note one bar late: reading that back turns it "
                "back, and writing MIDI adds it, so files and the two programs agree",
            ),
        ),
    ),
    # no page: a habit of the editor, kept between runs and never typed into the settings window
    Section(
        "editor",
        "Editor",
        (
            Field(
                "auto_page",
                "bool",
                False,
                "Auto page turn",
                "Take the next page of the roll once the playhead reaches the right of the window",
            ),
            Field(
                "overtone_highlight",
                "bool",
                False,
                "Overtone highlight",
                "Paint the overtones of the row under the mouse - f, 2f, 3f and 4f - as well",
            ),
        ),
    ),
    Section(
        "ass",
        "ASS subtitle",
        (
            Field(
                "font",
                "text",
                "sans-serif",
                "Font",
                "Font the subtitle is rendered in; the kara-templater matches it by name",
            ),
            Field(
                "overlay_color",
                "text",
                "0000FF",
                "Overlay colour",
                "RGB hex of the karaoke overlay, such as 6EB7E3; its blur and base shades follow from it",
            ),
            Field(
                "fade_in_ms",
                "int",
                800,
                "Fade in (ms)",
                "How long a line takes to fade in; it must not exceed the lead time",
                low=0,
                high=2147483647,
            ),
            Field(
                "fade_out_ms",
                "int",
                200,
                "Fade out (ms)",
                "How long a line takes to fade out",
                low=0,
                high=2147483647,
            ),
            Field(
                "lead_time_ms",
                "int",
                5000,
                "Lead time (ms)",
                "How early a line appears before its first word is sung",
                low=0,
                high=2147483647,
            ),
            Field(
                "guide_dot_duration_ms",
                "int",
                1000,
                "Guide dot (ms)",
                "Length of each of the three dots before a chapter's first line",
                low=1,
                high=2147483647,
            ),
            Field("offset_ms", "int", 0, "Subtitle offset (ms)", low=-2147483647, high=2147483647),
            Field("font_size", "int", 96, "Font size", low=1, high=2147483647),
            Field("spacing", "float", 5, "Font spacing", low=-2147483647, high=2147483647),
            Field("border", "int", 5, "Text border", high=2147483647),
            Field("border_furi", "int", 3, "Ruby border", high=2147483647),
            Field("margin_h", "int", 64, "Horizontal margin", high=2147483647),
            Field("margin_v", "int", 48, "Vertical margin", high=2147483647),
            Field("ruby_offset", "int", 10, "Ruby offset", low=-2147483647, high=2147483647),
            Field("base_outline_color", "text", "222222", "Base outline colour"),
            Field("overlay_outline_color", "text", "EFEFEF", "Overlay outline colour"),
            Field("blur", "int", 10, "Blur", high=2147483647),
            Field("blur_scale", "float", 2.0, "Blur border multiplier", high=1000),
            Field("overlay_blur_color", "text", "", "Overlay blur colour"),
            Field("base_blur_color", "text", "", "Base blur colour"),
            Field("clip_size", "int", 24, "Sweep range", high=2147483647),
        ),
    ),
    Section(
        "remote",
        "Remote",
        (
            Field(
                "enabled",
                "bool",
                False,
                "Remote control",
                "Listen for the namioto-ctl remote control interface; off unless it is asked for",
            ),
            Field(
                "address",
                "text",
                "",
                "Address",
                "Where to listen: a socket path, tcp://host:port, or empty for a per-user socket",
            ),
        ),
    ),
)

Settings = params.build(SECTIONS)
FIELD_SPECS = Settings.__field_specs__

get_value = params.get_value
set_value = params.set_value
clone = params.clone


def default_path() -> Path:
    """`$NAMIOTO_SETTINGS`, else the platform's config directory, where preferences belong."""
    from_env = os.environ.get("NAMIOTO_SETTINGS")
    if from_env:
        return Path(from_env)
    return config_dir("settings.json")


def to_dict(settings: Settings) -> dict:
    return {
        "version": VERSION,
        **{
            section.name: {item.name: getattr(getattr(settings, section.name), item.name) for item in section.fields}
            for section in SECTIONS
        },
    }


def from_dict(data) -> Settings:
    settings = Settings()
    if not isinstance(data, dict):
        return settings
    for section in SECTIONS:
        stored = data.get(section.name)
        if isinstance(stored, dict):
            for item in section.fields:
                if item.name in stored:
                    setattr(getattr(settings, section.name), item.name, params.coerce(item, stored[item.name]))
    return settings


def load(path: str | Path | None = None) -> Settings:
    """Read the preferences, falling back to the defaults for anything missing or unusable."""
    target = Path(path) if path is not None else default_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Settings()
    except (OSError, ValueError) as error:
        warnings.warn(f"{target} could not be read as settings ({error}); defaults are in use", stacklevel=2)
        return Settings()
    return from_dict(data)


def save(settings: Settings, path: str | Path | None = None) -> Path:
    """Write the preferences out whole: a half-written file would be read as a broken one."""
    return write_json(to_dict(settings), Path(path) if path is not None else default_path())
