# SPDX-License-Identifier: AGPL-3.0-only
"""Project palette persistence, legacy migration and fixed defaults."""

from namioto import project
from namioto.channels import Channel
from namioto.lyric_colors import DEFAULT_COLORS


def test_project_palette_round_trip(tmp_path):
    settings = project.default_settings()
    settings.lyric_colors.channel_2 = "#123456"
    path = tmp_path / "colors.nto"
    project.save(project.Project(settings=settings), path)
    loaded = project.load(path)
    assert loaded.settings.lyric_colors.channel_2 == "#123456"
    assert project.default_settings().lyric_colors.channel_2 == DEFAULT_COLORS[2]
    remembered = {}
    project.remember_changes(remembered, project.default_settings(), settings)
    assert "lyric_colors" not in remembered


def test_legacy_project_palette_uses_midi_colors_not_ass_colors():
    data = project.to_dict(project.Project(channels=(Channel(channel=2, color="#123456"),)))
    del data["lyric_colors"]
    data["ass"] = {"overlay_color": "FFFFFF", "channel_colors": "2=ABCDEF"}
    loaded = project.from_dict(data)
    assert loaded.settings.lyric_colors.channel_2 == "#123456"
    assert loaded.settings.lyric_colors.channel_1 == DEFAULT_COLORS[1]
    saved = project.to_dict(loaded)
    saved["channels"][0]["color"] = "#abcdef"
    assert project.from_dict(saved).settings.lyric_colors.channel_2 == "#123456"


def test_invalid_project_colors_use_fixed_defaults():
    settings = project.settings_from_dict(
        {"lyric_colors": {"channel_0": "invalid", "channel_1": "#123456", "channel_2": 123}}
    )
    assert settings.lyric_colors.channel_0 == DEFAULT_COLORS[0]
    assert settings.lyric_colors.channel_1 == "#123456"
    assert settings.lyric_colors.channel_2 == DEFAULT_COLORS[2]
