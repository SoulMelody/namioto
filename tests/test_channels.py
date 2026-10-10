# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the channel model and how it rides in the project file."""

from __future__ import annotations

import json

from namioto import project
from namioto.channels import (
    CHANNEL_COUNT,
    Channel,
    audible,
    free_channel,
    set_field,
    valid_color,
)


def make(**values) -> project.Project:
    return project.Project(settings=project.default_settings(), **values)


def test_a_channel_keeps_the_number_it_was_given() -> None:
    assert Channel().channel == 0
    assert Channel(channel=9).channel == 9  # the percussion channel is not special


def test_a_label_speaks_for_an_unnamed_channel() -> None:
    assert Channel(channel=6).label == "Channel 6"
    assert Channel(channel=6, name="Voice").label == "Voice"


def test_a_free_channel_is_the_lowest_one_nobody_plays_on() -> None:
    assert free_channel(()) == 0
    assert free_channel((Channel(channel=0), Channel(channel=2))) == 1
    assert free_channel(tuple(Channel(channel=channel) for channel in range(CHANNEL_COUNT))) is None


def test_a_channel_round_trips_through_the_project_file() -> None:
    channel = Channel(name="Drums", channel=9, program=0, volume=80)
    stored = project.to_dict(project.Project(channels=(channel,), notes=()))["channels"][0]
    assert stored["channel"] == 9
    assert project.from_dict({"format": "namioto", "channels": [stored]}).channels[0] == channel


def test_audible_leaves_out_the_muted_channels() -> None:
    channels = (Channel(), Channel(mute=True), Channel(channel=1))
    assert audible(channels) == [0, 1]


def test_set_field_returns_a_changed_copy() -> None:
    channel = Channel(name="Bass", program=32)
    changed = set_field(channel, program=33, mute=True)
    assert changed.program == 33 and changed.mute
    assert channel.program == 32 and not channel.mute


def test_a_colour_must_be_a_six_digit_hex() -> None:
    assert valid_color("#ff2f2f") == "#ff2f2f"
    assert valid_color("#FF2F2F") == "#ff2f2f"
    for broken in ("", "ff2f2f", "#ff2f2", "#ff2f2fx", "#gggggg", 123456):
        assert valid_color(broken) == ""


def test_a_file_without_channels_gets_one_default_channel() -> None:
    data = {"format": "namioto", "notes": [{"start": 0.0, "duration": 1.0, "pitch": 60}]}
    opened = project.from_dict(data)
    assert opened.channels == (Channel(),)
    assert opened.notes == (project.Note(0.0, 1.0, 60, 0),)


def test_channels_and_note_channels_survive_a_round_trip(tmp_path) -> None:
    path = tmp_path / "song.nto"
    channels = (
        Channel(name="Vocal", color="#ff2f2f", channel=0),
        Channel(name="Bass", channel=3, program=32, volume=90, mute=True),
    )
    notes = (project.Note(0.0, 1.0, 60, 0), project.Note(1.0, 1.0, 43, 3))
    project.save(make(channels=channels, notes=notes), path)
    opened = project.load(path)
    assert opened.channels == channels
    assert opened.notes == notes


def test_a_note_channel_past_the_sixteen_lands_on_the_last_channel() -> None:
    data = {"format": "namioto", "notes": [{"start": 0.0, "duration": 1.0, "pitch": 60, "channel": 99}]}
    opened = project.from_dict(data)
    assert opened.notes == (project.Note(0.0, 1.0, 60, 15),)
    # the default channel is there either way, and the note's is filled in beside it
    assert [channel.channel for channel in opened.channels] == [0, 15]


def test_a_note_on_a_channel_the_file_never_described_gets_a_plain_entry() -> None:
    data = {
        "format": "namioto",
        "channels": [{"name": "Only", "channel": 0}],
        "notes": [{"start": 0.0, "duration": 1.0, "pitch": 60, "channel": 4}],
    }
    opened = project.from_dict(data)
    assert [channel.channel for channel in opened.channels] == [0, 4]
    assert opened.channels[1].name == ""


def test_a_broken_channel_falls_back_to_its_defaults() -> None:
    data = {
        "format": "namioto",
        "channels": [
            {"name": 12, "color": "nope", "program": 999, "volume": "loud", "mute": "yes", "visible": "no", "lock": 3},
        ],
    }
    assert project.from_dict(data).channels == (Channel(),)


def test_a_repeated_channel_keeps_the_first_entry() -> None:
    data = {"format": "namioto", "channels": [{"channel": 2, "name": "First"}, {"channel": 2, "name": "Second"}]}
    assert [channel.name for channel in project.from_dict(data).channels] == ["First"]


def test_the_channels_come_back_in_channel_order() -> None:
    data = {"format": "namioto", "channels": [{"channel": 9}, {"channel": 0}, {"channel": 4}]}
    assert [channel.channel for channel in project.from_dict(data).channels] == [0, 4, 9]


def test_the_written_file_carries_the_channels(tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(make(channels=(Channel(name="Vocal", color="#ff2f2f", program=4),)), path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["channels"] == [
        {
            "name": "Vocal",
            "color": "#ff2f2f",
            "channel": 0,
            "program": 4,
            "volume": 100,
            "mute": False,
            "visible": True,
            "lock": False,
        }
    ]
    assert data["notes"] == []
