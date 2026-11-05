# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the channel model and how it rides in the project file."""

from __future__ import annotations

import json

from namioto import project
from namioto.channels import (
    CHANNEL_COUNT,
    Channel,
    audible,
    copied_track,
    copy_track,
    delete_track,
    free_channel,
    paste_track,
    set_field,
    valid_color,
)
from namioto.document import Note


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
            "solo": False,
            "visible": True,
            "lock": False,
        }
    ]
    assert data["notes"] == []


def test_solo_is_off_until_it_is_turned_on() -> None:
    assert Channel().solo is False
    assert Channel(solo=True).solo is True
    # a project written before solo existed comes back with every channel unsoloed
    data = {"format": "namioto", "channels": [{"channel": 0, "name": "Old"}]}
    assert project.from_dict(data).channels[0].solo is False


def test_solo_survives_a_round_trip(tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(make(channels=(Channel(name="Lead", solo=True),)), path)
    assert json.loads(path.read_text(encoding="utf-8"))["channels"][0]["solo"] is True
    assert project.load(path).channels[0].solo is True


def test_a_soloed_channel_sounds_even_while_it_is_muted() -> None:
    channels = (Channel(channel=0), Channel(channel=1, mute=True, solo=True), Channel(channel=2))
    assert audible(channels) == [1]  # solo is the last word: the mute beside it does not silence it


def test_two_soloed_channels_both_sound() -> None:
    channels = (Channel(channel=0, solo=True), Channel(channel=1, mute=True, solo=True), Channel(channel=2))
    assert audible(channels) == [0, 1]


def test_without_a_solo_only_the_muted_channels_are_left_out() -> None:
    channels = (Channel(channel=0), Channel(channel=1, mute=True), Channel(channel=2, solo=False))
    assert audible(channels) == [0, 2]


def test_a_copied_track_carries_its_channel_and_its_notes() -> None:
    channels = (Channel(channel=0), Channel(name="Bass", channel=1, program=32, volume=90, color="#ff2f2f"))
    notes = (Note(60, 0.0, 1.0, 1, 7), Note(64, 2.0, 0.5, 1), Note(67, 0.0, 1.0, 0))
    clip = copy_track(1, channels, notes)

    assert clip.channel == channels[1]
    assert [(note.pitch, note.start, note.duration, note.channel, note.id) for note in clip.notes] == [
        (60, 0.0, 1.0, 1, 7),
        (64, 2.0, 0.5, 1, 0),
    ]
    assert copied_track() is clip  # the clip is the clipboard until another track is copied


def test_copying_a_track_leaves_the_document_alone() -> None:
    channels = [Channel(name="Vocal")]
    notes = [Note(60, 0.0, 1.0, 0, 7)]
    clip = copy_track(0, channels, notes)

    assert clip.notes[0] is not notes[0]  # the clip holds copies, not the document's own notes
    assert clip.notes[0].pitch == 60 and clip.notes[0].id == 7
    assert notes[0].channel == 0 and notes[0].start == 0.0
    assert channels == [Channel(name="Vocal")]


def test_a_pasted_track_lands_on_the_target_channel() -> None:
    channels = [Channel(name="Vocal", channel=0, program=4, volume=80, color="#112233"), Channel(channel=1)]
    notes = [Note(60, 0.0, 1.0, 0), Note(64, 2.0, 0.5, 0, 9)]
    clip = copy_track(0, channels, notes)
    pasted_channels, pasted_notes = paste_track(clip, 1, channels, notes)

    assert pasted_channels[1] == Channel(name="Vocal", channel=1, program=4, volume=80, color="#112233")
    assert [note.channel for note in pasted_notes] == [0, 0, 1, 1]  # appended, and moved to the target
    assert [(note.pitch, note.start, note.duration) for note in pasted_notes[2:]] == [
        (60, 0.0, 1.0),
        (64, 2.0, 0.5),
    ]
    # nothing was touched in place, and the source notes keep their channel
    assert channels == [Channel(name="Vocal", channel=0, program=4, volume=80, color="#112233"), Channel(channel=1)]
    assert len(notes) == 2 and [note.channel for note in notes] == [0, 0]


def test_deleting_a_track_empties_it_and_leaves_the_other_one_alone() -> None:
    channels = [Channel(name="Vocal", channel=0, program=4), Channel(name="Bass", channel=1, mute=True)]
    notes = [Note(60, 0.0, 1.0, 0), Note(43, 1.0, 1.0, 1)]
    cleared_channels, cleared_notes = delete_track(0, channels, notes)

    assert cleared_channels == [Channel(channel=0), channels[1]]  # the card stays, its settings go
    assert cleared_channels[1] is channels[1]
    assert [(note.pitch, note.channel) for note in cleared_notes] == [(43, 1)]
    assert len(channels) == 2 and channels[0].name == "Vocal"  # the input is untouched


def test_pasting_a_track_writes_no_temporary_file(tmp_path) -> None:
    channels = [Channel(channel=0), Channel(channel=1)]
    notes = [Note(60, 0.0, 1.0, 0)]
    clip = copy_track(0, channels, notes)
    paste_track(clip, 1, channels, notes)

    assert list(tmp_path.iterdir()) == []  # the clip lives in memory; no temp file stands in for it
