# SPDX-License-Identifier: AGPL-3.0-only
"""The score model: note normalisation and the channel invariants, with no Qt in sight."""

from __future__ import annotations

from namioto.channels import Channel
from namioto.document import MIN_DURATION, PITCH_MAX, PITCH_MIN, Document, Note


def test_a_note_lands_inside_the_roll() -> None:
    note = Note(pitch=PITCH_MAX + 40, start=-3.0, duration=0.0, channel=99)
    assert note.pitch == PITCH_MAX
    assert note.start == 0.0
    assert note.duration == MIN_DURATION
    assert note.channel == 15
    assert Note(pitch=PITCH_MIN - 40, start=0.0, duration=1.0).pitch == PITCH_MIN


def test_a_note_keeps_its_end_with_it() -> None:
    note = Note(60, 2.0, 1.5)
    assert note.end == 3.5
    note.set_range(4.0, 72)
    note.set_duration(0.5)
    assert (note.start, note.pitch, note.end) == (4.0, 72, 4.5)


def test_two_notes_that_sound_alike_are_still_two() -> None:
    first = Note(60, 1.0, 1.0)
    twin = Note(60, 1.0, 1.0)
    document = Document(notes=[first, twin])
    assert first is not twin
    document.remove_note(twin)
    assert len(document.notes) == 1 and document.notes[0] is first


def test_only_the_same_pitch_on_the_same_channel_collides() -> None:
    document = Document(notes=[Note(60, 1.0, 1.0, channel=0)])
    assert document.collides(60, 1.5, 1.0, 0) is True
    assert document.collides(60, 0.5, 1.0, 0) is True
    assert document.collides(62, 1.0, 1.0, 0) is False  # another pitch may sound with it
    assert document.collides(60, 1.0, 1.0, 1) is False  # another channel is another track
    assert document.collides(60, 2.0, 1.0, 0) is False  # touching at the edge is not a clash


def test_a_note_is_no_obstacle_to_itself() -> None:
    note = Note(60, 1.0, 1.0)
    document = Document(notes=[note])
    assert document.collides(60, 1.0, 1.0, 0) is True
    assert document.collides(60, 1.0, 1.0, 0, ignore=(note,)) is False


def test_removing_a_channel_takes_its_notes_and_leaves_the_numbers_alone() -> None:
    document = Document(
        channels=[Channel(name="A", channel=0), Channel(name="B", channel=1), Channel(name="C", channel=3)]
    )
    document.add_note(Note(60, 0.0, 1.0, channel=0))
    document.add_note(Note(62, 0.0, 1.0, channel=1))
    document.add_note(Note(64, 0.0, 1.0, channel=3))

    removed = document.remove_channel(1)
    assert [note.pitch for note in removed] == [62]
    assert [channel.name for channel in document.channels] == ["A", "C"]
    assert [(note.pitch, note.channel) for note in document.notes] == [(60, 0), (64, 3)]


def test_the_last_channel_stays() -> None:
    document = Document()
    assert document.remove_channel(0) is None
    assert len(document.channels) == 1


def test_a_note_on_a_channel_the_document_never_heard_of_gets_an_entry() -> None:
    document = Document(channels=[Channel(name="A")], notes=[Note(60, 0.0, 1.0, channel=4)])
    assert [channel.channel for channel in document.channels] == [0, 4]


def test_setting_channels_keeps_the_notes_on_the_channels_left_out() -> None:
    document = Document(channels=[Channel(name="A"), Channel(name="B", channel=5)])
    document.add_note(Note(60, 0.0, 1.0, channel=5))
    document.set_channels([Channel(name="A")])
    assert [channel.channel for channel in document.channels] == [0, 5]
    assert [(note.pitch, note.channel) for note in document.notes] == [(60, 5)]


def test_setting_a_channel_field_reaches_the_right_channel() -> None:
    document = Document(channels=[Channel(name="A"), Channel(name="B", channel=5)])
    document.set_channel_field(5, name="Lead", program=40)
    assert [(channel.channel, channel.name, channel.program) for channel in document.channels] == [
        (0, "A", 0),
        (5, "Lead", 40),
    ]


def test_a_fresh_note_takes_the_next_stable_id() -> None:
    document = Document()
    first = document.add_note(Note(60, 0.0, 1.0))
    second = document.add_note(Note(62, 0.0, 1.0))
    assert (first.id, second.id) == (1, 2)
    assert document.next_id == 3


def test_a_note_keeps_the_id_it_came_with_and_the_next_one_carries_on() -> None:
    document = Document(notes=[Note(60, 0.0, 1.0, id=7)])
    assert document.notes[0].id == 7
    assert document.next_id == 8
    added = document.add_note(Note(62, 0.0, 1.0))
    assert added.id == 8


def test_a_note_from_a_file_without_ids_gets_them_in_order() -> None:
    document = Document(notes=[Note(60, 0.0, 1.0), Note(62, 0.0, 1.0), Note(64, 0.0, 1.0)])
    assert [note.id for note in document.notes] == [1, 2, 3]
    assert document.next_id == 4


def test_rebuilding_notes_does_not_reuse_removed_ids() -> None:
    document = Document(notes=[Note(60, 0.0, 1.0, id=100), Note(62, 1.0, 1.0, id=200)])
    document.replace_notes([Note(60, 0.0, 1.0, id=100)])
    assert document.next_id == 201
    document.replace_notes([])
    assert document.add_note(Note(64, 2.0, 1.0)).id == 201


def test_fresh_notes_do_not_take_ids_of_later_existing_notes() -> None:
    document = Document(notes=[Note(60, 0.0, 1.0), Note(62, 1.0, 1.0, id=1)])
    assert [note.id for note in document.notes] == [2, 1]
    assert document.next_id == 3


def test_replacing_the_notes_keeps_a_free_id_free() -> None:
    document = Document()
    document.replace_notes([Note(60, 0.0, 1.0, id=4)], next_id=9)
    assert document.next_id == 9
    assert document.add_note(Note(62, 0.0, 1.0)).id == 9
