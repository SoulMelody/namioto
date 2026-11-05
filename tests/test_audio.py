# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for external MIDI playback and silent fallback."""

from __future__ import annotations

import sys
import types

import pytest

from namioto.ui.audio import (
    NOTE_OFF,
    NOTE_ON,
    PROGRAM_CHANGE,
    VELOCITY,
    MidiPortOut,
    SilentPlayer,
    open_player,
)


class FakePort:
    """Collects what a synth would have received."""

    def __init__(self) -> None:
        self.messages: list[list[int]] = []

    def send(self, message) -> None:
        self.messages.append(list(message))


def note_messages(port) -> list[list[int]]:
    """Just the note events on any channel: the player also sends program and volume changes."""
    return [message for message in port.messages if (message[0] & 0xF0) in (NOTE_ON, NOTE_OFF)]


def test_clicking_the_same_note_again_restarts_it(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)

    player.preview(60, 0.15)
    player.preview(60, 0.15)  # the second click releases the note and starts it over
    pump(0.3)

    assert note_messages(port) == [
        [NOTE_ON, 60, VELOCITY],
        [NOTE_OFF, 60, 0],  # a synth that still holds the note would only layer a second one
        [NOTE_ON, 60, VELOCITY],
        [NOTE_OFF, 60, 0],
    ]


def test_stopping_silences_a_preview() -> None:
    port = FakePort()
    player = MidiPortOut(port)

    player.preview(60, 30.0)
    player.stop()

    assert note_messages(port)[-1] == [NOTE_OFF, 60, 0]


def test_the_midi_volume_becomes_a_control_change() -> None:
    port = FakePort()
    player = MidiPortOut(port)

    assert port.messages[0] == [0xB0, 0x07, 127]  # the base class starts it at full volume
    player.gain = 0.5
    assert port.messages[-1] == [0xB0, 0x07, round(0.5 * 127)]
    player.gain = 2.0  # a synth clamps at its own top
    assert port.messages[-1] == [0xB0, 0x07, 127]
    player.gain = -1.0
    assert port.messages[-1] == [0xB0, 0x07, 0]
    assert player.gain == 0.0


def test_each_channel_gets_its_program_and_volume() -> None:
    port = FakePort()
    player = MidiPortOut(port)

    player.set_program([(60, 0.0, 1.0, 0), (43, 0.0, 1.0, 10)], 1.0, ((0, 0, 100), (10, 32, 50)))
    changes = [message for message in port.messages if (message[0] & 0xF0) == PROGRAM_CHANGE]
    volumes = [message for message in port.messages if (message[0] & 0xF0) == 0xB0 and message[1] == 7]
    assert [message[0] & 0x0F for message in changes] == [0, 10]
    assert [message[1] for message in changes] == [0, 32]
    assert [message[2] for message in volumes][-2:] == [127, round(0.5 * 127)]


def test_the_notes_are_played_on_their_own_channel(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.set_program([(60, 0.0, 0.2, 3), (67, 0.0, 0.2, 0)], 1.0, ((0, 0, 100), (3, 0, 100)))
    port.messages.clear()
    player.play()
    pump(0.3)
    player.stop()

    assert note_messages(port) == [
        [NOTE_ON | 3, 60, VELOCITY],
        [NOTE_ON | 0, 67, VELOCITY],
        [NOTE_OFF | 3, 60, 0],
        [NOTE_OFF | 0, 67, 0],
    ]


def test_two_note_ons_for_one_pitch_are_both_released_on_stop(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)

    player.set_program([(69, 0.0, 1.0, 0), (69, 0.0, 1.0, 0)], 1.0, ((0, 0, 100),))
    port.messages.clear()
    player.play()
    pump(0.1)  # both note-ons of the one pitch are sounding
    player.pause()

    assert note_messages(port) == [
        [NOTE_ON, 69, VELOCITY],
        [NOTE_ON, 69, VELOCITY],
        [NOTE_OFF, 69, 0],
        [NOTE_OFF, 69, 0],  # one note-off per note-on, not one for the pitch
    ]


def test_a_note_ending_where_the_next_begins_is_released_first(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)

    player.set_program([(69, 0.1, 0.2, 0), (69, 0.3, 0.2, 0)], 1.0, ((0, 0, 100),))
    port.messages.clear()
    player.play()
    pump(0.35)  # 0.1 + 0.2 lands a bit past 0.3, where the next note starts
    player.stop()

    assert note_messages(port)[:3] == [
        [NOTE_ON, 69, VELOCITY],
        [NOTE_OFF, 69, 0],  # the pitch is let go before it is struck again
        [NOTE_ON, 69, VELOCITY],
    ]


def test_the_external_synth_is_preferred_when_one_is_listening() -> None:
    player, name = open_player()
    assert isinstance(player, MidiPortOut)
    assert name == "TiMidity:Fake TiMidity port 0 128:0"  # the port the suite is given instead of a real one


@pytest.mark.parametrize("unavailable", ["no_ports", "backend_error", "open_error"])
def test_without_midi_output_playback_and_preview_stay_silent(monkeypatch, unavailable) -> None:
    fake_midi = types.ModuleType("supriya_midi")

    class NoPorts:
        def get_ports(self) -> list[str]:
            if unavailable == "backend_error":
                raise RuntimeError("MIDI unavailable")
            return ["TiMidity"] if unavailable == "open_error" else []

        def open_port(self, index) -> None:
            raise RuntimeError("Port unavailable")

    fake_midi.MidiOut = NoPorts
    monkeypatch.setitem(sys.modules, "supriya_midi", fake_midi)

    player, name = open_player()
    assert isinstance(player, SilentPlayer)
    assert player.silent and name == "No MIDI output"

    player.set_program([(60, 0.0, 1.0)], 1.0)
    assert player.duration == pytest.approx(1.0)
    player.play(0.5)
    player.preview(60)
    assert not player.is_playing and player.position == 0.5
    player.pause()
    assert player.position == 0.5
    player.stop()
    assert player.position == 0.0


def offsets(events) -> list[float]:
    return [round(event[0], 6) for event in events]


def test_a_positive_delay_pushes_the_whole_timeline_later() -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.set_program([(60, 0.0, 0.5, 0), (62, 1.0, 0.5, 0)], 1.0)

    player.play(0.0)
    plain = offsets(player._timeline())

    player.delay_ms = 500
    player.play(0.0)
    delayed = offsets(player._timeline())

    assert plain == [0.0, 0.5, 1.0, 1.5]
    assert delayed == [0.5, 1.0, 1.5, 2.0]  # every event moved by exactly the delay
    player.stop()


def test_a_negative_delay_moves_the_cursor_forward_and_drops_what_it_passed() -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.set_program(
        [
            (59, 9.9, 0.2, 0),  # ends before the shifted cursor: dropped outright
            (60, 9.95, 0.2, 0),  # these three start left of the cursor: never struck again
            (61, 9.98, 0.2, 0),
            (62, 9.99, 0.2, 0),
            (63, 10.5, 0.7, 0),
            (64, 10.75, 0.15, 0),
            (65, 11.0, 0.1, 0),
        ],
        1.0,
    )
    player.delay_ms = -500
    player.play(10.0)

    events = player._timeline()
    assert offsets(events) == [0.0, 0.25, 0.4, 0.5, 0.6, 0.7]
    assert [event[1] for event in events] == [NOTE_ON, NOTE_ON, NOTE_OFF, NOTE_ON, NOTE_OFF, NOTE_OFF]
    assert [event[2] for event in events] == [63, 64, 64, 65, 65, 63]
    player.stop()


@pytest.mark.parametrize("milliseconds", [-500, 0, 500])
def test_a_note_left_of_the_cursor_is_never_struck_again(milliseconds) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.set_program([(60, 0.0, 3.0, 0), (62, 1.0, 2.0, 0)], 1.0)
    player.delay_ms = milliseconds
    player.play(1.0)

    first_on = next(event for event in player._timeline() if event[1] == NOTE_ON)
    assert first_on[2] == 62  # 60 began before the cursor, and a delay must not bring it back
    player.stop()


def test_a_positive_delay_holds_a_preview_before_it_sounds(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.delay_ms = 200

    player.preview(60, 0.15)
    assert note_messages(port) == []  # the note waits for the delay

    pump(0.4)
    assert note_messages(port) == [[NOTE_ON, 60, VELOCITY], [NOTE_OFF, 60, 0]]


def test_a_negative_delay_does_not_make_a_preview_sound_early() -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.delay_ms = -500

    player.preview(60, 0.15)
    assert note_messages(port)[0] == [NOTE_ON, 60, VELOCITY]  # a mouse click cannot play ahead of itself
    player.stop()


def test_stopping_cancels_a_preview_that_is_still_waiting(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.delay_ms = 500

    player.preview(60, 0.15)
    player.stop()

    pump(0.8)
    assert note_messages(port) == []  # nothing was ever struck, so nothing is owed a note-off


def test_clicking_a_waiting_preview_again_starts_it_once(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.delay_ms = 100

    player.preview(60, 0.3)
    player.preview(60, 0.3)  # the second click replaces the wait rather than layering a note
    pump(0.6)

    assert note_messages(port) == [[NOTE_ON, 60, VELOCITY], [NOTE_OFF, 60, 0]]


def test_clicking_a_sounding_preview_again_releases_it_first(pump) -> None:
    port = FakePort()
    player = MidiPortOut(port)
    player.delay_ms = 100

    player.preview(60, 0.3)
    pump(0.2)  # the delay has passed, so the note is sounding
    player.preview(60, 0.3)
    pump(0.6)

    assert note_messages(port) == [
        [NOTE_ON, 60, VELOCITY],
        [NOTE_OFF, 60, 0],  # a synth still holding the note only layers a second one
        [NOTE_ON, 60, VELOCITY],
        [NOTE_OFF, 60, 0],
    ]
