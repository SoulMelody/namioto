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

    def send_message(self, message) -> None:
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
