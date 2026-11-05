# SPDX-License-Identifier: AGPL-3.0-only
"""MIDI ports, one backend: RtMidi through supriya-midi.

The division of labour is deliberate. `namioto.midi` keeps mido, because a Standard MIDI File is a
format and mido is a format library; this file keeps the ports, and a process wants one port backend
rather than two. A mido backend would not even be the smaller adapter: its protocol is not a pair of
functions but `open_input`/`open_output`/`get_input_names`/`get_output_names` plus a whole `Port`
(`poll`, `receive`, `iter_pending`, a message queue of its own), while everything here sends two
bytes and receives two bytes. Building that adapter would mean wrapping RtMidi a second time, and the
project already opens its playback port through supriya-midi, so two RtMidi instances would be real
duplication rather than a seam.

Nothing here imports Qt. An input callback runs on RtMidi's own thread, and this layer hands the
bytes on untouched: what a message means, and how the reader gets back to the GUI thread, are the
caller's business. Both `import supriya_midi` statements sit inside functions so a machine without a
MIDI backend still imports this module - and so the test suite can put a fake one in its place.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress

SYNTH_NAMES = ("timidity", "fluidsynth", "qsynth", "wavetable")


def output_names() -> tuple[str, ...]:
    """The MIDI outputs this machine has; empty rather than an error when there are none."""
    return _names("MidiOut")


def input_names() -> tuple[str, ...]:
    """The MIDI inputs this machine has; empty rather than an error when there are none."""
    return _names("MidiIn")


def _names(device: str) -> tuple[str, ...]:
    try:
        import supriya_midi

        return tuple(getattr(supriya_midi, device)().get_ports())
    except Exception:  # no backend, no device, or a device that will not answer: all the same here
        return ()


def find_synth_port(port_names: Sequence[str]) -> int | None:
    """Index of the first port that looks like a software synth; Midi Through is a loopback."""
    for index, name in enumerate(port_names):
        lowered = name.lower()
        if "through" not in lowered and any(part in lowered for part in SYNTH_NAMES):
            return index
    return None


def find_port(port_names: Sequence[str], wanted: str = "") -> int | None:
    """The port a name asks for, else the first software synth: either way, None when there is none.

    The name is matched loosely as well, because a synthesiser's port is numbered by its client when
    it starts, and that number changes between sessions.
    """
    if wanted:
        for index, name in enumerate(port_names):
            if name == wanted:
                return index
        for index, name in enumerate(port_names):
            if wanted.lower() in name.lower():
                return index
    return find_synth_port(port_names)


class Port:
    """One open MIDI port: two bytes in or out, and nothing above that.

    A caller never imports supriya and never sees an RtMidi object, which is what lets the tests
    stand a fake in its place. `on_message` is settable after the port is open because the sync layer
    builds its input through a factory that knows the port's name but not yet its own callback; a
    port that receives nothing is a no-op there.

    `start` exists for the same reason: a receiver's protocol says when listening begins, and with
    RtMidi it begins as soon as the callback is set, so this only names that moment.
    """

    def __init__(self, port, name: str, *, receiving: bool = False) -> None:
        self._port = port
        self._name = name
        self._receiving = receiving
        self._on_message: Callable | None = None

    @property
    def name(self) -> str:
        """The port's name as the machine gave it, which is what the window shows."""
        return self._name

    @property
    def is_open(self) -> bool:
        return bool(self._port.is_port_open)

    def send(self, message: Iterable[int]) -> None:
        self._port.send_message(message)

    @property
    def on_message(self) -> Callable | None:
        return self._on_message

    @on_message.setter
    def on_message(self, callback: Callable | None) -> None:
        self._on_message = callback
        if callback is not None:
            self._port.set_callback(callback)

    def start(self) -> None:
        """Listening has already begun, because `on_message` set the callback; kept as the signal."""

    def close(self) -> None:
        if self._receiving:
            with suppress(Exception):
                self._port.cancel_callback()
        self._port.close_port()


def open_output(wanted: str = "") -> Port | None:
    """An open output: `wanted` by name, else the first software synth, else None."""
    return _open("MidiOut", wanted)


def open_input(wanted: str = "", on_message: Callable | None = None) -> Port | None:
    """An open input that hears quarter frames only: sysex, clock and sensing are noise here.

    MTC's quarter frame is 0xF1 and nothing else on this port matters, so the three other message
    types RtMidi would hand over are told to stay on the wire.
    """
    port = _open("MidiIn", wanted, receiving=True)
    if port is None:
        return None
    port._port.ignore_types(sysex=True, timing=True, active_sense=True)
    port.on_message = on_message
    return port


def _open(device: str, wanted: str, *, receiving: bool = False) -> Port | None:
    try:
        import supriya_midi

        port = getattr(supriya_midi, device)()
        index = find_port(port.get_ports(), wanted)
        if index is None:
            return None
        port.open_port(index)
        return Port(port, port.get_port_name(index), receiving=receiving)
    except Exception:  # no backend, or a port that refused to open: the caller has a quiet fallback
        return None
