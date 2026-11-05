# SPDX-License-Identifier: AGPL-3.0-only
"""The port layer on its own: which port a name picks, and what reaches the device.

Every device here is a fake `supriya_midi` put in `sys.modules`, so no test opens a real port - and
none needs MIDI installed, which is also why the module imports its backend inside each function.
"""

from __future__ import annotations

import sys
import types

import pytest

from namioto import midi_port


class FakeDevice:
    """One stand-in for a supriya-midi device: it records what it was asked to do."""

    def __init__(self, ports: list[str]) -> None:
        self._ports = list(ports)
        self.opened: int | None = None
        self.port_name = ""
        self.sent: list[list[int]] = []
        self.callback = None
        self.ignored: tuple[bool, bool, bool] | None = None
        self.is_port_open = False

    def get_ports(self) -> list[str]:
        return list(self._ports)

    def get_port_name(self, index: int) -> str:
        return self._ports[index]

    def open_port(self, index: int, name: str = "RtMidi") -> None:
        self.opened = index
        self.port_name = name
        self.is_port_open = True

    def send_message(self, message) -> None:
        self.sent.append(list(message))

    def set_callback(self, callback, data=None) -> None:
        self.callback = callback

    def cancel_callback(self) -> None:
        self.callback = None

    def ignore_types(self, sysex=True, timing=True, active_sense=True) -> None:
        self.ignored = (sysex, timing, active_sense)

    def close_port(self) -> None:
        self.is_port_open = False


@pytest.fixture
def install_backend(monkeypatch):
    """A fake backend listing `ports`, returning the devices it built so a test can inspect them."""

    def install(ports: list[str], *, error: Exception | None = None) -> list[FakeDevice]:
        made: list[FakeDevice] = []

        def device(kind: str):
            def open_device() -> FakeDevice:
                if error is not None:
                    raise error
                built = FakeDevice(ports)
                made.append(built)
                return built

            return open_device

        module = types.ModuleType("supriya_midi")
        module.MidiOut = device("output")
        module.MidiIn = device("input")
        monkeypatch.setitem(sys.modules, "supriya_midi", module)
        return made

    return install


# -- picking a port ---------------------------------------------------------------------------


def test_an_exact_name_picks_that_port() -> None:
    names = ["IAC Driver Bus 1", "TiMidity:TiMidity port 0 129:0"]
    assert midi_port.find_port(names, "IAC Driver Bus 1") == 0


def test_a_partial_name_still_finds_the_synth_whose_number_moved() -> None:
    assert midi_port.find_port(["TiMidity:TiMidity port 3 129:0"], "TiMidity") == 0


def test_without_a_name_the_first_synth_is_taken() -> None:
    names = ["Midi Through:Midi Through Port-0 14:0", "fluidsynth:synth 128:0"]
    assert midi_port.find_port(names) == 1


def test_the_loopback_is_never_mistaken_for_a_synth() -> None:
    assert midi_port.find_synth_port(["Midi Through:Midi Through Port-0 14:0"]) is None
    assert midi_port.find_port(["Midi Through:Midi Through Port-0 14:0"]) is None


# -- listing ports ----------------------------------------------------------------------------


def test_the_names_are_the_ones_the_backend_lists(install_backend) -> None:
    install_backend(["a", "b"])
    assert midi_port.output_names() == ("a", "b")
    assert midi_port.input_names() == ("a", "b")


def test_names_are_empty_when_no_backend_is_installed(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "supriya_midi", None)  # an import of this raises ImportError
    assert midi_port.output_names() == ()
    assert midi_port.input_names() == ()


def test_names_are_empty_when_the_backend_will_not_answer(install_backend) -> None:
    install_backend([], error=RuntimeError("no MIDI here"))
    assert midi_port.output_names() == ()


# -- opening ----------------------------------------------------------------------------------


def test_an_output_that_is_not_listed_is_none(install_backend) -> None:
    install_backend(["Midi Through:Midi Through Port-0 14:0"])
    assert midi_port.open_output("IAC Driver") is None


def test_a_port_that_is_not_listed_opens_nothing() -> None:
    assert midi_port.open_input("no such port") is None


def test_sending_hands_the_bytes_to_the_device(install_backend) -> None:
    made = install_backend(["TiMidity:TiMidity port 0 129:0"])
    port = midi_port.open_output()

    assert port is not None
    port.send([0x90, 60, 100])

    assert made[0].opened == 0
    assert made[0].sent == [[0x90, 60, 100]]
    assert port.name == "TiMidity:TiMidity port 0 129:0"
    assert port.is_open is True


def test_closing_a_port_closes_the_device(install_backend) -> None:
    made = install_backend(["TiMidity:TiMidity port 0 129:0"])
    port = midi_port.open_output()

    assert port is not None
    port.close()

    assert port.is_open is False
    assert made[0].is_port_open is False


def test_an_input_keeps_only_quarter_frames_and_takes_its_callback(install_backend) -> None:
    made = install_backend(["IAC Driver Bus 1"])
    heard: list[list[int]] = []

    port = midi_port.open_input("IAC Driver", on_message=heard.append)

    assert port is not None
    assert made[0].opened == 0
    assert made[0].ignored == (True, True, True)  # sysex, clock and active sensing stay on the wire
    assert made[0].callback == heard.append


def test_closing_an_input_drops_its_callback(install_backend) -> None:
    made = install_backend(["IAC Driver Bus 1"])
    port = midi_port.open_input("IAC Driver", on_message=lambda _message: None)

    assert port is not None
    port.close()

    assert made[0].callback is None
    assert made[0].is_port_open is False
