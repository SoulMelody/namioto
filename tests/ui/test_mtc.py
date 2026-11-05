# SPDX-License-Identifier: AGPL-3.0-only
"""The sync policy in the window: what a position from the port does to the transport.

No port is opened here - `MtcPort` has `test_mtc_port.py` of its own - because everything below is
the layer above it: a drift becomes a seek or a rate, the rate stays a nudge, and the playback delay
is taken out of the picture while an external clock drives.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QInputDialog

from namioto import midi_port
from namioto import settings as store
from namioto.mtc import Rate
from namioto.ui.app import clamp_rate, mtc_correction

from .helpers import RateOutput


def test_a_large_drift_seeks_the_transport(own_window) -> None:
    own_window.player = RateOutput(10.0)
    own_window.mtc_mode = "receive"
    own_window._on_mtc_position(10.2)  # 200 ms ahead: a jump is the cheap correction
    assert own_window.player.position == pytest.approx(10.2)
    assert "seek" in own_window.player.calls


def test_a_small_drift_only_trims_the_rate(own_window) -> None:
    player = RateOutput(10.0)
    own_window.player = player
    own_window.mtc_mode = "receive"
    own_window._on_mtc_position(10.02)  # 20 ms: half of it goes into the rate
    assert player._speed == pytest.approx(1.01)
    assert "seek" not in player.calls


def test_a_trimmed_rate_leaves_the_playhead_where_it_was(own_window) -> None:
    player = RateOutput(10.0)
    own_window.player = player
    own_window.mtc_mode = "receive"
    before = player.position
    own_window._on_mtc_position(10.02)
    assert player.position == pytest.approx(before)  # re-anchored, so no jump from the rate change


def test_the_rate_is_clamped_around_one() -> None:
    assert mtc_correction(0.05)[1] == pytest.approx(1.025)  # the most half of 50 ms can ask for
    assert mtc_correction(-0.05)[1] == pytest.approx(0.975)
    assert clamp_rate(9.0) == pytest.approx(1.05)
    assert clamp_rate(0.1) == pytest.approx(0.95)


def test_a_position_is_ignored_when_the_port_is_not_driving(own_window) -> None:
    player = RateOutput(10.0)
    own_window.player = player
    own_window.mtc_mode = ""
    own_window._on_mtc_position(20.0)
    assert player.calls == []


def test_receive_mode_takes_the_playback_delay_out(own_window) -> None:
    own_window.player = RateOutput()
    own_window.settings.midi.playback_delay_ms = 40
    own_window.mtc_mode = "receive"
    own_window._apply_delay_policy()
    assert own_window.player.delay_ms == 0  # the port says when the sound goes, not the delay
    own_window.mtc_mode = ""
    own_window._apply_delay_policy()
    assert own_window.player.delay_ms == 40


def test_a_send_without_a_frame_rate_is_refused_rather_than_guessed(own_window) -> None:
    # an empty rate is what a settings file that names none gives now that the section exists
    assert own_window._mtc_rate() is None
    own_window._set_mtc_mode("send")
    assert own_window.mtc_mode == ""  # nothing started, rather than a silent 25 fps fallback
    assert own_window._actions["mtc-send"].isChecked() is False
    assert "frame rate" in own_window.statusBar().currentMessage()


def test_the_settings_name_the_rate_the_port_speaks(own_window) -> None:
    own_window.settings.mtc.framerate = "25"
    assert own_window._mtc_rate() is Rate.FPS_25


def test_an_empty_rate_is_still_none(own_window) -> None:
    own_window.settings.mtc.framerate = ""
    assert own_window._mtc_rate() is None


def test_a_named_port_is_used_without_asking(own_window, monkeypatch) -> None:
    monkeypatch.setattr(midi_port, "output_names", lambda: ("IAC Driver", "Other"))
    asked = []
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: asked.append(args) or ("", False))
    own_window.settings.mtc.output_port = "IAC Driver"
    assert own_window._choose_midi_port("output") == "IAC Driver"
    assert asked == []


def test_a_picked_port_is_kept_for_the_next_start(own_window, monkeypatch) -> None:
    monkeypatch.setattr(midi_port, "input_names", lambda: ("IAC Driver", "Other"))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("Other", True))
    own_window.settings.mtc.input_port = ""
    assert own_window._choose_midi_port("input") == "Other"
    assert own_window.settings.mtc.input_port == "Other"


def test_a_port_that_is_gone_is_asked_for_again(own_window, monkeypatch) -> None:
    monkeypatch.setattr(midi_port, "output_names", lambda: ("Other",))
    monkeypatch.setattr(QInputDialog, "getItem", lambda *args, **kwargs: ("Other", True))
    own_window.settings.mtc.output_port = "IAC Driver"
    assert own_window._choose_midi_port("output") == "Other"
    assert own_window.settings.mtc.output_port == "Other"


def test_a_new_frame_rate_restarts_the_running_port(own_window, monkeypatch) -> None:
    # a port left sending at the old rate would put a timecode nobody can detect as wrong on the wire
    calls = []
    monkeypatch.setattr(own_window, "_start_mtc_send", lambda: calls.append("start") or True)
    monkeypatch.setattr(own_window.mtc, "stop", lambda: calls.append("stop"))
    own_window.mtc_mode = "send"
    changed = store.clone(own_window.settings)
    changed.mtc.framerate = "25"
    own_window._on_settings_changed(changed)
    assert calls == ["stop", "start"]


def test_the_frame_rate_comes_from_the_settings(own_window) -> None:
    own_window.settings.mtc.framerate = "29.97drop"
    assert own_window._mtc_rate() is Rate.FPS_29_97_DROP
    own_window.settings.mtc.framerate = "25"
    assert own_window._mtc_rate() is Rate.FPS_25
    own_window.settings.mtc.framerate = "wild"  # not one of the four: no rate rather than a guess
    assert own_window._mtc_rate() is None
