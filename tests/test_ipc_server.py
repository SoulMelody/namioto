# SPDX-License-Identifier: AGPL-3.0-only
"""The remote-control interface end to end: a real `MainWindow`, a real `IpcServer` on a unix
socket, and the Qt-free client driving both.

The server runs on the GUI thread, so the client is asked from a worker thread while this one keeps
pumping `processEvents`; `call` and `failure` below wrap that so a test reads as one line.
"""

from __future__ import annotations

import os
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from namioto import ipc, midi
from namioto import settings as store
from namioto.ui.app import MainWindow

TIMEOUT = 5.0


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    return app


@pytest.fixture(scope="module")
def window(qt_app, tmp_path_factory):
    """One window for the module, reset between tests: a fresh one per test would leave the closed
    ones to be collected, and a headless run cannot afford a pile of live top-level widgets."""
    root = tmp_path_factory.mktemp("ipc")
    previous = os.environ.get(ipc.ENV_RUN)
    previous_state = os.environ.get("NAMIOTO_STATE")
    os.environ[ipc.ENV_RUN] = str(root / "run.json")
    os.environ["NAMIOTO_STATE"] = str(root / "state.json")
    built = MainWindow()
    built.resize(1000, 640)
    built.show()
    yield built
    built.project_dirty = False  # a close prompt would block a headless run
    built.project_path = None
    built.stop_ipc()
    built.close()
    built.deleteLater()
    qt_app.processEvents()
    if previous is None:
        os.environ.pop(ipc.ENV_RUN, None)
    else:
        os.environ[ipc.ENV_RUN] = previous
    if previous_state is None:
        os.environ.pop("NAMIOTO_STATE", None)
    else:
        os.environ["NAMIOTO_STATE"] = previous_state


@pytest.fixture(autouse=True)
def clean_window(window):
    window.stop_ipc()
    window.reset_document()
    yield
    window.stop_ipc()


@pytest.fixture
def endpoint(window, tmp_path):
    bound = window.start_ipc(str(tmp_path / "control.sock"))
    assert bound is not None
    return bound


def _run(app, action, timeout=TIMEOUT):
    """Run a blocking client call while the server's event loop keeps running."""
    box: dict = {}

    def worker():
        try:
            box["value"] = action()
        except Exception as error:  # noqa: BLE001 - handed back to the test as it is
            box["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    deadline = time.monotonic() + timeout
    while thread.is_alive() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.002)
    thread.join(0.5)
    if "error" in box:
        raise box["error"]
    assert "value" in box, action
    return box["value"]


def call(app, endpoint, method, params=None):
    with ipc.Client(endpoint) as client:
        return _run(app, lambda: client.request(method, params, timeout=TIMEOUT))


def failure(app, endpoint, method, params=None) -> ipc.IpcError:
    with pytest.raises(ipc.IpcError) as raised:
        call(app, endpoint, method, params)
    return raised.value


class Watcher:
    """A long-lived client that collects the events the server broadcasts, off the main thread."""

    def __init__(self, endpoint):
        self.client = ipc.Client(endpoint)
        self.client.connect()
        self.events: list[dict] = []
        self.lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self):
        for event in self.client.events(timeout=0.2):
            if event is None:
                if self._stop:
                    return
                continue
            with self.lock:
                self.events.append(event)

    def wait_for(self, app, name, timeout=TIMEOUT) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            with self.lock:
                for event in self.events:
                    if event.get("event") == name:
                        return event
            time.sleep(0.005)
        raise AssertionError(f"no {name!r} event among {self.events}")

    def stop(self):
        self._stop = True
        self._thread.join(1.0)
        self.client.close()


@pytest.fixture
def watcher(endpoint):
    watching = Watcher(endpoint)
    yield watching
    watching.stop()


def test_ipc_command_line_flags():
    from namioto.ui.app import parse_args

    assert parse_args([]).ipc is None and parse_args([]).no_ipc is False
    assert parse_args(["--ipc"]).ipc == ""
    assert parse_args(["--ipc", "/tmp/x.sock"]).ipc == "/tmp/x.sock"
    assert parse_args(["--no-ipc"]).no_ipc is True


def test_ping_version_and_commands(qt_app, endpoint):
    assert call(qt_app, endpoint, "ping") == {"pong": True}
    version = call(qt_app, endpoint, "version")
    assert version["app"] == "namioto" and version["protocol"] == ipc.PROTOCOL
    methods = {entry["method"] for entry in call(qt_app, endpoint, "commands")["methods"]}
    assert {"transport.play", "transport.seek", "document.add_notes", "project.save", "channels.set"} <= methods


def test_state_describes_the_session(qt_app, endpoint):
    state = call(qt_app, endpoint, "state.get")
    assert state["project"]["path"] is None and state["project"]["dirty"] is False
    assert state["document"]["notes"] == 0
    assert state["transport"]["bpm"] == 120.0
    assert state["view"]["division"] == "beats"
    assert {channel["channel"] for channel in state["document"]["channels"]} == {0}


def test_transport_edits_reach_the_window(qt_app, window, endpoint):
    transport = call(qt_app, endpoint, "transport.set_bpm", {"bpm": 96.5})
    assert transport["bpm"] == 96.5 and window.view.bpm == 96.5
    assert call(qt_app, endpoint, "transport.set_speed", {"speed": 1.5})["speed"] == 1.5
    assert window.view.offset == 0.0
    call(qt_app, endpoint, "transport.set_grid_offset", {"ms": -120})
    assert window.view.offset == pytest.approx(-0.12)


def test_notes_round_trip(qt_app, endpoint):
    added = call(
        qt_app,
        endpoint,
        "document.add_notes",
        {"notes": [{"pitch": 60, "start": 0.5, "duration": 1.0}, {"pitch": 64, "start": 2.0, "duration": 0.5}]},
    )
    assert added == {"added": 2, "notes": 2}
    notes = call(qt_app, endpoint, "document.notes")["notes"]
    assert [(note["pitch"], note["start"], note["duration"]) for note in notes] == [(60, 0.5, 1.0), (64, 2.0, 0.5)]
    removed = call(qt_app, endpoint, "document.remove_notes", {"notes": [{"pitch": 60, "start": 0.5}]})
    assert removed == {"removed": 1, "notes": 1}
    assert call(qt_app, endpoint, "document.clear_notes") == {"notes": 0}


def test_remote_edits_preserve_existing_note_ids_and_allocator(qt_app, window, endpoint):
    view = window.view
    view.set_notes([(60, 0.0, 1.0, 0, 100), (62, 2.0, 1.0, 0, 200)], next_id=500)
    view.undo_stack.clear()
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 64, "start": 2.0, "duration": 0.5}]})
    assert [note.id for note in view.notes()] == [100, 200, 500]
    assert view.document.next_id == 501
    view.undo()
    assert [note.id for note in view.notes()] == [100, 200]
    assert view.document.next_id == 501
    view.redo()
    assert [note.id for note in view.notes()] == [100, 200, 500]
    call(qt_app, endpoint, "document.remove_notes", {"notes": [{"pitch": 64, "start": 2.0}]})
    assert [note.id for note in view.notes()] == [100, 200]
    assert view.document.next_id == 501
    view.undo()
    assert [note.id for note in view.notes()] == [100, 200, 500]
    view.redo()
    call(qt_app, endpoint, "document.replace_notes", {"notes": [{"pitch": 67, "start": 3.0, "duration": 0.5}]})
    assert [note.id for note in view.notes()] == [501]
    assert view.document.next_id == 502


def test_replace_notes_carries_channels(qt_app, endpoint):
    result = call(
        qt_app,
        endpoint,
        "document.replace_notes",
        {
            "channels": [{"channel": 3, "name": "Vocal", "program": 53, "volume": 90}],
            "notes": [{"pitch": 67, "start": 1.0, "duration": 0.5, "channel": 3}],
        },
    )
    assert result == {"notes": 1}
    state = call(qt_app, endpoint, "state.get")
    assert [channel["channel"] for channel in state["document"]["channels"]] == [3]
    assert state["document"]["channels"][0]["name"] == "Vocal"
    assert call(qt_app, endpoint, "document.notes")["notes"][0]["channel"] == 3


def test_quantize_snaps_to_the_named_cell(qt_app, endpoint):
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.31, "duration": 0.37}]})
    assert call(qt_app, endpoint, "document.quantize", {"cell": 0.5}) == {"quantized": True}
    note = call(qt_app, endpoint, "document.notes")["notes"][0]
    assert note["start"] == pytest.approx(0.25)
    assert note["duration"] == pytest.approx(0.5)


def test_channels_are_edited(qt_app, endpoint):
    added = call(qt_app, endpoint, "channels.add", {"channel": 4, "program": 24, "name": "Guitar"})
    assert added["channel"] == 4 and added["name"] == "Guitar"
    changed = call(qt_app, endpoint, "channels.set", {"channel": 4, "volume": 42, "mute": True})
    assert changed["volume"] == 42 and changed["mute"] is True
    assert call(qt_app, endpoint, "channels.activate", {"channel": 4}) == {"active_channel": 4}
    assert any(channel["channel"] == 4 for channel in call(qt_app, endpoint, "channels.list")["channels"])
    removed = call(qt_app, endpoint, "channels.remove", {"channel": 4})["channels"]
    assert removed == call(qt_app, endpoint, "channels.list")["channels"]
    assert failure(qt_app, endpoint, "channels.remove", {"channel": 0}).code == "conflict"


def test_project_save_and_open_round_trip(qt_app, window, endpoint, tmp_path):
    target = tmp_path / "song.nto"
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 72, "start": 1.0, "duration": 2.0}]})
    state = call(qt_app, endpoint, "project.save", {"path": str(target)})
    assert state["project"]["name"] == "song.nto" and state["project"]["dirty"] is False
    assert target.exists()
    call(qt_app, endpoint, "document.clear_notes")
    call(qt_app, endpoint, "project.open", {"path": str(target), "force": True})
    assert call(qt_app, endpoint, "document.notes")["notes"][0]["pitch"] == 72
    # a save to a bare name still lands on a .nto
    bare = tmp_path / "named"
    call(qt_app, endpoint, "project.save", {"path": str(bare)})
    assert (tmp_path / "named.nto").exists()


def test_unsaved_changes_need_force(qt_app, window, endpoint, tmp_path):
    target = tmp_path / "other.nto"
    call(qt_app, endpoint, "project.save", {"path": str(target)})
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.0, "duration": 0.5}]})
    assert window.project_dirty is True
    assert failure(qt_app, endpoint, "project.open", {"path": str(target)}).code == "dirty"
    assert call(qt_app, endpoint, "project.open", {"path": str(target), "force": True})["project"]["dirty"] is False


def test_unknown_method_and_bad_params(qt_app, endpoint):
    assert failure(qt_app, endpoint, "does.not.exist").code == "unknown_method"
    assert failure(qt_app, endpoint, "document.add_notes", {"notes": []}).code == "bad_params"
    bad_pitch = {"notes": [{"pitch": 200, "start": 0, "duration": 1}]}
    assert failure(qt_app, endpoint, "document.add_notes", bad_pitch).code == "bad_params"
    assert failure(qt_app, endpoint, "transport.set_bpm", {"bpm": 1000}).code == "bad_params"


def test_events_are_broadcast(qt_app, endpoint, watcher):
    hello = watcher.wait_for(qt_app, "hello")
    assert hello["protocol"] == ipc.PROTOCOL and hello["pid"] > 0
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.0, "duration": 1.0}]})
    event = watcher.wait_for(qt_app, "state")
    assert event["data"]["document"]["notes"] == 1


def test_transport_events_are_broadcast(qt_app, endpoint, watcher):
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.0, "duration": 1.0}]})
    watcher.wait_for(qt_app, "hello")
    call(qt_app, endpoint, "transport.toggle")
    assert watcher.wait_for(qt_app, "transport")["data"]["playing"] is True
    call(qt_app, endpoint, "transport.stop")


def test_status_messages_are_broadcast(qt_app, endpoint, watcher, tmp_path):
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.0, "duration": 1.0}]})
    watcher.wait_for(qt_app, "hello")
    call(qt_app, endpoint, "project.save", {"path": str(tmp_path / "song.nto")})
    message = watcher.wait_for(qt_app, "message")
    assert "song.nto" in message["data"]["text"]


def test_tcp_endpoint_needs_its_token(qt_app, window, tmp_path, monkeypatch):
    monkeypatch.setenv(ipc.ENV_TOKEN, "s3cret")
    endpoint = window.start_ipc("tcp://127.0.0.1:0")
    assert endpoint.kind == "tcp" and endpoint.token == "s3cret"
    assert failure(qt_app, ipc.Endpoint("tcp", endpoint.address, ""), "ping").code == "unauthorized"
    assert call(qt_app, endpoint, "ping") == {"pong": True}


def test_stop_removes_the_run_file(window, tmp_path, monkeypatch):
    run = tmp_path / "run.json"
    monkeypatch.setenv(ipc.ENV_RUN, str(run))
    window.reset_document()
    assert window.start_ipc(str(tmp_path / "control.sock")) is not None
    assert run.exists()
    window.stop_ipc()
    assert not run.exists()


def test_a_settings_change_starts_and_stops_the_interface(window):
    enabled = store.clone(window.settings)
    enabled.remote.enabled = True
    enabled.remote.address = ""
    window.settings_store.apply(enabled)
    assert window.ipc is not None
    disabled = store.clone(enabled)
    disabled.remote.enabled = False
    window.settings_store.apply(disabled)
    assert window.ipc is None


def test_import_midi_replaces_the_roll(qt_app, window, endpoint, tmp_path):
    from namioto import project
    from namioto.channels import Channel

    call(qt_app, endpoint, "project.save", {"path": str(tmp_path / "song.nto")})
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.0, "duration": 1.0}]})
    target = tmp_path / "in.mid"
    midi.write(
        target,
        (Channel(channel=0),),
        (project.Note(1.0, 0.5, 64, 0),),
        120.0,
        wavetone=window.settings.midi.wavetone,
    )
    call(qt_app, endpoint, "document.import_midi", {"path": str(target), "force": True})
    assert [note["pitch"] for note in call(qt_app, endpoint, "document.notes")["notes"]] == [64]


def test_view_mix_and_history_commands(qt_app, window, endpoint):
    assert call(qt_app, endpoint, "view.set_zoom", {"x": 60, "y": 20})["zoom"] == [60.0, 20.0]
    assert call(qt_app, endpoint, "view.set_snap", {"beats": 1.0})["snap"] == 1.0
    assert call(qt_app, endpoint, "view.set_division", {"division": "seconds"})["division"] == "seconds"
    call(qt_app, endpoint, "mix.set_volume", {"layer": "audio", "volume": 30})
    assert window.mix.audio_volume.value() == 30
    assert call(qt_app, endpoint, "spectrum.set", {"gain": 300, "contrast": 2.0})["gain"] == 300
    call(qt_app, endpoint, "document.add_notes", {"notes": [{"pitch": 60, "start": 0.0, "duration": 1.0}]})
    assert call(qt_app, endpoint, "undo")["notes"] == 0
    assert call(qt_app, endpoint, "redo")["notes"] == 1


def test_lyrics_get_and_set(qt_app, window, endpoint):
    assert call(qt_app, endpoint, "lyrics.get")["text"] == ""
    result = call(qt_app, endpoint, "lyrics.set", {"text": "ら[la] り[li] る[lu]\n"})
    assert result["lines"] == 1
    assert call(qt_app, endpoint, "lyrics.get")["text"].startswith("ら")
    window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]], "mms")
    assert call(qt_app, endpoint, "lyrics.set", {"text": "あい"}) == {"lines": 1}
    assert window.lyrics_text == "あい"
    assert "mode" not in call(qt_app, endpoint, "lyrics.get")


@pytest.mark.parametrize("keep_events", [True, False])
def test_polling_clients_can_discard_queued_events(qt_app, endpoint, keep_events):
    with ipc.Client(endpoint, keep_events=keep_events) as client:
        assert _run(qt_app, lambda: client.request("ping")) == {"pong": True}
        assert bool(client._events) is keep_events
