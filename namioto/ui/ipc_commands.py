# SPDX-License-Identifier: AGPL-3.0-only
"""What a remote caller may ask the editor to do: `WindowBridge` registers the methods of the
interface onto an `IpcServer`'s dispatcher and drives the `MainWindow` behind them.

This is the remote control's whole vocabulary. A method validates its parameters, does one thing to
the window, and answers with what the interface now holds, so a client never has to guess. Times on
the wire are seconds - the unit the audio, the files and the players already speak - while the roll
keeps its own beats and converts at this boundary, the way `PianoRollView.to_seconds` does
everywhere else.

The bridge also pushes events: any change to the notes, the channels, the tempo or the lyrics is
broadcast as a fresh `state`, a play, pause or stop as `transport`, and a status-bar message as
`message`, so a watcher can follow along without polling.
"""

from __future__ import annotations

import base64
import contextlib
import math
from typing import Any

from namioto import __version__, ipc, midi, project, subtitle
from namioto.channels import CHANNEL_COUNT, Channel, free_channel
from namioto.channels import set_field as channel_set_field
from namioto.ipc import IpcError
from namioto.karaoke import KrcError
from namioto.ui.roll import PITCH_MAX, PITCH_MIN

MATCH_SECONDS = 1e-3  # how close two notes have to be to count as the same one when removing


def _required(params: dict, name: str) -> Any:
    if name not in params:
        raise IpcError("bad_params", f"{name} is required")
    return params[name]


def _text(params: dict, name: str, *, required: bool = True, default: str | None = None) -> str:
    value = _string(params, name, required=required, default=default)
    if not value:
        raise IpcError("bad_params", f"{name} must be a non-empty string")
    return value


def _string(params: dict, name: str, *, required: bool = True, default: str | None = None) -> str:
    if name not in params:
        if required:
            raise IpcError("bad_params", f"{name} is required")
        return default if default is not None else ""
    value = params[name]
    if not isinstance(value, str):
        raise IpcError("bad_params", f"{name} must be a string")
    return value


def _number(
    params: dict,
    name: str,
    *,
    required: bool = True,
    default: float | None = None,
    low: float | None = None,
    high: float | None = None,
) -> float:
    if name not in params:
        if required:
            raise IpcError("bad_params", f"{name} is required")
        return float(default) if default is not None else 0.0
    value = params[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise IpcError("bad_params", f"{name} must be a finite number")
    number = float(value)
    if low is not None and number < low:
        raise IpcError("bad_params", f"{name} must be at least {low:g}")
    if high is not None and number > high:
        raise IpcError("bad_params", f"{name} must be at most {high:g}")
    return number


def _whole(
    params: dict,
    name: str,
    *,
    required: bool = True,
    default: int | None = None,
    low: int | None = None,
    high: int | None = None,
) -> int:
    value = _number(params, name, required=required, default=default, low=low, high=high)
    if value != int(value):
        raise IpcError("bad_params", f"{name} must be a whole number")
    return int(value)


def _flag(params: dict, name: str, default: bool = False) -> bool:
    value = params.get(name, default)
    if not isinstance(value, bool):
        raise IpcError("bad_params", f"{name} must be true or false")
    return value


class WindowBridge:
    """The remote methods over one `MainWindow`, and the events its changes raise."""

    def __init__(self, window, server):
        self.window = window
        self.server = server
        self.dispatcher = server.dispatcher
        self._connections: list[tuple] = []
        self._subtitle_revision = 0
        self._register()
        self._watch()

    # --- the vocabulary ---------------------------------------------------

    def _register(self) -> None:
        add = self.dispatcher.register
        add("ping", self.ping, summary="Check that the editor is listening")
        add("version", self.version, summary="The editor's and the protocol's versions")
        add("commands", self.commands, summary="Every method this editor offers")
        add("state.get", self.get_state, summary="The whole state of the open session")

        add("transport.play", self.play, summary="Start playing from the playhead")
        add("transport.pause", self.pause, summary="Pause, keeping the playhead")
        add("transport.toggle", self.toggle, summary="Play when paused, pause when playing")
        add("transport.stop", self.stop, summary="Stop and rewind to the start")
        add("transport.play_from_start", self.play_from_start, summary="Rewind, then play")
        add(
            "transport.seek",
            self.seek,
            summary="Move the playhead to seconds, a beat, or a fraction of the audio",
            params={"seconds?": "position in seconds", "beat?": "position in beats", "fraction?": "0 to 1"},
        )
        add("transport.position", self.position, summary="Where the transport is right now")
        add("transport.set_bpm", self.set_bpm, summary="Set the tempo of the beat grid", params={"bpm": "20 to 300"})
        add(
            "transport.set_speed",
            self.set_speed,
            summary="Set the playback speed, pitch unchanged",
            params={"speed": "0.10 to 2.00"},
        )
        add(
            "transport.set_grid_offset",
            self.set_grid_offset,
            summary="Slide the drawn grid by milliseconds",
            params={"ms": "-500 to 500"},
        )

        add("document.notes", self.notes, summary="Every note, in seconds")
        add(
            "document.add_notes",
            self.add_notes,
            summary="Add notes to the roll",
            params={"notes": "[{pitch, start, duration, channel?}] in seconds"},
        )
        add(
            "document.remove_notes",
            self.remove_notes,
            summary="Remove the notes that match these specs",
            params={"notes": "[{pitch, start, channel?, duration?}] in seconds"},
        )
        add("document.clear_notes", self.clear_notes, summary="Remove every note")
        add(
            "document.replace_notes",
            self.replace_notes,
            summary="Replace every note, and the channels when given",
            params={"notes": "[{pitch, start, duration, channel?}]", "channels?": "the channel list"},
        )
        add(
            "document.quantize",
            self.quantize,
            summary="Snap the notes onto the grid",
            params={"cell?": "grid cell in beats, the current snap when omitted"},
        )
        add(
            "document.import_midi",
            self.import_midi,
            summary="Replace the notes with a MIDI file",
            params={"path": "the .mid file", "force?": "discard unsaved changes"},
        )

        add("channels.list", self.channels, summary="Every MIDI channel")
        add(
            "channels.add",
            self.add_channel,
            summary="Add a channel on the first free number, or the one named",
            params={"channel?": "0 to 15", "program?": "0 to 127", "name?": "a label", "volume?": "0 to 127"},
        )
        add(
            "channels.set",
            self.set_channel,
            summary="Change one channel's fields",
            params={
                "channel": "0 to 15",
                "name?": "a label",
                "color?": "#rrggbb",
                "program?": "0 to 127",
                "volume?": "0 to 127",
                "mute?": "a switch",
                "visible?": "a switch",
                "lock?": "a switch",
            },
        )
        add(
            "channels.remove",
            self.remove_channel,
            summary="Drop a channel and the notes on it",
            params={"channel": "0 to 15"},
        )
        add(
            "channels.activate",
            self.activate_channel,
            summary="Make a channel the one notes are drawn into",
            params={"channel": "0 to 15"},
        )

        add(
            "view.set_zoom",
            self.set_zoom,
            summary="Set the horizontal and vertical zoom",
            params={"x?": "pixels per beat", "y?": "pixels per row"},
        )
        add("view.set_snap", self.set_snap, summary="Set the snap grid", params={"beats": "one of the snap cells"})
        add(
            "view.set_division",
            self.set_division,
            summary="Divide the time axis by beats or by seconds",
            params={"division": "beats or seconds"},
        )
        add(
            "spectrum.set",
            self.set_spectrum,
            summary="Set the spectrum's display values",
            params={"gain?": "10 to 600", "contrast?": "0.2 to 4.0"},
        )
        add(
            "mix.set_volume",
            self.set_volume,
            summary="Set one layer's volume",
            params={"layer": "audio or midi", "volume": "0 to 100"},
        )

        add("project.open", self.open_project, summary="Open a .nto project", params={"path": "the file"})
        add(
            "project.save",
            self.save_project,
            summary="Write the open project, to `path` when given",
            params={"path?": "where to write it"},
        )
        add(
            "project.export_midi",
            self.export_midi,
            summary="Write the notes out as MIDI",
            params={"path": "the .mid file"},
        )
        add(
            "project.export_krc",
            self.export_krc,
            summary="Write the lyrics out as a .krc",
            params={"path": "the .krc file"},
        )
        add(
            "project.export_ass",
            self.export_ass,
            summary="Build ASS from the current KRC and MIDI exports using GUI settings",
            params={"path": "the .ass file", "save_krc?": "KRC export path", "save_midi?": "MIDI export path"},
        )
        add(
            "project.ass_source",
            self.ass_source,
            summary="Snapshot the current KRC, base64 MIDI and ASS settings without writing files",
        )
        add(
            "audio.open",
            self.open_audio,
            summary="Start an untitled document from an audio file",
            params={
                "path": "the audio file",
                "analysis?": "channels, t_num, fft_points, a4",
                "force?": "discard unsaved changes",
            },
        )
        add("lyrics.get", self.get_lyrics, summary="The open project's lyrics text")
        add(
            "lyrics.set",
            self.set_lyrics,
            summary="Replace the lyrics text and re-map the sounds",
            params={"text": "a .krc document"},
        )
        add("undo", self.undo, summary="Undo the last edit")
        add("redo", self.redo, summary="Redo the last undone edit")

    def _watch(self) -> None:
        window = self.window
        for signal in (
            window.view.notes_changed,
            window.view.channels_changed,
            window.transport.bpm.valueChanged,
            window.view.lyrics_changed,
            window.subtitle_changed,
        ):
            signal.connect(self._changed)
            self._connections.append((signal, self._changed))
        status = window.statusBar().messageChanged
        status.connect(self._message)
        self._connections.append((status, self._message))
        window.transport_changed.connect(self._transport_changed)
        self._connections.append((window.transport_changed, self._transport_changed))

    def close(self) -> None:
        """Take the bridge's signal connections off, so a stopped server is not kept alive by them."""
        for signal, slot in self._connections:
            with contextlib.suppress(TypeError):
                signal.disconnect(slot)
        self._connections.clear()

    # --- events -----------------------------------------------------------

    def _changed(self, *_args) -> None:
        self._subtitle_revision += 1
        self.server.broadcast("state", self.get_state({}))

    def _transport_changed(self, _playing: bool) -> None:
        self.server.broadcast("transport", self._transport())

    def _message(self, text: str) -> None:
        if text:
            self.server.broadcast("message", {"text": text})

    def refresh(self) -> None:
        """Push the current state to every listener, after a change the signals do not carry."""
        self._changed()

    # --- the methods ------------------------------------------------------

    def ping(self, _params: dict) -> dict:
        return {"pong": True}

    def version(self, _params: dict) -> dict:
        return {"app": ipc.APP, "version": __version__, "protocol": ipc.PROTOCOL}

    def commands(self, _params: dict) -> dict:
        return {"methods": self.dispatcher.describe()}

    def get_state(self, _params: dict) -> dict:
        window = self.window
        view = window.view
        return {
            "app": ipc.APP,
            "protocol": ipc.PROTOCOL,
            "version": __version__,
            "project": {
                "path": str(window.project_path) if window.project_path is not None else None,
                "name": window.project_path.name if window.project_path is not None else None,
                "dirty": bool(window.project_dirty),
            },
            "audio": {
                "path": window.audio_path,
                "loaded": window.song.is_loaded,
            },
            "transport": self._transport(),
            "subtitle": {
                "revision": self._subtitle_revision,
                "pending": bool(view.lyric_mapping_pending),
            },
            "document": {
                "notes": len(view.notes()),
                "active_channel": view.active_channel,
                "channels": [self._channel(channel) for channel in view.channels],
            },
            "view": {
                "zoom": [round(view.zoom[0], 4), round(view.zoom[1], 4)],
                "offset_ms": round(view.offset * 1000.0, 3),
                "snap": view.snap,
                "division": view.division,
                "edit_mode": bool(view.edit_mode),
            },
            "mix": {
                "audio_volume": window.mix.audio_volume.value(),
                "midi_volume": window.mix.midi_volume.value(),
                "gain": view.gain,
                "contrast": view.contrast,
            },
        }

    # --- transport --------------------------------------------------------

    def _transport(self) -> dict:
        window = self.window
        return {
            "playing": bool(window._is_playing()),
            "position": round(window._position(), 6),
            "duration": round(window._duration(), 6),
            "bpm": round(window.view.bpm, 4),
            "speed": round(window.transport.speed.value(), 4),
        }

    def play(self, _params: dict) -> dict:
        self.window._play()
        return self._transport()

    def pause(self, _params: dict) -> dict:
        self.window._pause()
        return self._transport()

    def toggle(self, _params: dict) -> dict:
        self.window._toggle_play()
        return self._transport()

    def stop(self, _params: dict) -> dict:
        self.window._stop()
        return self._transport()

    def play_from_start(self, _params: dict) -> dict:
        self.window._play_from_start()
        return self._transport()

    def position(self, _params: dict) -> dict:
        return self._transport()

    def seek(self, params: dict) -> dict:
        window = self.window
        view = window.view
        choices = [name for name in ("seconds", "beat", "fraction") if name in params]
        if len(choices) != 1:
            raise IpcError("bad_params", "give exactly one of seconds, beat or fraction")
        name = choices[0]
        if name == "seconds":
            seconds = _number(params, "seconds", low=0.0)
        elif name == "beat":
            seconds = view.to_seconds(_number(params, "beat", low=0.0))
        else:
            seconds = _number(params, "fraction", low=0.0, high=1.0) * window._duration()
        window._seek(min(seconds, window._duration()))
        return self._transport()

    def set_bpm(self, params: dict) -> dict:
        self.window.transport.bpm.setValue(_number(params, "bpm", low=20.0, high=300.0))
        return self._transport()

    def set_speed(self, params: dict) -> dict:
        self.window.transport.speed.set_value(_number(params, "speed", low=0.1, high=2.0))
        return self._transport()

    def set_grid_offset(self, params: dict) -> dict:
        self.window.transport.grid_offset.setValue(round(_number(params, "ms", low=-500.0, high=500.0)))
        return self.get_state({})

    # --- the document -----------------------------------------------------

    def _channel(self, channel: Channel) -> dict:
        return {
            "channel": channel.channel,
            "name": channel.name,
            "color": channel.color,
            "program": channel.program,
            "volume": channel.volume,
            "mute": channel.mute,
            "visible": channel.visible,
            "lock": channel.lock,
        }

    def _note(self, item) -> dict:
        view = self.window.view
        return {
            "pitch": item.pitch,
            "start": round(view.to_seconds(item.start), 6),
            "duration": round(view.to_seconds(item.duration), 6),
            "channel": item.channel,
        }

    def _note_spec(self, entry: Any, *, duration: bool = True) -> tuple[int, float, float | None, int | None]:
        if not isinstance(entry, dict):
            raise IpcError("bad_params", "every note must be an object")
        pitch = _whole(entry, "pitch", low=PITCH_MIN, high=PITCH_MAX)
        start = _number(entry, "start", low=0.0)
        length = _number(entry, "duration", required=duration, low=0.0) if duration else None
        if duration and length is not None and length <= 0:
            raise IpcError("bad_params", "duration must be positive")
        channel = None
        if entry.get("channel") is not None:
            channel = _whole(entry, "channel", low=0, high=CHANNEL_COUNT - 1)
        return pitch, start, length, channel

    def notes(self, _params: dict) -> dict:
        view = self.window.view
        ordered = sorted(view.notes(), key=lambda item: (view.to_seconds(item.start), item.pitch))
        return {"notes": [self._note(item) for item in ordered]}

    def add_notes(self, params: dict) -> dict:
        entries = _required(params, "notes")
        if not isinstance(entries, list) or not entries:
            raise IpcError("bad_params", "notes must be a non-empty list")
        view = self.window.view
        kept = [(item.pitch, item.start, item.duration, item.channel, item.id) for item in view.notes()]
        arriving = []
        for entry in entries:
            pitch, start, duration, channel = self._note_spec(entry)
            arriving.append(
                (
                    pitch,
                    view.to_beats(start),
                    view.to_beats(duration if duration is not None else 0.0),
                    channel if channel is not None else view.active_channel,
                )
            )
        view.set_notes(kept + arriving)
        return {"added": len(arriving), "notes": len(view.notes())}

    def remove_notes(self, params: dict) -> dict:
        entries = _required(params, "notes")
        if not isinstance(entries, list) or not entries:
            raise IpcError("bad_params", "notes must be a non-empty list")
        view = self.window.view
        remaining = list(view.notes())
        removed = 0
        for entry in entries:
            pitch, start, duration, channel = self._note_spec(entry, duration=False)
            for item in list(remaining):
                if item.pitch != pitch or (channel is not None and item.channel != channel):
                    continue
                if abs(view.to_seconds(item.start) - start) > MATCH_SECONDS:
                    continue
                if duration is not None and abs(view.to_seconds(item.duration) - duration) > MATCH_SECONDS:
                    continue
                remaining.remove(item)
                removed += 1
                break
        view.set_notes([(item.pitch, item.start, item.duration, item.channel, item.id) for item in remaining])
        return {"removed": removed, "notes": len(remaining)}

    def clear_notes(self, _params: dict) -> dict:
        self.window.view.clear_notes()
        return {"notes": 0}

    def replace_notes(self, params: dict) -> dict:
        entries = _required(params, "notes")
        if not isinstance(entries, list):
            raise IpcError("bad_params", "notes must be a list")
        view = self.window.view
        arriving = []
        for entry in entries:
            pitch, start, duration, channel = self._note_spec(entry)
            arriving.append((pitch, view.to_beats(start), view.to_beats(duration or 0.0), channel or 0))
        channels = params.get("channels")
        if channels is None:
            view.set_notes(arriving)
        else:
            if not isinstance(channels, list):
                raise IpcError("bad_params", "channels must be a list")
            view.replace([self._channel_from(entry) for entry in channels], arriving, "Replace notes")
        return {"notes": len(view.notes())}

    def _channel_from(self, entry: Any) -> Channel:
        if not isinstance(entry, dict):
            raise IpcError("bad_params", "every channel must be an object")
        number = _whole(entry, "channel", low=0, high=CHANNEL_COUNT - 1)
        base = next(
            (channel for channel in self.window.view.channels if channel.channel == number),
            Channel(channel=number),
        )
        fields: dict[str, Any] = {}
        if "name" in entry:
            fields["name"] = _string(entry, "name")
        if "color" in entry:
            fields["color"] = _string(entry, "color")
        if "program" in entry:
            fields["program"] = _whole(entry, "program", low=0, high=127)
        if "volume" in entry:
            fields["volume"] = _whole(entry, "volume", low=0, high=127)
        for name in ("mute", "visible", "lock"):
            if name in entry:
                fields[name] = _flag(entry, name)
        return channel_set_field(base, **fields)

    def quantize(self, params: dict) -> dict:
        cell = params.get("cell")
        if cell is not None:
            index = self.window.edit.snap.findData(_number(params, "cell", low=0.0))
            if index < 0:
                raise IpcError("bad_params", f"{cell!r} is not one of the editor's snap cells")
            self.window.edit.snap.setCurrentIndex(index)
        applied = bool(self.window.view.quantize_notes())
        return {"quantized": applied}

    def import_midi(self, params: dict) -> dict:
        path = _text(params, "path")
        self._require_clean(params)
        if not midi.looks_like_midi(path):
            raise IpcError("bad_params", f"{path!r} is not a MIDI file")
        if self.window.project_path is None:
            raise IpcError("no_project", "open a project before importing a MIDI file")
        try:
            imported = midi.read(path, wavetone=self.window.settings.midi.wavetone)
        except (OSError, EOFError, ValueError) as error:
            raise IpcError("failed", f"the MIDI file could not be read: {error}") from error
        self.window.apply_midi(imported)
        return {"notes": len(self.window.view.notes())}

    # --- channels ---------------------------------------------------------

    def channels(self, _params: dict) -> dict:
        return {"channels": [self._channel(channel) for channel in self.window.view.channels]}

    def add_channel(self, params: dict) -> dict:
        view = self.window.view
        number = params.get("channel")
        if number is None:
            number = free_channel(view.channels)
            if number is None:
                raise IpcError("full", "all sixteen channels are in use")
        else:
            number = _whole(params, "channel", low=0, high=CHANNEL_COUNT - 1)
            if any(channel.channel == number for channel in view.channels):
                raise IpcError("conflict", f"channel {number} is already in use")
        entry: dict = {"channel": number}
        for name in ("name", "color", "program", "volume"):
            if name in params:
                entry[name] = params[name]
        view.set_channels([*view.channels, self._channel_from(entry)])
        return self._channel(next(channel for channel in view.channels if channel.channel == number))

    def set_channel(self, params: dict) -> dict:
        view = self.window.view
        number = _whole(params, "channel", low=0, high=CHANNEL_COUNT - 1)
        if not any(channel.channel == number for channel in view.channels):
            raise IpcError("not_found", f"there is no channel {number}")
        entry = {key: value for key, value in params.items() if key != "channel"}
        view.set_channel_field(number, **self._channel_fields(entry))
        return self._channel(next(channel for channel in view.channels if channel.channel == number))

    def _channel_fields(self, entry: dict) -> dict:
        fields: dict[str, Any] = {}
        if "name" in entry:
            fields["name"] = _string(entry, "name")
        if "color" in entry:
            fields["color"] = _string(entry, "color")
        if "program" in entry:
            fields["program"] = _whole(entry, "program", low=0, high=127)
        if "volume" in entry:
            fields["volume"] = _whole(entry, "volume", low=0, high=127)
        for name in ("mute", "visible", "lock"):
            if name in entry:
                fields[name] = _flag(entry, name)
        if not fields:
            raise IpcError("bad_params", "give at least one field to change")
        return fields

    def remove_channel(self, params: dict) -> dict:
        number = _whole(params, "channel", low=0, high=CHANNEL_COUNT - 1)
        if not self.window.view.remove_channel(number):
            raise IpcError("conflict", "the last channel cannot be removed")
        return {"channels": [self._channel(channel) for channel in self.window.view.channels]}

    def activate_channel(self, params: dict) -> dict:
        number = _whole(params, "channel", low=0, high=CHANNEL_COUNT - 1)
        if not any(channel.channel == number for channel in self.window.view.channels):
            raise IpcError("not_found", f"there is no channel {number}")
        self.window.view.set_active_channel(number)
        return {"active_channel": self.window.view.active_channel}

    # --- the view, the mix and the project --------------------------------

    def set_zoom(self, params: dict) -> dict:
        view = self.window.view
        x = _number(params, "x", required=False, default=view.zoom[0], low=12.0, high=900.0)
        y = _number(params, "y", required=False, default=view.zoom[1], low=8.0, high=64.0)
        view.set_zoom(x, y)
        return {"zoom": [round(view.zoom[0], 4), round(view.zoom[1], 4)]}

    def set_snap(self, params: dict) -> dict:
        value = _number(params, "beats", low=0.0)
        index = self.window.edit.snap.findData(value)
        if index < 0:
            raise IpcError("bad_params", f"{value:g} is not one of the editor's snap cells")
        self.window.edit.snap.setCurrentIndex(index)
        return {"snap": self.window.view.snap}

    def set_division(self, params: dict) -> dict:
        value = _text(params, "division")
        if value not in project.DIVISIONS:
            raise IpcError("bad_params", f"division must be one of {', '.join(project.DIVISIONS)}")
        self.window.transport.division.setChecked(value == "beats")
        return {"division": self.window.view.division}

    def set_spectrum(self, params: dict) -> dict:
        window = self.window
        if "gain" in params:
            window.mix.gain.set_value(_number(params, "gain", low=10.0, high=600.0))
        if "contrast" in params:
            window.mix.contrast.set_value(_number(params, "contrast", low=0.2, high=4.0))
        return {"gain": window.view.gain, "contrast": window.view.contrast}

    def set_volume(self, params: dict) -> dict:
        layer = _text(params, "layer")
        if layer not in ("audio", "midi"):
            raise IpcError("bad_params", "layer must be audio or midi")
        volume = _number(params, "volume", low=0.0, high=100.0)
        slider = self.window.mix.audio_volume if layer == "audio" else self.window.mix.midi_volume
        slider.set_value(volume)
        return {"layer": layer, "volume": slider.value()}

    def open_project(self, params: dict) -> dict:
        path = _text(params, "path")
        self._require_clean(params)
        if not project.looks_like_project(path):
            raise IpcError("bad_params", f"{path!r} is not a .nto project")
        if not self.window.load_project(path):
            raise IpcError("failed", f"the project {path!r} could not be opened")
        return self.get_state({})

    def save_project(self, params: dict) -> dict:
        window = self.window
        path = params.get("path")
        if path is None:
            if window.project_path is None:
                raise IpcError("bad_params", "this document has no file yet; give a path")
            target = window.project_path
        else:
            target = _text(params, "path")
            if not project.looks_like_project(target):
                target = f"{target}{project.SUFFIX}"
        if not window.save_project(target):
            raise IpcError("failed", f"the project could not be written to {target}")
        return self.get_state({})

    def export_midi(self, params: dict) -> dict:
        path = _text(params, "path")
        if not midi.looks_like_midi(path):
            path = f"{path}{midi.SUFFIXES[0]}"
        if not self.window.export_midi(path):
            raise IpcError("failed", f"the MIDI file could not be written to {path}")
        return {"path": path}

    def export_krc(self, params: dict) -> dict:
        path = _text(params, "path")
        if not self.window.export_krc(path):
            raise IpcError("failed", "the lyrics could not be written")
        return {"path": path}

    def export_ass(self, params: dict) -> dict:
        path = subtitle.ass_path(_text(params, "path"))
        paths = {name: _text(params, name) for name in ("save_krc", "save_midi") if name in params}
        if not self.window.export_ass(path, **paths):
            raise IpcError("failed", self.window.statusBar().currentMessage())
        return {"path": str(path), **paths}

    def ass_source(self, _params: dict) -> dict:
        try:
            krc, data = self.window.ass_source()
        except (KrcError, ValueError, OSError) as error:
            raise IpcError("failed", str(error)) from error
        return {
            "krc": krc,
            "midi": base64.b64encode(data).decode("ascii"),
            "settings": vars(self.window._ass_settings()).copy(),
            "wavetone": self.window.settings.midi.wavetone,
            "revision": self._subtitle_revision,
        }

    def open_audio(self, params: dict) -> dict:
        path = _text(params, "path")
        self._require_clean(params)
        if project.looks_like_project(path):
            return self.open_project(params)
        if midi.looks_like_midi(path):
            return self.import_midi(params)
        analysis = params.get("analysis") or {}
        if not isinstance(analysis, dict):
            raise IpcError("bad_params", "analysis must be an object")
        self.window.reset_document(analysis)
        self.window.load_audio(path)
        return self.get_state({})

    def get_lyrics(self, _params: dict) -> dict:
        window = self.window
        return {
            "text": window.lyrics_text,
            "lines": len(window.view.lyric_lines),
        }

    def set_lyrics(self, params: dict) -> dict:
        text = params.get("text")
        if not isinstance(text, str):
            raise IpcError("bad_params", "text must be a string")
        window = self.window
        if not window._on_lyrics_saved(text):
            raise IpcError("failed", window.statusBar().currentMessage())
        return {"lines": len(window.view.lyric_lines)}

    def undo(self, _params: dict) -> dict:
        self.window.view.undo()
        return {"notes": len(self.window.view.notes())}

    def redo(self, _params: dict) -> dict:
        self.window.view.redo()
        return {"notes": len(self.window.view.notes())}

    def _require_clean(self, params: dict) -> None:
        if self.window.project_dirty and not _flag(params, "force", False):
            raise IpcError("dirty", "the open project has unsaved changes; pass force=true to discard them")
