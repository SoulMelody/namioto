# SPDX-License-Identifier: AGPL-3.0-only
"""ASS uses the same KRC/MIDI pipeline for files and sessions, with explicit settings precedence."""

from __future__ import annotations

import base64

import pytest

from namioto import ass_cli, ipc, midi, project, subtitle
from namioto.channels import Channel
from namioto.karaoke import AssSettings, generate_ass


@pytest.fixture
def exports():
    krc = "あ.2\n@1{い}\nう.0"
    channels = (Channel(channel=0), Channel(channel=1))
    notes = (project.Note(0, 1, 60, 0), project.Note(2, 1, 62, 0), project.Note(4, 1, 64, 1))
    return krc, channels, notes


@pytest.mark.parametrize("wavetone", [False, True])
def test_memory_and_file_midi_exports_are_identical(exports, tmp_path, wavetone):
    krc, channels, notes = exports
    data = midi.to_bytes(channels, notes, 60, wavetone=wavetone)
    path = midi.write(tmp_path / "song.mid", channels, notes, 60, wavetone=wavetone)
    assert path.read_bytes() == data
    assert midi.read_bytes(data, wavetone=wavetone) == midi.read(path, wavetone=wavetone)
    expected = generate_ass(krc, [[(0, 3)], [(4, 5)], [None]])
    assert subtitle.build_ass(krc, data, wavetone=wavetone) == expected


@pytest.mark.parametrize("krc", ["あ.4", "@2{あ}", "あ.0", ""])
def test_unrenderable_input_is_rejected(exports, krc):
    _, channels, notes = exports
    with pytest.raises(ValueError):
        subtitle.build_ass(krc, midi.to_bytes(channels, notes, 60))


def test_overlapping_notes_are_not_silently_filtered():
    data = midi.to_bytes((Channel(channel=0),), (project.Note(0, 2, 60, 0), project.Note(1, 2, 62, 0)), 60)
    with pytest.raises(ValueError, match="conflict"):
        subtitle.build_ass("あい", data)


def test_settings_precedence_and_hardcoded_defaults(monkeypatch, tmp_path):
    parser = ass_cli.build_parser()
    args = parser.parse_args(["--out", "out.ass", "--spacing", "9"])
    monkeypatch.setenv("NAMIOTO_ASS_SPACING", "not a number")
    monkeypatch.setenv("NAMIOTO_ASS_FONT_SIZE", "72")
    settings = ass_cli.resolve_settings(args, {"spacing": 3, "font": "Session Font", "font_size": 48})
    assert (settings.spacing, settings.font_size, settings.font) == (9, 72, "Session Font")
    args.spacing = None
    monkeypatch.setenv("NAMIOTO_ASS_SPACING", "7.5")
    assert ass_cli.resolve_settings(args, {"spacing": 3}).spacing == 7.5
    monkeypatch.delenv("NAMIOTO_ASS_SPACING")
    assert ass_cli.resolve_settings(args, {"spacing": 3}).spacing == 3
    monkeypatch.delenv("NAMIOTO_ASS_FONT_SIZE")
    config = tmp_path / "settings.json"
    config.write_text('{"ass":{"spacing":99,"font":"Saved Font"}}')
    monkeypatch.setenv("NAMIOTO_SETTINGS", str(config))
    assert ass_cli.resolve_settings(args) == AssSettings()


@pytest.mark.parametrize(
    ("option", "value"),
    [("SPACING", "nan"), ("FONT_SIZE", "0"), ("FONT_SIZE", "bad"), ("OVERLAY_COLOR", "xyz"), ("FONT", "a,b")],
)
def test_invalid_environment_settings_are_rejected(monkeypatch, option, value):
    monkeypatch.setenv(f"NAMIOTO_ASS_{option}", value)
    args = ass_cli.build_parser().parse_args(["--out", "out.ass"])
    with pytest.raises(ValueError):
        ass_cli.resolve_settings(args)


def test_file_cli_keeps_previous_ass_on_failure_and_uses_explicit_settings(exports, tmp_path):
    krc, channels, notes = exports
    lyrics = tmp_path / "song.krc"
    lyrics.write_text(krc)
    source = midi.write(tmp_path / "song.mid", channels, notes, 60, wavetone=True)
    target = tmp_path / "song.ass"
    args = ["--krc", str(lyrics), "--midi", str(source), "--out", str(target), "--spacing", "8"]
    assert ass_cli.main(args) == 0
    before = target.read_bytes()
    assert b",100,100,8,0,1," in before
    lyrics.write_text("あ.10")
    assert ass_cli.main(args) == 1
    assert target.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_file_watch_handles_atomic_replacement_and_recovers(exports, tmp_path, monkeypatch):
    krc, channels, notes = exports
    source = midi.write(tmp_path / "song.mid", channels, notes, 60)
    lyrics = tmp_path / "song.krc"
    lyrics.write_text(krc)
    target = tmp_path / "song.ass"
    tick = 0
    initial = None

    def sleep(seconds):
        nonlocal tick, initial
        tick += 1
        if tick == 1:
            initial = target.read_bytes()
            lyrics.write_text("あ.100")
        if tick == 5:
            assert target.read_bytes() == initial
        if tick == 6:
            replacement = tmp_path / "replacement.krc"
            replacement.write_text(krc.replace("あ", "え"))
            replacement.replace(lyrics)
        if tick == 12:
            assert target.read_bytes() != initial
            assert "え" in target.read_text()
            raise KeyboardInterrupt

    monkeypatch.setattr(ass_cli.time, "sleep", sleep)
    monkeypatch.setattr(ass_cli.time, "monotonic", lambda: tick / 10)
    assert (
        ass_cli.main(
            [
                "--midi",
                str(source),
                "--krc",
                str(lyrics),
                "--out",
                str(target),
                "--watch",
                "--no-wavetone",
            ]
        )
        == 0
    )


def test_session_cli_builds_locally_and_saves_exact_intermediate_exports(exports, monkeypatch, tmp_path):
    krc, channels, notes = exports
    data = midi.to_bytes(channels, notes, 60, wavetone=True)
    calls = []

    class Client:
        def __init__(self, endpoint, **kwargs):
            assert endpoint.address == "/tmp/explicit.sock"
            assert kwargs["keep_events"] is False

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method):
            calls.append(method)
            return {
                "krc": krc,
                "midi": base64.b64encode(data).decode(),
                "settings": {"spacing": 2, "font": "GUI Font"},
                "wavetone": True,
            }

    monkeypatch.setattr(ipc, "Client", Client)
    monkeypatch.setenv("NAMIOTO_ASS_SPACING", "7")
    target, saved_krc, saved_midi = (tmp_path / name for name in ("out.ass", "out.krc", "out.mid"))
    assert (
        ass_cli.main(
            [
                "--session",
                "--endpoint",
                "/tmp/explicit.sock",
                "--out",
                str(target),
                "--save-krc",
                str(saved_krc),
                "--save-midi",
                str(saved_midi),
                "--spacing",
                "9",
            ]
        )
        == 0
    )
    assert calls == ["project.ass_source"]
    assert saved_krc.read_text() == krc
    assert saved_midi.read_bytes() == data
    expected = subtitle.build_ass(krc, data, settings=AssSettings(font="GUI Font", spacing=9), wavetone=True)
    assert target.read_text() == expected


def test_watch_builds_immediately_and_debounces_changes(monkeypatch, tmp_path):
    tick = 0
    revision = 0
    built = []

    def sleep(seconds):
        nonlocal tick, revision
        tick += 1
        if tick in (1, 3):
            revision += 1
        if tick == 10:
            raise KeyboardInterrupt

    monkeypatch.setattr(ass_cli.time, "sleep", sleep)
    monkeypatch.setattr(ass_cli.time, "monotonic", lambda: tick / 10)
    with pytest.raises(KeyboardInterrupt):
        ass_cli.watch(lambda: built.append(revision) or tmp_path / "out.ass", lambda: revision)
    assert built == [0, 2]


def test_watch_recovers_after_a_failed_build(monkeypatch, tmp_path, capsys):
    tick = 0
    built = []

    def sleep(seconds):
        nonlocal tick
        tick += 1
        if tick == 10:
            raise KeyboardInterrupt

    def build():
        built.append(tick)
        if tick == 0:
            raise ValueError("incomplete KRC")
        return tmp_path / "out.ass"

    monkeypatch.setattr(ass_cli.time, "sleep", sleep)
    monkeypatch.setattr(ass_cli.time, "monotonic", lambda: tick / 10)
    with pytest.raises(KeyboardInterrupt):
        ass_cli.watch(build, lambda: tick >= 1)
    assert len(built) == 2
    assert "incomplete KRC" in capsys.readouterr().err


def test_session_watch_stops_on_disconnect_without_rediscovery(monkeypatch, tmp_path, capsys):
    connected = []

    class Client:
        def __init__(self, endpoint, **kwargs):
            connected.append(endpoint)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, method):
            raise ipc.IpcError("closed", "the server closed the connection")

    monkeypatch.setattr(ipc, "Client", Client)
    assert ass_cli.main(["--session", "--watch", "--out", str(tmp_path / "out.ass")]) == 1
    assert len(connected) == 1
    assert "closed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "arguments",
    [[], ["--session", "--midi", "x.mid"], ["--midi", "x.mid"], ["--session", "--timeout", "nan"]],
)
def test_cli_requires_one_valid_source(arguments):
    with pytest.raises(SystemExit) as error:
        ass_cli.main(["--out", "out.ass", *arguments])
    assert error.value.code == 2


def test_outputs_cannot_overwrite_inputs_or_each_other(tmp_path):
    path = tmp_path / "out.ass"
    with pytest.raises(ValueError, match="distinct"):
        subtitle.check_paths([path, path])
    with pytest.raises(ValueError, match="input"):
        subtitle.check_paths([path], [path])


def test_atomic_output_failure_preserves_previous_ass(exports, tmp_path, monkeypatch):
    from namioto import utils

    krc, channels, notes = exports
    data = midi.to_bytes(channels, notes, 60)
    path = tmp_path / "out.ass"
    path.write_text("previous ASS")

    def fail(source, target):
        raise OSError("read-only output")

    monkeypatch.setattr(utils.os, "replace", fail)
    with pytest.raises(OSError):
        subtitle.save_export(path, krc, data, subtitle.build_ass(krc, data))
    assert path.read_text() == "previous ASS"
    assert not list(tmp_path.glob("*.tmp"))
