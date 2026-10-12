# SPDX-License-Identifier: AGPL-3.0-only
"""Checks for the project file: the values it holds, how a bad one is brought back, and what it refuses."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from namioto import project


def make(**values) -> project.Project:
    return project.Project(settings=project.default_settings(), **values)


def test_a_saved_project_can_be_read_back(tmp_path) -> None:
    path = tmp_path / "song.nto"
    saved = make(audio="vocal.wav", notes=(project.Note(0.73, 0.37, 63), project.Note(1.5, 2.0, 55)))
    project.save(saved, path)
    opened = project.load(path)
    assert opened.audio == "vocal.wav"
    assert opened.notes == (project.Note(0.73, 0.37, 63), project.Note(1.5, 2.0, 55))
    assert opened.settings == saved.settings


def test_the_file_is_text_with_a_name_and_a_version(tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(make(), path)
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    data = json.loads(text)
    assert (data["format"], data["version"]) == (project.FORMAT, project.VERSION)
    assert data["notes"] == []


def test_only_the_values_that_belong_to_the_document_are_written() -> None:
    settings = project.default_settings()
    settings.playback.speed = 0.75
    written = project.to_dict(project.Project(settings=settings))
    assert "wavetone" not in json.dumps(written)  # the program's own preference is not the song's
    assert "last_audio_dir" not in json.dumps(written)
    assert set(written) == {
        "format",
        "version",
        "audio",
        "channels",
        "notes",
        "note_id",
        "lyrics",
        "analysis",
        "spectrum",
        "playback",
        "editor",
        "tempo",
        "view",
    }
    assert set(written["playback"]) == {"audio_volume", "midi_volume", "speed"}
    assert set(written["editor"]) == {"snap", "division", "grid_offset_ms", "zoom_x", "zoom_y"}
    assert set(written["view"]) == {"center_x", "center_y"}


def test_the_machine_values_in_a_file_are_ignored(tmp_path) -> None:
    path = tmp_path / "song.nto"
    path.write_text(
        json.dumps(
            {
                "format": "namioto",
                "playback": {"speed": 0.75},
                "editor": {"grid_offset_ms": 120},
                "midi": {"wavetone": False},
                "paths": {"last_audio_dir": "/tmp"},
                "notes": [],
            }
        )
    )
    opened = project.load(path)
    assert opened.settings.playback.speed == 0.75
    assert opened.settings.editor.grid_offset_ms == 120
    assert not hasattr(opened.settings, "midi")
    assert not hasattr(opened.settings, "paths")


def test_a_field_from_an_older_project_is_ignored_rather_than_refused(tmp_path) -> None:
    path = tmp_path / "song.nto"
    path.write_text(json.dumps({"format": "namioto", "playback": {"latency_ms": 120}}))
    opened = project.load(path)
    assert opened.settings.editor.grid_offset_ms == 0


def test_an_empty_or_partial_file_still_gives_every_value(tmp_path) -> None:
    path = tmp_path / "song.nto"
    path.write_text(json.dumps({"format": "namioto", "tempo": {"bpm": 93.0}}))
    opened = project.load(path)
    assert opened.settings.tempo.bpm == 93.0
    assert opened.settings.analysis.a4 == 440.0
    assert opened.notes == ()


def test_a_value_out_of_range_is_brought_back_in_line(tmp_path) -> None:
    path = tmp_path / "song.nto"
    path.write_text(
        json.dumps(
            {
                "format": "namioto",
                "tempo": {"bpm": 9999.0},
                "analysis": {"channels": "middle", "a4": "high"},
                "playback": {"speed": "fast"},
            }
        )
    )
    opened = project.load(path)
    assert opened.settings.tempo.bpm == 300.0
    assert opened.settings.analysis.channels == "mono"
    assert opened.settings.analysis.a4 == 440.0
    assert opened.settings.playback.speed == 1.0


def test_another_json_file_is_refused() -> None:
    with pytest.raises(ValueError, match="not a namioto project"):
        project.from_dict({"version": 1})
    with pytest.raises(ValueError, match="not a namioto project"):
        project.from_dict([1, 2, 3])


def test_a_file_that_is_not_json_is_refused(tmp_path) -> None:
    path = tmp_path / "song.nto"
    path.write_text("this is not json")
    with pytest.raises(ValueError):
        project.load(path)
    with pytest.raises(OSError):
        project.load(tmp_path / "missing.nto")


def test_a_file_from_a_later_version_is_still_read(tmp_path) -> None:
    path = tmp_path / "song.nto"
    path.write_text(json.dumps({"format": "namioto", "version": 99, "tempo": {"bpm": 100.0}}))
    assert project.load(path).settings.tempo.bpm == 100.0


def test_notes_are_rounded_to_a_tenth_of_a_millisecond() -> None:
    written = project.to_dict(make(notes=(project.Note(0.7300000001, 0.37000001, 63),)))
    assert written["notes"] == [{"start": 0.73, "duration": 0.37, "pitch": 63, "channel": 0}]


def test_a_note_that_makes_no_sense_is_left_out(recwarn) -> None:
    data = {
        "format": "namioto",
        "notes": [
            {"start": 1.0, "duration": 1.0, "pitch": 60},
            {"start": -2.0, "duration": 1.0, "pitch": 60.4},  # brought back in line, not dropped
            {"start": 0.0, "duration": 0.0, "pitch": 60},
            {"start": 0.0, "duration": -1.0, "pitch": 60},
            {"start": "x", "duration": 1.0, "pitch": 60},
            {"start": 0.0, "duration": 1.0},
            {"start": 0.0, "duration": 1.0, "pitch": True},
            {"start": float("inf"), "duration": 1.0, "pitch": 60},
            [1.0, 1.0, 60],
        ],
    }
    assert project.from_dict(data).notes == (project.Note(1.0, 1.0, 60), project.Note(0.0, 1.0, 60))
    assert "7 of the notes" in str(recwarn[0].message)


def test_notes_that_are_not_a_list_leave_an_empty_project() -> None:
    assert project.from_dict({"format": "namioto", "notes": "lots"}).notes == ()


def test_the_lyric_times_survive_a_round_trip() -> None:
    lyrics = project.Lyrics(
        text="あん\n", key="abc", model="mms", mode="read", raw=(((0.0, 1.0), (None, None)),), flagged=(True,)
    )
    written = project.to_dict(make(lyrics=lyrics))
    assert written["lyrics"] == {
        "text": "あん\n",
        "key": "abc",
        "model": "mms",
        "raw": [[[0.0, 1.0], [None, None]]],
        "scores": [],
        "flagged": [True],
        "problems": [],
    }
    assert project.from_dict(written).lyrics == lyrics


def test_a_lyrics_block_without_a_text_still_reads() -> None:
    data = {"format": "namioto", "lyrics": {"key": "abc", "model": "mms", "raw": [[[0.0, 1.0]]]}}
    assert project.from_dict(data).lyrics == project.Lyrics(key="abc", model="mms", raw=(((0.0, 1.0),),))


def test_an_unknown_lyric_mode_without_legacy_operations_reads_krc() -> None:
    data = {"format": "namioto", "lyrics": {"key": "a", "raw": [], "mode": "sideways"}}
    assert project.from_dict(data).lyrics.mode == "read"


def test_a_broken_lyric_block_is_dropped() -> None:
    assert project.from_dict({"format": "namioto", "lyrics": "lots"}).lyrics is None
    assert project.from_dict({"format": "namioto", "lyrics": {"key": "a"}}).lyrics is None
    assert project.from_dict({"format": "namioto", "lyrics": {"key": "a", "raw": [[1.0]]}}).lyrics is None


def test_the_audio_is_stored_beside_the_project_when_it_can_be(tmp_path) -> None:
    (tmp_path / "song.wav").write_bytes(b"")
    assert project.store_audio(tmp_path / "song.nto", tmp_path / "song.wav") == "song.wav"
    assert project.store_audio(tmp_path / "song.nto", str(tmp_path / "song.wav")) == "song.wav"
    assert project.store_audio(tmp_path / "song.nto", "/elsewhere/song.wav") == "/elsewhere/song.wav"
    assert project.store_audio(tmp_path / "song.nto", None) == ""
    assert project.store_audio(tmp_path / "song.nto", "") == ""


def test_the_audio_is_found_next_to_the_project_first(tmp_path) -> None:
    assert project.resolve_audio(tmp_path / "song.nto", "song.wav") == tmp_path / "song.wav"
    assert project.resolve_audio(tmp_path / "song.nto", "/elsewhere/song.wav") == Path("/elsewhere/song.wav")
    assert project.resolve_audio(tmp_path / "song.nto", "~/song.wav") == Path.home() / "song.wav"
    assert project.resolve_audio(tmp_path / "song.nto", "") is None


def test_the_suffix_says_whether_a_file_is_one_of_ours() -> None:
    assert project.SUFFIX == ".nto"
    assert project.looks_like_project("song.nto")
    assert project.looks_like_project(Path("SONG.NTO"))
    assert not project.looks_like_project("song.wav")
    assert not project.looks_like_project("song")


def test_saving_leaves_no_half_written_file_behind(tmp_path) -> None:
    path = tmp_path / "song.nto"
    project.save(make(notes=(project.Note(1.0, 1.0, 60),)), path)
    project.save(make(notes=(project.Note(2.0, 1.0, 62),)), path)
    assert [item.name for item in tmp_path.iterdir()] == ["song.nto"]
    assert project.load(path).notes == (project.Note(2.0, 1.0, 62),)


def test_only_the_habits_are_worth_carrying_across_projects() -> None:
    reusable = {key for key, spec in project.FIELD_SPECS.items() if spec.reuse}
    assert reusable == {
        ("analysis", "channels"),
        ("analysis", "t_num"),
        ("analysis", "fft_points"),
        ("spectrum", "gain"),
        ("spectrum", "contrast"),
        ("playback", "audio_volume"),
        ("playback", "midi_volume"),
        ("editor", "snap"),
        ("editor", "division"),
        ("editor", "zoom_x"),
        ("editor", "zoom_y"),
    }


def test_a_new_document_takes_the_remembered_values_it_is_worth_carrying() -> None:
    settings = project.default_settings({"spectrum": {"gain": 300.0, "contrast": 1.4}, "editor": {"zoom_x": 96.0}})
    assert settings.spectrum.gain == 300.0
    assert settings.spectrum.contrast == 1.4
    assert settings.editor.zoom_x == 96.0
    assert settings.editor.zoom_y == project.FIELD_SPECS[("editor", "zoom_y")].default


def test_a_remembered_value_is_checked_like_any_other() -> None:
    settings = project.default_settings({"spectrum": {"gain": 9999.0}, "editor": {"snap": "wide"}})
    assert settings.spectrum.gain == 600.0
    assert settings.editor.snap == 0.5


def test_a_remembered_value_the_song_owns_is_left_out() -> None:
    settings = project.default_settings(
        {"tempo": {"bpm": 93.0}, "editor": {"grid_offset_ms": 120}, "view": {"center_x": 5.0}}
    )
    assert settings.tempo.bpm == 120.0
    assert settings.editor.grid_offset_ms == 0
    assert settings.view.center_x == 8.0


def test_remembering_a_habit_checks_it_like_any_other_value() -> None:
    remembered: dict[str, dict] = {}
    project.remember(remembered, "spectrum", "gain", 9999.0)
    project.remember(remembered, "editor", "zoom_x", 72.0)
    assert remembered["spectrum"]["gain"] == 600.0
    assert remembered["editor"]["zoom_x"] == 72.0


def test_remembering_changes_keeps_only_the_reusable_fields_that_moved() -> None:
    before = project.default_settings()
    after = project.default_settings()
    after.analysis.fft_points = 4096
    after.analysis.a4 = 442.0  # the song's own, so not a habit
    remembered: dict[str, dict] = {}

    assert project.remember_changes(remembered, before, after) is True
    assert remembered == {"analysis": {"fft_points": 4096}}
    assert project.remember_changes(remembered, before, before) is False


def test_a_remembered_map_that_is_not_a_map_gives_the_defaults() -> None:
    assert project.default_settings(None) == project.default_settings()
    assert project.default_settings("nonsense") == project.default_settings()
    assert project.default_settings({"spectrum": "loud"}) == project.default_settings()


def test_note_ids_and_the_next_free_id_survive_a_round_trip(tmp_path) -> None:
    path = tmp_path / "song.nto"
    notes = (project.Note(1.0, 1.0, 60, 0, 4), project.Note(2.0, 1.0, 62, 0, 7))
    project.save(make(notes=notes, next_id=9), path)
    opened = project.load(path)
    assert opened.notes == notes
    assert opened.next_id == 9


def test_a_project_without_ids_leaves_them_for_the_document_to_fill() -> None:
    data = {
        "format": "namioto",
        "notes": [
            {"start": 1.0, "duration": 1.0, "pitch": 60},
            {"start": 2.0, "duration": 1.0, "pitch": 62},
        ],
    }
    opened = project.from_dict(data)
    assert [note.id for note in opened.notes] == [0, 0]
    assert opened.next_id == 1


def test_a_next_id_that_would_reuse_a_taken_one_is_pushed_past_it() -> None:
    data = {
        "format": "namioto",
        "note_id": 2,
        "notes": [{"start": 1.0, "duration": 1.0, "pitch": 60, "id": 5}],
    }
    assert project.from_dict(data).next_id == 6


def test_the_mapping_survives_a_round_trip(tmp_path) -> None:
    from namioto.karaoke.operations import Match, Merge, SoundRef

    lyrics = project.Lyrics(
        text="@3{胡椒[こ,しょう]}\n",
        mode="edit",
        key="abc",
        raw=(((0.0, 0.9), (1.0, 0.4), (2.0, 0.0)),),
        scores=((0.9, None, 0.8),),
        flagged=(True,),
        problems=(("a problem",),),
        operations=(
            Match(SoundRef(0, 0), (1,), confirmed=True),
            Merge((SoundRef(0, 1), SoundRef(0, 2)), 2),
        ),
        version=2,
    )
    path = tmp_path / "song.nto"
    project.save(make(lyrics=lyrics), path)
    opened = project.load(path).lyrics
    assert opened == lyrics


def test_span_based_lyric_evidence_is_not_read_as_reference_durations() -> None:
    data = {
        "format": "namioto",
        "lyrics": {"text": "あ\n", "key": "k", "lines": [[[0.0, 1.0]]]},
    }
    assert project.from_dict(data).lyrics is None


def test_a_malformed_operation_is_left_out() -> None:
    from namioto.karaoke.operations import Match, SoundRef

    data = {
        "format": "namioto",
        "lyrics": {
            "text": "あい\n",
            "key": "k",
            "raw": [[[0.0, 1.0], [1.0, 1.0]]],
            "operations": [
                {"kind": "match", "sound": [0, 0], "notes": [1], "confirmed": True},
                {"kind": "match", "sound": [0, 1]},
                "not an operation",
            ],
        },
    }
    assert project.from_dict(data).lyrics.operations == (Match(SoundRef(0, 0), (1,), confirmed=True),)
