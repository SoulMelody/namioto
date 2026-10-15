# SPDX-License-Identifier: AGPL-3.0-only
import pytest

from namioto import project
from namioto.channels import Channel
from namioto.karaoke.operations import SoundRef

from .helpers import wait_for_lyric_mapping


@pytest.mark.parametrize(
    ("text", "count"),
    [("あ\nい", 1), ("あ\n@1{い}", 1)],
)
def test_failed_subtitle_export_reveals_the_affected_sound_on_the_roll(own_window, tmp_path, text, count):
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes(tuple((60, float(at), 1.0, 0) for at in range(count)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text=text)
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    gate = own_window._verification(subtitle=True)
    incomplete = next(problem for problem in gate.problems if problem.code == "incomplete_alignment")
    assert incomplete.refs == (SoundRef(1, 0),)
    assert own_window.lyric_editor.isHidden()
    target = tmp_path / "incomplete.ass"
    assert not own_window.export_ass(target)
    assert not target.exists()
    assert own_window.lyric_editor.isHidden()
    assert own_window.sound_strip._hover == (1, 0)
    assert own_window.view.active_channel == (1 if "@1" in text else 0)
    rect = own_window.sound_strip._blocks()[(1, 0)]
    origin = own_window.sound_strip.origin().x()
    assert origin <= rect.center().x() <= origin + own_window.view.viewport().width()
    assert "line 2, sound 1" in own_window.statusBar().currentMessage()


def test_subtitle_export_silently_truncates_the_final_sound_note_count(own_window, tmp_path):
    own_window.view.set_notes(((60, 0.0, 1.0, 0), (62, 1.0, 1.0, 0)))
    own_window.project_path = tmp_path / "song.nto"
    own_window._stored_lyrics = project.Lyrics(text="あ\nい.2")
    own_window._watch_lyrics()
    wait_for_lyric_mapping(own_window)

    assert own_window._verification(subtitle=True).open()
    target = tmp_path / "truncated.ass"
    assert own_window.export_ass(target)
    assert target.exists()
    assert own_window.lyrics_text == "あ\nい.2"
    assert own_window.lyric_editor.isHidden()
