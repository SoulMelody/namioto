# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtTest import QTest

from namioto import project
from namioto.channels import Channel

from .helpers import drag_sound, sound_mouse, wait_for_lyric_mapping


def _gesture_lyrics(window, tmp_path, text, count=6):
    window.project_path = tmp_path / "gestures.nto"
    window.transport.bpm.setValue(60)
    window.view.set_notes(tuple((60 + index, index, 1, 0, index + 1) for index in range(count)))
    window._stored_lyrics = project.Lyrics(text=text)
    window._lyric_key = ""
    window._restore_lyrics()
    wait_for_lyric_mapping(window)


def _shift_lyric(window, seconds, target, *, cancel=False):
    strip = window.sound_strip
    shift = Qt.KeyboardModifier.ShiftModifier
    sound_mouse(window, QEvent.Type.MouseButtonPress, strip._x(seconds), modifiers=shift)
    sound_mouse(window, QEvent.Type.MouseMove, strip._x(target), modifiers=shift)
    if cancel:
        QTest.keyClick(strip, Qt.Key.Key_Escape)
    else:
        sound_mouse(window, QEvent.Type.MouseButtonRelease, strip._x(target), modifiers=shift)
    wait_for_lyric_mapping(window)


def test_plain_sound_drag_merges_and_moves_following_lyrics_forward(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あいう")
    before = own_window.view.undo_stack.count()
    drag_sound(own_window, (0, 1), own_window.sound_strip._x(0.5))
    assert own_window.lyrics_text == "(あい).1う"
    assert own_window.view.lyric_mapped == (((1,), (1,), (2,)),)
    assert own_window.view.undo_stack.count() == before + 1
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あいう"
    assert own_window.view.lyric_mapped == (((1,), (2,), (3,)),)
    own_window.view.redo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "(あい).1う"


def test_shift_right_resizes_the_whole_group_without_changing_its_neighbor(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あい).1う.2")
    before = own_window.view.undo_stack.count()
    _shift_lyric(own_window, 0.75, 1.75)
    assert own_window.lyrics_text == "(あい).2う.2"
    assert own_window.view.lyric_group[0][0] == own_window.view.lyric_group[0][1] >= 0
    assert own_window.view.lyric_mapped == (((1,), (2,), (3, 4)),)
    assert own_window.view.undo_stack.count() == before + 1
    own_window.view.undo()
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "(あい).1う.2"


def test_shift_left_only_exchanges_counts_with_the_left_neighbor(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あ.3い.2う")
    _shift_lyric(own_window, 3.25, 2.25)
    assert own_window.lyrics_text == "あ.2い.3う"
    assert own_window.view.lyric_mapped == (((1, 2), (3, 4, 5), (6,)),)
    _shift_lyric(own_window, 2.25, -1.0)
    assert own_window.lyrics_text == "あ.0い.5う"
    assert own_window.view.lyric_mapped == (((), (1, 2, 3, 4, 5), (6,)),)


def test_shift_left_cannot_move_the_first_unit_start(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あ.2い")
    before = own_window.view.undo_stack.count()
    _shift_lyric(own_window, 0.25, -1.0)
    assert own_window.lyrics_text == "あ.2い"
    assert own_window.view.undo_stack.count() == before


def test_unit_boundary_drag_keeps_the_total_without_merging(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あ.2い.2う")
    strip = own_window.sound_strip
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(2.0))
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(3.0))
    sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(3.0))
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == "あ.3いう"
    assert own_window.view.lyric_mapped == (((1, 2, 3), (4,), (5,)),)
    assert own_window.view.lyric_group == ((-1, -1, -1),)


def test_internal_group_boundary_is_a_body_drag_not_a_resize(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あい).2う")
    strip = own_window.sound_strip
    assert strip._boundary_at(strip._x(1.0), 15) is None
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, strip._x(1.0))
    assert strip._drag[0] == "body"
    QTest.keyClick(strip, Qt.Key.Key_Escape)
    assert own_window.lyrics_text == "(あい).2う"


def test_shift_group_can_shrink_to_zero_without_losing_its_members(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あい).1う")
    _shift_lyric(own_window, 0.75, 0.0)
    assert own_window.lyrics_text == "(あい).0う"
    blocks = own_window.sound_strip._blocks()
    assert all(blocks[(0, index)].top() > 36 for index in (0, 1))
    assert own_window.view.lyric_group[0][0] == own_window.view.lyric_group[0][1] >= 0


def test_group_edge_drag_keeps_remaining_count_and_moves_following_lyrics_back(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あいう).2え")
    drag_sound(own_window, (0, 2), own_window.sound_strip._x(3.5))
    assert own_window.lyrics_text == "(あい).2うえ"
    assert own_window.view.lyric_mapped == (((1,), (2,), (3,), (4,)),)


def test_plain_drag_keeps_receiver_count_and_does_not_reorder_sounds(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あ.2い.3う")
    drag_sound(own_window, (0, 0), own_window.sound_strip._x(4.0))
    assert own_window.lyrics_text == "(あい).3う"
    assert own_window.view.lyric_mapped[0][-1] == (4,)
    before = own_window.view.undo_stack.count()
    drag_sound(own_window, (0, 0), own_window.sound_strip._x(3.5))
    assert own_window.lyrics_text == "(あい).3う"
    assert own_window.view.undo_stack.count() == before


def test_zero_sound_drag_to_upper_blank_restores_one_sequential_note(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あ.0い")
    drag_sound(own_window, (0, 0), own_window.sound_strip._x(2.5))
    assert own_window.lyrics_text == "あい"
    assert own_window.view.lyric_mapped == (((1,), (2,)),)


def test_unmatched_labels_are_visible_even_when_the_channel_has_no_notes(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あ.0(いう).2", count=0)
    blocks = own_window.sound_strip._blocks()
    assert set(blocks) == {(0, 0), (0, 1), (0, 2)}
    assert all(rect.top() > 36 and rect.width() > 0 for rect in blocks.values())
    drag_sound(own_window, (0, 1), own_window.sound_strip._x(5.0))
    assert own_window.lyrics_text == "あ.0(いう).2"


def test_unmatched_sound_keeps_its_requested_count_when_dragged_to_upper_blank(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あい.3", count=1)
    drag_sound(own_window, (0, 1), own_window.sound_strip._x(4.0))
    assert own_window.lyrics_text == "あい.3"
    assert own_window.view.lyric_mapped == (((1,), ()),)


def test_shift_resize_truncates_only_the_display_and_is_persisted(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "あい.3", count=3)
    _shift_lyric(own_window, 0.75, 2.75)
    assert own_window.lyrics_text == "あ.3い.3"
    assert own_window.view.lyric_mapped == (((1, 2, 3), ()),)
    assert own_window.view.lyric_zero == ((False, True),)  # い.3 has no notes left to hold it
    assert own_window.save_project(own_window.project_path)
    assert project.load(own_window.project_path).lyrics.text == "あ.3い.3"
    own_window.view.set_notes(tuple((60 + index, index, 1, 0, index + 1) for index in range(6)))
    wait_for_lyric_mapping(own_window)
    assert own_window.view.lyric_mapped == (((1, 2, 3), (4, 5, 6)),)
    assert own_window.lyrics_text == "あ.3い.3"


def test_escape_restores_shift_resize_and_does_not_add_an_undo_step(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あい).1う.2")
    before = own_window.view.undo_stack.count()
    raw = own_window.view.lyric_raw
    _shift_lyric(own_window, 0.75, 1.75, cancel=True)
    assert own_window.lyrics_text == "(あい).1う.2"
    assert own_window.view.lyric_raw == raw
    assert own_window.view.undo_stack.count() == before


@pytest.mark.parametrize("text, count", [("(あい).2う", 3), ("(あい).0う", 1), ("泣[な.2]い.+う", 3)])
def test_auto_mapping_preserves_literal_groups_and_their_counts(own_window, tmp_path, text, count):
    _gesture_lyrics(own_window, tmp_path, text, count=count)
    own_window._adopt_alignment([[(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)]], "mms")
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == text
    assert own_window._lyric_model == "mms"
    assert own_window.view.lyric_group[0][0] == own_window.view.lyric_group[0][1] >= 0


def test_channel_rebuild_preserves_another_channels_literal_multi_note_group(own_window, tmp_path):
    text = "@1{(あい).2}@0{う}"
    _gesture_lyrics(own_window, tmp_path, text, count=1)
    own_window.view.set_channels((Channel(channel=0), Channel(channel=1)))
    own_window.view.set_notes(((60, 0, 1, 1, 10), (62, 1, 1, 1, 20), (64, 0, 1, 0, 30)))
    wait_for_lyric_mapping(own_window)
    own_window._adopt_alignment([[(0.0, 1.0)]], "mms", use_constraints=False, channel=0)
    wait_for_lyric_mapping(own_window)
    assert own_window.lyrics_text == text
    assert own_window.view.lyric_mapped == (((10,), (20,), (30,)),)


def test_shift_can_resize_a_group_across_ruby_containers(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "泣[な]い.+う", count=3)
    _shift_lyric(own_window, 0.75, 1.75)
    assert own_window.lyrics_text == "泣[な.2]い.+う"
    assert own_window.view.lyric_mapped == (((1,), (2,), (3,)),)
    assert own_window.view.lyric_group[0][0] == own_window.view.lyric_group[0][1] >= 0


@pytest.mark.parametrize(
    "text, ref, near, deep, merged",
    [
        ("(あい).1う", (0, 1), 1.1, 1.5, "あ(いう).1"),
        ("あ(いう).1", (0, 1), 0.9, 0.5, "(あい).1う"),
    ],
)
@pytest.mark.parametrize("finish", ["detach", "merge", "cancel"])
def test_group_edge_drag_previews_detach_before_receiving_neighbor(
    own_window, tmp_path, text, ref, near, deep, merged, finish
):
    _gesture_lyrics(own_window, tmp_path, text)
    strip = own_window.sound_strip
    before = own_window.view.undo_stack.count()
    raw = own_window.view.lyric_raw
    rect = strip._blocks()[ref]
    sound_mouse(own_window, QEvent.Type.MouseButtonPress, rect.center().x(), rect.center().y())
    sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(near), 15.0)
    assert own_window.lyrics_text == "あいう"
    assert own_window.view.lyric_mapped == (((1,), (2,), (3,)),)
    assert own_window.view.undo_stack.count() == before
    if finish != "detach":
        sound_mouse(own_window, QEvent.Type.MouseMove, strip._x(deep), 15.0)
        assert own_window.lyrics_text == merged
        assert own_window.view.undo_stack.count() == before
    if finish == "cancel":
        QTest.keyClick(strip, Qt.Key.Key_Escape)
        wait_for_lyric_mapping(own_window)
        assert own_window.lyrics_text == text
        assert own_window.view.undo_stack.count() == before
    else:
        destination = near if finish == "detach" else deep
        sound_mouse(own_window, QEvent.Type.MouseButtonRelease, strip._x(destination), 15.0)
        wait_for_lyric_mapping(own_window)
        expected = "あいう" if finish == "detach" else merged
        assert own_window.lyrics_text == expected
        assert own_window.view.undo_stack.count() == before + 1
        own_window.view.undo_stack.undo()
        wait_for_lyric_mapping(own_window)
        assert own_window.lyrics_text == text
        own_window.view.undo_stack.redo()
        wait_for_lyric_mapping(own_window)
        assert own_window.lyrics_text == expected
    assert own_window.view.lyric_raw == raw


def test_group_edge_stays_detached_over_a_nonadjacent_receiving_unit(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あい).1うえ")
    drag_sound(own_window, (0, 1), own_window.sound_strip._x(2.5))
    assert own_window.lyrics_text == "あいうえ"
    assert own_window.view.lyric_mapped == (((1,), (2,), (3,), (4,)),)


def test_group_edge_can_be_pulled_out_above_an_occupied_neighbor(own_window, tmp_path):
    _gesture_lyrics(own_window, tmp_path, "(あいう).2え")
    drag_sound(own_window, (0, 2), own_window.sound_strip._x(2.1), 0.0)
    assert own_window.lyrics_text == "(あい).2うえ"
    assert own_window.view.lyric_mapped == (((1,), (2,), (3,), (4,)),)
