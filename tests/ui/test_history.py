# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPointF, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from namioto import project
from namioto.channels import Channel
from namioto.interaction import Interaction, Tool
from namioto.ui.controls import EditBar
from namioto.ui.roll import PITCH_MAX, SNAP_CHOICES, PianoRollView

from .helpers import draw_note, roll_mouse


def test_copy_and_paste_drops_the_selection_at_the_playhead() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.add_note(60, 2.0, 1.0)
    view.add_note(64, 3.0, 0.5)
    for note in view.notes():
        note.setSelected(True)
    view.set_playhead(3.0)  # six beats at 120 BPM
    view.undo_stack.clear()

    assert view.copy_selection()
    assert view.paste_notes()
    assert [(note.pitch, note.start, note.duration) for note in view.notes()] == [
        (60, 2.0, 1.0),
        (64, 3.0, 0.5),
        (60, 6.0, 1.0),
        (64, 7.0, 0.5),
    ]
    assert [note.pitch for note in view.selected_notes()] == [60, 64]  # the paste is what is selected
    assert view.undo_stack.count() == 1
    view.undo()
    assert len(view.notes()) == 2


def test_paste_snaps_the_anchor_and_the_spacing() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5  # the 1/8 default
    view.add_note(60, 2.0, 1.0)
    view.add_note(62, 3.3, 0.5)  # off the grid, so the paste has to align it
    for note in view.notes():
        note.setSelected(True)
    view.set_playhead(1.05)  # 2.1 beats, which snap to 2.0

    assert view.copy_selection()
    assert view.paste_notes()
    assert [(note.pitch, note.start) for note in view.notes()[2:]] == [(60, 2.0), (62, 3.5)]


@pytest.mark.parametrize("bpm", [90.0, 120.0, 180.0])
@pytest.mark.parametrize("grid_offset", [-0.07, 0.07])
def test_paste_aligns_to_the_offset_grid_at_the_current_bpm(bpm, grid_offset) -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.bpm = bpm
    view.set_offset(grid_offset)
    view.snap = 0.5
    view.add_note(60, 2.0, 1.0)
    view.add_note(62, 3.3, 0.5)
    for note in view.notes():
        note.setSelected(True)
    view.set_playhead(view.to_seconds(4.1) + grid_offset)

    assert view.copy_selection()
    assert view.paste_notes()

    pasted = view.notes()[2:]
    assert [note.start for note in pasted] == pytest.approx([4.0 + view.offset_beats, 5.5 + view.offset_beats])
    assert [note.duration for note in pasted] == [1.0, 0.5]


def test_paste_without_a_copy_does_nothing() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_playhead(1.0)
    assert not view.paste_notes()
    assert view.notes() == []


def test_copy_and_paste_need_edit_mode() -> None:
    view = PianoRollView()
    view.set_channels((Channel(channel=0),))
    view.add_note(60, 2.0, 1.0).setSelected(True)
    assert not view.copy_selection()
    assert not view.paste_notes()


def test_quantize_puts_the_notes_on_the_snap_grid() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 0.6, 0.7)  # starts 0.5, ends 1.5
    view.add_note(62, 1.9, 0.2)  # both ends round to the same cell, so one cell is the length
    view.undo_stack.clear()

    assert view.quantize_notes()
    assert [(note.start, note.duration) for note in view.notes()] == [(0.5, 1.0), (2.0, 0.5)]
    assert view.undo_stack.count() == 1
    view.undo()
    assert [(note.start, note.duration) for note in view.notes()] == [(0.6, 0.7), (1.9, 0.2)]


def test_quantize_leaves_a_note_off_its_neighbour_rather_than_on_it() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 1.1, 0.2)  # both 60s round to the cell 1.0-1.5
    view.add_note(60, 1.2, 0.2)
    view.add_note(62, 3.1, 0.2)  # another pitch, free to snap

    assert view.quantize_notes()
    assert [(note.pitch, round(note.start, 2), round(note.duration, 2)) for note in view.notes()] == [
        (60, 1.1, 0.2),  # the grid would fold it onto the other 60, so it stays
        (60, 1.2, 0.2),
        (62, 3.0, 0.5),
    ]


def test_quantize_takes_only_the_selection_when_there_is_one() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 0.6, 1.0).setSelected(True)
    view.add_note(62, 1.6, 1.0)

    assert view.quantize_notes()
    assert [(note.pitch, note.start) for note in view.notes()] == [(60, 0.5), (62, 1.6)]


def test_quantize_leaves_a_locked_channel_where_it_is() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0, lock=True),))
    view.snap = 0.5
    view.add_note(60, 0.6, 1.0)

    assert not view.quantize_notes()
    assert view.notes()[0].start == 0.6


def test_a_note_already_on_the_grid_is_not_an_edit() -> None:
    view = PianoRollView()
    view.apply_interaction(Interaction.editing_with(Tool.PEN))
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 1.0, 1.0)
    view.undo_stack.clear()

    assert not view.quantize_notes()
    assert view.undo_stack.count() == 0


def test_quantize_works_outside_edit_mode() -> None:
    view = PianoRollView()
    view.set_channels((Channel(channel=0),))
    view.snap = 0.5
    view.add_note(60, 0.6, 1.0)

    assert view.edit_mode is False
    assert view.quantize_notes()
    assert view.notes()[0].start == 0.5


def test_the_quantize_button_quantizes_the_roll(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.snap = 0.5
    window.view.add_note(60, 0.6, 1.0)
    window.view.undo_stack.clear()

    window.edit.quantize.click()
    assert window.view.notes()[0].start == 0.5
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    window.view.clear_notes()


def test_the_snap_grid_and_quantize_stay_usable_outside_edit_mode() -> None:
    bar = EditBar(SNAP_CHOICES)
    assert bar.snap.isEnabled() and bar.quantize.isEnabled()
    bar.mode.click()
    assert bar.snap.isEnabled() and bar.quantize.isEnabled()


def test_the_quantize_button_asks_the_roll_to_quantize() -> None:
    bar = EditBar(SNAP_CHOICES)
    asked: list[bool] = []
    bar.quantize_requested.connect(lambda: asked.append(True))
    bar.quantize.click()
    assert asked == [True]


def test_ctrl_c_and_ctrl_v_carry_the_selection(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 1.0).setSelected(True)
    window.view.set_playhead(2.0)  # four beats at 120 BPM

    QApplication.setActiveWindow(window)  # an offscreen window is never active on its own
    QTest.keyClick(window, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier)
    QTest.keyClick(window, Qt.Key.Key_V, Qt.KeyboardModifier.ControlModifier)
    assert [(note.pitch, note.start) for note in window.view.notes()] == [(69, 2.0), (69, 4.0)]
    window.view.set_playhead(None)
    window.view.clear_notes()


def test_a_new_edit_drops_the_redo_branch(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()

    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))
    window.view.undo()
    assert window.view.undo_stack.canRedo()
    draw_note(window, QPointF(6.0, 40.0), QPointF(7.0, 40.0))
    assert not window.view.undo_stack.canRedo()
    window.view.clear_notes()


def test_channel_edits_and_a_removal_can_be_undone(window) -> None:
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_channel(program=4)
    window.view.add_note(60, 0.0, 1.0, 0)
    window.view.add_note(64, 1.0, 1.0, 1)
    window.view.undo_stack.clear()

    window.view.set_channel_field(1, mute=True)
    assert window.view.channels[1].mute is True
    window.view.undo()
    assert window.view.channels[1].mute is False

    assert window.view.remove_channel(0)
    assert [note.pitch for note in window.view.notes()] == [64]
    window.view.undo()
    assert [(note.pitch, note.channel) for note in window.view.notes()] == [(60, 0), (64, 1)]
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()


def test_a_whole_document_replacement_is_one_undo_step(window) -> None:
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()

    window.view.replace((Channel(channel=0),), [(60, 0.0, 1.0, 0), (64, 1.0, 1.0, 0)], "Import MIDI")
    assert window.view.undo_stack.count() == 1
    window.view.undo()
    assert window.view.notes() == []


def test_undo_keeps_a_note_on_the_audio_across_a_tempo_change(window) -> None:
    window.edit.pen.click()
    window.transport.bpm.setValue(120.0)
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.add_note(69, 2.0, 2.0)
    window.view.undo_stack.clear()
    seconds = 2.0 * window.view.seconds_per_beat

    row = float(PITCH_MAX - 69) + 0.5
    window.view.centerOn(QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(3.0, row))
    roll_mouse(window, QEvent.Type.MouseMove, QPointF(4.0, row))
    roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(4.0, row))

    window.transport.bpm.setValue(60.0)  # beats halve, seconds do not
    window.view.undo()
    restored = window.view.notes()[0]
    assert restored.start * window.view.seconds_per_beat == pytest.approx(seconds)
    window.transport.bpm.setValue(120.0)
    window.view.clear_notes()


def test_the_time_conversion_runs_both_ways(window) -> None:
    window.transport.bpm.setValue(120.0)
    assert window.view.to_seconds(2.0) == pytest.approx(1.0)
    assert window.view.to_beats(1.0) == pytest.approx(2.0)
    window.transport.bpm.setValue(60.0)
    assert window.view.to_seconds(0.5) == pytest.approx(0.5)
    assert window.view.to_beats(2.0) == pytest.approx(2.0)
    window.transport.bpm.setValue(120.0)


def test_ctrl_z_and_ctrl_shift_z_drive_the_history(window) -> None:
    window.edit.pen.click()
    window.view.set_channels((Channel(channel=0),))
    window.view.clear_notes()
    window.view.undo_stack.clear()
    draw_note(window, QPointF(2.0, 40.0), QPointF(3.0, 40.0))

    QApplication.setActiveWindow(window)  # an offscreen window is never active on its own
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    assert len(window.view.notes()) == 1
    QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
    assert window.view.notes() == []
    QTest.keyClick(window, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier)
    assert len(window.view.notes()) == 1
    window.view.clear_notes()


def test_opening_a_project_starts_the_history_over(own_window, tmp_path) -> None:
    own_window.view.set_channels((Channel(channel=0),))
    own_window.view.add_note(60, 0.0, 1.0)
    assert own_window.view.undo_stack.canUndo()

    path = tmp_path / "song.nto"
    project.save(
        project.Project(settings=project.default_settings(), notes=(project.Note(1.0, 0.5, 62),)),
        path,
    )
    assert own_window.load_project(path)
    assert not own_window.view.undo_stack.canUndo()
    assert not own_window.view.undo_stack.canRedo()


def test_dragging_a_box_fills_the_selection_under_any_style(window) -> None:
    QApplication.setStyle("Windows")  # its own rubber band ignores a stylesheet, and the box is ours
    window.edit.select.click()
    window.view.centerOn(QPointF(5.0, 63.0))
    try:
        roll_mouse(window, QEvent.Type.MouseButtonPress, QPointF(2.0, 60.0))
        roll_mouse(window, QEvent.Type.MouseMove, QPointF(8.0, 66.0))
        rubber = window.view._rubber
        assert rubber.isVisible()
        area = window.view.mapFromScene(rubber.rect()).boundingRect().intersected(window.view.viewport().rect())
        image = window.view.viewport().grab().toImage()
        accent = window.view.palette().highlight().color()
        canvas = image.pixelColor(area.left() - 20, area.center().y())
        inside = image.pixelColor(area.center())

        def from_accent(colour: QColor) -> int:
            """How far one pixel is from the palette's accent, across the three channels."""
            return (
                abs(colour.red() - accent.red())
                + abs(colour.green() - accent.green())
                + abs(colour.blue() - accent.blue())
            )

        # the playhead sits on the box's left edge, so the outline is read off the top one, wherever
        # the antialiased pixel of it landed
        border = min(
            (image.pixelColor(area.center().x(), y) for y in range(area.top() - 1, area.top() + 2)),
            key=from_accent,
        )

        # the accent is whatever the palette carries - blue, grey, red - so the pixels are measured
        # against it rather than against a colour this test would be guessing
        assert inside != canvas, "the region is filled"
        assert from_accent(inside) < from_accent(canvas), "the fill is the accent over the canvas"
        assert from_accent(border) < from_accent(canvas), "the outline is the accent, not the style's own"
        roll_mouse(window, QEvent.Type.MouseButtonRelease, QPointF(8.0, 66.0))
    finally:
        QApplication.setStyle("Fusion")
        window.edit.mode.click()
