# SPDX-License-Identifier: AGPL-3.0-only
"""The meter dialog and the command that opens it.

The dialog is a list of spin boxes and nothing more - `MeterMap` does the normalising - so these
tests check the two halves that can go wrong: rows to points, and the window marking the project
dirty after the map changes.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog

from namioto.meter_map import MeterMap, MeterPoint
from namioto.ui.app import MeterDialog


def test_the_dialog_starts_from_the_map_in_force(own_window) -> None:
    meter = MeterMap((MeterPoint(0.0, 4.0, 1), MeterPoint(8.0, 3.0, 1)))
    dialog = MeterDialog(meter, own_window)

    assert dialog.meter_map() == meter
    dialog.deleteLater()


def test_an_added_row_reaches_the_map(own_window) -> None:
    dialog = MeterDialog(MeterMap.single(), own_window)
    dialog._append(20.0, 6.0, 7)

    points = dialog.meter_map().points
    assert points[-1] == MeterPoint(20.0, 6.0, 7)
    dialog.deleteLater()


def test_a_removed_row_leaves_the_map(own_window) -> None:
    dialog = MeterDialog(MeterMap((MeterPoint(0.0, 4.0, 1), MeterPoint(8.0, 3.0, 1))), own_window)
    dialog._remove(dialog._rows[1][0])

    assert dialog.meter_map().points == (MeterPoint(0.0, 4.0, 1),)
    dialog.deleteLater()


def test_applying_a_new_map_marks_the_project_dirty(own_window, monkeypatch, tmp_path) -> None:
    """A meter is song data, unlike the A-B loop: it lands in the project file, so it is an edit."""
    changed = MeterMap((MeterPoint(0.0, 3.0, 1),))

    class Stub:
        def __init__(self, meter, parent) -> None:
            pass

        def exec(self) -> int:
            return QDialog.DialogCode.Accepted

        def meter_map(self) -> MeterMap:
            return changed

    monkeypatch.setattr("namioto.ui.app.MeterDialog", Stub)
    own_window.project_dirty = False

    own_window._edit_meter()

    assert own_window.view.meter_map == changed
    assert own_window.project_dirty is True


def test_cancelling_the_dialog_changes_nothing(own_window, monkeypatch) -> None:
    class Stub:
        def __init__(self, meter, parent) -> None:
            pass

        def exec(self) -> int:
            return QDialog.DialogCode.Rejected

    monkeypatch.setattr("namioto.ui.app.MeterDialog", Stub)
    before = own_window.view.meter_map
    own_window.project_dirty = False

    own_window._edit_meter()

    assert own_window.view.meter_map == before
    assert own_window.project_dirty is False
