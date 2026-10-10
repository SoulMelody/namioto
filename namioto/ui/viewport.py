# SPDX-License-Identifier: AGPL-3.0-only
"""The strips that share the roll's columns.

The roll's ruler and keyboard and the lyrics strip all read the view's scroll position rather than
keeping one of their own, so a strip and the roll never drift apart. The view is typed as a
`QGraphicsView` on purpose: the strip only needs its viewport, and the concrete view lives in
`namioto.ui.roll`, which imports this module.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QGraphicsView, QWidget


class ViewportStrip(QWidget):
    """A strip sharing a view's columns: the viewport's top left in this widget's coordinates."""

    def __init__(self, view: QGraphicsView):
        super().__init__()
        self.view = view

    def origin(self) -> QPoint:
        return self.mapFromGlobal(self.view.viewport().mapToGlobal(QPoint(0, 0)))
