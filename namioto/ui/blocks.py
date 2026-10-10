# SPDX-License-Identifier: AGPL-3.0-only
"""The pointer grammar the notes and the lyrics sounds share.

Both editors draw a timed block over the roll's shared time axis and answer the pointer the same
way: an edge resizes the block, its body moves it, and a press stays a click - narrowing the
selection - until it travels past `CLICK_SLOP_PX`. The block and the constraints on its span stay
with each editor; only the hit test and the click/drag boundary live here.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QPointF

CLICK_SLOP_PX = 4


def block_part(x: float, start: float, end: float, grab: float) -> str:
    """Which part of a block a pointer at `x` is on: `"left"`, `"right"` or `"move"`.

    The nearer edge wins when the block is too narrow for both grab bands, so a resize always
    takes the edge the pointer is closest to.
    """
    if abs(x - start) <= abs(x - end) and abs(x - start) <= grab:
        return "left"
    if abs(x - end) <= grab:
        return "right"
    return "move"


@dataclass
class Press:
    """A press that is a click until it travels past `CLICK_SLOP_PX`.

    `block` is what the pointer landed on, whatever the editor calls it - a `NoteItem`, a sound's
    (row, column). A click narrows a multi-selection to it; a drag carries the selection instead.
    """

    block: object
    origin: QPointF
    dragged: bool = False

    def follow(self, position: QPointF) -> bool:
        if not self.dragged and (position - self.origin).manhattanLength() > CLICK_SLOP_PX:
            self.dragged = True
        return self.dragged
