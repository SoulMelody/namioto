# SPDX-License-Identifier: AGPL-3.0-only
"""The MIDI import dialog."""

from __future__ import annotations

from collections.abc import Collection

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
)

from namioto.channels import CHANNEL_COUNT, Channel
from namioto.i18n import tr


def _title(channel: Channel) -> str:
    """A channel as the dialog names it: its own name when it has one, its number either way."""
    number = channel.channel
    return (
        tr("{name} (channel {number})", name=channel.name, number=number)
        if channel.name
        else tr("Channel {number}", number=number)
    )


class MidiImportDialog(QDialog):
    """What an import does to the roll, and which channel each file channel lands on.

    A merge is the additive choice: every file channel is pointed at one of the roll's channels or at
    a new one. The file's i-th channel starts on the roll's i-th channel while that one carries no
    notes, and the ones whose place is taken fill the empty channels left, in order; the dialog is
    where that is changed before the notes arrive.
    """

    def __init__(self, imported, channels, source: str, occupied: Collection[int] = (), parent=None):
        super().__init__(parent)
        self.setWindowTitle(tr("Import MIDI"))
        self._channels = list(channels)
        self._occupied = set(occupied)
        self._mode = "merge"
        self._targets: list[QComboBox] = []

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                tr(
                    "{notes} notes in {channels} channels from {source}.",
                    notes=len(imported.notes),
                    channels=len(imported.channels),
                    source=source,
                )
            )
        )
        layout.addWidget(QLabel(tr("The notes already on the roll are replaced, or merged into the channels below.")))

        grid = QGridLayout()
        grid.addWidget(QLabel(tr("File channel")), 0, 0)
        grid.addWidget(QLabel(tr("Lands on")), 0, 1)
        defaults = self._default_mapping(len(imported.channels))
        for row, channel in enumerate(imported.channels, start=1):
            grid.addWidget(QLabel(_title(channel)), row, 0)
            combo = QComboBox()
            for current in self._channels:
                combo.addItem(_title(current), current.channel)
            combo.addItem(tr("New channel"), -1)
            combo.setCurrentIndex(max(0, combo.findData(defaults[row - 1])))
            combo.currentIndexChanged.connect(self._refresh)
            self._targets.append(combo)
            grid.addWidget(combo, row, 1)
        layout.addLayout(grid)

        buttons = QDialogButtonBox()
        self.merge_button = buttons.addButton(tr("Merge"), QDialogButtonBox.ButtonRole.AcceptRole)
        self.replace_button = buttons.addButton(tr("Replace"), QDialogButtonBox.ButtonRole.DestructiveRole)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        self.merge_button.setDefault(True)
        self.merge_button.clicked.connect(self._on_merge)
        self.replace_button.clicked.connect(self._on_replace)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._refresh()

    def _default_mapping(self, count: int) -> list[int]:
        """Where each file channel starts: its own place while that is empty, else an empty one left.

        A channel that already carries notes keeps its place, so the file channel aimed at it is the
        one that moves - the notes arriving are added to a channel of their own, not mixed into the
        roll's.
        """
        spare = [channel.channel for channel in self._channels if channel.channel not in self._occupied]
        mapping = [-1] * count
        for source in range(count):
            if source in spare:
                mapping[source] = source
                spare.remove(source)
        for source, target in enumerate(mapping):
            if target == -1 and spare:
                mapping[source] = spare.pop(0)
        return mapping

    def _refresh(self) -> None:
        fresh = sum(1 for combo in self._targets if combo.currentData() == -1)
        room = len(self._channels) + fresh <= CHANNEL_COUNT
        self.merge_button.setEnabled(room)
        self.merge_button.setToolTip(
            "" if room else tr("Point the file at existing channels to stay within {count}", count=CHANNEL_COUNT)
        )

    def _on_merge(self) -> None:
        self._mode = "merge"
        self.accept()

    def _on_replace(self) -> None:
        self._mode = "replace"
        self.accept()

    def mode(self) -> str:
        return self._mode

    def mapping(self) -> list[int]:
        """One target per file channel: a channel number of the roll, or -1 for a new one."""
        return [combo.currentData() for combo in self._targets]
