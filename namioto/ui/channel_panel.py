# SPDX-License-Identifier: AGPL-3.0-only
"""The channel sidebar: one card per MIDI channel, noteDigger style, toggled from the edit bar.

The panel is a view of `PianoRollView.channels`, never a copy; edits go through
`set_channel_field`, `add_channel`, `remove_channel` and `set_channel_number`. A card shows the MIDI
channel number in the channel's own note colour, and the card of `PianoRollView.active_channel` is
outlined in the palette's `Highlight`. Its three switches are `checkable` and mark the exceptional
state - locked, hidden or muted - and hiding a channel locks it too. `Channel.color` is the palette
entry of the channel's number, and `theme.note_shades` derives a note's body and bevels from it.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPalette, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from namioto.channels import GM_PROGRAMS, PROGRAM_LABELS, Channel, free_channel
from namioto.i18n import tr
from namioto.ui import icons
from namioto.ui.controls import CORNER_RADIUS, FIELD_HEIGHT, icon_button
from namioto.ui.roll import PianoRollView

# wide enough that the longest General MIDI name fits the combo whole, scrollbar included
CARD_WIDTH = 300
SWATCH_WIDTH = 5
ACTIVE_BORDER = 2


class _NameLabel(QLabel):
    """The channel name, elided so a long one cannot push the card wider than its column."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setMinimumWidth(1)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        metrics = painter.fontMetrics()
        painter.setPen(self.palette().color(QPalette.ColorRole.WindowText))
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignVCenter,
            metrics.elidedText(self.text(), Qt.TextElideMode.ElideRight, self.width()),
        )


class _Card(QWidget):
    """One channel: colour swatch, name and instrument, the lock/eye/mute buttons.

    The active channel — where new notes land, and the only one the roll's frame selects — wears the
    palette's selection colour as an outline, so which one it is can be read off the sidebar.
    """

    def __init__(self, panel: ChannelPanel, channel: Channel):
        super().__init__()
        self._panel = panel
        self._number = channel.channel

        self.swatch = QFrame()
        self.swatch.setFixedSize(SWATCH_WIDTH, 34)

        self.name = _NameLabel(channel.name)
        self.id = QLabel(f"({channel.channel + 1})")
        self.id.setMinimumWidth(16)
        self.id.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.program = QComboBox()
        self.program.addItems(GM_PROGRAMS)  # the index is the program number
        # the card is a fixed column: the combo must be allowed to shrink below its longest item
        self.program.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.program.setMinimumContentsLength(12)
        self.program.setFixedHeight(FIELD_HEIGHT)
        self.program.currentIndexChanged.connect(self._on_program)

        self.lock_button = icon_button("unlock", tr("Lock: notes on this channel cannot be edited"), checkable=True)
        self.eye_button = icon_button("eye", tr("Show or hide the notes of this channel"), checkable=True)
        self.mute_button = icon_button("sound", tr("Mute this channel during playback"), checkable=True)
        for button, field in ((self.lock_button, "lock"), (self.eye_button, "visible"), (self.mute_button, "mute")):
            button.clicked.connect(lambda _c, field=field: self._toggle(field))

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(2)
        head.addWidget(self.id)
        head.addWidget(self.name, 1)
        head.addWidget(self.lock_button)
        head.addWidget(self.eye_button)
        head.addWidget(self.mute_button)

        text = QVBoxLayout()
        text.setContentsMargins(0, 2, 0, 2)
        text.setSpacing(2)
        text.addLayout(head)
        text.addWidget(self.program)

        row = QHBoxLayout()
        row.setContentsMargins(6, 4, 4, 4)
        row.setSpacing(6)
        row.addWidget(self.swatch)
        row.addLayout(text, 1)
        self.setLayout(row)

        self.apply(channel)

    def apply(self, channel: Channel) -> None:
        """Take the channel over in place, so a field change never rebuilds the whole sidebar."""
        self._channel = channel
        self.set_active(channel.channel == self._panel.view.active_channel)
        self.swatch.setStyleSheet(f"background: {channel.color}; border-radius: 2px;")
        self.id.setText(f"({channel.channel + 1})")
        self.id.setStyleSheet(f"color: {self._panel.view.channel_color(channel.channel).name()};")
        self.name.setText(channel.name or tr("Channel"))
        self.program.blockSignals(True)
        self.program.setCurrentIndex(channel.program)
        self.program.setToolTip(PROGRAM_LABELS[channel.program])
        self.program.blockSignals(False)
        self.lock_button.setIcon(icons.icon("lock" if channel.lock else "unlock"))
        self.lock_button.setChecked(channel.lock)
        # the fill marks the exceptional state, so the eye is checked while the channel is hidden
        self.eye_button.setIcon(icons.icon("eyeoff" if not channel.visible else "eye"))
        self.eye_button.setChecked(not channel.visible)
        self.mute_button.setIcon(icons.icon("mute" if channel.mute else "sound"))
        self.mute_button.setChecked(channel.mute)

    def set_active(self, active: bool) -> None:
        self.setProperty("active", active)
        self.update()

    def paintEvent(self, _event) -> None:
        if not self.property("active"):
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(QPen(self.palette().color(QPalette.ColorRole.Highlight), ACTIVE_BORDER))
        inset = ACTIVE_BORDER // 2
        painter.drawRoundedRect(
            self.rect().adjusted(inset, inset, -inset - 1, -inset - 1), CORNER_RADIUS, CORNER_RADIUS
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._panel.view.set_active_channel(self._number)
        super().mousePressEvent(event)

    def contextMenuEvent(self, event) -> None:
        panel, channel = self._panel, self._channel
        menu = QMenu(self)
        rename = menu.addAction(tr("Rename…"))
        volume = menu.addAction(tr("Volume…"))
        number = menu.addAction(tr("Channel ID…"))
        remove = menu.addAction(tr("Delete channel"))
        chosen = menu.exec(event.globalPos())
        view = panel.view
        if chosen is rename:
            name, ok = QInputDialog.getText(self, tr("Rename channel"), tr("Name:"), text=channel.name)
            if ok and name.strip():
                view.set_channel_field(self._number, name=name.strip())
        elif chosen is volume:
            value, ok = QInputDialog.getInt(self, tr("Channel volume"), tr("Volume (0-127):"), channel.volume, 0, 127)
            if ok:
                view.set_channel_field(self._number, volume=value)
        elif chosen is number:
            value, ok = QInputDialog.getInt(self, tr("Channel ID"), tr("MIDI channel (1-16):"), self._number + 1, 1, 16)
            if ok and not view.set_channel_number(self._number, value - 1):
                QMessageBox.warning(self, tr("Channel ID"), tr("Channel {number} is already in use", number=value))
        elif chosen is remove:
            view.remove_channel(self._number)

    def _toggle(self, field: str) -> None:
        channel = self._channel
        if field == "visible":
            visible = not channel.visible
            # hiding also locks, the way noteDigger couples the eye and the padlock
            self._panel.view.set_channel_field(self._number, visible=visible, lock=not visible)
        else:
            self._panel.view.set_channel_field(self._number, **{field: not getattr(channel, field)})

    def _on_program(self, program: int) -> None:
        self.program.setToolTip(PROGRAM_LABELS[program])
        self._panel.view.set_channel_field(self._number, program=program)


class ChannelPanel(QWidget):
    """The list of the document's channels; a view over `PianoRollView.channels`, never a copy."""

    def __init__(self, view: PianoRollView, parent=None):
        super().__init__(parent)
        self.view = view
        self.setFixedWidth(CARD_WIDTH)
        view.channels_changed.connect(self._sync)
        view.active_channel_changed.connect(self._on_active_changed)

        self.cards = QVBoxLayout()
        self.cards.setContentsMargins(6, 6, 6, 6)
        self.cards.setSpacing(4)
        self._cards: dict[int, _Card] = {}

        body = QWidget()
        body.setLayout(self.cards)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)  # the card is the column width
        scroll.setWidget(body)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)
        self._sync()

    def _sync(self) -> None:
        """Bring the cards in step with the channels, rebuilding only when the set of them changes."""
        channels = {channel.channel: channel for channel in self.view.channels}
        if set(channels) != set(self._cards):
            self._rebuild()
            return
        for number, card in self._cards.items():
            card.apply(channels[number])

    def _on_active_changed(self, _number: int) -> None:
        for number, card in self._cards.items():
            card.set_active(number == self.view.active_channel)

    def _rebuild(self) -> None:
        while self.cards.count():
            item = self.cards.takeAt(0)
            if widget := item.widget():
                widget.deleteLater()
        self._cards.clear()
        for channel in self.view.channels:
            card = _Card(self, channel)
            self._cards[channel.channel] = card
            self.cards.addWidget(card)
        self.cards.addStretch(1)

    def contextMenuEvent(self, event) -> None:
        menu = QMenu(self)
        add = menu.addAction(tr("Add channel"))
        add.setEnabled(free_channel(self.view.channels) is not None)
        if menu.exec(event.globalPos()) is add:
            self.view.add_channel()
