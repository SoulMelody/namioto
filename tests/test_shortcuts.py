# SPDX-License-Identifier: AGPL-3.0-only
"""The shortcut table on its own: the shape the help dialog and the i18n catalog both rely on."""

from __future__ import annotations

from namioto import shortcuts


def test_every_group_says_what_it_is_and_holds_something() -> None:
    assert shortcuts.GROUPS
    for group in shortcuts.GROUPS:
        assert isinstance(group.title, str) and group.title.strip()
        assert group.shortcuts, group.title


def test_every_row_is_a_key_and_an_action() -> None:
    for group in shortcuts.GROUPS:
        for shortcut in group.shortcuts:
            assert isinstance(shortcut.keys, str) and shortcut.keys.strip()
            assert isinstance(shortcut.action, str) and shortcut.action.strip()


def test_a_key_is_bound_once_in_the_whole_table() -> None:
    # one binding per command: two rows with the same keys would mean either a copy-paste slip or
    # two actions fighting over one gesture. There is no exception to whitelist today.
    keys = [shortcut.keys for group in shortcuts.GROUPS for shortcut in group.shortcuts]
    assert len(keys) == len(set(keys))


def test_a_title_is_used_once_so_the_tabs_stay_apart() -> None:
    titles = [group.title for group in shortcuts.GROUPS]
    assert len(titles) == len(set(titles))
