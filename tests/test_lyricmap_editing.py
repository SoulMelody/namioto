import pytest

from namioto.karaoke.canonical import rebuild
from namioto.karaoke.operations import NoteRun, SoundRef
from namioto.lyricmap.editing import detach, insert_sound, move_boundary, note_runs, operation_at, resize

A, B, C, D = (SoundRef(0, index) for index in range(4))


def test_literal_groups_keep_counts_even_without_notes():
    assert note_runs("(あい).2う.0") == (NoteRun((A, B), 2), NoteRun((C,), 0))
    assert note_runs("(あい).0う.3") == (NoteRun((A, B), 0), NoteRun((C,), 3))


def test_continuations_are_one_literal_group():
    assert note_runs("あい.+う.+") == (NoteRun((A, B, C), 1),)
    assert note_runs("泣[な]い.+") == (NoteRun((A, B), 1),)


@pytest.mark.parametrize("slots", [0, 1, 2, 3, 4])
def test_boundary_redistributes_counts_without_joining(slots):
    base = note_runs("あ.2い.2う")
    changed = move_boundary(base, A, B, slots)
    assert changed == (NoteRun((A,), slots), NoteRun((B,), 4 - slots), NoteRun((C,), 1))
    assert sum(run.slots for run in changed) == sum(run.slots for run in base)


def test_boundary_between_groups_preserves_members():
    base = note_runs("(あい).2(うえ).3")
    assert move_boundary(base, B, C, 4) == (NoteRun((A, B), 4), NoteRun((C, D), 1))
    assert move_boundary(base, A, B, 1) is None
    assert move_boundary(base, B, C, 6) is None


def test_right_resize_does_not_change_the_next_run():
    base = note_runs("(あい).1う.3")
    changed = resize(base, A, 2)
    assert changed == (NoteRun((A, B), 2), NoteRun((C,), 3))
    assert rebuild("(あい).1う.3", changed) == "(あい).2う.3"


def test_left_resize_only_borrows_from_the_immediate_previous_run():
    base = note_runs("あ.3(いう).2")
    assert resize(base, B, 4, left=True) == (NoteRun((A,), 1), NoteRun((B, C), 4))
    assert resize(base, B, 5, left=True) == (NoteRun((A,), 0), NoteRun((B, C), 5))
    assert resize(base, B, 6, left=True) is None
    assert resize(base, A, 4, left=True) is None


def test_dragging_two_sounds_together_keeps_the_receivers_count():
    base = note_runs("あ.2い.3う")
    assert insert_sound(base, A, operation_at(base, B)) == (NoteRun((A, B), 3), NoteRun((C,), 1))
    assert insert_sound(base, B, operation_at(base, A)) == (NoteRun((A, B), 2), NoteRun((C,), 1))


def test_group_can_receive_a_sound_that_already_owns_notes():
    base = note_runs("(あい).2う.3え")
    changed = insert_sound(base, C, operation_at(base, A))
    assert changed == (NoteRun((A, B, C), 2), NoteRun((D,), 1))
    assert rebuild("(あい).2う.3え", changed) == "(あいう).2え"


def test_only_edge_members_can_leave_a_group():
    base = note_runs("(あいう).2")
    assert detach(base, B) is None
    assert detach(base, A) == (NoteRun((A,), 1), NoteRun((B, C), 2))
    assert detach(base, C) == (NoteRun((A, B), 2), NoteRun((C,), 1))
    assert rebuild("(あいう).2", detach(base, C)) == "(あい).2う"


def test_drop_and_restore_preserve_the_remaining_group_count():
    base = note_runs("(あいう).2")
    dropped = detach(base, C, 0)
    assert dropped == (NoteRun((A, B), 2), NoteRun((C,), 0))
    assert rebuild("(あいう).2", dropped) == "(あい).2う.0"
    assert detach(dropped, C) == (NoteRun((A, B), 2), NoteRun((C,), 1))


def test_whole_group_can_be_resized_to_zero_and_restored():
    base = note_runs("(あい).1う")
    zero = resize(base, A, 0)
    assert rebuild("(あい).1う", zero) == "(あい).0う"
    assert resize(zero, B, 2) == (NoteRun((A, B), 2), NoteRun((C,), 1))


def test_moves_cannot_reorder_sounds_or_cross_lines():
    base = note_runs("あいう")
    assert insert_sound(base, A, operation_at(base, C)) is None
    base = note_runs("あ\nい")
    other = SoundRef(1, 0)
    assert insert_sound(base, A, operation_at(base, other)) is None
    assert resize(base, other, 2, left=True) is None


def test_groups_cannot_lose_their_middle_member_to_another_group():
    base = note_runs("(あいう).1え")
    assert insert_sound(base, B, operation_at(base, D)) is None


@pytest.mark.parametrize("slots", [0, 1, 2, 3])
def test_group_counts_can_change_across_ruby_containers(slots):
    text = "泣[な]い.+"
    changed = resize(note_runs(text), A, slots)
    rebuilt = rebuild(text, changed)
    assert note_runs(rebuilt) == changed


def test_rewriting_selected_lines_keeps_other_literal_structures():
    text = "あい\n@1{(うえ).2}"
    base = note_runs(text)
    changed = insert_sound(base, B, operation_at(base, A))
    assert rebuild(text, changed, line_indices=(0,)) == "(あい).1\n@1{(うえ).2}"
