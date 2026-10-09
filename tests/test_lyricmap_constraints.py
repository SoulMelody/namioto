import pytest

from namioto.karaoke.operations import Drop, Match, Merge, SoundRef
from namioto.karaoke.sounds import natural_sounds
from namioto.lyricmap.constraints import read
from namioto.lyricmap.notes import TimedNote
from namioto.lyricmap.problems import MappingError
from namioto.lyricmap.raw import Raw
from namioto.lyricmap.solver import solve


@pytest.mark.parametrize(
    "text,expected",
    [
        ("あい", {}),
        ("あ.2い", {SoundRef(0, 0): (1, 2)}),
        ("(あい).1う", {SoundRef(0, 0): (2, 1)}),
        ("泣[な]い.+", {SoundRef(0, 0): (2, 1)}),
        ("(あい).0", {SoundRef(0, 0): (1, 0), SoundRef(0, 1): (1, 0)}),
        ("(しょ).2", {SoundRef(0, 0): (1, 2)}),
    ],
)
def test_only_explicit_krc_shapes_become_constraints(text, expected):
    assert read(text) == expected


@pytest.mark.parametrize(
    "text,times,count,expected",
    [
        ("あ.2い", [(0.0, 1.0), (1.0, 1.0)], 3, [Match(SoundRef(0, 0), (1, 2)), Match(SoundRef(0, 1), (3,))]),
        (
            "(あい).1う",
            [(0.0, 1.0), (1.0, 1.0), (2.0, 1.0)],
            2,
            [Merge((SoundRef(0, 0), SoundRef(0, 1)), 1), Match(SoundRef(0, 2), (2,))],
        ),
        ("あ.0い", [(0.0, 1.0), (1.0, 1.0)], 1, [Drop(SoundRef(0, 0)), Match(SoundRef(0, 1), (1,))]),
        ("あい.2", [(0.0, 1.0), (1.0, 1.0)], 3, [Match(SoundRef(0, 0), (1,)), Match(SoundRef(0, 1), (2, 3))]),
    ],
)
def test_solver_preserves_explicit_structure_without_pinning_note_identity(text, times, count, expected):
    notes = [TimedNote(float(index), float(index + 1), 60, index + 1) for index in range(count)]
    result = solve(natural_sounds(text), [[Raw(*time) for time in times]], notes, constraints=read(text))
    assert result == expected


def test_session_edits_can_override_an_existing_krc_group():
    text = "(あい).1"
    notes = [TimedNote(0.0, 1.0, 60, 1), TimedNote(1.0, 2.0, 62, 2)]
    anchor = Match(SoundRef(0, 0), (1,), confirmed=True)
    result = solve(
        natural_sounds(text), [[Raw(0.0, 1.0), Raw(1.0, 1.0)]], notes, anchors=[anchor], constraints=read(text)
    )
    assert result == [anchor, Match(SoundRef(0, 1), (2,))]


def test_incompatible_constraints_do_not_fall_back_to_unconstrained_mapping():
    text = "あ.3"
    notes = [TimedNote(0.0, 1.0, 60, 1)]
    with pytest.raises(MappingError, match="cannot be partitioned"):
        solve(natural_sounds(text), [[Raw(0.0, 1.0)]], notes, constraints=read(text))
    assert solve(natural_sounds(text), [[Raw(0.0, 1.0)]], notes) == [Match(SoundRef(0, 0), (1,))]
