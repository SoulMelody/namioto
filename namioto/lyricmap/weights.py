# SPDX-License-Identifier: AGPL-3.0-only
"""Pattern weights for mapping costs, independent of the aligner and the raw evidence.

A merge whose first or last Sound is a sokuon takes the edge weight; otherwise three or more
members take the size weight, and two members may take the empirical weight. The first rule wins.
Empirical vowel connections are not decided here; an explicit long-vowel mark makes the pair
empirical on its own. Natural Sounds and their readings are never changed. Qt-free.
"""

from __future__ import annotations

from functools import lru_cache

from namioto.karaoke.sounds import SoundLine

MERGE_SIZE_SLOPE = 0.5
EMPIRICAL_MERGE_WEIGHT = 0.8
EDGE_SOKUON_MERGE_WEIGHT = 1.5


def merge_weights(line: SoundLine) -> tuple[tuple[float, ...], ...]:
    """One row per starting Sound, indexed by merge member count minus two."""
    pairs = _empirical_pairs(line)
    return tuple(
        tuple(_weight(line, start, count, pairs[start]) for count in range(2, len(line.sounds) - start + 1))
        for start in range(len(line.sounds))
    )


def _weight(line: SoundLine, start: int, count: int, empirical: bool) -> float:
    if line.sounds[start].reading in ("っ", "ッ") or line.sounds[start + count - 1].reading in ("っ", "ッ"):
        return EDGE_SOKUON_MERGE_WEIGHT
    if count >= 3:
        return 1.0 + MERGE_SIZE_SLOPE * (count - 2)
    if empirical:
        return EMPIRICAL_MERGE_WEIGHT
    return 1.0


@lru_cache(maxsize=256)
def _empirical_pairs(line: SoundLine) -> tuple[bool, ...]:
    """A pair whose second Sound is an explicit long-vowel mark; it needs no word boundary."""
    pairs = [False] * len(line.sounds)
    for index, (first, second) in enumerate(zip(line.sounds[:-1], line.sounds[1:], strict=True)):
        if first.token[-1:] not in ("a", "i", "u", "e", "o"):
            continue
        if second.reading == "ー":
            pairs[index] = True
    return tuple(pairs)
