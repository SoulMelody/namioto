# SPDX-License-Identifier: AGPL-3.0-only
"""The marker list: named points a long score is navigated by."""

from __future__ import annotations

from namioto.markers import Marker, MarkerList


def test_a_marker_can_be_found_by_where_it_sits() -> None:
    markers = MarkerList()
    markers.add(4.0, "chorus")
    assert len(markers) == 1
    assert markers.markers == (Marker(4.0, "chorus"),)
    assert markers.nearest(5.0) == Marker(4.0, "chorus")
    assert markers.nearest(2.0) == Marker(4.0, "chorus")
    # past the last marker the last one is still the nearest
    assert markers.nearest(99.0) == Marker(4.0, "chorus")
    assert MarkerList().nearest(1.0) is None


def test_one_beat_holds_one_marker_and_the_last_name_wins() -> None:
    markers = MarkerList()
    markers.add(2.0, "first")
    markers.add(2.0, "second")
    assert len(markers) == 1
    assert [marker.name for marker in markers] == ["second"]


def test_markers_iterate_in_beat_order_however_they_were_added() -> None:
    markers = MarkerList()
    for beat, name in ((9.0, "outro"), (0.0, "intro"), (4.0, "verse")):
        markers.add(beat, name)
    assert [marker.beat for marker in markers] == [0.0, 4.0, 9.0]
    assert [marker.name for marker in markers] == ["intro", "verse", "outro"]


def test_a_list_round_trips_and_junk_is_skipped() -> None:
    markers = MarkerList.from_list([{"beat": 4.0, "name": "verse"}, {"beat": 0.0, "name": "intro"}])
    assert MarkerList.from_list(markers.to_list()).markers == markers.markers
    assert markers.to_list() == [{"beat": 0.0, "name": "intro"}, {"beat": 4.0, "name": "verse"}]
    assert MarkerList.from_list([{"name": "nowhere"}, "junk", {"beat": "high"}]).markers == ()
    assert MarkerList.from_list(None).markers == ()


def test_a_shifted_marker_keeps_the_list_sorted() -> None:
    markers = MarkerList.from_list([{"beat": 1.0, "name": "a"}, {"beat": 5.0, "name": "b"}])
    markers.shift(2.0, 4.0)
    assert [(marker.beat, marker.name) for marker in markers] == [(1.0, "a"), (9.0, "b")]
    # shifting one marker onto a beat that already holds another keeps one, not two
    markers.shift(9.0, -8.0)
    assert [(marker.beat, marker.name) for marker in markers] == [(1.0, "b")]
