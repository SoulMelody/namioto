# SPDX-License-Identifier: AGPL-3.0-only

import json

import numpy as np
import pytest

from namioto import utils
from namioto.analysis import align, model_store


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    return tmp_path


class FakeBackend:
    """A CTC model that answers every window with the same hand-built posteriors."""

    def __init__(self, logits):
        self._logits = np.asarray(logits, dtype=np.float32)

    def logits(self, waveform):
        return self._logits


class WindowBackend:
    """A CTC model that answers a window with one frame per 20 ms, and remembers the windows."""

    def __init__(self, vocabulary: int = 3):
        self.windows: list[int] = []
        self.vocabulary = vocabulary

    def logits(self, waveform):
        self.windows.append(waveform.size)
        return np.zeros((max(0, waveform.size // align.FRAME_SAMPLES - 1), self.vocabulary), dtype=np.float32)


AUDIO = np.zeros(align.SAMPLE_RATE * 2, dtype=np.float32)

# blank, "a", "b": a over frames 0-1, a rest, b over frames 3-4, a rest
LOGITS = [
    [0.0, 20.0, 0.0],
    [0.0, 20.0, 0.0],
    [20.0, 0.0, 0.0],
    [0.0, 0.0, 20.0],
    [0.0, 0.0, 20.0],
    [20.0, 0.0, 0.0],
]
DICTIONARY = {"[pad]": 0, "a": 1, "b": 2}


def test_the_alignment_parameters_are_remembered_and_checked():
    assert align.load_parameters() == align.default_parameters()

    align.save_parameters({"model": "yohane", "device": "gpu", "quantize": 4, "chunk": "off"})
    assert align.load_parameters() == {
        "model": "yohane",
        "device": "gpu",
        "quantize": 4,
        "chunk": "off",
        "use_constraints": True,
    }
    align.save_parameters({"use_constraints": False})
    assert align.load_parameters()["use_constraints"] is False

    align.save_parameters({"model": "nope", "device": 7, "quantize": "eight"})
    assert align.load_parameters() == align.default_parameters()  # every value goes through its check

    align.parameter_path().write_text("{not json", encoding="utf-8")
    with pytest.warns(UserWarning):
        assert align.load_parameters() == align.default_parameters()


def test_the_alignment_cache_holds_the_raw_lines_and_reuses_them_by_their_inputs():
    rows = [[(0.0, 0.5), (0.5, 1.0), (None, None)]]
    problems = ["あい: empty"]
    flagged = [True]
    align.save_alignment("/tmp/song.wav", "mms", "cpu", "あい\n", rows, problems, flagged)

    assert align.find_alignment("/tmp/song.wav", "mms", "cpu", "あい\n") == (rows, problems, flagged)
    assert align.find_alignment("/tmp/song.wav", "yohane", "cpu", "あい\n") is None
    assert align.find_alignment("/tmp/song.wav", "mms", "cpu", "うえ\n") is None


def test_the_alignment_cache_tells_the_cutting_modes_apart():
    rows = [[(0.0, 0.5)]]
    align.save_alignment("/tmp/song.wav", "mms", "cpu", "あい\n", rows, [], [], chunk="off")

    assert align.find_alignment("/tmp/song.wav", "mms", "cpu", "あい\n", chunk="off") == (rows, [], [])
    assert align.find_alignment("/tmp/song.wav", "mms", "cpu", "あい\n", chunk="silence") is None


def test_a_broken_alignment_cache_is_no_alignment():
    target = align._store_path("/tmp/song.wav")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{not json", encoding="utf-8")

    assert align.load_alignments("/tmp/song.wav") == {}
    assert align.find_alignment("/tmp/song.wav", "mms", "cpu", "あい\n") is None


def test_a_window_under_a_chunk_is_one_pass():
    backend = WindowBackend()
    chunked = align.ChunkedBackend(backend, chunk_seconds=1.0, overlap_seconds=0.2)
    audio = np.zeros(align.SAMPLE_RATE // 2, dtype=np.float32)

    got = chunked.logits(audio)

    assert backend.windows == [audio.size]
    assert got.shape[0] == audio.size // align.FRAME_SAMPLES - 1


def test_chunking_a_long_window_keeps_the_one_pass_frame_grid():
    backend = WindowBackend()
    chunked = align.ChunkedBackend(backend, chunk_seconds=1.0, overlap_seconds=0.2)
    audio = np.zeros(align.SAMPLE_RATE * 3, dtype=np.float32)

    got = chunked.logits(audio)

    assert got.shape[0] == audio.size // align.FRAME_SAMPLES - 1  # the frames one long pass would have
    assert len(backend.windows) > 1
    assert max(backend.windows) <= chunked.chunk


def test_every_chunk_of_a_long_window_is_reported():
    backend = WindowBackend()
    seen: list[tuple[int, int]] = []
    chunked = align.ChunkedBackend(
        backend, chunk_seconds=1.0, overlap_seconds=0.2, progress=lambda done, total: seen.append((done, total))
    )

    chunked.logits(np.zeros(align.SAMPLE_RATE * 3, dtype=np.float32))

    assert [done for done, _total in seen] == list(range(1, len(seen) + 1))
    assert all(total == len(backend.windows) for _done, total in seen)


def _tone_with_rest(seconds: float, rest: tuple[float, float]) -> np.ndarray:
    audio = np.full(int(align.SAMPLE_RATE * seconds), 0.5, dtype=np.float32)
    audio[int(rest[0] * align.SAMPLE_RATE) : int(rest[1] * align.SAMPLE_RATE)] = 0.0
    return audio


def test_silence_chunking_pulls_a_seam_into_the_rest():
    chunked = align.ChunkedBackend(WindowBackend(), chunk_seconds=2.0, overlap_seconds=0.2, silence=True)

    regions = chunked._regions(_tone_with_rest(5.0, (1.0, 1.5)))

    assert align.SAMPLE_RATE <= regions[0][1] <= int(1.5 * align.SAMPLE_RATE)


def test_silence_chunking_keeps_the_one_pass_frame_grid():
    backend = WindowBackend()
    chunked = align.ChunkedBackend(backend, chunk_seconds=2.0, overlap_seconds=0.2, silence=True)
    audio = _tone_with_rest(5.0, (1.0, 1.5))

    got = chunked.logits(audio)

    assert got.shape[0] == audio.size // align.FRAME_SAMPLES - 1
    assert len(backend.windows) > 1
    assert max(backend.windows) <= chunked.chunk


def test_silence_chunking_without_a_rest_cuts_like_the_fixed_one():
    audio = np.full(align.SAMPLE_RATE * 5, 0.5, dtype=np.float32)
    fixed = align.ChunkedBackend(WindowBackend(), chunk_seconds=2.0, overlap_seconds=0.2)
    silence = align.ChunkedBackend(WindowBackend(), chunk_seconds=2.0, overlap_seconds=0.2, silence=True)

    assert silence._regions(audio) == fixed._regions(audio)


def test_forced_align_visits_every_target_in_order():
    emission = align.log_softmax(np.asarray(LOGITS, dtype=np.float32))
    path, _scores = align.forced_align(emission, [1, 2])
    assert list(path) == [1, 1, 0, 2, 2, 0]


def test_forced_align_gives_up_when_the_window_cannot_hold_the_targets():
    emission = align.log_softmax(np.zeros((2, 3), dtype=np.float32))
    assert align.forced_align(emission, [1, 1, 1]) is None


def test_merge_tokens_collapses_runs_and_drops_the_blank():
    path = np.asarray([1, 1, 0, 2, 2, 0])
    scores = np.asarray([0.5, 1.0, 0.0, 1.0, 1.0, 0.0])
    spans = align.merge_tokens(path, scores)
    assert [(token, start, end) for token, start, end, _score in spans] == [(1, 0, 2), (2, 3, 5)]
    assert spans[0][3] == pytest.approx(0.75)


def test_align_gives_each_token_the_span_of_its_frames():
    aligned = align.align([align.Segment(0.0, 1.0, ("a", "b"))], FakeBackend(LOGITS), DICTIONARY, AUDIO)
    first, second = aligned[0].tokens
    assert first.text == "a" and second.text == "b"
    assert first.start == pytest.approx(0.0, abs=1e-6)
    assert first.end == pytest.approx(0.04, abs=1e-6)
    assert second.start == pytest.approx(0.06, abs=1e-6)
    assert second.end == pytest.approx(0.10, abs=1e-6)
    assert first.start < second.start


def test_align_groups_a_multi_character_token_into_one_span():
    aligned = align.align([align.Segment(0.0, 1.0, ("ab",))], FakeBackend(LOGITS), DICTIONARY, AUDIO)
    (only,) = aligned[0].tokens
    assert only.text == "ab"
    assert only.start == pytest.approx(0.0, abs=1e-6)
    assert only.end == pytest.approx(0.10, abs=1e-6)


def test_align_keeps_the_original_case_of_a_token():
    aligned = align.align([align.Segment(0.0, 1.0, ("A",))], FakeBackend(LOGITS), DICTIONARY, AUDIO)
    assert aligned[0].tokens[0].text == "A"
    assert aligned[0].tokens[0].start is not None


def test_align_times_a_character_outside_the_dictionary():
    aligned = align.align([align.Segment(0.0, 1.0, ("ab",))], FakeBackend(LOGITS), {"[pad]": 0, "a": 1}, AUDIO)
    assert aligned[0].tokens[0].start is not None


def test_align_returns_untimed_tokens_when_the_window_is_empty():
    aligned = align.align([align.Segment(1.0, 1.0, ("a", "b"))], FakeBackend(LOGITS), DICTIONARY, AUDIO)
    assert all(token.start is None for token in aligned[0].tokens)


def test_align_keeps_an_empty_token_untimed():
    aligned = align.align([align.Segment(0.0, 1.0, ("a", "", "b"))], FakeBackend(LOGITS), DICTIONARY, AUDIO)
    assert aligned[0].tokens[1].start is None
    assert aligned[0].tokens[0].start is not None and aligned[0].tokens[2].start is not None


def test_voice_segments_finds_the_stretch_that_is_sounding():
    audio = np.zeros(align.SAMPLE_RATE * 2, dtype=np.float32)
    audio[align.SAMPLE_RATE // 2 : align.SAMPLE_RATE * 3 // 2] = 0.5
    (only,) = align.voice_segments(audio)
    assert only[0] == pytest.approx(0.5, abs=0.05)
    assert only[1] == pytest.approx(1.5, abs=0.05)


def test_voice_segments_finds_nothing_in_silence():
    assert align.voice_segments(np.zeros(align.SAMPLE_RATE, dtype=np.float32)) == []


def test_correct_times_fills_a_tail_up_to_the_next_onset(monkeypatch):
    monkeypatch.setattr(align, "voice_segments", lambda audio, frame_seconds=align.TAIL_FRAME_SECONDS: [(0.5, 1.5)])
    rows = [[(0.6, 0.8)], [(1.1, 1.4)]]
    got = align.correct_times(rows, np.zeros(1))
    assert got[0] == [(0.6, pytest.approx(1.08))]  # the voice runs on, so it stops a frame before
    assert got[1] == [(1.1, 1.5)]  # nothing follows, so it reaches the stretch's end


def test_correct_times_moves_a_head_that_straddles_a_rest_to_the_stretch_of_its_end(monkeypatch):
    monkeypatch.setattr(
        align, "voice_segments", lambda audio, frame_seconds=align.TAIL_FRAME_SECONDS: [(0.4, 0.9), (1.2, 2.0)]
    )
    rows = [[(0.5, 0.8), (1.3, 1.6)]]
    got = align.correct_times(rows, np.zeros(1))
    assert got[0][0][0] == pytest.approx(0.8)  # no one stretch holds the line, so the head takes its own end
    assert got[0][1][1] == pytest.approx(2.0)  # the last sound reaches the stretch's end


def test_correct_times_leaves_a_line_inside_one_stretch_alone(monkeypatch):
    monkeypatch.setattr(align, "voice_segments", lambda audio, frame_seconds=align.TAIL_FRAME_SECONDS: [(0.0, 1.1)])
    rows = [[(0.5, 0.8), (0.8, 1.1)]]
    assert align.correct_times(rows, np.zeros(1)) == [[(0.5, 0.8), (0.8, 1.1)]]


def test_correct_times_leaves_an_unaligned_line_alone(monkeypatch):
    monkeypatch.setattr(align, "voice_segments", lambda audio, frame_seconds=align.TAIL_FRAME_SECONDS: [(0.0, 1.0)])
    rows = [[(None, None)], [(0.5, 0.8)]]
    assert align.correct_times(rows, np.zeros(1))[0] == [(None, None)]


def test_load_dictionary_lower_cases_and_finds_the_blank(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps({"[PAD]": 0, "A": 1, "あ": 2}), encoding="utf-8")
    dictionary, blank_id = align.load_dictionary(path)
    assert dictionary == {"[pad]": 0, "a": 1, "あ": 2}
    assert blank_id == 0


def test_load_dictionary_rejects_a_file_that_is_not_one(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        align.load_dictionary(path)


def test_read_segments_accepts_a_list_or_an_object(tmp_path):
    path = tmp_path / "segments.json"
    path.write_text(json.dumps([{"start": 0, "end": 1, "tokens": ["a"]}]), encoding="utf-8")
    assert align.read_segments(path) == [align.Segment(0.0, 1.0, ("a",))]
    path.write_text(json.dumps({"segments": [{"start": 1, "end": 2, "tokens": ["b"]}]}), encoding="utf-8")
    assert align.read_segments(path) == [align.Segment(1.0, 2.0, ("b",))]


def test_model_dir_separates_the_two_models(tmp_path, monkeypatch):
    monkeypatch.setattr(utils.platformdirs, "user_data_dir", lambda name: str(tmp_path))
    assert align.model_dir("mms", "ja") == tmp_path / "models" / "mms" / "ja"
    assert align.model_dir("yohane", "ja") == tmp_path / "models" / "yohane" / "ja"
    with pytest.raises(ValueError):
        align.model_dir("other", "ja")


def test_resolve_model_falls_back_to_the_environment(tmp_path, monkeypatch):
    (tmp_path / align.MODEL_FILE).write_bytes(b"")
    (tmp_path / align.VOCAB_FILE).write_text("{}", encoding="utf-8")
    monkeypatch.setenv(align.MODEL_ENV, str(tmp_path))
    assert align.resolve_model() == tmp_path


def test_resolve_model_reads_a_file_as_its_directory(tmp_path):
    model = tmp_path / align.MODEL_FILE
    model.write_bytes(b"")
    (tmp_path / align.VOCAB_FILE).write_text("{}", encoding="utf-8")
    assert align.resolve_model(model) == tmp_path


def test_model_file_prefers_the_half_precision_one_on_a_gpu(tmp_path):
    (tmp_path / align.MODEL_FILE).write_bytes(b"")
    assert align.model_file(tmp_path, "webgpu") == tmp_path / align.MODEL_FILE
    assert align.model_file(tmp_path, "cpu") == tmp_path / align.MODEL_FILE
    (tmp_path / align.FP16_FILE).write_bytes(b"")
    assert align.model_file(tmp_path, "webgpu") == tmp_path / align.FP16_FILE
    assert align.model_file(tmp_path, "cpu") == tmp_path / align.MODEL_FILE


def test_a_gpu_run_fetches_the_half_precision_model(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(align.model_store, "install", lambda name, parts, progress=None: fetched.append((name, parts)))
    directory = align.model_dir("mms", "ja")

    align.fetch_fp16(directory, "webgpu", "mms")
    align.fetch_fp16(directory, "cpu", "mms")
    align.fetch_fp16(tmp_path, "webgpu", "mms")  # a conversion of one's own, not the store's copy

    assert fetched == [(align.FP16_KEY, ("mms", "ja"))]


def test_an_installed_half_precision_model_is_not_fetched_again(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(align.model_store, "install", lambda name, parts, progress=None: fetched.append((name, parts)))
    directory = align.model_dir("mms", "ja")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / align.FP16_FILE).write_bytes(b"")

    align.fetch_fp16(directory, "webgpu", "mms")

    assert fetched == []


def test_a_fetch_that_fails_leaves_the_shipped_model(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("no network")

    monkeypatch.setattr(align.model_store, "install", refuse)
    directory = align.model_dir("mms", "ja")

    align.fetch_fp16(directory, "webgpu", "mms")

    assert align.model_file(directory, "webgpu") == directory / align.MODEL_FILE


def test_resolve_model_rejects_a_directory_without_the_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        align.resolve_model(tmp_path)


def test_resolve_model_downloads_what_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.delenv(align.MODEL_ENV, raising=False)
    monkeypatch.setattr(utils.platformdirs, "user_data_dir", lambda name: str(tmp_path))
    asked: list[tuple] = []

    def install(name, parts=(), progress=None, **rest):
        asked.append((name, parts))
        return tmp_path

    monkeypatch.setattr(model_store, "install", install)

    assert align.resolve_model() == tmp_path
    assert asked == [("aligner", ("mms", "ja"))]


def test_the_model_pass_is_cached_apart_from_the_lyrics():
    emission = np.zeros((6, 3), dtype=np.float32)
    align.save_emissions("/tmp/song.wav", "mms", "cpu", "silence", emission)

    assert align.has_emissions("/tmp/song.wav", "mms", "cpu", "silence")
    assert not align.has_emissions("/tmp/song.wav", "yohane", "cpu", "silence")
    assert not align.has_emissions("/tmp/song.wav", "mms", "cuda", "silence")
    assert not align.has_emissions("/tmp/song.wav", "mms", "cpu", "off")
    found = align.load_emissions("/tmp/song.wav", "mms", "cpu", "silence")
    assert found is not None and found.shape == (6, 3)
    assert align.load_emissions("/tmp/song.wav", "mms", "cpu", "off") is None


def test_align_whole_matches_a_live_pass_over_the_same_audio():
    audio = np.zeros(align.SAMPLE_RATE, dtype=np.float32)
    segment = align.Segment(0.0, 1.0, ("ab",))
    emission = align.whole_emissions(FakeBackend(LOGITS), audio)

    cached = align.align_whole(segment, emission, DICTIONARY)
    live = align.align([segment], FakeBackend(LOGITS), DICTIONARY, audio)[0]

    assert [token.start for token in cached.tokens] == [token.start for token in live.tokens]
    assert [token.end for token in cached.tokens] == [token.end for token in live.tokens]
