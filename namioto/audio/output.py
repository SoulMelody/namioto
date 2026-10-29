# SPDX-License-Identifier: AGPL-3.0-only
"""The realtime output: a sounddevice stream whose callback only copies from a ring a background
thread keeps full by pulling from the `Engine`.

That split is the whole point of this module. The device callback runs on a thread of PortAudio's,
where a Python allocation or a lock held across the DSP would glitch, so it does nothing but copy
already-rendered frames out of a ring and count them; everything that allocates or thinks happens on
the producer thread, which has the whole ring's worth of slack to be slow in. It is also what lets
the phase vocoder afford to exist in Python at all.

`open(null_backend=True)` keeps the suite device-free: no stream is opened and a plain thread paces
the producer at realtime, so the transport still answers the way it does with a device behind it.
Nothing here imports sounddevice until a real device is asked for, so a machine without PortAudio
can still run the tests, the analysis and the CLI.
"""

from __future__ import annotations

import threading
import time
from math import gcd

import numpy as np

from namioto.audio.engine import Engine

POLL_SECONDS = 0.02  # how long the producer sleeps when it has nothing to do
MAX_EXTRAPOLATION = 0.05  # seconds of wall time the playhead may be carried over between callbacks
FINISHED_SLACK = 1e-3


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Move a buffer between two rates, which is all a device that refuses 48 kHz ever needs."""
    from scipy.signal import resample_poly

    if not len(samples) or source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)
    divisor = gcd(int(target_rate), int(source_rate))
    return resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)


class FrameRing:
    """Single-producer/single-consumer ring of mono float frames, guarded by its own short lock.

    The lock is only ever held across a copy, never across the engine: the producer renders outside
    it and the device callback reads inside it. `flush` lets the producer discard everything unplayed
    so a seek does not replay the buffer behind it, and the consumer notices the drop and skips to
    it, which is the one value either side writes for the other.
    """

    def __init__(self, capacity: int):
        self._buffer = np.zeros(capacity, np.float32)
        self._capacity = capacity
        self._mask = capacity - 1
        self._lock = threading.Lock()
        self._written = 0
        self._read = 0
        self._dropped = 0

    def _space(self) -> int:
        return self._capacity - (self._written - self._read)

    @property
    def space(self) -> int:
        with self._lock:
            return self._space()

    def write(self, frames: np.ndarray) -> int:
        with self._lock:
            count = min(len(frames), self._space())
            start = self._written & self._mask
            first = min(count, self._capacity - start)
            self._buffer[start : start + first] = frames[:first]
            if count > first:
                self._buffer[: count - first] = frames[first:count]
            self._written += count
            return count

    def flush(self) -> None:
        with self._lock:
            self._dropped = self._written

    def read_into(self, out: np.ndarray, count: int) -> int:
        """Consumer: copy up to `count` frames into `out`, returning how many were read."""
        with self._lock:
            read = max(self._read, self._dropped)
            count = min(count, self._written - read)
            if count:
                start = read & self._mask
                first = min(count, self._capacity - start)
                out[:first] = self._buffer[start : start + first]
                if count > first:
                    out[first:count] = self._buffer[: count - first]
                self._read = read + count
            return count


def _capacity_for(block_frames: int) -> int:
    capacity = 1
    while capacity < block_frames * 4:
        capacity <<= 1
    return capacity


class Output:
    """The loaded song, a device, and the transport surface the window drives."""

    def __init__(self, sample_rate: int, block_frames: int = 1024):
        self.block_frames = max(1, int(block_frames))
        self._sample_rate = int(sample_rate)
        self._stream_rate = self._sample_rate  # what the device took, once one was asked for
        self._engine = Engine(self._sample_rate)
        self._ring = FrameRing(_capacity_for(self.block_frames))
        self._lock = threading.RLock()  # the engine and the anchors, never the ring copy
        self._wake = threading.Condition(self._lock)
        self._producer: threading.Thread | None = None
        self._pacer: threading.Thread | None = None
        self._stream = None
        self._scratch = np.zeros(self.block_frames, np.float32)

        self._is_open = False
        self._playing = False
        self._finished = False
        self._played_frames = 0
        self._last_callback_ns = 0
        self._anchor_position = 0.0
        self._anchor_played = 0
        self._error = ""
        self.source = np.zeros(0, np.float32)  # what `load` was handed, kept for a late rate change

    # -- the surface song.py drives -------------------------------------------------

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def stream_rate(self) -> int:
        """The rate the device took: what `load` moves a song to once one is open."""
        return self._stream_rate

    def error(self) -> str:
        return self._error

    def open(self, null_backend: bool = False) -> bool:
        """Start the device and the producer. `null_backend` paces a plain thread instead."""
        with self._lock:
            if self._is_open:
                return True
            if null_backend:
                self._stream = None
            else:
                chosen = self._open_stream(self._sample_rate)
                if chosen is None:
                    return False
                self._stream_rate = chosen
            if self._stream_rate != self._engine.sample_rate:
                self._rebuild_engine(self._stream_rate)
            self._is_open = True
            self._wake.notify_all()

        self._producer = threading.Thread(target=self._producer_loop, name="namioto-audio", daemon=True)
        self._producer.start()
        if null_backend:
            self._pacer = threading.Thread(target=self._pace, name="namioto-null", daemon=True)
            self._pacer.start()
        return True

    def close(self) -> None:
        with self._lock:
            if not self._is_open:
                return
            self._is_open = False
            self._playing = False
            self._wake.notify_all()
        for thread in (self._producer, self._pacer):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)
        self._producer = self._pacer = None
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # a device that went away is not worth failing the window over
                pass
            self._stream = None

    def load(self, samples: np.ndarray) -> None:
        with self._lock:
            self.source = np.ascontiguousarray(samples, dtype=np.float32)
            if self._stream_rate != self._sample_rate:
                self._rebuild_engine(self._stream_rate)
            else:
                self._engine.load(self.source)
            self._anchor_position = 0.0
            self._anchor_played = self._played_frames
            self._playing = False
            self._finished = False
            self._ring.flush()

    def unload(self) -> None:
        with self._lock:
            self.source = np.zeros(0, np.float32)
            self._engine.unload()
            self._anchor_position = 0.0
            self._anchor_played = self._played_frames
            self._playing = False
            self._finished = False
            self._ring.flush()

    @property
    def loaded(self) -> bool:
        return self._engine.loaded

    @property
    def duration(self) -> float:
        return self._engine.duration

    @property
    def position(self) -> float:
        with self._lock:
            return self._position()

    @property
    def playing(self) -> bool:
        return self._playing and not self.finished

    @property
    def finished(self) -> bool:
        # the source running out is not the song ending: the frames already in the ring still have to
        # be heard, so this is judged on the played position rather than on the producer's flag alone
        with self._lock:
            return self._finished and self._position() >= self._engine.duration - FINISHED_SLACK

    @property
    def speed(self) -> float:
        return self._engine.speed

    @speed.setter
    def speed(self, value: float) -> None:
        with self._lock:
            if value == self._engine.speed:
                return
            # the stretcher takes the new rate from the next frame on, so nothing is reset: re-seeking
            # would replay its pre-roll and dropping the buffered frames would skip source already read
            self._anchor_position = self._position()
            self._anchor_played = self._played_frames
            self._engine.speed = value
            self._wake.notify_all()

    @property
    def gain(self) -> float:
        return self._engine.gain

    @gain.setter
    def gain(self, value: float) -> None:
        self._engine.gain = value

    def play(self, seconds: float = 0.0) -> None:
        with self._lock:
            self._engine.play(seconds)
            self._anchor_position = self._clamp(seconds)
            self._anchor_played = self._played_frames
            self._ring.flush()
            self._playing = True
            self._finished = False
            self._wake.notify_all()

    def pause(self) -> None:
        with self._lock:
            self._reanchor()
            self._engine.pause()
            self._playing = False
            self._ring.flush()

    def seek(self, seconds: float) -> None:
        with self._lock:
            self._engine.seek(seconds)
            self._anchor_position = self._clamp(seconds)
            self._anchor_played = self._played_frames
            self._ring.flush()
            self._finished = False
            self._wake.notify_all()

    # -- the callback body ----------------------------------------------------------

    def render(self, frames: int) -> np.ndarray:
        """What the ring holds, padded with silence, and counted as played: the whole callback."""
        out = self._scratch_for(frames)
        done = self._ring.read_into(out, frames)
        if done < frames:
            out[done:frames] = 0.0
        self._last_callback_ns = time.monotonic_ns()
        if self._playing:
            self._played_frames += frames
        return out[:frames]

    def _callback(self, outdata, frames, _time_info, _status) -> None:
        outdata[:, 0] = self.render(frames)

    def _scratch_for(self, frames: int) -> np.ndarray:
        if frames > len(self._scratch):
            self._scratch = np.zeros(frames, np.float32)
        return self._scratch

    # -- threads --------------------------------------------------------------------

    def _producer_loop(self) -> None:
        while True:
            with self._wake:
                self._wake.wait_for(
                    lambda: not self._is_open or (self._playing and not self._finished and self._ring.space > 0),
                    POLL_SECONDS,
                )
                if not self._is_open:
                    return
                if not self._playing or self._finished:
                    continue
                space = self._ring.space
                if space == 0:
                    continue
                rendered = self._engine.pull(min(self.block_frames, space))
                if self._engine.finished:
                    self._finished = True
            self._ring.write(rendered)

    def _pace(self) -> None:
        """The null backend's clock: consume at realtime, so the transport ages as a device's does."""
        period = self.block_frames / self._sample_rate
        deadline = time.monotonic()
        while True:
            with self._wake:
                if not self._is_open:
                    return
            self.render(self.block_frames)
            deadline += period
            slept = deadline - time.monotonic()
            if slept < -period:  # a stalled run resyncs rather than catching up in a burst
                deadline = time.monotonic()
            else:
                time.sleep(max(0.0, slept))

    # -- internals ------------------------------------------------------------------

    def _open_stream(self, rate: int) -> int | None:
        """A mono float32 stream at `rate`, or at the device's own rate when it refuses that one."""
        try:
            import sounddevice as sd
        except Exception as exc:  # PortAudio missing: playback stays silent rather than fatal
            self._error = f"no audio device: {exc}"
            return None
        chosen = rate
        if not self._device_takes(sd, rate):
            try:
                chosen = int(sd.query_devices(kind="output")["default_samplerate"])
            except Exception as exc:
                self._error = str(exc)
                return None
        try:
            self._stream = sd.OutputStream(
                samplerate=chosen,
                channels=1,
                dtype="float32",
                blocksize=self.block_frames,
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:
            self._error = str(exc)
            self._stream = None
            return None
        return chosen

    @staticmethod
    def _device_takes(sd, rate: int) -> bool:
        try:
            sd.check_output_settings(channels=1, dtype="float32", samplerate=rate)
        except Exception:
            return False
        return True

    def _rebuild_engine(self, rate: int) -> None:
        """Move the loaded song to the device's rate by rebuilding the engine around it."""
        gain, speed = self._engine.gain, self._engine.speed
        self._engine = Engine(rate)
        self._engine.gain = gain
        self._engine.speed = speed
        self._engine.load(_resample(self.source, self._sample_rate, rate))

    def _clamp(self, seconds: float) -> float:
        if self._engine.loaded:
            return min(max(0.0, float(seconds)), self._engine.duration)
        return max(0.0, float(seconds))

    def _position(self) -> float:
        rate = self._engine.sample_rate
        frames = self._played_frames - self._anchor_played
        if self._playing:
            # the device clock is stepped, so wall time since its last tick fills the gap; it is capped
            # in case the callbacks stall, and the next tick re-anchors it
            now = time.monotonic_ns()
            last = self._last_callback_ns
            if last and now > last:
                frames += min((now - last) * 1e-9 * rate, rate * MAX_EXTRAPOLATION)
        position = self._anchor_position + frames / rate * self._engine.speed
        if self._engine.loaded:
            return min(max(0.0, position), self._engine.duration)
        return max(0.0, position)

    def _reanchor(self) -> None:
        self._anchor_position = self._position()
        self._anchor_played = self._played_frames
