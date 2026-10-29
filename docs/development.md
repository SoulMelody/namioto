# Development

[README](../README.md) · [CLI guide](cli.md)

## Source setup

Install uv, Python 3.12+ and PortAudio, then run from the repository root:

```bash
uv sync --extra cpu
uv run namioto
```

Choose `--extra cuda` or `--extra webgpu` instead if needed; see [devices](cli.md#devices-and-models).
Audio output goes through `sounddevice`, which needs the system PortAudio its Linux wheel does not
carry - `libportaudio2` on Debian, `portaudio` on Arch. Everything else is a prebuilt wheel, MIDI
ports included, so no compiler, meson or pybind11 is involved at any point.

## Tests and lint

```bash
uv run pytest
uv run pytest -m slow
uv run pytest -m ''
uvx ruff check namioto tests
uvx ruff format namioto tests
uv run scripts/bench_spectrum.py
```

The default test run excludes slow beat-tracker tests. Tests must not require a display,
a synthesiser or an audio device. See `tests/conftest.py` and `tests/test_ui.py`
for the enforced boundaries.

## Packaging

```bash
./build.sh wheel   # dist/*.whl and dist/*.tar.gz
./build.sh cli     # dist/namioto-cli/
./build.sh ctl     # dist/namioto-ctl/
./build.sh app     # dist/namioto/
./build.sh all
./build.sh clean
```

Neither wheels nor frozen builds bundle analysis models. Model publishing uses
`scripts/upload_models.sh`; preserve each model's license and attribution in [NOTICE](../NOTICE).

## Model conversion

The aligner runs converted ONNX models; conversion requires dependencies outside the normal
runtime:

```bash
uv run --group export scripts/export_align_model.py --model mms
uv run --group export scripts/export_align_model.py --model yohane
```

MMS conversion uses torch and torchaudio; yohane uses torch and transformers. Both need ONNX.
The checkpoints are Meta's MMS forced aligner and
`NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn`, respectively.

## Code map

| Location | Responsibility |
| --- | --- |
| `namioto/ui/` | Qt editor, dialogs, spectrum rendering and remote-control server |
| `namioto/analysis/` | Spectrum, tempo, transcription, alignment, model downloads and devices |
| `namioto/karaoke/` | The project's `.krc` lyric text: parsing, natural sounds, canonical rebuild, mapping operations |
| `namioto/lyricmap/` | Lyrics-to-notes mapping and export validation |
| `namioto/document.py`, `project.py`, `midi.py` | Notes, project persistence and MIDI interchange |
| `namioto/settings.py`, `state.py` | Preferences and remembered UI/new-project values |
| `namioto/cli.py`, `ipc.py` | Headless tasks and remote-control client/protocol |
| `namioto/audio/` | Song playback: the phase vocoder, the transport and the `sounddevice` output |
| `tests/`, `scripts/` | Tests and developer tools |

Read module docstrings before changes. Keep Qt imports under `namioto/ui/`; core analysis,
project, MIDI and lyrics modules must remain usable without Qt. Note editing uses beats
internally; project files and MIDI use seconds. Keep conversions at the piano-roll boundary,
and keep each user gesture as one undo step.
