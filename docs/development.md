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

### GitHub Actions and releases

The CI workflow runs lint and the default headless tests on Ubuntu 24.04 for pull requests,
main-branch commits, version tags and manual runs. Slow tests are excluded.

Every run builds the CPU-only GUI on Ubuntu 24.04 x86_64 and Windows x64 after checks pass,
and uploads it as an Actions artifact: the AppImage for Linux, the frozen application directory
for Windows (extract the downloaded artifact and run `namioto.exe`). Only a `v*` tag push creates
a GitHub Release **draft** containing:

- `namioto-linux-x86_64.AppImage`
- `namioto-windows-x86_64.zip` (extract the entire archive and run `namioto/namioto.exe`)
- `SHA256SUMS`

Review the draft and manually test the applications before publishing. The workflow verifies
build/package success, not application startup, desktop interaction or audio-device operation.
Windows builds are unsigned and may trigger SmartScreen. Linux builds target Ubuntu 24.04 and
newer compatible systems; older glibc distributions such as Ubuntu 22.04 are not supported.
AppImage execution normally requires FUSE; where unavailable, use its
`--appimage-extract-and-run` option.

No wheel, sdist or separate CLI package is published by this workflow. Both applications keep
settings and downloaded models in the normal user directories, not beside the executable.

To package locally, first run `bash build.sh app`, then:

```bash
# Linux: download appimagetool 1.9.1 for x86_64 and make it executable first
python3 scripts/package_release.py appimage --appimagetool /path/to/appimagetool-x86_64.AppImage
# Windows: zip the frozen application
python3 scripts/package_release.py windows
```

Packages are written to `dist/release/`.

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
| `namioto/lyrics.py` | The word a note sings, and how a text becomes one word per note |
| `namioto/document.py`, `project.py`, `midi.py` | Notes, project persistence and MIDI interchange |
| `namioto/settings.py`, `state.py` | Preferences and remembered UI/new-project values |
| `namioto/cli.py`, `ipc.py` | Headless tasks and remote-control client/protocol |
| `namioto/audio/` | Song playback: the phase vocoder, the transport and the `sounddevice` output |
| `tests/`, `scripts/` | Tests and developer tools |

Read module docstrings before changes. Keep Qt imports under `namioto/ui/`; core analysis,
project, MIDI and lyrics modules must remain usable without Qt. Note editing uses beats
internally; project files and MIDI use seconds. Keep conversions at the piano-roll boundary,
and keep each user gesture as one undo step.
