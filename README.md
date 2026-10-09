# Namioto

An open-source editor that follows [WaveTone](https://ackiesound.ifdef.jp/)'s feature set: analyse an
audio file into a note-domain spectrum, draw that behind a piano roll, then transcribe what you see
into notes - time on the horizontal axis, pitch on the vertical axis, with notes you can draw, move,
resize and delete. Notes play back through a synthesiser - the machine's MIDI one when it has one,
or a built-in fallback - and the file plays along with them.

![namioto](docs/screenshot.png)

## Requirements

- [uv](https://docs.astral.sh/uv/) (Python 3.12 or newer)
- a C++17 compiler to build from the source tree or the sdist (the audio backend is native; the
  published wheel carries it prebuilt)

## Run

```bash
uv sync --extra cpu                       # the onnxruntime build the analysis models use (or --extra cuda/webgpu)
uv run namioto                            # the editor
uv run namioto song.mp3                   # the editor with the audio analysed into a spectrum
uv run namioto song.nto                   # open a project (the notes and the audio together)
uv run namioto song.mp3 --channels both --gain 300   # analysis options
uv run namioto-wavetone song.mp3          # WaveTone's own volume-envelope analysis (the editor's default)
uv run namioto-tempo song.mp3             # the librosa beat tracker + least-squares fit
uv run namioto-tempo song.mp3 --local     # per-window estimates, 12 s wide, 6 s apart
uv run namioto-tempo song.mp3 --json      # machine readable, includes every beat
uv run namioto-tempocnn song.mp3          # the TempoCNN model, on the CPU or a GPU backend
uv run namioto-spectrum song.mp3          # analyse into 84 note bands (C1-B7)
uv run namioto-spectrum song.mp3 --bench  # plus per-stage timings
uv run namioto-spectrum song.mp3 --threshold 1.2   # plus auto-filled note spans
uv run namioto-game song.mp3 --model DIR  # extract the notes of a singing voice (GAME's models)
uv run namioto-align vocal.wav lines.json # time known lyrics against a separated vocal
uv run pytest                             # tests (the beat tracker's slow ones left out)
uv run pytest -m slow                     # just those, minutes of synthetic audio
uv run scripts/bench_spectrum.py          # spectrum benchmark
```

`uv` creates the virtual environment and installs the dependencies on first run. `--extra cpu` adds
ONNX Runtime, which the analysis models need and which is optional: `--extra webgpu` puts the WebGPU
(Vulkan) provider on top of that same build, while a CUDA build is a separate one that replaces it,
so an installed copy takes `pip install namioto[cpu]` instead (or `[cuda]`/`[webgpu]` for a GPU). No
model is part of the program: the ones it uses are downloaded from this project's `models` release
(or GAME's, for GAME's own), so the first run of an analysis is the one that fetches them.

## Spectrum

Passing an audio file draws its note-domain spectrum behind the piano roll, the way noteDigger and
WaveTone show it: one column per 25 ms frame (40 frames per second), one row per note band (C1 to B7),
coloured from dark
blue through green to red as the energy rises, with the octave lines of the pitch axis. The
**Spectrum** block sets the two display parameters — gain (how much energy reaches full red) and
contrast (the exponent applied to the energy). The analysis runs in a background thread and reports
progress in the status bar. The channels, the frames per second, the FFT window and the tuning are
chosen in the window that opens an audio file, filled in from the habits the last one left;
`--channels` and `--t-num` only start that window off with the values they name.

![spectrum](docs/spectrum.png)

## Projects

The **Project** block of the transport row holds `Open` and `Save` (`Ctrl+O`, `Ctrl+S`, `Ctrl+Shift+S`
for Save As), and `Export` beside them, a menu that writes the notes as a MIDI file, the lyrics as a
`.krc`, or a karaoke subtitle as `.ass`. A `.nto` project is a plain
JSON file that keeps the notes together with what they were drawn over and the values that belong to
that piece of work: the audio file, the tempo, the analysis parameters, the spectrum display, the snap
grid and the view. It is a few kilobytes, so it is diffable, searchable and editable by hand.

Opening an audio file is how a project starts: namioto asks where its `.nto` goes, defaulting to the
audio's own name beside it, and how the audio is analysed - the channels, the frames per second, the
FFT window and the tuning - then writes the project there. A MIDI file is imported into the project
that is open, never opened on its own, and switching to another project or another audio file asks
about unsaved notes first.

Notes are kept in seconds, so a different tempo moves the grid and never the notes, and the file
lists them in time order. The audio is recorded as a path - relative to the project when it sits
beside it - and never copied into the project. A project whose audio is missing still opens, since
the notes are worth having, and says so in the status bar.

The window title shows the project name, with a `*` while there are unsaved changes; closing, or
opening another project, asks before they are lost. Moving the view, or turning a volume down, is
written when you save but does not count as a change, so looking around never nags. A roll that has
not been saved under a name yet is treated as a sketch and closes without asking.

Settings and projects stay apart: the values inside a project belong to that project and never
overwrite the program's own preferences, and only a handful of them - the ones that are habits
rather than a song's - carry into the next new project.

## MIDI files

`Open` (`Ctrl+O`) takes audio files and `.nto` projects — the suffix picks whether the file is
analysed behind the roll or opened as a project — and, once a project is open, `.mid`/`.midi` files
to import into it: a MIDI is a part of a project and is never opened on its own. `Export MIDI`
writes the roll out for a DAW, a score program or a synth, saving the project
separately. A MIDI is read by channel: every MIDI channel with notes in it becomes one channel here,
carrying its instrument and channel volume, and one track chunk holding several channels is read as
several. A channel carries no name -
a name belongs to a track chunk, which may hold any number of channels - so names live in the project
file only, and none is read from or written to MIDI. Importing into a window that already has audio
loaded keeps the audio, the view and the snap grid, which is how a transcription made elsewhere is
checked against the sound it came from; what it cannot use (a note that never ends, a channel past the
sixteenth, a tempo change after the first, since the grid holds one tempo) is counted in the
status bar rather than dropped silently. An import over a roll that already has notes asks first, and
along with **Replace** it offers **Merge**: each of the file's channels is pointed at one of the
roll's channels or at a new one. The file's i-th channel starts on the roll's i-th channel while that
one carries no notes, and the ones whose place is taken fill the empty channels left, in order, so
the notes arriving are added to a channel of their own rather than mixed into one already in use; the
mapping is edited in the same dialog, and the tempo stays where the audio put it.

**Export MIDI** asks for one thing only, the file name, and writes every channel on the project's own
tempo, one track chunk per channel and no name on any of them. A note drawn on the roll's grid lands
exactly on the tick that grid names, and one taken from the audio keeps the time it has, rounded to
the nearest tick; a hidden channel goes in like any other, since hiding is about the drawing and not
the notes. **Export lyrics** folds the current mapping back into a `.krc` - a run of sounds sharing a
note becomes `(...)` and every word gets the notes it covers as its `.N` - writing a file of its own,
for another karaoke tool or a read-only pass. **Export ASS subtitle** times the same mapping onto a
`.ass` karaoke subtitle - the `\k` of every syllable, its ruby and the gaps between notes - with the
header the Aegisub kara-templater expects, so an existing video pipeline picks it up as it is.
**WaveTone compatibility**, in the settings, is
on by default: WaveTone's own MIDI export starts every note one bar late, so a file it wrote is read
back with that bar removed, and a file written here carries it again - which is what its own tools
expect. Turn it off to exchange plain MIDI with anything else,
since a file whose notes start before that bar was never WaveTone's and is read as it stands.

## Settings

The gear at the right end of the Mix row opens the settings window, and it is short on purpose: it
holds only what has no control in the bars - **General** (the interface **Language**, where the
default `Follow system` takes the machine's own and a change takes effect on the next run,
**Auto-save**, off by default, which writes the open project once editing stops and when the window
loses focus, and **Style**, the widget style that draws the window), **Devices** (which GPU backend a
run that asks for the GPU uses; the line under it says whether its runtime is installed), **Tempo**
(the **Algorithm** that estimates the tempo - `wavetone` by default, or the librosa beat tracker or
TempoCNN - with the two windows the beat tracker fits under an `ADVANCED` heading),
**WaveTone compatibility**, **Network** (the **GitHub mirror** releases are fetched through, and
an HTTP **Proxy** for downloads and the lyrics API), and **Lyrics** (the OpenAI-compatible endpoint
the lyrics window may call - **API base** up to its `/v1`, **API key**, **Model**, the temperature
and timeout under an `ADVANCED` heading - and the **External editor** command a `.krc` is opened
with), and **ASS subtitle** (the **Font** and **Overlay colour** a subtitle is drawn with, the
**Fade in** and **Fade out** lengths, the **Lead time** a line appears before its first word, and
the **Guide dot** length) - a page per group. `Restore defaults` puts everything back, and `Apply`
lets the change go live without closing the window.

![settings](docs/settings.png)

Everything else is remembered rather than configured, and the file it goes to says what it belongs
to. A value that describes the song - how it is analysed, drawn and played back, the tempo, the grid
offset, the speed, the tuning, the view centre - is written into the `.nto`, so a project carries
its own. The habits a new document should inherit are kept as well, in
`~/.config/namioto/state.json` (`$NAMIOTO_STATE` points elsewhere): the spectrum gain and contrast,
the audio and MIDI volumes, the analysis channels and resolution, the snap grid, the time division
and the zoom. Change one of those while working and the next new project starts from it; opening a
project never changes what is remembered. The program's own preferences - the interface language,
the wanted GPU, the tempo algorithm, the lyrics endpoint, the GitHub mirror and proxy, WaveTone
compatibility and the two editor switches - go to `~/.config/namioto/settings.json`
(`$NAMIOTO_SETTINGS` points elsewhere), and the window's size, position and last folder also to
`state.json`. The files are plain JSON, so they can be edited by hand - and a hand-mangled or
half-written one falls back to the defaults field by field instead of refusing to start.

The command line still wins for one run: `--channels`, `--t-num`, `--gain` and `--contrast` shape
this run alone and are never written back into the file.

## Remote control

The editor can listen for a small JSON-lines interface, so a script or another machine can drive
the transport, the notes and the project without touching the window. It is off unless it is asked
for: `namioto --ipc` turns it on for that run, and **Remote control** on the **Remote** page of
the settings turns it on for good. The command line wins for one run: `--ipc ADDRESS` chooses where
it listens, `--no-ipc` keeps it off even when the setting is on.

The default endpoint is a per-user unix socket (`$XDG_RUNTIME_DIR/namioto/control.sock`, mode 0600
inside a 0700 directory); `--ipc tcp://host:port` listens over TCP instead, which always carries a
generated token. A listening editor records its endpoint in `$XDG_RUNTIME_DIR/namioto/ipc.json`, so
`namioto-ctl` finds it without arguments; `$NAMIOTO_IPC`, `$NAMIOTO_IPC_TOKEN` and `--endpoint`
override that for one call. A client passes the token with `--token` or `$NAMIOTO_IPC_TOKEN`, and
the server takes one from `$NAMIOTO_IPC_TOKEN`, generating one when it is unset.

`namioto-ctl` is the client:

```bash
namioto-ctl status                       # the whole session as JSON
namioto-ctl play                         # also pause, stop, toggle
namioto-ctl seek 12.5                    # seconds; also `seek beat=8` or `seek fraction=0.5`
namioto-ctl bpm 132                      # tempo of the beat grid
namioto-ctl notes                        # every note, in seconds
namioto-ctl document.add_notes notes='[{"pitch":60,"start":0.5,"duration":1.0}]'
namioto-ctl project.save path=song.nto
namioto-ctl audio.open path=song.mp3 analysis='{"channels":"both"}'
namioto-ctl methods                      # what this editor offers, with each method's parameters
namioto-ctl --watch toggle               # keep printing the events: hello, state, message
```

Parameters are `NAME=VALUE`, their JSON type kept, and a single positional value fills the obvious
parameter of a short alias. Every message on the wire is one JSON object per line: a request
`{"id":1,"method":"transport.play"}`, an answer `{"id":1,"ok":true,"result":{...}}` or
`{"id":1,"ok":false,"error":{"code":...,"message":...}}`, and an event
`{"event":"state","data":{...}}` the server pushes to every client, so a watcher can follow
without polling. The protocol is small enough to speak from any language, and
`namioto.ipc.Client` is a Python client for it. A call that would discard unsaved notes is refused
with `dirty` unless it carries `force=true`.

## Build

```bash
./build.sh wheel   # sdist + wheel             -> dist/*.whl, dist/*.tar.gz
./build.sh cli     # frozen tempo CLI          -> dist/namioto-tempo/
./build.sh cli spectrum                        # -> dist/namioto-spectrum/
./build.sh app     # frozen editor             -> dist/namioto/
./build.sh all
./build.sh clean
```

Neither the wheel nor the frozen builds carry a model: they are downloaded at run time into the
data directory, and each keeps its own license — see [NOTICE](NOTICE) and
`scripts/upload_models.sh`, which publishes them.

## Layout

```
namioto/analysis/    the analysis engines, each also a CLI, no Qt
  beats.py            tempo estimation by beat tracking (librosa) + least-squares fit
  tempo.py            TempoCNN tempo estimation (ONNX Runtime) + tempo map helpers
  spectrum.py         note-domain spectrum analysis (STFT → 84 note bands)
  wavetone.py         WaveTone's own volume-envelope tempo analysis
  bpm.py              the tempo algorithms behind one setting
  align.py            forced alignment of known lyrics onto the audio (FA-Kara's core, ONNX)
  game.py             singing-voice note extraction with GAME's ONNX models
  transcription.py    GAME's parameters, their store and the spawned child entry point
  model_store.py      where a model comes from: one registry, one download, one session
  devices.py          the CPU/GPU device registry the models run on
  choices.py          the values a setting may take
namioto/playback.py   note synthesis: pitches rendered into one audio buffer, no Qt
namioto/channels.py   the MIDI channels a note plays on, and the values they play with, no Qt
namioto/interaction.py the roll's normal/edit mode and its tool, as one value, no Qt
namioto/document.py   the notes and the MIDI channels they play on, in beats, no Qt
namioto/midi.py       reading and writing MIDI files (mido), no Qt
namioto/project.py    the .nto file: the notes, the audio they were drawn over, and the document's values, no Qt
namioto/settings.py   the program's own preferences: the spec table, its file and the defaults, no Qt
namioto/params.py     the Field primitive every parameter table is built from, no Qt
namioto/state.py      the window's size, position, last folder and the habits a new document starts from, no Qt
namioto/lyrics.py     KRC text I/O and the model call that fills it, no Qt
namioto/ipc.py        the remote control's protocol, endpoint and `namioto-ctl` client, no Qt
namioto/karaoke/      the .krc model, its parser, its writer, its timeline and the ASS subtitle (lark), no Qt
namioto/utils.py      the app's directories, a file's identity, kana to romaji tokens, no Qt
namioto/i18n.py       the language catalogs and the language in force, no Qt
LICENSE-CC-BY-NC-SA-4.0.txt  the license text of the downloaded TempoCNN model
namioto/ui/           PyQt6 editor (app.py: window, controls.py: control bars,
                      roll.py: the view, its items, undo, ruler and keyboard, strips.py: the
                      lyrics strip, lyric_map.py: the lyrics-to-notes mapping and its thread,
                      blocks.py: the pointer grammar both editors share, viewport.py: the strips
                      that share the roll's columns, theme.py: colours, icons.py: glyphs,
                      text.py: shared labels, loading.py: one-shot worker, spectrogram.py:
                      spectrum colour map, image cache and loader, audio.py: note playback
                      outputs - the external MIDI synth, or the built-in synth on the native engine,
                      song.py: the audio file streamed to the native engine, channel_panel.py:
                      the sidebar, form.py: the pieces a settings form is built from,
                      ipc_server.py: the remote control's socket on the Qt event loop,
                      ipc_commands.py: the methods it offers, and one
                      module per dialog: settings, lyrics, transcription, align, MIDI import,
                      open audio)
audio/                the native audio backend (C++17): Signalsmith Stretch for the speed, miniaudio
                      for the output device; built as the namioto._audio extension
vendor/               the backend's vendored dependencies (miniaudio, Signalsmith Stretch and
                      Linear), each with its license text
meson.build           the native extension's build (meson-python + pybind11)
tests/                pytest
scripts/              developer tools (spectrum benchmark, aligner model export, model upload)
build.sh              packaging script
```

## License

Namioto is licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE).
The AGPL is required because the program reuses code from the GPL-3.0 licensed NoteDigger
project, reimplements the AGPL-3.0 licensed Essentia TempoCNN front end, and links PyQt6
(GPL-3.0 or commercial). See [NOTICE](NOTICE) for the full reasoning and the attributions.

None of the program's models is AGPL, and none of them is part of a build: `deeptemp-k16-3.onnx`
comes from Essentia/TempoCNN and the aligners are converted from Meta's MMS and NextFire's yohane
checkpoint, all of them under **non-commercial** Creative Commons licenses (CC BY-NC-SA 4.0 /
CC BY-NC 4.0). They are downloaded from this project's `models` release, so the program itself can
be; using them makes the *output* non-commercial. See [NOTICE](NOTICE) for each one.

## Controls

The window has three control bars, each split into captioned blocks of related controls
(WaveTone's grouping):

| Bar | Blocks |
| --- | --- |
| Transport | **Project** (open, save, export MIDI), **Playback** (rewind, stop, play from the beginning, play/pause, forward, position readout, and the display switches: auto page turn and overtone highlight, then a divider, MIDI channels, lyrics on MIDI notes, and time division - the metronome icon checked means the time axis follows the beats of the tempo map, unchecked the seconds), **Speed** (0.10x-2.00x in 5% steps, pitch unchanged, with a reset icon back to 1.00x), **Tempo** (BPM, the estimated tempo of the audio, and the grid offset in ms) |
| Edit | **Tools** (edit mode, pen, select, snap grid, quantize, the GAME transcription, the lyrics window) |
| Mix | **Spectrum** (gain, contrast), **Volume** (**Audio** for the file, **MIDI** for the notes), and the gear that opens the settings window (the few options the bars do not hold; the analysis parameters belong to the project) |

**Volume** has a slider for each layer: the audio file is streamed at the level of the first one, and
the second is the note playback - through the machine's external MIDI synth when one is listening,
sent as control change 7 (channel volume) since that synth does its own mixing, or through the
built-in synth otherwise.

**Channels** are toggled by the layers icon among the playback switches: a sidebar with one card per MIDI channel.
The notes of each channel are painted in its colour, and the card holds the name (double-click to
rename), the GM instrument the channel plays, and the lock, show and mute switches. Clicking a card
makes it the drawing channel; the context menu adds a channel, sets its volume or deletes it (the
last one stays). Deleting one leaves the numbers of the others alone, so a note keeps the MIDI
channel it plays on. Any of the sixteen can be used, percussion included.
A locked channel cannot be edited, a hidden one is not drawn, and a muted one stays silent; an
external synth receives the real GM program on the channel's own number.

| Action | Input |
| --- | --- |
| Edit mode | The button in front of the tools: notes are drawn only while it is on (WaveTone keeps its graph to the spectrum outside note edit mode), and only then can they be drawn, moved, resized and selected; the spectrum behind them fades so that they stand out over it (WaveTone does the same), and picking the pen or the select tool turns the mode on as well - entering it starts on the pen. The snap grid and the quantize button are the two that work in either mode: they set the grid and apply it, and the notes they move are drawn once the mode is on. The window opens with it off, so a click in the roll moves the playhead until a tool is picked. Hovering marks the row under the mouse, and the piano key with it, in either mode; with **Overtone highlight** on, the overtones of that row - `f`, `2f`, `3f` and `4f` - are marked the same way, in either mode, as WaveTone does |
| Draw note | Pen tool: left drag on the empty grid. Horizontal movement sets the length, vertical movement sets the pitch, so the note follows the pointer |
| Move note(s) | Left drag a note, which steps in whole snap cells and keeps where it sits inside its cell; Quantize is what lands it on the grid |
| Resize note | Left drag either edge of a note, or Shift + left drag anywhere on it: its left half moves the start, its right half the end. A note is never left shorter than one snap cell |
| Select note | Left click; a click on one that is already selected leaves only that one, while a press that drags carries the whole selection |
| Add to selection | Ctrl + left click |
| Box select | Select tool: left drag on the grid, or Ctrl + left drag with either tool |
| Select all | Ctrl + A |
| Delete | Right click a note, or Delete / Backspace for the selection |
| Copy notes | `Ctrl+C` takes the selected notes as one block, timed from their earliest note |
| Paste notes | `Ctrl+V` drops that block at the playhead, its first note on the snap grid and the spacing between them quantised to the same cell, and selects what it pasted |
| Quantize notes | The grid icon beside the Snap combo, in edit mode: the starts and ends of the notes land on the snap grid, so they sit on the beats of the current tempo. The selection when there is one, the whole roll otherwise; a locked channel is never touched, and a note shorter than one cell is left one cell long |
| Cancel a drag | Escape |
| Play or pause | `Space` or the play/pause button |
| Open a project | `Ctrl+O`, or `Open` in the Project block |
| Save a project | `Ctrl+S` (Save As on the first save, or `Ctrl+Shift+S`), or `Save` in the Project block |
| Export MIDI / lyrics / subtitle | `Export` in the Project block, next to `Save`: a menu with `Export MIDI…`, `Export lyrics (.krc)…` and `Export ASS subtitle…` |
| Move the playhead | A press anywhere in the roll - over the grid or over a note, in either mode - or a click in the timeline ruler (any mode, and it works while the file plays), or the rewind / forward buttons for the ends. A press on a note moves, resizes or selects it *and* moves the playhead. Dragging carries the playhead along with the pointer, and sounding every row it crosses like a glissando. While the audio plays the roll keeps its cursor so editing does not jump the sound; seeking then is what the ruler is for, and the file carries on from there |
| Double / halve the tempo | Right-click the Tempo field, or press `*` / `/` while it has the focus |
| Pan | Middle drag, a scrollbar, or drag in the timeline ruler to scroll horizontally |
| Zoom time | Ctrl + wheel |
| Zoom pitch | Ctrl + Shift + wheel |
| Scroll | Wheel along the timeline, Shift + wheel up and down the pitches |

The value sliders - **Speed**, **Gain**, **Contrast**, **MIDI** - land on the spot the track is clicked
and step with the wheel, one notch to a step, turning the value down as the wheel turns up.

Loading an audio file also estimates its tempo in the background with the chosen **Algorithm**
(WaveTone's own volume-envelope analysis by default; the librosa beat tracker also reports how many
of its 12 s windows agree and the beat-fit residual). It never changes the tempo by itself:
when it is done, the **Tempo** block shows a suggestion such as `≈93 BPM` next to the BPM field,
with a tick to use it and a cross to drop it. The balloon
floats inside the window, so it takes neither the keyboard nor the click: drawing, selecting or
playing underneath it goes on, and typing a tempo, dropping the suggestion or loading another file
discards it; the round arrow estimates again.
Changing the tempo never re-times the notes: they are timed against the audio, so only the beat
grid re-divides underneath them (and the roll keeps the audio at the same scale on screen, which is
how the notes stay where they are relative to the spectrum and to the time ruler).
TempoCNN (`namioto/analysis/tempo.py`, `namioto-tempocnn`) and the librosa beat tracker are the other
two algorithms the **Algorithm** setting offers, each on its own command line too: TempoCNN is strong
on full mixes, but its 256 integer-BPM classes and its training data (full mixes only) make it a poor
fit for stems, where the other two win.

**Extracting notes with GAME.** A singing voice can be transcribed in one pass with
[GAME](https://github.com/openvpi/GAME)'s ONNX models:

```bash
uv run namioto-game song.wav --language zh          # fetches the small model on first use
uv run namioto-game song.wav --size medium          # one of small, medium, large
uv run namioto-game song.wav --device gpu           # on the GPU backend the settings name
uv run namioto-game song.wav --quantize 4 --tempo 93 --midi out.mid
uv run namioto-game                                 # only fetch a model, do not transcribe
```

The editor can do the same over the file it has open: the wand button in the tools opens a
window with GAME's options (model size, device, language, the quantisation grid and its inference
parameters), runs the model in a process of its own so a crash cannot take the editor down, and
watches it there with a progress bar and a log. The notes arrive on a channel of their own, or over
the active channel's own — the window's **Target** says which, and asks before overwriting a channel
that already holds notes. A finished run inserts itself, so there is no second click to make. The
options are remembered for the next run, and so is the result: asking for exactly the same run again
offers the saved notes instead of loading the model a second time.

GAME is a PyTorch project and ships its models separately, in three sizes, so they are not packaged
here: `--size` picks one and it is downloaded from GAME's GitHub release into the data directory
(`~/.local/share/namioto/models/game/<size>`, via `platformdirs`), unpacked, and reused from then
on. An explicit `--model DIR` or `$NAMIOTO_GAME_MODEL` skips all of that and loads a directory as it
is.

| size | download | 20 s of a singing voice, 16-core CPU |
| --- | --- | --- |
| small | 46 MB | 1.9 s, 46 notes |
| medium | 180 MB | — |
| large | 362 MB | 7.1 s, 47 notes |

The notes come back as floating-point pitches with the onsets the model found; `--quantize` snaps
them to a beat grid first, and `--midi` writes them out. The **Device** option (the dialog's own
field, `--device` on the command line) picks where the models run: `cpu` by default, or `gpu` for
the GPU backend named under **Devices** in the settings (CUDA, which needs ONNX Runtime's GPU build
together with CUDA 12 and cuDNN 9, or WebGPU/Vulkan). A GPU provider that cannot be created is not
an error - ONNX Runtime says so on stderr and the run carries on the CPU. The code is MIT (Team
OpenVPI, like GAME
itself); the models are CC BY-NC-SA 4.0, so anything produced with them is non-commercial, and they
are downloaded rather than redistributed here - see NOTICE.

## Importing lyrics

A project can carry rubies for the words its notes sing. The text itself lives in the `.nto`.
External editing uses a temporary `.krc` copy, not a file beside the project; existing sibling
`.krc` files are left untouched and can be imported manually. The text-box button beside
the GAME wand opens the lyrics window. The window is one pipeline: the **Project lyrics
(.krc)** box at the top is what the project sings, and the plain text below it is the optional
upstream that fills it. `Import .krc…` puts another `.krc` into that box, and `Open in external
editor` hands the file to the command named under **External editor** in the settings, or to the
platform's own choice when that is left empty. `Save` updates the project lyrics through the temporary
UTF-8 copy. External saves are validated and reloaded without reopening the project or automatically
running Align. Temporary copies are removed when the document closes. The plain-text panel folds
away when there are already lyrics to work on, and opens by itself when the box is empty.

Plain text is the exception, and only so a model can make a `.krc` of it: the panel takes pasted
lyrics or a `.txt`/`.md`/`.lrc` (`Load text…`), and `Convert with the API` or `Copy prompt` fills
the lyrics box above. The prompt is built in and adds the rubies: it is the same rule set an external
`lyrics.md` role would carry, asking for the kana of each kanji in square brackets, grouped per word
and comma-separated. `Copy prompt` puts it on the clipboard together with the lyrics, for a web
model - paste it there, paste the answer back into the lyrics box and save. With an OpenAI-compatible
endpoint set up under **Lyrics** in the settings, `Convert with the API` does that round trip in
the background instead, and a box under the plain-text lyrics streams the model's own output as it
arrives - its reasoning first, then the answer. An answer is only taken when it parses, carries the
same lyrics and gives every kanji a reading; a refusal is sent back to the model with what it wrote
and why, up to a few tries, and one that outlives them is not applied at all - the plain-text lyrics
then offer the refusal as the next prompt to copy. Until an endpoint is set up the button is off and a
line under the lyrics says to copy the prompt instead. The streamed box appears only once the API is
asked for; the clipboard and hand-editing routes never need it. Closing with unsaved lyrics asks
before they are dropped. The key lives in `~/.config/namioto/settings.json` in plain text; a `.krc`
an external hand edit leaves broken is kept as it is for you to fix where it lives, but the window
will not write one that does not parse.

Lyrics normally follow the KRC's own `.N` and groups (faithful mapping). A successful **Align**
starts a mapping-editing session. Its **Use KRC mapping constraints** switch, on by default,
preserves explicit NOTE counts and shared-sound groups; turn it off to compute a new structure.
Text editing is unavailable during this session. Right-click the lyrics timeline, including its
blank area, and choose **Save lyrics map** to write the final structure into the project and return
to faithful mapping, or **Discard lyrics map** to restore the KRC structure from before Align.
Saving the project normally keeps unfinished mapping work so it can be resumed next time.

Once a `.krc` is open and its audio is loaded, the clock button beside the text-box one puts a time
on every sound: the whole stream is forced onto the frames of a wav2vec2 CTC model in one pass, and
each sound gets the span of frames it won. The lyrics are drawn above the roll as one row of sounds, on the roll's own columns: each sound
starts at a `|` with its label, and its block covers the note span the mapping gives it - green while
it sits on its notes, red while the mapping doubts it, grey and blockless while it covers none.
Dragging a `|` slides the boundary between two sounds' raw aligned starts; the drag is smooth and
sticks to a drawn beat line when the pointer comes near it, and one gesture is one undo step.
Hovering a sound that shares a note with its neighbours lights the whole run. The mapping is the only
thing that decides a sound's look; **Save lyrics map** writes the final mapping into the project lyrics.
The text and the times ride in the `.nto` beside the notes, saved and undone with them.
`Export lyrics` folds the current mapping back into a `.krc`: a run of sounds sharing a note becomes
`(...)`, and every word gets the notes it covers as its `.N` - `0` for a sound that covers none - so a
read-only pass reads the same layout. The align window also offers a Quantize choice (Off / 1/4 / 1/8 / …)
that snaps the result onto the BPM beat grid, and remembers the model, the device and the Quantize
choice for the next run. A finished run is kept as a cache, so changing the
tempo or the grid offset and running again re-snaps it without touching the model.

## Aligning lyrics to a vocal

Given a separated vocal and the tokens of each line, `namioto-align` puts a time on every token: the
tokens are forced onto the frames of a wav2vec2 CTC model, which never recognises anything, it only
says where the words it is given fall. The vocal has to be the isolated singing voice and not the
mix, the tokens its kana romanised, and each line a roughly right window - a list of
`{start, end, tokens}` in a JSON file, one token per sound - because the aligner refines a window and
cannot find one: it is passed as a whole, only the model's own frames carry the times.

```bash
uv run namioto-align vocal.wav lines.json --out aligned.json
```

Each line comes back with one entry per input token - its onset, its end and the model's probability
- and a token the model could not place stays untimed. Only the onset is worth reading: a token's end
is the next token's onset, so the last one before a rest reaches into the rest. Lines the run could
not place are named on stderr: `empty` when nothing was timed, `nonmonotonic` when the times run
backwards, `collapsed` when a run of tokens was squeezed into no time at all, and `diverged` against
reference times handed to the module's `problems()`. A line wearing one of them is worth running
again with another window before it is believed.

`namioto.utils.kana_tokens` turns a line's kana into those tokens, one hepburn token per sound. The
converted graph is downloaded from this project's `models` release on first use, into the data
directory; `scripts/export_align_model.py` builds one yourself, and `--dir` or
`$NAMIOTO_ALIGN_MODEL` points at a converted one instead.
Two models can be exported: `mms` (Meta's MMS forced-alignment checkpoint, the default) and `yohane`
(the karaoke fine-tune `NextFire/mms-300m-ForcedAligner-karaoke-ja-Latn`).

```bash
uv run --group export scripts/export_align_model.py --model mms
uv run --group export scripts/export_align_model.py --model yohane
```

The script is a development tool and needs torch (plus torchaudio for `mms`, transformers for
`yohane`) and ONNX, which the program itself does not install.

The transport plays the loaded audio file and the notes on top of it. A press in the roll moves its
playhead wherever it lands - over a note it edits it as well, over the empty grid the pen draws one
there, so a click writes a note where the sound has just moved to. Held down, the drag keeps
carrying that playhead with the pointer and sounds every row it slides over, which is a glissando
up and down the keyboard. While the file is playing that
press only edits: the cursor is left alone, and the timeline ruler is what seeks, from where the
sound carries on. `Space` starts and pauses; the playhead, the readout and the timeline grid all
follow the file. The audio is decoded once at 48 kHz mono and handed to the native engine, which
streams it to the output device: the stretching, the buffering and the device live in the
`namioto._audio` extension, off the Python interpreter, so seeking and speed changes never interrupt
the sound. **Speed** stretches the song with a Signalsmith Stretch phase vocoder as it streams: the
rhythm moves and the pitch does not, the way WaveTone's speed and pitch sliders are separate. Only
the frames the output asks for are generated, so a change of rate costs nothing but the new ratio,
and it applies while the transport runs too - the notes and the song both pick the new speed up at
once, carrying on from where they had got to.

The notes go to a **software MIDI synth** when one is listening on the
MIDI bus - TiMidity and FluidSynth are recognised by name, and the tooltip of the **MIDI** slider
says which one is in use, because that is where the sound comes from (patches included). With no
such synth the notes fall back to a built-in synth, so they are heard either way. **Grid offset (ms)** slides the grid lines - `-` draws them to the left, `+` to
the right, and snapping follows them - while the notes and the playback keep their exact
timestamps. A click in the roll auditions what it lands on - the pitch of the row, whether or
not it is a note - in either mode, and so does a key on the keyboard: listening and editing are
separate. Auditions never wait for the one before them: each click is its own note, and clicking the
same row again releases that note and starts it over - one note-off before the new note-on on the
synth, a 40 ms release ramp over the note still ringing inside - instead of stacking on it or waiting
for it to finish, so fast clicking on one row sounds every time. Hovering the roll tints the
row under the mouse and paints that piano key red, in
either mode, white or black; with **Overtone highlight** (the wave icon, off by default) on it also
tints the overtones of that row - `f`, `2f`, `3f` and `4f` - while editing or not, and shows the note
name and frequency next to the status bar; the notes are drawn only while edit mode is on, so outside it
the roll is a plain view of the spectrum.

The **Snap** combo sets the quantisation applied when drawing, moving, and resizing notes; the grid icon beside it quantises the notes already on the roll onto that same grid, which is how an imported MIDI or a transcription made with another tempo is brought onto the beats here. The **Division** icon changes how the time axis is divided — into
beats and bars of the tempo map, or into a 1-2-5 ladder of seconds — and nothing else: the ruler
shows the clock above the measure numbers under either of them.
