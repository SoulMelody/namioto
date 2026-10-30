# Lyrics

[README](../README.md) · [Editor guide](usage.md) · [CLI](cli.md)

## The word on a note

A lyric belongs to the note that sings it: every note carries one word, and the `.nto` keeps that
word with the note. There is no separate lyric file, no lyric text of the project's own, and no
lyric list beside the notes — a note with no word simply sings nothing.

Open the lyrics window with the text-box button beside the GAME wand. It shows the words of the
active channel, one word per note, in the order the notes sound:

```text
き み の こ え
```

Whitespace separates words. A text with no whitespace is read one character at a time, which is how
Japanese lyrics are usually typed. **Save** hands the words to the notes, one per note in time
order; a note past the end of the text keeps no word. Saving is one undo step.

Words show on the notes themselves while **Show lyrics on MIDI notes** is on; the toggle sits in the
playback cluster and does not change the data.

## Align against a vocal

Use a separated singing voice for alignment, not a full mix. Save the project and load the audio
first, then the clock button opens **Align** for the active channel.

The aligner reads the words of the active channel's notes as one token stream and forces it onto
the audio: each note takes the span from its first token's onset to its last token's end, so a word
of several kana covers all of them. A note whose word is empty is not aligned.

- **Quantize** optionally snaps the onsets onto the BPM grid, preserving the aligned end.
- Model, device and chunk choices are remembered between runs.
- A successful run rewrites the note times in one undo step without saving the project.

Model and device choices live in the alignment window's parameter file; the model is an ONNX graph,
so torch stays out of the program. See [CLI alignment](cli.md#align-a-vocal) for the headless path.

![Alignment setup for the selected channel](alignment.png)