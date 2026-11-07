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

The aligner was trained on speech and hears a voice best without the band, so the window splits the
audio first when **Separate first** is on and reads only the vocal stem. With the audio loaded, the
clock button beside the wand opens **Align** for the active channel.

The window holds a **Lyrics** box, which is what a run aligns — not the words already on the notes.
Type or paste a lyric into it, or let the model hear it, and then **Align**:

```text
き み の こ え
```

- **Recognize lyrics** runs Qwen3-ASR over the same audio (or vocal stem), and fills the box with
  what it heard. Recognition language is picked beside **Language**; empty lets the model identify
  it. The text is left in the box to correct: a rounded character costs a wrong time on every note
  after it, so the model's answer is a draft, never the words themselves.
- **Align** runs Qwen3-ForcedAligner over the box's text. It answers a time per unit it read, and
  the units are matched back onto the words by their text, so a word split across two units keeps
  the whole span. A word the model could not place keeps the note's current time, and the log
  counts those instead of inventing a time for them.
- Accepting writes the box's words onto the notes of the active channel, one per note in time
  order, and moves those notes onto the aligned spans. Both are one undo step each; a note past the
  end of the text keeps no word and is not moved.

- **Language** picks how the aligner reads the text; it knows eleven, from Chinese and Japanese to
  Spanish.
- **Quantize** optionally puts the aligned onsets on the drawn grid and leaves each end where the
  model put it.
- The weights are fetched on first use from the Hugging Face endpoint in Settings → Network, which
  defaults to the domestic mirror. A GPU run needs a CUDA build of torch, which is what the
  `ort-cuda` extra installs; the CPU works without one.
- A run over the same audio, words and language is remembered, so re-aligning after a tempo change
  only re-snaps the stored spans. A recognized text is not remembered: it lands in the box where
  the corrections are yours.