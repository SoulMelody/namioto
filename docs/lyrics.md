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
audio first when **Separate first** is on (the default) and aligns only the vocal stem. With the
audio loaded and the words on the notes, the clock button beside the wand opens **Align** for the
active channel.

The words are handed to Qwen3-ForcedAligner as one text; it answers a time per unit it read, and
the units are matched back onto the notes by their text, so a word split across two units keeps the
whole span. A note whose word the model could not place keeps the time it has, and the log counts
those instead of inventing a time for them.

- **Language** picks how the model reads the text; it knows eleven, from Chinese and Japanese to
  Spanish.
- **Quantize** optionally puts the aligned onsets on the drawn grid and leaves each end where the
  model put it.
- The weights are fetched on first use from the Hugging Face endpoint in Settings → Network, which
  defaults to the domestic mirror. A GPU run needs a CUDA build of torch, which is what the
  `ort-cuda` extra installs; the CPU works without one.
- A run over the same audio, words and language is remembered, so re-aligning after a tempo change
  only re-snaps the stored spans.