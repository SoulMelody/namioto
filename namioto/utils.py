# SPDX-License-Identifier: AGPL-3.0-only
"""Small helpers shared across the Qt-free modules: the app's own directories, the identity of a
file under a path, and `kana_tokens()`.

`kana_tokens()` is the ruby `namioto-cli align` takes: it turns a `.krc` line's bracketed kana into
one hepburn token per sound, the unit FA-Kara's models are trained on. It is deliberately small - a
`.krc` file already carries the kana ruby of every kanji, so the kanji surface and the annotation
punctuation only have to be dropped, with no kanji lookup and no dictionary. A run of Latin letters
or digits is kept as one token; everything else is dropped, so the same call reads a bare kana
string and a whole `.krc` line.
"""

from __future__ import annotations

import json
import os
import tempfile
import unicodedata
from pathlib import Path

import platformdirs

# The name the platform's own directories are filed under: every path the program keeps is built
# from it, so a renamed program takes its files with it.
APP_DIRECTORY = "namioto"


def data_dir(*parts: str) -> Path:
    """A path under the app's data root, where derived results and downloaded models belong."""
    return Path(platformdirs.user_data_dir(APP_DIRECTORY), *parts)


def config_dir(*parts: str) -> Path:
    """A path under the app's config root, where preferences belong."""
    return Path(platformdirs.user_config_dir(APP_DIRECTORY), *parts)


def runtime_dir(*parts: str) -> Path:
    """A path under the app's runtime root, where a socket or a lock file belongs.

    `$XDG_RUNTIME_DIR` when the session has one - it is cleared on logout, which is exactly right
    for something that only exists while the program runs - and the config directory otherwise.
    """
    base = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(base) / APP_DIRECTORY if base else config_dir()
    return root.joinpath(*parts)


def resolved(path: str | Path) -> Path:
    """`path` made absolute, with `~` and symlinks taken out, so two names for one file compare equal."""
    return Path(path).expanduser().resolve()


def file_stamp(path: str | Path) -> tuple[int, int]:
    """Size and mtime, so the same path holding a different file is not the same file."""
    try:
        info = resolved(path).stat()
    except OSError:
        return (0, 0)
    return (info.st_size, info.st_mtime_ns)


def write_text(path: str | Path, text: str) -> Path:
    """Write a whole file at once, so a half-written one never exists to be read."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(dir=target.parent, prefix=f"{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(name, target)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return target


def write_json(data: dict, path: str | Path) -> Path:
    """Write a whole file at once, with the layout both settings and projects use."""
    return write_text(path, json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


_VOWELS = "aiueo"
# A sokuon or a long vowel mark with nothing after it is still a mora, so it stands as itself.
_LONE_SOKUON = "'"
_LONE_LONG_VOWEL = "-"

# the gojuuon, dakuten and handakuten, after katakana is normalised to hiragana
_KANA = {
    "あ": "a",
    "い": "i",
    "う": "u",
    "え": "e",
    "お": "o",
    "か": "ka",
    "き": "ki",
    "く": "ku",
    "け": "ke",
    "こ": "ko",
    "が": "ga",
    "ぎ": "gi",
    "ぐ": "gu",
    "げ": "ge",
    "ご": "go",
    "さ": "sa",
    "し": "shi",
    "す": "su",
    "せ": "se",
    "そ": "so",
    "ざ": "za",
    "じ": "ji",
    "ず": "zu",
    "ぜ": "ze",
    "ぞ": "zo",
    "た": "ta",
    "ち": "chi",
    "つ": "tsu",
    "て": "te",
    "と": "to",
    "だ": "da",
    "ぢ": "ji",
    "づ": "zu",
    "で": "de",
    "ど": "do",
    "な": "na",
    "に": "ni",
    "ぬ": "nu",
    "ね": "ne",
    "の": "no",
    "は": "ha",
    "ひ": "hi",
    "ふ": "fu",
    "へ": "he",
    "ほ": "ho",
    "ば": "ba",
    "び": "bi",
    "ぶ": "bu",
    "べ": "be",
    "ぼ": "bo",
    "ぱ": "pa",
    "ぴ": "pi",
    "ぷ": "pu",
    "ぺ": "pe",
    "ぽ": "po",
    "ま": "ma",
    "み": "mi",
    "む": "mu",
    "め": "me",
    "も": "mo",
    "や": "ya",
    "ゆ": "yu",
    "よ": "yo",
    "ら": "ra",
    "り": "ri",
    "る": "ru",
    "れ": "re",
    "ろ": "ro",
    "わ": "wa",
    "ゐ": "i",
    "ゑ": "e",
    "を": "o",
    "ん": "n",
    "ゔ": "vu",
    "ぁ": "a",
    "ぃ": "i",
    "ぅ": "u",
    "ぇ": "e",
    "ぉ": "o",
    "ゃ": "ya",
    "ゅ": "yu",
    "ょ": "yo",
}

# a base kana followed by a small ya, yu or yo, which is one sound
_DIGRAPHS = {
    "きゃ": "kya",
    "きゅ": "kyu",
    "きょ": "kyo",
    "ぎゃ": "gya",
    "ぎゅ": "gyu",
    "ぎょ": "gyo",
    "しゃ": "sha",
    "しゅ": "shu",
    "しょ": "sho",
    "じゃ": "ja",
    "じゅ": "ju",
    "じょ": "jo",
    "ちゃ": "cha",
    "ちゅ": "chu",
    "ちょ": "cho",
    "ぢゃ": "ja",
    "ぢゅ": "ju",
    "ぢょ": "jo",
    "にゃ": "nya",
    "にゅ": "nyu",
    "にょ": "nyo",
    "ひゃ": "hya",
    "ひゅ": "hyu",
    "ひょ": "hyo",
    "びゃ": "bya",
    "びゅ": "byu",
    "びょ": "byo",
    "ぴゃ": "pya",
    "ぴゅ": "pyu",
    "ぴょ": "pyo",
    "みゃ": "mya",
    "みゅ": "myu",
    "みょ": "myo",
    "りゃ": "rya",
    "りゅ": "ryu",
    "りょ": "ryo",
    "ふぁ": "fa",
    "ふぃ": "fi",
    "ふぇ": "fe",
    "ふぉ": "fo",
    "うぃ": "wi",
    "うぇ": "we",
    "うぉ": "wo",
    "ゔぁ": "va",
    "ゔぃ": "vi",
    "ゔぇ": "ve",
    "ゔぉ": "vo",
    "てぃ": "ti",
    "でぃ": "di",
    "とぅ": "tu",
    "どぅ": "du",
    "しぇ": "she",
    "じぇ": "je",
    "ちぇ": "che",
}


def _normalize(text: str) -> str:
    """Full-width and half-width forms resolved, katakana folded onto hiragana."""
    folded = []
    for char in unicodedata.normalize("NFKC", text):
        code = ord(char)
        folded.append(chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else char)
    return "".join(folded)


def _geminate(token: str, count: int) -> str:
    """A sokuon doubles the next token's consonant; `ch` becomes `tch`, Hepburn's own spelling."""
    head = "t" if token.startswith("ch") or token[:1] in _VOWELS else token[0]
    return head * count + token


def kana_tokens(text: str) -> list[str]:
    """The ruby of `text` as one hepburn token per sound, Latin and digits kept as runs.

    `text` is a kana string or a whole `.krc` line; the kanji and the annotation punctuation are
    dropped, so `漢字[かんじ]` reads as `kanji`.
    """
    folded = _normalize(text)
    found: list[str] = []
    geminate = 0
    index = 0
    while index < len(folded):
        char = folded[index]
        if char == "っ":
            geminate += 1
            index += 1
            continue
        if char == "ー":
            if geminate:
                found.append(_LONE_SOKUON * geminate)
                geminate = 0
            if found and found[-1][-1:] in _VOWELS:
                found[-1] += found[-1][-1]
            elif found:
                found.append(found[-1][-1])
            else:
                found.append(_LONE_LONG_VOWEL)
            index += 1
            continue
        if char.isascii() and char.isalnum():
            end = index + 1
            while end < len(folded) and folded[end].isascii() and folded[end].isalnum():
                end += 1
            run = folded[index:end].lower()
            found.append(_geminate(run, geminate) if geminate else run)
            geminate = 0
            index = end
            continue
        pair = folded[index : index + 2]
        if pair in _DIGRAPHS:
            token, index = _DIGRAPHS[pair], index + 2
        elif char in _KANA:
            token, index = _KANA[char], index + 1
        else:
            index += 1
            continue
        found.append(_geminate(token, geminate) if geminate else token)
        geminate = 0
    if geminate:
        found.append(_LONE_SOKUON * geminate)
    return found
