# SPDX-License-Identifier: AGPL-3.0-only
# ruff: noqa: E501
"""KRC text I/O, the prompt that fills it, and the model call.

Qt-free on purpose. The project keeps the text itself as its baseline; external editing uses a
temporary working copy, while imports and exports may live anywhere. `load` and `save` do not check the syntax
- a file with a mistake in it is still the user's to fix in an editor - while `conversion_error` and
`syntax_error` are what tell a model's answer, or a box about to be written, from a readable one.
`convert` takes its opener as an argument, so a call can be exercised without a network.

Align starts a mapping session; saving its final map writes NOTE counts and groups into the KRC
and returns to faithful reading. Text edits never start alignment automatically. The endpoint,
the key and the model are the settings'; the built-in `DEFAULT_PROMPT` is the one here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from namioto import net
from namioto.karaoke import KrcError, parse
from namioto.utils import write_text

SUFFIX = ".krc"
DEFAULT_PROMPT = """请你为输入的歌词中的汉字进行注音标注，规则如下：
1. 对于输入的歌词中的汉字，在其后用方括号进行注音。
示例：
输入：覚えてる　君がくれた
输出：覚[おぼ]えてる　君[きみ]がくれた
2. 对于由连续的多个汉字组成的词语，将其作为一个整体进行注音；同时，使用逗号对注音进行分隔，使注音与汉字一一对应（如果是熟字训等无法将读音与汉字对应的无需分隔）。
示例：
输入1：季節は移ろい
正确输出：季節[き,せつ]は移[うつ]ろい
错误输出1：季節[きせつ]は移[うつ]ろい  # 没有使用逗号分隔注音
错误输出2：季[き]節[せつ]は移[うつ]ろい  # 季節没有作为一个整体进行标注

输入2：明日をえがく
正确输入：明日[あした]を描[えが]く
错误输出：明日[あ,した]を描[えが]く  # 明日（あした）是一个整体，不存在音节与单字的对应关系
3. 对于汉字与假名混合的词语，只对汉字部分进行注音标注，不需要为词语的完整性对整个词语进行标注。
示例：
输入：映し出す　気持ちは
输出：映[うつ]し出[だ]す　気持[きも]ちは
错误输出：映し出す[うつしだす]　気持ち[きもち]は
4. 对于片假名词语，不进行注音标注

现在请对以下歌词进行注音标注："""

_FENCE = re.compile(r"```[^\n]*\n(?P<body>.*?)\n?```", re.DOTALL)


def text_key(text: str) -> str:
    """A short key for a `.krc` text, so the times stored beside it can tell when it changed."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def path_for(project: str | Path) -> Path:
    """A suggested KRC export path beside a project file."""
    return Path(project).with_suffix(SUFFIX)


def load(path: str | Path) -> str:
    """The `.krc` text; a missing or unreadable file is an empty one."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def save(path: str | Path, text: str) -> Path:
    """Write the whole file at once, so a half-written `.krc` never exists to be read."""
    return write_text(path, text)


def build_prompt(lyrics: str, prompt: str = DEFAULT_PROMPT, retry: str = "") -> str:
    """The prompt that goes to a model, with the text to annotate under it.

    `retry` is what an earlier refusal said, appended so a second try knows what to fix.
    """
    text = f"{prompt}\n{lyrics}"
    return f"{text}\n\n{retry}" if retry else text


def retry_prompt(answer: str, error: str) -> str:
    """What a rejected answer earns: the model's own output again, and why it was refused."""
    return (
        f"你上一次的输出：\n{answer}\n\n"
        f"它没有通过检查：{error}\n"
        "请严格按上面的规则重新注音，只输出修正后的歌词本身，不要任何解释。"
    )


def conversion_error(text: str, source: str) -> str:
    """Why `text` is not `source` with the readings added, or "" when it is.

    The model is asked for the readings alone, never to rewrite, so once the readings and the `.krc`
    markers are taken away the two must carry the same characters; a model that dropped a line,
    changed a word or answered with prose is caught here. Every kanji must also carry a reading: an
    unread one has no sound to align. Whether the readings themselves are right is the user's to
    judge.
    """
    try:
        converted = parse(text)
    except KrcError as error:
        return str(error)
    spoken = "".join(unit.text for chapter in converted.chapters for line in chapter.lines for unit in line.units)
    if _squeezed(spoken) != _squeezed(source):
        return "the answer changed the lyrics instead of annotating them"
    unread = [
        unit.text
        for chapter in converted.chapters
        for line in chapter.lines
        for unit in line.units
        if unit.ruby is None and unit.is_kanji()
    ]
    if unread:
        return f"these kanji have no reading: {' '.join(unread)}"
    return ""


def syntax_error(text: str) -> str:
    """Why `text` is not a readable `.krc`, or "" when it is; empty text is readable."""
    if not text.strip():
        return ""
    try:
        parse(text)
    except KrcError as error:
        return str(error)
    return ""


def _squeezed(text: str) -> str:
    return "".join(char for char in text if not char.isspace())


def convert(
    lyrics: str,
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    retry: str = "",
    temperature: float = 0.2,
    timeout: float = 120.0,
    stream: bool = False,
    on_delta: Callable[[str, str], None] | None = None,
    opener: Callable[..., Any] | None = None,
) -> str:
    """Ask an OpenAI-compatible endpoint to annotate `lyrics`, and return what it wrote.

    `base_url` is the endpoint up to its `/v1`. The key rides in the header and nowhere else, and
    nothing raised here repeats it. `retry` is what an earlier refusal said, sent with the lyrics so
    the model can correct itself. With `stream`, every delta is handed to `on_delta(kind, text)`
    as it arrives - `kind` being `"reasoning"` or `"content"` - and the answer is still returned.
    Without an `opener`, the request goes through the proxy named in the settings.
    """
    body: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{lyrics}\n\n{retry}" if retry else lyrics},
        ],
    }
    if stream:
        body["stream"] = True
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    open_url = opener or net.opener()
    try:
        with open_url(request, timeout=timeout) as response:
            if stream:
                return _read_stream(response, on_delta)
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ValueError(f"the service answered HTTP {error.code} {error.reason}") from None
    except urllib.error.URLError as error:
        raise ValueError(f"the service could not be reached: {error.reason}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"the service answered something that is not JSON: {error}") from None
    return _content(payload)


def _read_stream(response: Any, on_delta: Callable[[str, str], None] | None) -> str:
    """Read an SSE completion as it arrives, handing every delta on before the answer is done."""
    raw: list[str] = []
    pieces: list[str] = []
    found = False
    for line in response:
        text = line.decode("utf-8")
        raw.append(text)
        entry = text.strip()
        if not entry.startswith("data:"):
            continue
        data = entry[5:].strip()
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"]
        except (ValueError, KeyError, IndexError, TypeError):
            continue
        for key, kind in (("reasoning_content", "reasoning"), ("reasoning", "reasoning"), ("content", "content")):
            chunk = delta.get(key) if isinstance(delta, dict) else None
            if not isinstance(chunk, str) or not chunk:
                continue
            found = True
            if kind == "content":
                pieces.append(chunk)
            if on_delta is not None:
                on_delta(kind, chunk)
    if not found:  # an endpoint that ignored the stream and answered whole
        return _content(json.loads("".join(raw)))
    return _unfence("".join(pieces))


def _content(payload: Any) -> str:
    """The message out of a chat completion, whether or not it came wrapped in a code fence."""
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("the service answered without a message") from None
    if not isinstance(content, str):
        raise ValueError("the service answered without a message")
    return _unfence(content)


def _unfence(text: str) -> str:
    """A model that wrapped its answer in a code fence still said what it was asked."""
    text = text.strip()
    match = _FENCE.search(text)
    return match.group("body") if match else text


def editor_command(
    configured: str = "",
    *,
    platform: str | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> list[str]:
    """The program that opens a `.krc`: the setting, else the platform's own choice of editor."""
    if configured.strip():
        return shlex.split(configured)
    system = sys.platform if platform is None else platform
    values = os.environ if environ is None else environ
    if system.startswith("win"):
        for name in ("code", "notepad++", "subl"):
            found = which(name)
            if found:
                return [found]
        return ["notepad"]
    if system == "darwin":
        return ["open"]
    editor = values.get("EDITOR", "").strip()
    return shlex.split(editor) if editor else ["xdg-open"]
