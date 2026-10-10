# SPDX-License-Identifier: AGPL-3.0-only
"""The remote-control interface: the wire protocol, where a running editor is found, and the
`namioto-ctl` command line that talks to one.

Qt-free on purpose. The editor wraps this into a `QLocalServer` / `QTcpServer` in
`namioto.ui.ipc_server`, and the client below is its own program, so a script or a remote machine
can drive the editor without importing Qt or anything else.

The protocol is one JSON object per line, UTF-8, with no framing beyond the newline: a request
`{"id": 1, "method": "transport.play", "params": {}}` is answered by
`{"id": 1, "ok": true, "result": {...}}`, or by `{"id": 1, "ok": false, "error": {...}}`. The server
may also send an event, `{"event": "state", "data": {...}}`, to every client connected. A request
carries `token` when the endpoint asks for one, which a unix socket does not and a TCP one does.

`resolve_endpoint` is the one place the two sides agree on where to meet: an explicit address wins,
then `$NAMIOTO_IPC`, then the run file a listening editor left behind, then the per-user socket.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import secrets
import socket
import sys
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from namioto import __version__
from namioto.utils import runtime_dir, write_json

PROTOCOL = 1
APP = "namioto"
ENV_ENDPOINT = "NAMIOTO_IPC"  # the address both sides listen on or connect to
ENV_TOKEN = "NAMIOTO_IPC_TOKEN"  # the token for a TCP endpoint, when it is not in the run file
ENV_RUN = "NAMIOTO_IPC_RUN"  # where the running editor records its endpoint
DEFAULT_SOCKET = "control.sock"
RUN_FILE = "ipc.json"
LINE_LIMIT = 1 << 20  # one line may carry a whole transcription; past this the client is cut off
CONNECT_TIMEOUT = 5.0
DEFAULT_HOST = "127.0.0.1"

# the words `namioto-ctl` accepts in place of a full method name, and the parameter a lone
# positional argument fills in
ALIASES: dict[str, tuple[str, str]] = {
    "play": ("transport.play", ""),
    "pause": ("transport.pause", ""),
    "stop": ("transport.stop", ""),
    "toggle": ("transport.toggle", ""),
    "status": ("state.get", ""),
    "state": ("state.get", ""),
    "notes": ("document.notes", ""),
    "channels": ("channels.list", ""),
    "methods": ("commands", ""),
    "seek": ("transport.seek", "seconds"),
    "bpm": ("transport.set_bpm", "bpm"),
    "speed": ("transport.set_speed", "speed"),
}


class IpcError(Exception):
    """A request that could not be served, with the machine-readable code that says why."""

    def __init__(self, code: str, message: str, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def error_for(ident, code: str, message: str, details: dict | None = None) -> dict:
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return {"id": ident, "ok": False, "error": error}


# --- the network shape of an endpoint ---------------------------------------


@dataclass(frozen=True)
class Endpoint:
    """Where the interface listens: a unix socket path, or `host:port` over TCP."""

    kind: str  # unix or tcp
    address: str
    token: str = ""

    def describe(self) -> str:
        return f"unix://{self.address}" if self.kind == "unix" else f"tcp://{self.address}"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "address": self.address, "token": self.token}

    @classmethod
    def from_dict(cls, data: Any) -> Endpoint | None:
        if not isinstance(data, dict) or data.get("kind") not in ("unix", "tcp"):
            return None
        address = data.get("address")
        if not isinstance(address, str) or not address:
            return None
        token = data.get("token")
        return cls(data["kind"], address, token if isinstance(token, str) else "")


def split_tcp(address: str) -> tuple[str, int]:
    """`host:port` as its two halves; a bare port listens on the loopback."""
    host, _, port = address.rpartition(":")
    if not port.isdigit():
        raise IpcError("bad_endpoint", f"{address!r} is not a host:port address")
    return (host or DEFAULT_HOST), int(port)


def parse_endpoint(value: str, *, token: str = "") -> Endpoint:
    """An address as the command line spells it: `path`, `tcp://host:port`, or a bare name.

    A bare name is a unix socket under the runtime directory, so several editors can sit side by
    side without anyone typing a full path. An empty value is the default per-user socket.
    """
    text = (value or "").strip()
    if text.startswith("unix://"):
        return Endpoint("unix", text[len("unix://") :] or str(default_socket_path()), token)
    if text.startswith("tcp://"):
        return Endpoint("tcp", text[len("tcp://") :] or f"{DEFAULT_HOST}:0", token)
    if not text:
        return Endpoint("unix", str(default_socket_path()), token)
    if text.startswith(("/", "./", "../", "~")) or "/" in text or "\\" in text:
        return Endpoint("unix", str(Path(text).expanduser()), token)
    if ":" in text and text.rsplit(":", 1)[1].isdigit():
        return Endpoint("tcp", text, token)
    return Endpoint("unix", str(runtime_dir(text)), token)


def default_socket_path() -> Path:
    return runtime_dir(DEFAULT_SOCKET)


def default_endpoint() -> Endpoint:
    return Endpoint("unix", str(default_socket_path()))


def run_path() -> Path:
    """Where a listening editor leaves its endpoint for the client to find."""
    from_env = os.environ.get(ENV_RUN)
    return Path(from_env) if from_env else runtime_dir(RUN_FILE)


def write_run(endpoint: Endpoint, pid: int | None = None) -> Path:
    """Record the live endpoint, so a client started with no address of its own can find it."""
    return write_json(
        {
            "protocol": PROTOCOL,
            "app": APP,
            "version": __version__,
            "pid": os.getpid() if pid is None else pid,
            "endpoint": endpoint.to_dict(),
        },
        run_path(),
    )


def read_run() -> tuple[Endpoint, int] | None:
    """The endpoint a running editor recorded, or None when there is none to trust.

    A record whose process is gone is stale and is ignored: the socket file may still be there, but
    nothing will answer on it.
    """
    try:
        data = json.loads(run_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("protocol") != PROTOCOL:
        return None
    endpoint = Endpoint.from_dict(data.get("endpoint"))
    pid = data.get("pid")
    if endpoint is None or not isinstance(pid, int) or pid <= 0:
        return None
    if pid != os.getpid() and not _alive(pid):
        return None
    return endpoint, pid


def remove_run(pid: int | None = None) -> None:
    """Clear the record on the way out, but only when it is still this process's own."""
    target = run_path()
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if pid is not None and data.get("pid") != pid:
        return
    with contextlib.suppress(OSError):
        target.unlink()


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # another user's process: alive as far as this one can tell
        return True
    return True


def resolve_endpoint(
    address: str | None = None,
    *,
    token: str | None = None,
    discover: bool = True,
) -> Endpoint:
    """The endpoint to use, from the address, the environment, the run file, or the default.

    `discover` is off for a listener, which should not adopt another editor's endpoint: it decides
    where to listen, not where to connect.
    """
    chosen = token if token is not None else os.environ.get(ENV_TOKEN, "")
    if address:
        return parse_endpoint(address, token=chosen)
    from_env = os.environ.get(ENV_ENDPOINT, "")
    if from_env:
        return parse_endpoint(from_env, token=chosen)
    if discover:
        found = read_run()
        if found is not None:
            endpoint, _pid = found
            if endpoint.token and not chosen:
                return endpoint
            return Endpoint(endpoint.kind, endpoint.address, chosen)
    return default_endpoint()


def listen_endpoint(address: str | None = None, *, token: str | None = None) -> Endpoint:
    """Where a new server should listen, with a token generated for a TCP one that names none."""
    endpoint = resolve_endpoint(address, token=token, discover=False)
    chosen = token if token is not None else os.environ.get(ENV_TOKEN, "")
    if endpoint.kind == "tcp" and not chosen:
        chosen = secrets.token_hex(16)
    return Endpoint(endpoint.kind, endpoint.address, chosen)


# --- the protocol messages --------------------------------------------------


def encode(message: dict) -> bytes:
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line: bytes | str) -> dict:
    """One line as a message; anything that is not a JSON object is a protocol error."""
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    data = json.loads(line)
    if not isinstance(data, dict):
        raise ValueError("a message must be a JSON object")
    return data


def unwrap(reply: dict) -> Any:
    """The result of a response, or the error it carries raised for the caller."""
    if reply.get("ok"):
        return reply.get("result")
    error = reply.get("error")
    if not isinstance(error, dict):
        raise IpcError("failed", "the server returned no error")
    raise IpcError(str(error.get("code", "failed")), str(error.get("message", "")), error.get("details"))


# --- dispatching a request --------------------------------------------------


@dataclass(frozen=True)
class Command:
    name: str
    summary: str
    params: dict[str, str]
    handler: Callable[[dict], Any]


class Dispatcher:
    """The methods a server offers, each a plain callable over a parameter dict.

    It knows nothing about sockets or Qt: the server hands it a decoded request and writes back what
    it returns, and `commands` lets a client discover what is there. A handler raises `IpcError` to
    name its own failure; anything else it raises becomes a `failed` error.
    """

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(
        self,
        name: str,
        handler: Callable[[dict], Any],
        *,
        summary: str = "",
        params: dict[str, str] | None = None,
    ) -> Callable[[dict], Any]:
        self._commands[name] = Command(name, summary, dict(params or {}), handler)
        return handler

    def describe(self) -> list[dict]:
        return [
            {"method": command.name, "summary": command.summary, "params": command.params}
            for command in sorted(self._commands.values(), key=lambda item: item.name)
        ]

    def handle(self, request: Any) -> dict:
        if not isinstance(request, dict):
            return error_for(None, "bad_request", "a request must be a JSON object")
        ident = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method:
            return error_for(ident, "bad_request", "the request names no method")
        params = request.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return error_for(ident, "bad_params", "params must be a JSON object")
        command = self._commands.get(method)
        if command is None:
            return error_for(
                ident,
                "unknown_method",
                f"no method named {method!r}",
                {"methods": sorted(self._commands)},
            )
        try:
            return {"id": ident, "ok": True, "result": command.handler(params)}
        except IpcError as error:
            return error_for(ident, error.code, error.message, error.details)
        except Exception as error:  # a handler's own bug must come back as an answer, not a crash
            return error_for(ident, "failed", f"{type(error).__name__}: {error}")


# --- the client -------------------------------------------------------------


class Client:
    """A blocking JSON-lines client over a unix socket or TCP.

    One call opens the connection; `request` sends one method and waits for its answer, skipping
    (and keeping) any event that arrives in between, and `events` yields those and everything the
    server pushes afterwards.
    """

    def __init__(self, endpoint: Endpoint, timeout: float = CONNECT_TIMEOUT):
        self.endpoint = endpoint
        self.timeout = timeout
        self._socket: socket.socket | None = None
        self._stream: Any = None
        self._id = 0
        self._events: deque[dict] = deque()

    def connect(self) -> Client:
        if self._socket is not None:
            return self
        if self.endpoint.kind == "unix":
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(self.timeout)
            connection.connect(self.endpoint.address)
        else:
            host, port = split_tcp(self.endpoint.address)
            connection = socket.create_connection((host, port), timeout=self.timeout)
        self._socket = connection
        self._stream = connection.makefile("rwb")
        return self

    def close(self) -> None:
        stream, self._stream = self._stream, None
        connection, self._socket = self._socket, None
        if stream is not None:
            stream.close()
        if connection is not None:
            connection.close()

    def request(self, method: str, params: dict | None = None, *, timeout: float | None = None) -> Any:
        self.connect()
        self._id += 1
        ident = self._id
        message: dict = {"id": ident, "method": method}
        if params:
            message["params"] = params
        if self.endpoint.token:
            message["token"] = self.endpoint.token
        assert self._socket is not None and self._stream is not None
        previous = self._socket.gettimeout()
        if timeout is not None:
            self._socket.settimeout(timeout)
        try:
            self._stream.write(encode(message))
            self._stream.flush()
            while True:
                line = self._stream.readline()
                if not line:
                    raise IpcError("closed", "the server closed the connection")
                reply = decode(line)
                if "event" in reply:
                    self._events.append(reply)
                    continue
                if reply.get("id") != ident:
                    continue  # a late answer to a request that already timed out
                return unwrap(reply)
        except TimeoutError as error:
            raise IpcError("timeout", f"{method} did not answer in time") from error
        finally:
            if timeout is not None:
                self._socket.settimeout(previous)

    def events(self, timeout: float | None = None) -> Iterator[dict]:
        """Every event the server sends, from now on; blocks between them.

        `timeout` bounds one wait, after which `None` is yielded and the loop carries on, so a caller
        can stop the stream without closing the socket.
        """
        self.connect()
        while self._events:
            yield self._events.popleft()
        assert self._socket is not None and self._stream is not None
        previous = self._socket.gettimeout()
        self._socket.settimeout(timeout)
        try:
            while True:
                try:
                    line = self._stream.readline()
                except TimeoutError:
                    yield None
                    continue
                if not line:
                    return
                try:
                    message = decode(line)
                except ValueError:
                    continue
                if "event" in message:
                    yield message
        finally:
            self._socket.settimeout(previous)

    def __enter__(self) -> Client:
        return self.connect()

    def __exit__(self, *_error) -> None:
        self.close()


# --- the command line -------------------------------------------------------


def _value(text: str) -> Any:
    """A command-line value with its JSON type when it has one: `132`, `0.25`, `[1, 2]`."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def _parse_params(pairs: list[str]) -> dict:
    """`NAME=VALUE` arguments as a parameter dict; a value that parses as JSON keeps its type."""
    params: dict[str, Any] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name:
            raise ValueError(f"{pair!r} is not a NAME=VALUE parameter")
        params[name] = _value(value)
    return params


def _resolve_call(args: argparse.Namespace) -> tuple[str, dict]:
    method = args.method
    positional = [item for item in args.params if "=" not in item]
    if positional:
        if len(positional) != len(args.params):
            raise ValueError("positional values and NAME=VALUE parameters cannot be mixed")
        if len(positional) > 1:
            raise ValueError("give at most one positional value")
        name = ALIASES.get(method, ("", ""))[1]
        if not name:
            raise ValueError(f"{method} takes no positional argument; use NAME=VALUE")
        params = {name: _value(positional[0])}
    else:
        params = _parse_params(args.params)
    return ALIASES.get(method, (method,))[0], params


def _print(value: Any, compact: bool) -> None:
    if compact:
        print(json.dumps(value, ensure_ascii=False, separators=(",", ":")), flush=True)
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="namioto-ctl",
        description="Remote control for a running Namioto editor.",
    )
    parser.add_argument(
        "method",
        nargs="?",
        default="state.get",
        help="a method name (see `namioto-ctl methods`), or one of "
        + ", ".join(sorted(ALIASES))
        + " (default: state.get)",
    )
    parser.add_argument(
        "params",
        nargs="*",
        help="NAME=VALUE parameters, values read as JSON when they parse (notes='[...]')",
    )
    parser.add_argument("-e", "--endpoint", default=None, help=f"address to connect to; overrides ${ENV_ENDPOINT}")
    parser.add_argument("--token", default=None, help=f"token for a TCP endpoint; overrides ${ENV_TOKEN}")
    parser.add_argument(
        "--run",
        default=None,
        help=f"the run file a listening editor wrote; overrides ${ENV_RUN}",
    )
    parser.add_argument("-t", "--timeout", type=float, default=CONNECT_TIMEOUT, help="seconds to wait for an answer")
    parser.add_argument("-c", "--compact", action="store_true", help="print the result on one line")
    parser.add_argument("-w", "--watch", action="store_true", help="keep reading the server's events after the answer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.run:
        os.environ[ENV_RUN] = args.run
    try:
        method, params = _resolve_call(args)
    except ValueError as error:
        print(f"namioto-ctl: {error}", file=sys.stderr)
        return 2
    endpoint = resolve_endpoint(args.endpoint, token=args.token)
    try:
        with Client(endpoint) as client:
            result = client.request(method, params, timeout=args.timeout)
            _print(result, args.compact)
            if args.watch:
                for event in client.events():
                    if event is not None:
                        _print(event, args.compact)
    except IpcError as error:
        if error.code == "closed" and args.endpoint is None:
            print(f"namioto-ctl: no editor is listening on {endpoint.describe()}", file=sys.stderr)
        else:
            print(f"namioto-ctl: {error.message} ({error.code})", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"namioto-ctl: cannot reach {endpoint.describe()}: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
