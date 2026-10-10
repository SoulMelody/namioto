# SPDX-License-Identifier: AGPL-3.0-only
"""The editor side of the remote-control interface: a JSON-lines server on the Qt event loop.

`namioto.ipc` owns the protocol, the endpoint and the client; this module only carries them over a
socket. A unix endpoint listens through `QLocalServer`, a TCP one through `QTcpServer`, and both
hand their accepted sockets to the same line-framed `_Connection`, so a method is served exactly
the same way whichever way the caller arrived. The socket permissions are the unix endpoint's whole
authentication (0700 directory, 0600 socket); a TCP one carries a token instead.

Everything runs on the GUI thread, so a handler may touch the window directly. The editor starts
one of these from `namioto.ui.app`, hands it a `WindowBridge` for the methods, and stops it when
the window closes.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from PySide6.QtCore import QObject
from PySide6.QtNetwork import QHostAddress, QLocalServer, QLocalSocket, QTcpServer, QTcpSocket

from namioto import __version__
from namioto.ipc import APP, LINE_LIMIT, PROTOCOL, Dispatcher, Endpoint, decode, encode, error_for, split_tcp


def _chmod(path: str | Path, mode: int) -> None:
    """Best-effort permissions: a filesystem that has none is not a reason to fail to listen."""
    with contextlib.suppress(OSError):
        os.chmod(path, mode)


class _Connection(QObject):
    """One client, its input buffer and the requests it has sent in.

    Framing is one JSON message per newline, so a partial line waits here until the rest of it
    arrives; a line past `LINE_LIMIT` is refused and the socket closed rather than buffered without
    end.
    """

    def __init__(self, socket: QLocalSocket | QTcpSocket, server: IpcServer):
        super().__init__(socket)
        self.socket = socket
        self.server = server
        self.buffer = bytearray()
        self._closed = False
        socket.readyRead.connect(self._read)
        socket.disconnected.connect(self._close)

    def send(self, message: dict) -> None:
        self.socket.write(encode(message))
        self.socket.flush()

    def close(self) -> None:
        self.socket.close()
        self._close()

    def _read(self) -> None:
        data = bytes(self.socket.readAll())
        if not data:
            return
        self.buffer.extend(data)
        while True:
            index = self.buffer.find(b"\n")
            if index < 0:
                break
            line = bytes(self.buffer[:index])
            del self.buffer[: index + 1]
            self._handle(line)
        if len(self.buffer) > LINE_LIMIT:
            self.send(error_for(None, "too_large", "the request line is past the limit"))
            self.close()

    def _handle(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            request = decode(line)
        except (ValueError, UnicodeDecodeError):
            self.send(error_for(None, "bad_request", "a request must be one line of JSON"))
            return
        ident = request.get("id") if isinstance(request, dict) else None
        authorized = not self.server.endpoint.token or (
            isinstance(request, dict) and request.get("token") == self.server.endpoint.token
        )
        if not authorized:
            self.send(error_for(ident, "unauthorized", "the endpoint's token is missing or wrong"))
            self.close()
            return
        self.send(self.server.dispatcher.handle(request))

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.server.forget(self)


class IpcServer(QObject):
    """A listening endpoint with a `Dispatcher`; `broadcast` reaches every client connected."""

    def __init__(self, endpoint: Endpoint, parent=None):
        super().__init__(parent)
        self.requested = endpoint
        self.endpoint = endpoint
        self.dispatcher = Dispatcher()
        self._connections: list[_Connection] = []
        self._local: QLocalServer | None = None
        self._tcp: QTcpServer | None = None

    @property
    def listening(self) -> bool:
        return self._local is not None or self._tcp is not None

    def start(self) -> Endpoint:
        """Open the socket, filling in the address actually bound; raises OSError when it cannot."""
        if self.requested.kind == "unix":
            self._start_unix(Path(self.requested.address))
        else:
            self._start_tcp(self.requested.address)
        return self.endpoint

    def _start_unix(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        _chmod(path.parent, 0o700)
        QLocalServer.removeServer(str(path))  # a socket left by a dead editor is not a live one
        server = QLocalServer(self)
        if not server.listen(str(path)):
            raise OSError(f"cannot listen on {path}: {server.errorString()}")
        _chmod(path, 0o600)
        server.newConnection.connect(self._accept)
        self._local = server
        self.endpoint = Endpoint("unix", str(path), self.requested.token)

    def _start_tcp(self, address: str) -> None:
        host, port = split_tcp(address)
        target = QHostAddress(QHostAddress.SpecialAddress.LocalHost) if host == "localhost" else QHostAddress(host)
        if target.isNull():
            raise OSError(f"{host!r} is not a local address to listen on")
        server = QTcpServer(self)
        if not server.listen(target, port):
            raise OSError(f"cannot listen on {host}:{port}: {server.errorString()}")
        server.newConnection.connect(self._accept)
        self._tcp = server
        self.endpoint = Endpoint(
            "tcp",
            f"{server.serverAddress().toString()}:{server.serverPort()}",
            self.requested.token,
        )

    def _accept(self) -> None:
        source = self._local if self._local is not None else self._tcp
        if source is None:
            return
        while source.hasPendingConnections():
            socket = source.nextPendingConnection()
            if socket is None:
                continue
            connection = _Connection(socket, self)
            self._connections.append(connection)
            connection.send(
                {
                    "event": "hello",
                    "protocol": PROTOCOL,
                    "app": APP,
                    "version": __version__,
                    "pid": os.getpid(),
                    "endpoint": self.endpoint.to_dict(),
                }
            )

    def forget(self, connection: _Connection) -> None:
        if connection in self._connections:
            self._connections.remove(connection)
        connection.deleteLater()

    def broadcast(self, event: str, data=None) -> None:
        """Push an event to every client connected; a server with none does nothing."""
        if not self._connections:
            return
        message: dict = {"event": event}
        if data is not None:
            message["data"] = data
        for connection in list(self._connections):
            connection.send(message)

    def stop(self) -> None:
        for connection in list(self._connections):
            connection.close()
        self._connections.clear()
        if self._local is not None:
            name = self._local.serverName()
            self._local.close()
            QLocalServer.removeServer(name)
            self._local = None
        if self._tcp is not None:
            self._tcp.close()
            self._tcp = None
