"""
A static file server with HTTP range requests, for local testing of cloud-native files.

Python's built-in ``http.server`` ignores ``Range`` headers, so cloud-native
readers (GDAL ``/vsicurl/``, PyArrow/fsspec, DuckDB ``httpfs``, PMTiles and COG
viewers in the browser) either fail on it or download whole files. This server
answers single ``bytes=a-b`` ranges with ``206 Partial Content``, adds CORS
headers so browser maps can read from it, and counts the requests and bytes it
serves, which makes it easy to measure how much a cloud-native read transfers.

Example:
    from geolattice.web import static_server

    with static_server("outputs") as srv:
        print(srv.url)                                   # http://127.0.0.1:PORT
        gdf = read_geoparquet(srv.url + "/unions.parquet", bbox=(90.35, 23.7, 90.45, 23.82))
        print(srv.stats)                                 # {'requests': ..., 'bytes': ...}

Readers that hold Python's GIL while waiting for the network, notably
``rasterio.open`` (rasterio ships its own GDAL), deadlock against a server
running in the same process. Use ``static_server(folder, subprocess=True)``
for them: the server then runs in a child process with the same interface.
"""

from __future__ import annotations

import functools
import http.server
import json
import os
import re
import subprocess as _subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union

_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


class _RangeHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files with single-range support, CORS and byte counting."""

    server: "_CountingServer"

    def log_message(self, *args: Any) -> None:  # keep notebooks quiet
        if self.server.verbose:
            super().log_message(*args)

    def handle(self) -> None:
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):  # clients cancel read-ahead requests
            pass

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, ETag")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # CORS pre-flight for Range requests from browsers
        self.send_response(204)
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
        self.end_headers()

    def send_head(self) -> Any:
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        header = (self.headers.get("Range") or "").strip()
        match = _RANGE.match(header) if header else None
        if header and not match:
            # Multi-range or malformed: refuse rather than silently sending the whole file.
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        handle = open(path, "rb")
        if match:
            first, last = match.groups()
            if first == "":
                start, end = max(size - int(last or 0), 0), size - 1
            else:
                start, end = int(first), min(int(last) if last else size - 1, size - 1)
            if start >= size or start > end:
                handle.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            handle.seek(start)
            self._remaining = end - start + 1
        else:
            self.send_response(200)
            self._remaining = size
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(self._remaining))
        stat = os.stat(path)
        self.send_header("ETag", f'"{stat.st_mtime_ns:x}-{size:x}"')
        self.end_headers()
        return handle

    def copyfile(self, source: Any, destination: Any) -> None:
        # Count before writing: a client may check the counters as soon as it has the last byte.
        remaining = getattr(self, "_remaining", None)
        self.server.count(0, request=True)
        while remaining is None or remaining > 0:
            chunk = source.read(65536 if remaining is None else min(65536, remaining))
            if not chunk:
                break
            self.server.count(len(chunk))
            destination.write(chunk)
            if remaining is not None:
                remaining -= len(chunk)


class _CountingServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, *args: Any, verbose: bool = False, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.verbose = verbose
        self._lock = threading.Lock()
        self.requests = 0
        self.bytes = 0

    def count(self, sent: int, request: bool = False) -> None:
        with self._lock:
            self.requests += int(request)
            self.bytes += sent


class StaticServer:
    """A running :func:`static_server`; use it as a context manager or call :meth:`stop`."""

    def __init__(self, directory: Union[str, Path], host: str, port: int, verbose: bool) -> None:
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(f"{self.directory} is not a directory")
        handler = functools.partial(_RangeHandler, directory=str(self.directory))
        self._server = _CountingServer((host, port), handler, verbose=verbose)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.host, self.port = self._server.server_address[:2]

    @property
    def url(self) -> str:
        """Base URL, e.g. ``http://127.0.0.1:8765``."""
        return f"http://{self.host}:{self.port}"

    @property
    def stats(self) -> Dict[str, int]:
        """Requests answered and bytes sent since the last :meth:`reset`."""
        return {"requests": self._server.requests, "bytes": self._server.bytes}

    def reset(self) -> None:
        """Set the request and byte counters to zero."""
        with self._server._lock:
            self._server.requests = 0
            self._server.bytes = 0

    def stop(self) -> None:
        """Stop serving and release the port."""
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> "StaticServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"StaticServer({self.url} → {self.directory})"


_CHILD = """
import json, sys
from geolattice.web.static import StaticServer
srv = StaticServer(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1")
print(json.dumps({"host": srv.host, "port": srv.port}), flush=True)
try:
    for line in sys.stdin:
        command = line.strip()
        if command == "reset":
            srv.reset()
        print(json.dumps(srv.stats), flush=True)
except KeyboardInterrupt:                # the parent was interrupted; exit quietly
    pass
"""


class SubprocessStaticServer:
    """A :class:`StaticServer` running in a child process (same ``url``, ``stats``, ``reset``, ``stop``)."""

    def __init__(self, directory: Union[str, Path], host: str, port: int, verbose: bool) -> None:
        self.directory = Path(directory).resolve()
        if not self.directory.is_dir():
            raise NotADirectoryError(f"{self.directory} is not a directory")
        self._proc = _subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(self.directory), host, str(port), "1" if verbose else "0"],
            stdin=_subprocess.PIPE, stdout=_subprocess.PIPE, text=True)
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("The static server process did not start.")
        address = json.loads(line)
        self.host, self.port = address["host"], address["port"]
        self._lock = threading.Lock()

    def _ask(self, command: str) -> Dict[str, int]:
        with self._lock:
            self._proc.stdin.write(command + "\n")
            self._proc.stdin.flush()
            return json.loads(self._proc.stdout.readline())

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def stats(self) -> Dict[str, int]:
        """Requests answered and bytes sent since the last :meth:`reset`."""
        return self._ask("stats")

    def reset(self) -> None:
        self._ask("reset")

    def stop(self) -> None:
        if self._proc.poll() is None:
            self._proc.terminate()
            self._proc.wait(timeout=10)

    def __enter__(self) -> "SubprocessStaticServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"SubprocessStaticServer({self.url} → {self.directory})"


def static_server(directory: Union[str, Path] = ".", *, host: str = "127.0.0.1", port: int = 0,
                  verbose: bool = False, subprocess: bool = False) -> Union[StaticServer, SubprocessStaticServer]:
    """
    Serve a folder over HTTP with range requests, CORS and transfer counters.

    Args:
        directory: Folder to serve.
        host: Interface; ``"0.0.0.0"`` exposes it on the network.
        port: Port; ``0`` picks a free one (see ``server.url``).
        verbose: Log every request.
        subprocess: Run the server in a child process. Needed when the client in this
            process holds the GIL during network reads (``rasterio``), which would deadlock
            against an in-process server.

    Returns:
        A running server (a context manager) with ``url``, ``stats``, ``reset()`` and ``stop()``.
    """
    if subprocess:
        return SubprocessStaticServer(directory, host, port, verbose)
    return StaticServer(directory, host, port, verbose)


__all__ = ["StaticServer", "SubprocessStaticServer", "static_server"]
