"""Bearer-защищённый HTTP-прокси перед локальным API Ollama."""

from __future__ import annotations

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


TOKEN = os.environ.get("OLLAMA_PROXY_TOKEN", "")
UPSTREAM = os.environ.get("OLLAMA_UPSTREAM", "http://127.0.0.1:11436").rstrip("/")
HOST = os.environ.get("OLLAMA_PROXY_HOST", "0.0.0.0")
PORT = int(os.environ.get("OLLAMA_PROXY_PORT", "11435"))
TIMEOUT = int(os.environ.get("OLLAMA_PROXY_TIMEOUT", "3600"))
MAX_BODY = int(os.environ.get("OLLAMA_PROXY_MAX_BODY", str(128 * 1024 * 1024)))
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate",
              "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade"}


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.0 закрывает соединение после ответа и позволяет передавать SSE
    # по мере генерации, не буферизуя весь результат.
    protocol_version = "HTTP/1.0"

    def _error(self, status: int, message: str, *, auth: bool = False) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if auth:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self) -> None:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {TOKEN}"
        if not hmac.compare_digest(supplied.encode("latin-1"), expected.encode("ascii")):
            self._error(401, "unauthorized", auth=True)
            return

        if self.headers.get("Transfer-Encoding"):
            self._error(501, "chunked request bodies are not supported")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "invalid content length")
            return
        if length < 0 or length > MAX_BODY:
            self._error(413, "request body is too large")
            return
        body = self.rfile.read(length) if length else None

        headers = {
            key: value for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP | {"host", "authorization", "content-length"}
        }
        headers["Accept-Encoding"] = "identity"
        if length:
            headers["Content-Length"] = str(length)
        request = Request(UPSTREAM + self.path, data=body,
                          headers=headers, method=self.command)
        try:
            response = build_opener(ProxyHandler({})).open(request, timeout=TIMEOUT)
        except HTTPError as error:
            response = error
        except (URLError, TimeoutError, OSError):
            self._error(502, "upstream unavailable")
            return

        with response:
            self.send_response(response.status)
            for key, value in response.headers.items():
                if key.lower() not in HOP_BY_HOP | {"content-encoding", "server", "date"}:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            if self.command != "HEAD":
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/event-stream" in content_type or "ndjson" in content_type:
                    while line := response.readline():
                        self.wfile.write(line)
                        self.wfile.flush()
                else:
                    while chunk := response.read(64 * 1024):
                        self.wfile.write(chunk)

    do_GET = _proxy
    do_HEAD = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_PATCH = _proxy
    do_DELETE = _proxy


    def log_message(self, fmt: str, *args: object) -> None:
        # Стандартный журнал не содержит заголовков Authorization.
        super().log_message(fmt, *args)


def main() -> None:
    if len(TOKEN) < 32:
        raise SystemExit("VLM_API_KEY должен содержать не менее 32 символов")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    print(f"Ollama auth proxy listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
