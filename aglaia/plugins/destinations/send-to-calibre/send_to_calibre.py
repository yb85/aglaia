# SPDX-License-Identifier: MIT
"""Send a finished export into a calibre library, over the content server.

calibre's content server exposes the library's own database over HTTP, and
`/cdb/add-book/` is how the web interface's "+" button adds a book. It is a
real API, not a scrape: the reply is JSON carrying the new `book_id`, or the
list of duplicates it refused to create.

    POST {base}{prefix}/cdb/add-book/{job_id}/{add_duplicates}/{filename}/{library_id}

Two things about it surprise people, so they are handled explicitly here
rather than left to fail:

* **The file goes in the raw request body.** Not multipart, not a form field.
  There is no field name to get wrong, and sending multipart gets you a book
  whose contents are a MIME envelope.
* **The parameters are path segments**, so every one of them has to survive
  URL-quoting — a filename with a slash or a space silently reshapes the
  route otherwise.

Auth: the route needs database write access, so the server must run with
`--enable-auth` and the user must not be restricted to a read-only library.
calibre's default auth mode is **digest**; `--auth-mode=basic` is what the
manual recommends behind a TLS reverse proxy. Both are supported, because
guessing wrong produces a 401 that says nothing about which one was expected.

<https://manual.calibre-ebook.com/server.html>
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from urllib.parse import quote

from aglaia.plugin_api import (
    BookMeta, CheckResult, Destination, Field, SendResult, http_client,
    register_destination,
)


@register_destination
class CalibreDestination(Destination):
    name = "send-to-calibre"
    display = "Export to Calibre server"
    description = "Add the export to a calibre content server's library."
    # calibre reads metadata out of the file itself, and happily takes any
    # format it knows. These are the ones Aglaïa produces.
    accepts = ("pdf", "md", "txt")

    CONFIG_FIELDS = (
        Field("base_url", "Server URL", "str",
              "http://127.0.0.1:8080", required=True,
              placeholder="http://127.0.0.1:8080",
              help="Where the content server answers. Include the scheme and "
                   "the port."),
        Field("url_prefix", "URL prefix", "str", "",
              placeholder="/calibre",
              help="Only if the server runs under a path (calibre's "
                   "--url-prefix), e.g. behind a reverse proxy."),
        Field("library_id", "Library", "str", "",
              placeholder="Calibre_Library",
              help="Which library to add to. Empty = the server's default. "
                   "This is the library's id, which is its directory name."),
        Field("auth_mode", "Authentication", "choice", "digest",
              choices=("digest", "basic"),
              help="calibre defaults to digest. Basic is what the manual "
                   "recommends behind an HTTPS reverse proxy."),
        Field("add_duplicates", "Add even if already present", "bool", False,
              help="Off: a book calibre recognises as a duplicate is "
                   "reported as already there and nothing is written."),
        Field("timeout_s", "Timeout (seconds)", "int", 120,
              help="Raise it for large files or a slow link."),
    )
    SECRET_FIELDS = (
        Field("username", "Username", "secret", "", required=True,
              help="A calibre server user with write access to the library."),
        Field("password", "Password", "secret", "", required=True),
    )

    # ── plumbing ──────────────────────────────────────────────────────
    def _root(self) -> str:
        base = str(self.conf("base_url") or "").strip().rstrip("/")
        prefix = str(self.conf("url_prefix") or "").strip().strip("/")
        return f"{base}/{prefix}" if prefix else base

    def _auth(self):
        import httpx
        user, pwd = self.secret("username"), self.secret("password")
        if not user:
            return None
        if str(self.conf("auth_mode")) == "basic":
            return httpx.BasicAuth(user, pwd)
        return httpx.DigestAuth(user, pwd)

    def _client(self):
        # The host's client, not a bare httpx one: it prefers IPv4, which on
        # a network advertising IPv6 without routing it is the difference
        # between 0.05 s and 24 s per request.
        return http_client(float(self.conf("timeout_s") or 120),
                           auth=self._auth())

    # ── check ─────────────────────────────────────────────────────────
    def check(self) -> CheckResult:
        """Three failures with three different fixes, told apart.

        Unreachable means the URL or the server is wrong. 401 means the
        credentials are wrong *or* the auth mode is — calibre answers the same
        way for both, so the message says so instead of picking one. 403 means
        the account exists but cannot write, which is a calibre user setting
        and nothing this plugin can fix."""
        import httpx
        # Settings first, network second. A round trip that can only fail
        # costs the user the wait and then reports "could not connect", which
        # sends them to look at the server instead of at the empty field.
        missing = self.missing_settings()
        if missing:
            return CheckResult(False, "Still needed: " + ", ".join(missing),
                               kind=CheckResult.CONFIG)
        root = self._root()
        if not root:
            return CheckResult(False, "No server URL set.",
                               kind=CheckResult.CONFIG)
        try:
            with self._client() as c:
                r = c.get(f"{root}/ajax/library-info")
        except httpx.HTTPError as e:
            return CheckResult(
                False, f"Could not reach {root}. Is the content server "
                       f"running?",
                {"error": f"{type(e).__name__}: {e}"},
                kind=CheckResult.NETWORK)
        if r.status_code == 401:
            return CheckResult(
                False, "The username or password was rejected — or the server "
                       "expects the other authentication mode.",
                {"status": 401, "auth_mode": self.conf("auth_mode")},
                kind=CheckResult.AUTH)
        if r.status_code == 403:
            return CheckResult(
                False, "Signed in, but this user cannot write to the library. "
                       "Give it write access in calibre's user manager.",
                {"status": 403}, kind=CheckResult.PERMISSION)
        if r.status_code >= 400:
            return CheckResult(False, "calibre could not answer.",
                               {"status": r.status_code},
                               kind=CheckResult.SERVER)
        libs, current = (), ""
        try:
            data = r.json()
            libs = tuple((data.get("library_map") or {}).keys())
            current = str(data.get("default_library") or "")
        except Exception:
            pass
        chosen = str(self.conf("library_id") or "") or current
        if libs and chosen and chosen not in libs:
            return CheckResult(
                False, f"Connected, but there is no library called {chosen}. "
                       f"This server has: {', '.join(libs)}.",
                {"libraries": list(libs)}, kind=CheckResult.CONFIG)
        where = f" — library {chosen}" if chosen else ""
        return CheckResult(True, f"Connected to calibre{where}.",
                           {"libraries": list(libs)})

    # ── send ──────────────────────────────────────────────────────────
    def send(self, path: Path, meta: BookMeta) -> SendResult:
        import httpx
        path = Path(path)
        if not path.is_file():
            return SendResult(False, f"No such file: {path}")
        root = self._root()
        if not root:
            return SendResult(False, "No server URL set.")

        # Every parameter is a PATH SEGMENT, so each is quoted with an empty
        # safe list: an unquoted '/' in a filename would reshape the route and
        # a space would break the request line.
        job_id = uuid.uuid4().hex[:12]
        dupes = "y" if self.conf("add_duplicates") else "n"
        # calibre takes the book's name from this filename, so send the
        # document's title when we have one rather than `project_003_A.pdf`.
        stem = (meta.title or path.stem).strip() or path.stem
        filename = f"{stem}{path.suffix}"
        url = (f"{root}/cdb/add-book/{quote(job_id, safe='')}/"
               f"{quote(dupes, safe='')}/{quote(filename, safe='')}")
        library = str(self.conf("library_id") or "").strip()
        if library:
            url += f"/{quote(library, safe='')}"

        try:
            with self._client() as c:
                r = c.post(url, content=path.read_bytes(),
                           headers={"Content-Type": "application/octet-stream"})
        except httpx.HTTPError as e:
            return SendResult(False, f"Upload failed — {type(e).__name__}: {e}")

        if r.status_code == 401:
            return SendResult(False, "Rejected (401) — check the credentials "
                                     "and the authentication mode.")
        if r.status_code == 403:
            return SendResult(False, "This calibre user may not write to the "
                                     "library.")
        if r.status_code >= 400:
            return SendResult(False, f"calibre answered {r.status_code}: "
                                     f"{r.text[:200]}")
        try:
            data = r.json()
        except json.JSONDecodeError:
            return SendResult(False, "calibre answered with something that is "
                                     "not JSON — is this really a content "
                                     "server?")

        # A duplicate is not a failure. The book is in the library, which is
        # what the user wanted to be true.
        if data.get("duplicates"):
            return SendResult(
                True, f"Already in the library: {filename}", already_there=True,
                detail={"duplicates": data["duplicates"]})
        book_id = data.get("book_id")
        if book_id is None:
            return SendResult(False, f"calibre accepted the request but "
                                     f"returned no book id: {data}")
        return SendResult(
            True, f"Added to calibre as #{book_id}.",
            url=f"{root}/#book_id={book_id}", detail={"book_id": book_id})
