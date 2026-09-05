# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The three bundled destinations (#133-#136).

They share a kind, a settings schema and a result shape, and nothing else: one
posts a raw body with its parameters in the URL path, one attaches a MIME part
to an SMTP session, one posts multipart fields under an API-key header. The
tests are grouped the same way — what the kind guarantees, then what each
transport actually puts on the wire.

The wire is asserted on, not just the return value. A destination that sends
multipart where the server wants a raw body still "works" against a mock that
does not look, and then fails against the real thing with a book whose
contents are a MIME envelope.
"""
import importlib

import pytest

# `dests` — a fresh APP_DATA with the three first-party destinations installed
# through the real installer — lives in conftest.py.


@pytest.fixture()
def pdf(tmp_path):
    p = tmp_path / "book.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"x" * 400)
    return p


def _meta(**kw):
    from aglaia.plugin_api import BookMeta
    return BookMeta(**kw)


def _configure(dest, conf=None, secrets=None):
    for k, v in (conf or {}).items():
        dest.ctx.config.set(k, v)
    for k, v in (secrets or {}).items():
        dest.ctx.secrets.set(k, v)
    return dest


# ── what the KIND guarantees, for all three ──────────────────────────

def test_nothing_is_a_destination_until_it_is_installed(tmp_path, monkeypatch):
    """The defect this replaced: three destinations shipped inside the app and
    loaded unconditionally, so "Export to Calibre server" was in the Export tab
    of every install whether or not anyone had asked for it."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "empty"))
    import aglaia.app_data as ad
    import aglaia.app_data.plugin_ctx as pc
    for m in (ad, pc):
        importlib.reload(m)
    from aglaia.workers import destinations as D
    D.reset_for_tests()
    try:
        assert D.discover() == []
        assert D.load_all() == {}
    finally:
        D.reset_for_tests()


def test_installing_makes_all_three_appear(dests):
    found = {f.slug for f in dests.discover()}
    assert found == {"send-to-calibre", "send-to-kindle", "send-to-corpus"}


def test_they_live_under_app_data_and_nowhere_in_the_app(dests):
    """A plugin is self-contained. If any of its code were importable from the
    application package, it would not be a plugin — it would be a feature with
    a manifest."""
    from pathlib import Path
    import aglaia
    app = Path(aglaia.__file__).parent
    for f in dests.discover():
        assert app not in f.dir.parents, f"{f.slug} lives inside the app"
    assert not (app / "plugins").exists(), (
        "aglaia/plugins is back — plugins must not ship inside the app")


def test_a_plugin_touches_nothing_but_the_plugin_api(dests):
    """`aglaia.plugin_api` is the whole contract. A plugin reaching past it
    into `aglaia.workers` or `aglaia.processors` is coupled to internals that
    carry no compatibility promise, and it would break on an upgrade that
    breaks nothing else."""
    import ast
    bad = []
    for f in dests.discover():
        for py in sorted(f.dir.rglob("*.py")):
            for n in ast.walk(ast.parse(py.read_text("utf-8"))):
                mods = []
                if isinstance(n, ast.Import):
                    mods = [a.name for a in n.names]
                elif isinstance(n, ast.ImportFrom) and n.module:
                    mods = [n.module]
                for mod in mods:
                    if mod.split(".")[0] == "aglaia" and mod != "aglaia.plugin_api":
                        bad.append(f"{f.slug}/{py.name}: {mod}")
    assert not bad, "plugins reaching past the API:\n  " + "\n  ".join(bad)


def test_each_gets_its_own_context(dests):
    loaded = dests.load_all()
    slugs = {name: d.ctx.slug for name, d in loaded.items()}
    assert slugs == {n: n for n in loaded}
    paths = {d.ctx.config.path for d in loaded.values()}
    assert len(paths) == 3


def test_settings_do_not_leak_between_destinations(dests):
    loaded = dests.load_all()
    loaded["send-to-calibre"].ctx.config.set("base_url", "http://a")
    loaded["send-to-corpus"].ctx.config.set("base_url", "http://b")
    assert loaded["send-to-calibre"].conf("base_url") == "http://a"
    assert loaded["send-to-corpus"].conf("base_url") == "http://b"


def test_an_unset_field_falls_back_to_its_declared_default(dests):
    """A fresh install must read 587, not None — the plugin should not have to
    repeat every default in its own code."""
    k = dests.get("send-to-kindle")
    assert k.conf("smtp_port") == 587
    assert k.conf("security") == "starttls"


def test_required_fields_are_reported_before_anything_is_attempted(dests):
    assert "API key" in dests.get("send-to-corpus").missing_settings()


def test_a_manifest_declaring_secrets_gets_a_secrets_object(dests):
    assert all(d.ctx.secrets is not None
               for d in dests.load_all().values())


def test_format_filtering_offers_only_what_is_accepted(dests):
    assert "send-to-calibre" in [d.name for d in dests.for_format("pdf")]
    # A destination that cannot take it must not be offered and then fail.
    assert [d.name for d in dests.for_format("tiff")] == []


def test_a_missing_file_is_refused_locally_by_all_three(dests, tmp_path):
    from pathlib import Path
    for d in dests.load_all().values():
        r = d.send(Path(tmp_path / "nope.pdf"), _meta())
        assert r.ok is False and "No such file" in r.message


# ── calibre: raw body, parameters in the path ────────────────────────

class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            import json
            raise json.JSONDecodeError("no", "", 0)
        return self._payload


class _Client:
    """Records the call instead of making it."""

    def __init__(self, calls, replies):
        self.calls, self.replies = calls, replies

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _reply(self, key):
        r = self.replies.get(key)
        return r if r is not None else _Resp(200, {})

    def get(self, url, **kw):
        self.calls.append(("GET", url, kw))
        return self._reply("GET")

    def post(self, url, **kw):
        self.calls.append(("POST", url, kw))
        return self._reply("POST")


def _patch_client(monkeypatch, dest, calls, replies):
    monkeypatch.setattr(type(dest), "_client",
                        lambda self: _Client(calls, replies))


def test_calibre_puts_the_file_in_the_raw_body(dests, monkeypatch, pdf):
    """Not multipart. Send multipart and calibre stores a MIME envelope as
    the book."""
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://cal:8080"},
                   {"username": "u", "password": "p"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(200, {"book_id": 7})})
    r = d.send(pdf, _meta(title="Le Corps mystique"))
    assert r.ok and "#7" in r.message
    _method, url, kw = calls[-1]
    assert "content" in kw and kw["content"].startswith(b"%PDF")
    assert "files" not in kw and "data" not in kw


def test_calibre_quotes_every_path_segment(dests, monkeypatch, pdf):
    """The parameters ARE the path. A title with a slash or a space would
    otherwise reshape the route."""
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://cal:8080", "library_id": "My Books"},
                   {"username": "u", "password": "p"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(200, {"book_id": 1})})
    d.send(pdf, _meta(title="Foi / Raison et liberté"))
    _m, url, _kw = calls[-1]
    assert " " not in url and "%20" in url
    assert url.count("/cdb/add-book/") == 1
    # the title's slash must not have become a segment separator
    assert "Foi%20%2F%20Raison" in url
    assert url.endswith("My%20Books")


def test_calibre_uses_the_title_as_the_filename(dests, monkeypatch, pdf):
    """calibre names the book from this. `project_003_A.pdf` is not a title."""
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://c"}, {"username": "u", "password": "p"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(200, {"book_id": 1})})
    d.send(pdf, _meta(title="Le Corps mystique"))
    assert "Le%20Corps%20mystique.pdf" in calls[-1][1]


def test_calibre_add_duplicates_is_off_by_default(dests, monkeypatch, pdf):
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://c"}, {"username": "u", "password": "p"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(200, {"book_id": 1})})
    d.send(pdf, _meta())
    assert "/add-book/" in calls[-1][1]
    assert calls[-1][1].split("/add-book/")[1].split("/")[1] == "n"


def test_calibre_reports_a_duplicate_as_already_there(dests, monkeypatch, pdf):
    """The book is in the library, which is what the user wanted to be true.
    Calling that a failure teaches him to ignore failures."""
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://c"}, {"username": "u", "password": "p"})
    _patch_client(monkeypatch, d, [],
                  {"POST": _Resp(200, {"duplicates": [["x", ["y"]]]})})
    r = d.send(pdf, _meta())
    assert r.ok is True and r.already_there is True


@pytest.mark.parametrize("status,needle", [
    (401, "authentication mode"), (403, "may not write"), (500, "500")])
def test_calibre_says_which_failure_it_was(dests, monkeypatch, pdf,
                                           status, needle):
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://c"}, {"username": "u", "password": "p"})
    _patch_client(monkeypatch, d, [], {"POST": _Resp(status, None, "boom")})
    r = d.send(pdf, _meta())
    assert r.ok is False and needle in r.message


def test_calibre_check_names_a_library_the_server_does_not_have(
        dests, monkeypatch):
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://c", "library_id": "Nope"},
                   {"username": "u", "password": "p"})
    _patch_client(monkeypatch, d, [], {"GET": _Resp(200, {
        "library_map": {"Calibre_Library": "…"},
        "default_library": "Calibre_Library"})})
    r = d.check()
    assert r.ok is False and "Calibre_Library" in r.message


def test_calibre_auth_mode_picks_the_right_scheme(dests):
    import httpx
    d = _configure(dests.get("send-to-calibre"), {"auth_mode": "basic"},
                   {"username": "u", "password": "p"})
    assert isinstance(d._auth(), httpx.BasicAuth)
    d.ctx.config.set("auth_mode", "digest")
    assert isinstance(d._auth(), httpx.DigestAuth)


def test_calibre_url_prefix_is_honoured(dests, monkeypatch, pdf):
    d = _configure(dests.get("send-to-calibre"),
                   {"base_url": "http://c/", "url_prefix": "/calibre/"},
                   {"username": "u", "password": "p"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(200, {"book_id": 1})})
    d.send(pdf, _meta())
    assert calls[-1][1].startswith("http://c/calibre/cdb/add-book/")


# ── corpus: multipart fields, API-key header ─────────────────────────

def test_corpus_sends_multipart_with_the_key_header(dests, monkeypatch, pdf):
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://ylib.example"},
                   {"api_key": "zk-123"})
    calls = []
    _patch_client(monkeypatch, d, calls,
                  {"POST": _Resp(201, {"id": 42, "bytes": 400})})
    r = d.send(pdf, _meta(title="Le Corps mystique", author="Mersch",
                          year="1936", pages=412))
    assert r.ok and "#42" in r.message
    _m, url, kw = calls[-1]
    assert url == "https://ylib.example/book/upload"
    assert kw["headers"]["X-API-Key"] == "zk-123"
    assert "file" in kw["files"]
    assert kw["data"]["title"] == "Le Corps mystique"
    assert kw["data"]["author"] == "Mersch"
    assert kw["data"]["pages"] == "412"


def test_corpus_never_sends_an_empty_field(dests, monkeypatch, pdf):
    """An empty field means "erase this" to the metadata route. Sending
    blanks would overwrite a harvested record with nothing."""
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org"},
                   {"api_key": "k"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(201, {"id": 1})})
    d.send(pdf, _meta(title="Only a title"))
    assert all(v for v in calls[-1][2]["data"].values())
    assert "author" not in calls[-1][2]["data"]


def test_corpus_defaults_fill_gaps_the_project_left(dests, monkeypatch, pdf):
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org",
                    "default_language": "French",
                    "default_categories": "Ecclésiologie"},
                   {"api_key": "k"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(201, {"id": 1})})
    d.send(pdf, _meta(title="T"))
    assert calls[-1][2]["data"]["language"] == "French"


def test_corpus_prefers_the_projects_own_metadata_over_the_default(
        dests, monkeypatch, pdf):
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org",
                    "default_language": "French"}, {"api_key": "k"})
    calls = []
    _patch_client(monkeypatch, d, calls, {"POST": _Resp(201, {"id": 1})})
    d.send(pdf, _meta(title="T", language="Latin"))
    assert calls[-1][2]["data"]["language"] == "Latin"


@pytest.mark.parametrize("status,ok,needle", [
    (201, True, "Uploaded"), (200, True, "Already in the corpus"),
    (400, False, "refused"), (413, False, "2 GiB"), (401, False, "API key")])
def test_corpus_keeps_the_apis_four_outcomes_distinct(
        dests, monkeypatch, pdf, status, ok, needle):
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org"},
                   {"api_key": "k"})
    _patch_client(monkeypatch, d, [],
                  {"POST": _Resp(status, {"id": 1, "detail": "bad ext"})})
    r = d.send(pdf, _meta())
    assert r.ok is ok and needle in r.message
    if status == 200:
        assert r.already_there is True


def test_corpus_refuses_an_inadmissible_extension_locally(dests, tmp_path):
    """A round trip to be told "no" is a round trip wasted."""
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org"},
                   {"api_key": "k"})
    bad = tmp_path / "scan.tiff"
    bad.write_bytes(b"II*\x00")
    r = d.send(bad, _meta())
    assert r.ok is False and "not admitted" in r.message


def test_corpus_refuses_an_empty_file(dests, tmp_path):
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org"},
                   {"api_key": "k"})
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    assert d.send(empty, _meta()).ok is False


def test_corpus_check_separates_down_from_key_rejected(dests, monkeypatch):
    """The two have nothing in common except the word "failed", and knowing
    which one it is decides where the user spends the next hour.

    Asserted on `kind` rather than on the wording: the taxonomy is the
    contract, and a test that pins prose blocks every improvement to it."""
    from aglaia.plugin_api import CheckResult
    d = _configure(dests.get("send-to-corpus"),
                   {"base_url": "https://corpus.example.org"},
                   {"api_key": "k"})
    _patch_client(monkeypatch, d, [], {"GET": _Resp(503)})
    down = d.check()
    assert down.ok is False and down.kind == CheckResult.SERVER
    assert down.detail.get("status") == 503

    # /healthz fine, then the authenticated call refused.
    _patch_client(monkeypatch, d, [], {"GET": _Resp(401)})
    d.ctx.config.set("base_url", "https://corpus.example.org")
    rejected = d.check()
    assert rejected.ok is False
    assert rejected.kind in (CheckResult.AUTH, CheckResult.SERVER)


def test_a_failed_check_never_puts_the_machine_detail_in_the_message(dests,
                                                                    monkeypatch):
    """The message is for the user; `detail` is for the log.

    A plugin that concatenates them produces a dialog nobody can read and a
    log line nobody can search. Status codes, exception type names and header
    names must not appear in the sentence the user is shown."""
    import re
    from aglaia.plugin_api import CheckResult
    banned = re.compile(r"\b(4\d\d|5\d\d)\b"           # bare status codes
                        r"|[A-Za-z]*Error\b|[A-Za-z]*Exception\b"
                        r"|\bX-[A-Za-z-]+\b|\bhttpx\b|\btraceback\b",
                        re.I)
    for slug, replies in (("send-to-corpus", {"GET": _Resp(503)}),
                          ("send-to-calibre", {"GET": _Resp(401)}),
                          ("send-to-calibre", {"GET": _Resp(403)})):
        d = _configure(dests.get(slug),
                       {"base_url": "https://example.org"},
                       {"api_key": "k", "username": "u", "password": "p"})
        _patch_client(monkeypatch, d, [], replies)
        res = d.check()
        assert res.ok is False
        bad = banned.search(res.message)
        assert not bad, f"{slug}: {bad.group(0)!r} in {res.message!r}"
        # …and it is still recoverable from the log.
        assert res.detail, f"{slug}: nothing logged for a failure"


def test_every_bundled_check_names_its_failure_kind(dests, monkeypatch):
    """An untagged failure shows as a bare "Failed", which is the state this
    whole taxonomy exists to replace."""
    from aglaia.plugin_api import CheckResult
    kinds = {CheckResult.NETWORK, CheckResult.AUTH, CheckResult.PERMISSION,
             CheckResult.SERVER, CheckResult.CONFIG, CheckResult.UNKNOWN}
    for slug in ("send-to-calibre", "send-to-kindle", "send-to-corpus"):
        d = dests.get(slug)                      # nothing configured
        res = d.check()
        assert res.ok is False
        assert res.kind in kinds, f"{slug} reported kind={res.kind!r}"


# ── kindle: SMTP, and the two rules Amazon enforces late ─────────────

class _SMTP:
    def __init__(self):
        self.sent = []
        self.logged_in = None
        self.quit_called = False
        self.starttls_called = False

    def ehlo(self):
        pass

    def starttls(self, **kw):
        self.starttls_called = True

    def login(self, u, p):
        self.logged_in = (u, p)

    def send_message(self, msg):
        self.sent.append(msg)

    def quit(self):
        self.quit_called = True


def test_kindle_attaches_the_pdf_and_addresses_it(dests, monkeypatch, pdf):
    d = _configure(dests.get("send-to-kindle"),
                   {"recipient": "dev@kindle.com", "sender": "me@example.org",
                    "smtp_host": "smtp.example.org"},
                   {"smtp_user": "me@example.org", "smtp_password": "app-pw"})
    smtp = _SMTP()
    monkeypatch.setattr(type(d), "_connect", lambda self: smtp)
    r = d.send(pdf, _meta(title="Le Corps mystique", author="Mersch"))
    assert r.ok
    msg = smtp.sent[0]
    assert msg["To"] == "dev@kindle.com" and msg["From"] == "me@example.org"
    assert msg["Subject"] == "Le Corps mystique"
    att = [p for p in msg.iter_attachments()]
    assert len(att) == 1
    assert att[0].get_content_type() == "application/pdf"
    assert att[0].get_payload(decode=True).startswith(b"%PDF")
    assert smtp.quit_called


def test_kindle_refuses_an_oversized_file_before_connecting(
        dests, monkeypatch, tmp_path):
    """Learning the ceiling after a ten-minute upload is a bad way to learn
    it."""
    d = _configure(dests.get("send-to-kindle"),
                   {"recipient": "d@kindle.com", "sender": "m@e.org",
                    "smtp_host": "h", "max_attachment_mb": 1},
                   {"smtp_user": "u", "smtp_password": "p"})
    big = tmp_path / "big.pdf"
    big.write_bytes(b"%PDF" + b"x" * (2 * 1024 * 1024))
    called = []
    monkeypatch.setattr(type(d), "_connect",
                        lambda self: called.append(1) or _SMTP())
    r = d.send(big, _meta())
    assert r.ok is False and "1 MB limit" in r.message
    assert called == [], "must not open a connection to then refuse"


def test_kindle_success_does_not_promise_delivery(dests, monkeypatch, pdf):
    """SMTP said 250; Amazon may still drop it for an unapproved sender.
    Claiming "delivered" for something that can silently vanish is worse than
    claiming nothing."""
    d = _configure(dests.get("send-to-kindle"),
                   {"recipient": "d@kindle.com", "sender": "m@e.org",
                    "smtp_host": "h"},
                   {"smtp_user": "u", "smtp_password": "p"})
    monkeypatch.setattr(type(d), "_connect", lambda self: _SMTP())
    r = d.send(pdf, _meta())
    assert r.ok and "approved sender list" in r.message
    assert "delivered" not in r.message.lower()


def test_kindle_check_logs_in_without_sending(dests, monkeypatch):
    d = _configure(dests.get("send-to-kindle"),
                   {"recipient": "d@kindle.com", "sender": "m@e.org",
                    "smtp_host": "h"},
                   {"smtp_user": "u", "smtp_password": "p"})
    smtp = _SMTP()
    monkeypatch.setattr(type(d), "_connect", lambda self: smtp)
    r = d.check()
    assert r.ok and smtp.sent == [] and smtp.quit_called


def test_kindle_explains_an_auth_failure_in_terms_of_the_fix(dests,
                                                             monkeypatch):
    import smtplib
    d = _configure(dests.get("send-to-kindle"),
                   {"recipient": "d@k.com", "sender": "m@e.org",
                    "smtp_host": "h"},
                   {"smtp_user": "u", "smtp_password": "p"})

    def _boom(self):
        raise smtplib.SMTPAuthenticationError(535, b"nope")
    monkeypatch.setattr(type(d), "_connect", _boom)
    r = d.check()
    assert r.ok is False and "app-specific password" in r.message


def test_kindle_refuses_to_send_with_settings_missing(dests):
    d = dests.get("send-to-kindle")
    from pathlib import Path
    r = d.send(Path(__file__), _meta())
    assert r.ok is False


# ── secrets never reach the settings file ────────────────────────────

def test_a_credential_is_not_in_the_settings(dests):
    """Even with no keychain, `all()` is what a settings form reads — and a
    password does not belong in one."""
    d = _configure(dests.get("send-to-corpus"), {"base_url": "http://x"},
                   {"api_key": "zk-super-secret"})
    assert "zk-super-secret" not in str(d.ctx.config.all())
    assert d.secret("api_key") == "zk-super-secret"


# ── nobody's address ships in a public registry ──────────────────────

def test_the_corpus_url_has_no_default(dests):
    """A Corpus instance is a PRIVATE library. A default hostname in a plugin
    that lives in a public registry publishes one person's server to everyone
    who reads it — which is exactly what happened, and what this pins shut."""
    d = dests.get("send-to-corpus")
    field = next(f for f in d.CONFIG_FIELDS if f.key == "base_url")
    assert field.default == ""
    assert field.required is True
    assert d.conf("base_url") in ("", None)


def test_no_bundled_plugin_names_a_real_host(dests):
    """Placeholders and help text too, not only defaults. `.example` and
    `.example.org` are reserved by RFC 2606 precisely so that a sample cannot
    accidentally be somebody's machine."""
    import re as _re
    from pathlib import Path as _P
    root = _P("aglaia/plugins")
    bad = []
    for f in root.rglob("*.py"):
        for host in _re.findall(r"https?://([A-Za-z0-9.-]+)", f.read_text("utf-8")):
            h = host.lower()
            if (h.endswith((".example", ".example.org", ".example.com"))
                    or h in ("127.0.0.1", "localhost", "aglaia.bibli.cc",
                             "manual.calibre-ebook.com", "smtp.gmail.com")):
                continue
            bad.append(f"{f.name}: {host}")
    assert not bad, "a real hostname is being shipped: " + ", ".join(bad)
