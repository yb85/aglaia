# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""The UI is written for the user; the log is written for us.

Two registers, not two catalogues (docs/ui-writing.md). The failure is almost
never that a log line is literally reused in a dialog — it is that UI text gets
written in log voice: mechanism first, an exception spliced in, an identifier
the reader cannot act on.

Style is normally a review matter, not a test matter. These four rules are here
because they are mechanical, because a violation is invisible in review (the
sentence reads fine; it is simply addressed to the wrong person), and because
the catalogue is about to be frozen for translation — every string rewritten
after that is one translated twice.

Everything else in the guide stays a matter of judgement.
"""
import ast
import pathlib
import re

import pytest

GUI = pathlib.Path("aglaia/gui")
PLUGINS = pathlib.Path("aglaia/plugins")

#: The calls that put text in front of a person.
UI_CALLS = {"toast", "setText", "setToolTip", "setPlaceholderText",
            "showMessage", "warning", "critical", "information", "question"}


def _literal(node):
    """The string a node evaluates to, as far as we can tell statically."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str)
            else "{}" for v in node.values)
    if isinstance(node, ast.Call):
        fn = node.func
        if getattr(fn, "attr", None) == "tr" and node.args:
            return _literal(node.args[0])
        if getattr(fn, "attr", None) == "format":
            return _literal(fn.value)
    return None


def _strings():
    """Every user-visible string we can find, as (file, line, text)."""
    out = []
    for f in sorted(GUI.rglob("*.py")) + sorted(PLUGINS.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text("utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            fn = n.func
            name = getattr(fn, "attr", None) or getattr(fn, "id", None)
            args = []
            if name == "tr" and n.args:
                args = [n.args[0]]
            elif name in UI_CALLS:
                args = list(n.args)
            elif name == "Field":
                args = [kw.value for kw in n.keywords
                        if kw.arg in ("help", "placeholder")]
            elif name in ("SendResult", "CheckResult") and len(n.args) >= 2:
                args = [n.args[1]]
            for a in args:
                s = _literal(a)
                if s:
                    out.append((str(f), n.lineno, s))
    return out


@pytest.fixture(scope="module")
def strings():
    found = _strings()
    # A scan that silently found nothing would make every rule below pass.
    assert len(found) > 500, f"only {len(found)} strings — did the scan break?"
    return found


def _report(bad):
    return "\n".join(f"  {f}:{ln}\n      {s[:120]}" for f, ln, s in bad)


# ── the do-not-write list (docs/ui-writing.md §10) ───────────────────

BANNED = [
    (re.compile(r"\b[A-Z][A-Z0-9]{2,}(?:_[A-Z0-9]+)+\b"),
     "an environment variable or constant — the user did not set it"),
    (re.compile(r"\bX-[A-Za-z]+-[A-Za-z-]+\b|\bContent-Type\b"),
     "an HTTP header — wire detail"),
    # `(?<![\w.])` so an email address is not a decorator.
    (re.compile(r"(?<![\w.])@[a-z_]{4,}\b"),
     "a decorator — addressed to whoever writes plugins, not who uses them"),
    (re.compile(r"\buv sync\b|--extra\s+\w|\bmaturin\b|\bpip install\b"),
     "an install command — they installed a .dmg"),
    # A file EXTENSION the user picks in a dialog is fair game — `.agl` and
    # `.scanproj.sqlite` are things they can see and act on. A bare module
    # name is not.
    (re.compile(r"(?<!\.)\bsys\.path\b|(?<![.\w])\bsqlite3?\b|(?<![\w])\.env\b"),
     "an internal file or module"),
]

#: Strings that name an internal on purpose, with the reason. Keep this list
#: short; each entry is a promise that a user can act on the word.
ALLOWED = {
    # The bug report tells the user to review a file they are about to send.
    "Saved to <b>{folder}</b>",
    # Diagnostics the user is asked to attach to a report.
}


def test_no_user_visible_string_names_an_internal(strings):
    bad = []
    for f, ln, s in strings:
        if any(s.startswith(a) for a in ALLOWED):
            continue
        for rx, why in BANNED:
            m = rx.search(s)
            if m:
                bad.append((f, ln, f"[{m.group(0)}: {why}] {s}"))
                break
    assert not bad, "user-visible text naming internals:\n" + _report(bad)


# ── log voice in the UI ──────────────────────────────────────────────

def test_no_raw_exception_is_spliced_into_user_visible_text():
    """`TypeError: 'NoneType' object is not subscriptable` tells the user
    nothing and costs them the sentence that would have.

    The exception belongs in the log, in the same handler — not in the dialog.
    """
    bad = []
    for f in sorted(GUI.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text("utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if not (isinstance(n, ast.Call)
                    and getattr(n.func, "attr", None) in UI_CALLS):
                continue
            for a in n.args:
                for m in ast.walk(a):
                    if isinstance(m, ast.Attribute) and m.attr == "__name__":
                        bad.append((str(f), n.lineno, "type(e).__name__"))
                    elif isinstance(m, ast.FormattedValue):
                        for x in ast.walk(m.value):
                            if isinstance(x, ast.Name) and x.id in (
                                    "e", "exc", "err", "ex"):
                                bad.append((str(f), n.lineno, "{e}"))
    assert not bad, "raw exceptions shown to the user:\n" + _report(bad)


# ── the two mechanical consistency rules ─────────────────────────────

def test_the_ellipsis_is_one_character(strings):
    """`...` and `…` are two msgids for one sentence: every translator renders
    it twice, and the two drift."""
    bad = [(f, ln, s) for f, ln, s in strings if s.rstrip().endswith("...")]
    assert not bad, "ASCII ellipsis, use …:\n" + _report(bad)


def test_short_labels_are_sentence_case(strings):
    """Title case survives only in the native macOS menu bar, where Apple's
    HIG governs and the OS convention is more visible than ours. Everything we
    draw ourselves is sentence case."""
    small = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "for",
             "with", "by", "as", "at", "from", "is", "it"}
    #: Menu-bar items and proper nouns. Not a dumping ground — an entry here
    #: says "the OS decides this one", not "I liked the capitals".
    menu_bar = {
        "About Aglaïa", "About Aglaïa…", "Quit Aglaïa", "Close Project",
        "New Project…", "Open Project…", "Export Markdown",
        "Aglaïa Documentation", "Aglaïa Scanner", "Set up Aglaïa",
        "UNREVIEWED PLUGIN", "by Aglaïa",
    }
    #: Names that are capitalised because they are names.
    proper = {"aglaïa", "aglaia", "apple", "intelligence", "vision",
              "markdown", "mistral", "kindle", "calibre", "corpus", "surya",
              "github", "finder", "python", "macos", "linux", "windows",
              "wolf", "sauvola", "vosk", "jbig2", "sift"}
    bad = []
    for f, ln, s in strings:
        if s in menu_bar or "{" in s or s.startswith("<") or "\n" in s:
            continue
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", s)
        if not (2 <= len(s.split()) <= 4) or s.endswith((".", "!", "?", ":")):
            continue
        rest = [w for w in words[1:]
                if len(w) >= 4 and w.lower() not in small
                and w.lower() not in proper]
        if rest and all(w[0].isupper() for w in rest):
            bad.append((f, ln, s))
    assert not bad, "Title Case outside the menu bar:\n" + _report(bad)
