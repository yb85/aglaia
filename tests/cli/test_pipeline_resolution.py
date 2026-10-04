"""`--pipeline NAME` resolution (#161).

The user pipelines dir (`<APP_DATA>/pipelines`) is seeded from the bundle so
the shipped files can be edited. An edit only wins if the CLI looks there
first — the bug was that it looked ONLY in the bundle, so a user pipeline was
unreachable by name and an edited shipped one was silently replaced by the
stock copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aglaia.workers import cli as wcli


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    """A user dir and a bundle dir, both empty, wired into the resolver."""
    user = tmp_path / "user"
    bundled = tmp_path / "bundled"
    user.mkdir()
    bundled.mkdir()
    monkeypatch.setattr(wcli, "_pipeline_search_dirs", lambda: [user, bundled])
    return user, bundled


def _write(d: Path, stem: str, body: str = "name: x\npipeline: []\n") -> Path:
    p = d / f"{stem}.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_unset_returns_none(dirs):
    assert wcli.resolve_pipeline_path(None) is None
    assert wcli.resolve_pipeline_path("") is None


def test_name_only_in_user_dir_resolves(dirs):
    user, _ = dirs
    want = _write(user, "my-book")
    assert wcli.resolve_pipeline_path("my-book") == want.resolve()


def test_name_only_in_bundle_resolves(dirs):
    _, bundled = dirs
    want = _write(bundled, "book_curved_x2")
    assert wcli.resolve_pipeline_path("book_curved_x2") == want.resolve()


def test_user_copy_wins_over_the_bundled_one(dirs):
    """The regression: both dirs hold `book_curved_x2`, and the user's edit
    is the one that must run."""
    user, bundled = dirs
    want = _write(user, "book_curved_x2", "name: book_curved_x2\npipeline: [edited]\n")
    _write(bundled, "book_curved_x2")
    assert wcli.resolve_pipeline_path("book_curved_x2") == want.resolve()


def test_explicit_path_wins_over_both(tmp_path, dirs):
    user, _ = dirs
    _write(user, "elsewhere")
    direct = _write(tmp_path, "elsewhere")
    assert wcli.resolve_pipeline_path(str(direct)) == direct.resolve()


def test_unknown_name_names_both_directories(dirs):
    user, bundled = dirs
    with pytest.raises(SystemExit) as e:
        wcli.resolve_pipeline_path("nope")
    msg = str(e.value)
    assert "nope" in msg and str(user) in msg and str(bundled) in msg


def test_every_listed_pipeline_resolves():
    """What `aglaia list pipelines` prints is what `-p` accepts."""
    from aglaia.app_data import pipelines_dir

    for p in sorted(pipelines_dir().glob("*.yaml")):
        assert wcli.resolve_pipeline_path(p.stem) is not None
