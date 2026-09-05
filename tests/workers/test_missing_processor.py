# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A pipeline step whose processor is gone must stop the run, not vanish.

The chain builder used to `print("Warning: Unknown processor … Skipping.")` and
carry on. The failure that produces is the expensive kind: three hundred pages
process to completion, every status is green, and the book still has its stamps
in it — because the step that removes them was silently dropped.

The common cause is a plugin: uninstalled, switched off in the Plugins tab, or
a pipeline shared with someone who never had it. A pipeline is a description of
what the user wants done, and quietly doing less of it produces output that
looks finished and is not.
"""
import pytest

from aglaia.workers.Initializer import MissingProcessorError


class TestTheError:
    def test_it_names_what_is_missing(self):
        e = MissingProcessorError([("remove_stamps", "StampRemover")], "book")
        assert "StampRemover" in str(e)
        assert e.processors == ["StampRemover"]

    def test_it_says_what_to_do_about_it(self):
        """Three routes, because the cause is one of three things and the user
        cannot be expected to know which."""
        msg = str(MissingProcessorError([("s", "StampRemover")]))
        assert "Plugins tab" in msg          # install it
        assert "switched it off" in msg      # re-enable it
        assert "remove the step" in msg      # or drop it

    def test_one_processor_used_twice_is_named_once(self):
        e = MissingProcessorError([("a", "StampRemover"), ("b", "StampRemover")])
        assert e.processors == ["StampRemover"]
        assert str(e).count("StampRemover") == 1

    def test_several_missing_processors_are_all_named(self):
        e = MissingProcessorError([("a", "Foo"), ("b", "Bar")])
        assert e.processors == ["Bar", "Foo"]
        assert "Foo" in str(e) and "Bar" in str(e)

    def test_it_keeps_the_step_names_for_the_log(self):
        """The message names processors, because that is what the user
        installs. The steps are kept so the log can say WHERE in the pipeline
        — a pipeline may use one processor at three different points."""
        e = MissingProcessorError([("deskew_pass", "Foo")])
        assert e.steps == [("deskew_pass", "Foo")]

    def test_it_is_catchable_as_a_runtime_error(self):
        """Callers that do not know about it still handle it as a failure
        rather than letting it escape as a bare traceback."""
        assert issubclass(MissingProcessorError, RuntimeError)


@pytest.fixture()
def no_plugins(tmp_path, monkeypatch):
    """An install with no plugins — which is what "uninstalled" means.

    Isolated deliberately: this test first passed because StampRemover was
    installed on the developer's machine, so the very case it exists to cover
    could not occur.
    """
    import importlib
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path / "appdata"))
    import aglaia.app_data as ad
    import aglaia.app_data.plugins as pl
    from aglaia.processors import registry as R
    for m in (ad, pl, R):
        importlib.reload(m)
    R._discover_once()
    yield R
    importlib.reload(R)


def test_the_chain_refuses_to_build(no_plugins, tmp_path, monkeypatch):
    """End to end: a pipeline naming a processor nobody has."""
    import multiprocessing
    from types import SimpleNamespace
    from aglaia.workers import Initializer

    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "name: test\npipeline:\n"
        "  - name: clamp\n    processor: DPIfixer\n"
        "  - name: remove_stamps\n    processor: StampRemover\n",
        encoding="utf-8")

    args = SimpleNamespace(
        pipeline=yaml_path, workers=1, debug=False, options={"paths": {}},
        db_path=str(tmp_path / "p.agl"), workspace_dir=tmp_path,
        max_pages=None, dpi=300,
    )
    with pytest.raises(MissingProcessorError) as got:
        Initializer.create_processing_chain(
            args, multiprocessing.Queue(), db_path=str(tmp_path / "p.agl"))
    assert got.value.processors == ["StampRemover"]
    assert ("remove_stamps", "StampRemover") in got.value.steps


def test_a_pipeline_with_everything_present_still_builds(no_plugins,
                                                        tmp_path):
    """The guard must not fire on the ordinary case."""
    import multiprocessing
    from types import SimpleNamespace
    from aglaia.workers import Initializer

    yaml_path = tmp_path / "p.yaml"
    yaml_path.write_text(
        "name: test\npipeline:\n  - name: clamp\n    processor: DPIfixer\n",
        encoding="utf-8")
    args = SimpleNamespace(
        pipeline=yaml_path, workers=1, debug=False, options={"paths": {}},
        db_path=str(tmp_path / "p.agl"), workspace_dir=tmp_path,
        max_pages=None, dpi=300,
    )
    chain = Initializer.create_processing_chain(
        args, multiprocessing.Queue(), db_path=str(tmp_path / "p.agl"))
    assert chain is not None
