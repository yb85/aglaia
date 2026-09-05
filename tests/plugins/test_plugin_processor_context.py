# Aglaïa — book scanner
# Copyright (c) 2026 Yann Barbotin <aglaia@bibli.cc>
# https://aglaia.bibli.cc
# SPDX-License-Identifier: LicenseRef-PolyForm-Shield-1.0.0
# Source-available under the PolyForm Shield License 1.0.0; any use except
# building a competing product. See LICENSE or https://polyformproject.org/licenses/shield/1.0.0/

"""A plugin processor gets its `PluginContext` (#141).

Processors were the one plugin kind the host never handed a context to.
Destinations got one in `destinations.load_all`; plugin windows got one at open
time; a processor got the `ctx = None` class attribute its own source comments
as "set by the host after construction", and nothing ever set it.

The symptom had no error in it. StampRemover could not reach its stamp library,
found nothing to look for, and returned the page untouched in 0.014 ms. The
node was written, the status was OK, the pipeline was green, and the stamp was
still there.

The context is attached to the CLASS as a lazy property rather than to an
instance, because processors are constructed inside spawned workers from a
pickled `ChainElement` — a context built in the parent would never reach them.
"""
import importlib

import numpy as np
import pytest

MANIFEST = ('[plugin]\nslug = "ctx-probe"\nname = "Ctx probe"\n'
            'version = "2.1.0"\nentry = "ctx_probe.py"\nlicense = "MIT"\n'
            '[requires]\napi = 1\n[capabilities]\nconfig = true\nsecrets = true\n')

SOURCE = '''
from aglaia.processors.abstraction import AbstractImageProcessor
from aglaia.plugin_api import MetaKind, declare_meta
for _k in ("saw_ctx", "slug", "data_dir"):
    declare_meta(_k, MetaKind.LABEL)


class CtxProbe(AbstractImageProcessor):
    SUMMARY = "records what the host gave it"
    OPTIONS = {}

    def process(self, image_buffer):
        image_buffer.meta["saw_ctx"] = self.ctx is not None
        if self.ctx is not None:
            image_buffer.meta["slug"] = self.ctx.slug
            image_buffer.meta["data_dir"] = str(self.ctx.data_dir)
        return image_buffer
'''


@pytest.fixture()
def installed(tmp_path, monkeypatch):
    """A clean APP_DATA with one plugin processor installed."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugins as pl
    import aglaia.app_data.plugin_ctx as pc
    import aglaia.app_data.plugin_registry as reg
    for m in (ad, pl, pc, reg):
        importlib.reload(m)

    d = pl.plugins_dir("processors") / "ctx-probe"
    d.mkdir(parents=True, exist_ok=True)
    (d / "aglaia-plugin.toml").write_text(MANIFEST, encoding="utf-8")
    entry = d / "ctx_probe.py"
    entry.write_text(SOURCE, encoding="utf-8")
    # Consent, the way an install records it.
    from aglaia.app_data import db as cfg
    with cfg.session() as conn:
        cfg.acknowledge_plugin(conn, "processors", entry,
                               pl.sha256_file(entry))
        conn.commit()

    from aglaia.processors import registry as R
    importlib.reload(R)
    R._discover_once()
    yield R, tmp_path


def _run(R):
    from aglaia.ImageBuffer import ImageBuffer
    info = R.get_processor("CtxProbe")
    assert info is not None, "the plugin processor did not register"
    inst = info.processor_cls(info.option_cls())
    img = np.full((8, 8, 3), 255, np.uint8)
    return inst, inst.process(ImageBuffer(img, "COLOR", 300)).meta


def test_a_plugin_processor_is_given_a_context(installed):
    R, _ = installed
    _, meta = _run(R)
    assert meta["saw_ctx"] is True
    assert meta["slug"] == "ctx-probe"


def test_the_context_points_at_the_plugins_own_data_dir(installed):
    """Where a plugin keeps a library, a cache, a model. StampRemover's stamps
    live here, and reading `None.data_dir` is what left it with an empty
    library and no error."""
    R, app_data = installed
    _, meta = _run(R)
    assert meta["data_dir"].startswith(str(app_data))
    assert "ctx-probe" in meta["data_dir"]


def test_the_context_carries_settings_and_secrets(installed):
    R, _ = installed
    inst, _ = _run(R)
    inst.ctx.config.set("k", 7)
    assert inst.ctx.config.get("k") == 7
    assert inst.ctx.secrets is not None


def test_it_is_built_per_process_not_pickled(installed):
    """Processors are constructed inside spawned workers from a pickled
    ChainElement. A context built in the parent would never arrive, so it is a
    lazy class property — every process that asks builds its own."""
    R, _ = installed
    info = R.get_processor("CtxProbe")
    assert isinstance(
        getattr(info.processor_cls, "ctx", None), property), (
        "ctx must be a lazy property, not a value fixed at registration")
    a = info.processor_cls(info.option_cls())
    b = info.processor_cls(info.option_cls())
    assert a.ctx is b.ctx          # one per class, per process


def test_a_core_processor_gets_no_context(installed):
    """A context belongs to a plugin. Handing one to Binarizer would invent a
    slug for something that has none."""
    R, _ = installed
    info = R.get_processor("Binarizer")
    assert getattr(info.processor_cls, "ctx", None) is None


def test_a_hand_dropped_processor_gets_no_context(tmp_path, monkeypatch):
    """A loose .py never declared a manifest, so it never declared that it
    wanted settings, secrets or a data directory — and there is no slug to
    namespace them under."""
    monkeypatch.setenv("AGLAIA_APP_DATA_DIR", str(tmp_path))
    import aglaia.app_data as ad
    import aglaia.app_data.plugins as pl
    for m in (ad, pl):
        importlib.reload(m)
    d = pl.plugins_dir("processors")
    d.mkdir(parents=True, exist_ok=True)
    entry = d / "loose_proc.py"
    entry.write_text(SOURCE.replace("CtxProbe", "LooseProc"), encoding="utf-8")
    from aglaia.app_data import db as cfg
    with cfg.session() as conn:
        cfg.acknowledge_plugin(conn, "processors", entry,
                               pl.sha256_file(entry))
        conn.commit()
    from aglaia.processors import registry as R
    importlib.reload(R)
    R._discover_once()
    info = R.get_processor("LooseProc")
    assert info is not None
    assert getattr(info.processor_cls, "ctx", None) is None
