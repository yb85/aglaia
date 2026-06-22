# Static imports so PyInstaller bundles the built-in processor modules into the
# PYZ (collect_submodules alone doesn't make them importable when this is a
# dynamically-discovered package). Importing here is also what the registry
# relies on for discovery; from source it's redundant with the filesystem walk.
from lib.processors import (  # noqa: F401
    Binarizer, DPIfixer, MarginSetter, PageDetector, PageDewarper,
    SkewFinder, TrapezoidalCorrection,
)
