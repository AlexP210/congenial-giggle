"""Making DINO-WM's own modules importable from inside this package.

DINO-WM imports rootlessly — `train.py` does `from metrics.image_metrics import ...`, not
`from dino_wm.metrics...`, and `env/__init__.py` registers its gym ids under entry points
like `env.pusht.pusht_wrapper:PushTWrapper` — all of which resolve only when
agents/dino_wm/dino_wm is itself on `sys.path`. Running its own scripts gets that for free,
because Python puts the script's directory first; importing it from here does not, and the
failure is a bare `ModuleNotFoundError: No module named 'metrics'` (or `'env'`) from inside
DINO-WM.

Every place in this package that reaches into DINO-WM calls `ensure_dino_wm_importable()`
first, so there is one definition of where its root is.
"""

import os
import sys

import dino_wm as _dino_wm_package


def dino_wm_package_root() -> str:
    """
    Locate the directory DINO-WM's own modules import each other relative to.

    Found through the installed `dino_wm` package rather than a path relative to this file,
    so it survives the repo moving. Both resolutions of that name are checked: the editable
    install maps it onto the inner package directory, but with agents/ on sys.path it instead
    resolves as a namespace package over the outer one.
    """
    for path in getattr(_dino_wm_package, "__path__", []):
        for candidate in (path, os.path.join(path, "dino_wm")):
            if os.path.isfile(os.path.join(candidate, "train.py")):
                return candidate
    raise ImportError(
        "Could not locate DINO-WM's package root (the directory holding train.py) under "
        f"{list(getattr(_dino_wm_package, '__path__', []))}. Is agents/dino_wm installed?"
    )


def ensure_dino_wm_importable() -> str:
    """Put DINO-WM's root on `sys.path` (idempotent) and return it."""
    root = dino_wm_package_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
