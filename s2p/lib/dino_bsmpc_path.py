"""Making DINO-Bisim's own modules importable from inside this package.

DINO-Bisim (agents/dino_bsmpc) imports rootlessly, as its DINO-WM ancestor does — `train.py`
does `from metrics.image_metrics import ...`, `from utils import cfg_to_dict` and
`from models.bisim import BisimModel`, `plan.py` does `from preprocessor import Preprocessor`,
its Hydra `_target_`s name `models.visual_world_model.VWorldModel` and
`datasets.pushcube_dset...`, and `env/__init__.py` registers its gym ids under entry points
like `env.pusht.pusht_wrapper:PushTWrapper`. All of that resolves only when
agents/dino_bsmpc is itself on `sys.path`. Running its own scripts gets that for free,
because Python puts the script's directory first; importing it from here does not, and the
failure is a bare `ModuleNotFoundError: No module named 'metrics'` (or `'utils'`).

Like TC-WM and Sparse Imagination, and unlike DINO-WM, it ships no installable package, so
there is no `import dino_bsmpc` to find its root through. It is located by path instead —
see `dino_bsmpc_package_root` for the order.

### Why this cannot coexist with the other DINO-WM forks in one process

DINO-WM and all three forks of it vendored under agents/ own the *same* top-level module
names (`models`, `env`, `planning`, `metrics`, `utils`, ...), and none of them namespaces
them. Whichever root reaches `sys.path` first wins, and — worse — whichever repo imports
first populates `sys.modules` with its own `models`, so the others silently get its classes:
a `VWorldModel` that is a different class than the checkpoint was written against, or an
`ImportError` for a symbol only one of them has.

There is no fix short of vendoring, so `ensure_dino_bsmpc_importable` detects the situation
and raises instead. This is why `s2p.models.dino_bisim_world_model` imports the repo lazily,
at construction rather than at module import: a config that merely has another wrapper on an
import path does not poison this one until something actually builds a
DINOBisimWorldModel.
"""

import os
import sys

# Top-level names DINO-Bisim claims -- every name it resolves rootlessly, whether through an
# `import` statement or through a Hydra `_target_`. `train` is included because
# `s2p.models.dino_bisim_world_model` reaches its Trainer through a bare
# `from train import Trainer`; `plan` because train.py imports ALL_MODEL_KEYS from it.
# DINO-WM (agents/dino_wm/dino_wm), TC-WM (agents/TC-WM) and Sparse Imagination
# (agents/sparse_imagination) claim most of the same names; `loss_history` is this repo's
# alone, which is what makes it a usable root marker below.
_ROOTLESS_MODULE_NAMES = (
    "train",
    "plan",
    "models",
    "datasets",
    "env",
    "planning",
    "metrics",
    "utils",
    "preprocessor",
    "custom_resolvers",
    "distributed_fn",
    "loss_history",
)

# Files whose presence identifies a candidate directory as DINO-Bisim's root rather than one
# of its sibling forks: all of them have train.py and models/visual_world_model.py, but only
# this one has the bisimulation encoder and the loss_history package.
_ROOT_MARKERS = (
    "train.py",
    os.path.join("models", "visual_world_model.py"),
    os.path.join("models", "bisim.py"),
    "loss_history",
)


def _is_dino_bsmpc_root(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, marker)) for marker in _ROOT_MARKERS)


def dino_bsmpc_package_root() -> str:
    """
    Locate the directory DINO-Bisim's own modules import each other relative to.

    Checked in order:

    1. `$DINO_BSMPC_ROOT`, for a checkout somewhere else entirely. Set but wrong is an error
       rather than something to fall through from, since the whole point of setting it is to
       override the guesses below.
    2. agents/dino_bsmpc as a sibling of the installed `s2p` package — the layout of this
       project, resolved through `__file__` so it follows an editable install wherever the
       repo is checked out.
    3. `$PROJECT_ROOT/agents/dino_bsmpc`, which is the variable the repo's own configs
       already read (`conf/encoder/dinov3.yaml` uses `${oc.env:PROJECT_ROOT}`), for a
       process whose install layout says nothing useful.
    """
    override = os.environ.get("DINO_BSMPC_ROOT")
    if override:
        override = os.path.abspath(override)
        if not _is_dino_bsmpc_root(override):
            raise ImportError(
                f"DINO_BSMPC_ROOT is set to {override!r}, but that is not a DINO-Bisim "
                f"checkout (expected {', '.join(_ROOT_MARKERS)} under it)."
            )
        return override

    candidates = []
    # .../agents/squeeze2plan/s2p/lib/dino_bsmpc_path.py -> .../agents
    agents_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    candidates.append(os.path.join(agents_dir, "dino_bsmpc"))

    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidates.append(os.path.join(project_root, "agents", "dino_bsmpc"))

    for candidate in candidates:
        if _is_dino_bsmpc_root(candidate):
            return candidate

    raise ImportError(
        "Could not locate DINO-Bisim's root (the directory holding train.py, "
        f"models/visual_world_model.py and models/bisim.py). Looked in {candidates}. Set "
        "DINO_BSMPC_ROOT to the checkout, since the repo ships no installable package to "
        "find it through."
    )


def _module_locations(module) -> list:
    """Every directory a loaded module could have come from, for the conflict check."""
    locations = [os.path.abspath(path) for path in getattr(module, "__path__", []) or []]
    filename = getattr(module, "__file__", None)
    if filename:
        locations.append(os.path.dirname(os.path.abspath(filename)))
    # A package reports the same directory through both, and the error message reads better
    # without the duplicate. dict.fromkeys rather than set() to keep the order stable.
    return list(dict.fromkeys(locations))


def _assert_no_rootless_conflict(root: str) -> None:
    """
    Refuse to proceed if another rootless repo already owns these top-level names.

    Checked before the `sys.path` insert, because by the time the repo's own
    `from models...` runs, `sys.modules["models"]` has already resolved to whatever got
    there first and the import quietly succeeds with the wrong classes. See the module
    docstring.
    """
    root = os.path.abspath(root)
    for name in _ROOTLESS_MODULE_NAMES:
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = _module_locations(module)
        if any(location == root or location.startswith(root + os.sep) for location in locations):
            continue
        raise ImportError(
            f"The top-level module {name!r} is already imported in this process from "
            f"{locations or 'an unknown location'}, not from DINO-Bisim's root ({root}). "
            "DINO-Bisim imports its own modules rootlessly under that name, so it would "
            "silently get the other repo's classes instead.\n\n"
            "DINO-WM (agents/dino_wm/dino_wm), TC-WM (agents/TC-WM) and Sparse Imagination "
            "(agents/sparse_imagination) are the usual culprits: every fork claims the same "
            f"names ({', '.join(_ROOTLESS_MODULE_NAMES)}). They cannot share a process -- run "
            "each wrapper in its own process (separate Hydra runs) rather than composing more "
            "than one into a single config."
        )


def ensure_dino_bsmpc_importable() -> str:
    """Put DINO-Bisim's root on `sys.path` (idempotent) and return it."""
    root = dino_bsmpc_package_root()
    _assert_no_rootless_conflict(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
