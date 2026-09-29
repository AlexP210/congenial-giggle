"""Making TC-WM's own modules importable from inside this package.

TC-WM imports rootlessly, exactly as DINO-WM does — `train.py` does
`from metrics.image_metrics import ...` and `from utils import cfg_to_dict`, `plan.py` does
`from planning.evaluator import PlanEvaluator`, and `env/__init__.py` registers its gym ids
under entry points like `env.pusht.pusht_wrapper:PushTWrapper`. All of that resolves only
when agents/TC-WM is itself on `sys.path`. Running its own scripts gets that for free,
because Python puts the script's directory first; importing it from here does not, and the
failure is a bare `ModuleNotFoundError: No module named 'metrics'` (or `'utils'`) from
inside TC-WM.

Unlike DINO-WM, TC-WM ships no `pyproject.toml` and is not pip-installed, so there is no
`import tc_wm` to find its root through. It is located by path instead — see
`tc_wm_package_root` for the order.

### Why this cannot coexist with DINO-WM in one process

Both repos own the *same* top-level module names (`models`, `datasets`, `env`, `planning`,
`metrics`, `distributed_fn`, `utils`), and neither namespaces them. Whichever root reaches
`sys.path` first wins, and — worse — whichever repo imports first populates `sys.modules`
with its own `models`, so the second repo silently gets the first one's classes: a
`VWorldModel` with the wrong latent layout, or an `ImportError` for a symbol only one of
them has.

There is no fix short of vendoring one of them under a package name, so
`ensure_tc_wm_importable` detects the situation and raises instead. This is why
`s2p.models.tc_world_model` imports TC-WM lazily, at construction rather than at module
import: a config that merely has `s2p.models.dino_world_model` on an import path does not
poison this one until something actually builds a TCWorldModel.
"""

import os
import sys

# Top-level names TC-WM claims, and which DINO-WM (agents/dino_wm/dino_wm) claims too --
# `train` included, since `s2p.models.tc_world_model` reaches TC-WM's Trainer through a bare
# `from train import Trainer`.
# Anything already bound to one of these from outside TC-WM's root makes TC-WM
# unimportable in this process; see the module docstring.
_ROOTLESS_MODULE_NAMES = (
    "train",
    "models",
    "datasets",
    "env",
    "planning",
    "metrics",
    "distributed_fn",
    "utils",
)

# Files whose presence identifies a candidate directory as TC-WM's root rather than some
# other repo. DINO-WM's root also holds train.py, but its projector-less models package has
# no visual_world_model.py alongside a `models/projector`.
_ROOT_MARKERS = (
    "train.py",
    os.path.join("models", "visual_world_model.py"),
    os.path.join("models", "projector"),
)


def _is_tc_wm_root(path: str) -> bool:
    return all(os.path.exists(os.path.join(path, marker)) for marker in _ROOT_MARKERS)


def tc_wm_package_root() -> str:
    """
    Locate the directory TC-WM's own modules import each other relative to.

    Checked in order:

    1. `$TC_WM_ROOT`, for a checkout somewhere else entirely. Set but wrong is an error
       rather than something to fall through from, since the whole point of setting it is
       to override the guesses below.
    2. agents/TC-WM as a sibling of the installed `s2p` package — the layout of this
       project, resolved through `__file__` so it follows an editable install wherever the
       repo is checked out.
    3. `$PROJECT_ROOT/agents/TC-WM`, which is the variable TC-WM's own configs already read
       (`conf/encoder/dinov3.yaml` uses `${oc.env:PROJECT_ROOT}`), for a process whose
       install layout says nothing useful.
    """
    override = os.environ.get("TC_WM_ROOT")
    if override:
        override = os.path.abspath(override)
        if not _is_tc_wm_root(override):
            raise ImportError(
                f"TC_WM_ROOT is set to {override!r}, but that is not a TC-WM checkout "
                f"(expected {', '.join(_ROOT_MARKERS)} under it)."
            )
        return override

    candidates = []
    # .../agents/squeeze2plan/s2p/lib/tc_wm_path.py -> .../agents
    agents_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    candidates.append(os.path.join(agents_dir, "TC-WM"))

    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidates.append(os.path.join(project_root, "agents", "TC-WM"))

    for candidate in candidates:
        if _is_tc_wm_root(candidate):
            return candidate

    raise ImportError(
        "Could not locate TC-WM's root (the directory holding train.py and "
        f"models/visual_world_model.py). Looked in {candidates}. Set TC_WM_ROOT to the "
        "checkout, since TC-WM ships no installable package to find it through."
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
    Refuse to proceed if another rootless repo already owns TC-WM's top-level names.

    Checked before the `sys.path` insert, because by the time TC-WM's own `from models...`
    runs, `sys.modules["models"]` has already resolved to whatever got there first and the
    import quietly succeeds with the wrong classes. See the module docstring.
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
            f"{locations or 'an unknown location'}, not from TC-WM's root ({root}). "
            "TC-WM imports its own modules rootlessly under that name, so it would "
            "silently get the other repo's classes instead.\n\n"
            "DINO-WM (agents/dino_wm/dino_wm) is the usual culprit: it claims the same "
            f"names ({', '.join(_ROOTLESS_MODULE_NAMES)}). The two cannot be used in one "
            "process — run TCWorldModel and DINOWorldModel in separate processes (separate "
            "Hydra runs) rather than composing both into one config."
        )


def ensure_tc_wm_importable() -> str:
    """Put TC-WM's root on `sys.path` (idempotent) and return it."""
    root = tc_wm_package_root()
    _assert_no_rootless_conflict(root)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root
