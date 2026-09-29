"""
Checkpoint references: how a frozen model earns its slot in an agent checkpoint folder
without a second copy of its weights.

`AgentModel.save_to_folder` writes one entry per submodel and `load_from_folder` reads them
back by the same names, so every model has to leave *something* behind or the folder stops
round-tripping and `AgentModel.cfg.checkpoint_folder` breaks. For a model that was frozen
for the whole run, that something does not have to be the weights: they are byte-for-byte
the file it was constructed from.

Each model decides this for itself, in its own `save_to_file`/`load_from_file`, using the
two functions below -- there is no base-class machinery. That is deliberate: whether a
model's checkpoint path can be *read back* by its own `load_from_file` is knowledge local to
the class that defines both, and guessing it from config keys gets it wrong. The DINOv3
encoders are the cautionary case: they keep backbone weights in `cfg.checkpoint` and hand
them to `torch.hub.load(weights=...)`, which their `load_from_file` cannot read, so they
write their weights out in full and say so in a comment where the guard would have gone. Re-saving them copies hundreds of MiB per checkpoint --
a run whose encoder, dynamics and value all resolve to one frozen DINO-WM writes the same
378 MiB three times, and does it again for every "best" folder an evaluator keeps.

So a frozen model writes a *reference* instead: a small file naming where its weights
already live. `load_from_folder` resolves it and hands the model that path, which is the
same file it would have loaded at construction, so a folder round-trips either way.

A reference occupies the model's ordinary `<name>.pt` slot rather than a file of its own,
so a checkpoint folder keeps exactly one entry per model and nothing downstream has to
learn a second filename:

    checkpoints/best_reward_loss/
        encoder.pt       <- 1.4 KB: frozen DINO-WM, points at the run it was trained in
        dynamics.pt      <- the same model again, same pointer
        value.pt         <- and again
        reward.pt        <- 14 MB: the head this run actually trained, weights and all

Telling the two apart therefore means looking inside. `resolve_checkpoint_path` skips that
read for any file too large to be a reference -- see `_MAX_REFERENCE_BYTES`, which is an
optimisation only: a file under it is still opened and checked, so no threshold can make
the answer wrong. Folders written before this existed hold only real weights and load
unchanged.
"""

import os
import typing
from datetime import datetime

import torch

_REFERENCE_MAGIC = "s2p_frozen_checkpoint_reference_v1"
# Written by checkpoints saved before the project was renamed from TSD; still accepted on load.
_LEGACY_REFERENCE_MAGICS = ("tsd_frozen_checkpoint_reference_v1",)

_MAX_REFERENCE_BYTES = 64 * 1024
"""
Files larger than this are not opened by `resolve_checkpoint_path`, which is worth doing
when the alternative is reading a 378 MiB checkpoint just to discover it is a checkpoint.

Purely an optimisation: a reference is a four-key dict and comes out around 1.4 KB, so this
sits orders of magnitude above one, and anything below it is opened and checked properly.
Raising or lowering it changes how much is read, never what is concluded -- which matters,
because an empty state dict (1285 bytes) and a reference (1413 bytes) are the same size to
within noise, and only the contents can separate them.
"""


class CheckpointReferenceError(RuntimeError):
    """A checkpoint entry is a reference whose source file cannot be used."""


DEFAULT_RESUME_CHECKPOINT = "model_latest.pth"
"""
The filename every vendored `Trainer` resumes from, and the only one it will.

`TC-WM/train.py:739` and `dino_wm/train.py:225` both build the path as
`<resume_folder>/checkpoints/model_latest.pth` with the filename written out literally -- there
is no config key for it. A run whose best epoch is not its last therefore cannot be reached
through `resume_folder` at all, which is what `resolve_resume_checkpoint` below exists to work
around.
"""


def resume_folder_checkpoint(
    resume_folder: typing.Optional[str],
    filename: str = DEFAULT_RESUME_CHECKPOINT,
) -> typing.Optional[str]:
    """
    The checkpoint a repo-backed world model resumes from, if it is actually there.

    DINO-WM and its relatives (TC-WM, DINO-Bisim, Sparse Imagination) are all built by a
    vendored `Trainer`, which resumes from `<resume_folder>/checkpoints/model_latest.pth`
    and otherwise leaves the model at its initialisation. In that second case there is no
    source to point at and the weights have to be written out like anyone else's -- so this
    checks rather than assumes.

    `filename` names a different checkpoint in that same folder, for a model whose wrapper
    loaded one itself (see `resolve_resume_checkpoint`). The save-by-reference path has to be
    given the file the weights *actually* came from, or a checkpoint folder would point at the
    one the Trainer happened to read on the way past.
    """
    if resume_folder is None:
        return None
    checkpoint = os.path.join(resume_folder, "checkpoints", filename)
    return checkpoint if os.path.exists(checkpoint) else None


def resolve_resume_checkpoint(
    resume_checkpoint: typing.Optional[str],
    resume_folder: typing.Optional[str],
    owner: str,
) -> typing.Optional[str]:
    """
    The file a wrapper should load over whatever its `Trainer` resumed, or None for neither.

    `resume_checkpoint` is either a bare filename inside `<resume_folder>/checkpoints/`
    (`model_best.pth`, the usual case) or an absolute path to a checkpoint anywhere. `None`
    means take the Trainer's own resume and load nothing further.

    Raises rather than falling back. A wrapper that silently kept `model_latest.pth` after
    being asked for `model_best.pth` would report a full set of numbers for the wrong epoch,
    and the two files differ by exactly as much as training improved -- which is to say, by
    enough to matter and not enough to look wrong.
    """
    if resume_checkpoint is None:
        return None

    if os.path.isabs(resume_checkpoint):
        path = resume_checkpoint
    else:
        if resume_folder is None:
            raise CheckpointReferenceError(
                f"{owner}: cfg.resume_checkpoint is {resume_checkpoint!r}, a filename to be "
                "looked up under the resume folder, but resume_folder is null. Give an "
                "absolute path, or set the resume folder."
            )
        path = os.path.join(resume_folder, "checkpoints", resume_checkpoint)

    if not os.path.exists(path):
        raise CheckpointReferenceError(
            f"{owner}: cfg.resume_checkpoint points at {path!r}, which does not exist. "
            f"(The Trainer's own default is {DEFAULT_RESUME_CHECKPOINT!r} in the same folder; "
            "set cfg.resume_checkpoint to null to use it.)"
        )
    return path


def save_checkpoint_reference(filepath: str, source: str, owner: str) -> None:
    """
    Write, at `filepath`, a note that these weights live at `source` instead.

    The source is resolved to an absolute path: a checkpoint folder is routinely read from a
    different working directory than the one that wrote it (Hydra gives every run its own),
    and a relative pointer would quietly resolve somewhere else.
    """
    source = os.path.abspath(source)
    if not os.path.exists(source):
        raise CheckpointReferenceError(
            f"Cannot reference {source!r} for {owner}: the file does not exist. A frozen "
            "model may only be saved by reference while the checkpoint it was loaded from "
            "is still in place."
        )
    torch.save(
        {
            "magic": _REFERENCE_MAGIC,
            "source": source,
            "owner": owner,
            "written_at": datetime.now().isoformat(timespec="seconds"),
        },
        filepath,
    )


def resolve_checkpoint_path(filepath: str) -> str:
    """
    `filepath` itself, or -- if it holds a reference -- the file that reference names.

    Safe to call on any checkpoint: a file that is not a reference is returned untouched,
    and one too large to be a reference is not even opened.
    """
    if not os.path.exists(filepath):
        raise CheckpointReferenceError(f"No checkpoint at {filepath!r}.")
    if os.path.getsize(filepath) > _MAX_REFERENCE_BYTES:
        return filepath

    payload = torch.load(filepath, map_location="cpu", weights_only=False)
    if not (isinstance(payload, dict) and payload.get("magic") in (_REFERENCE_MAGIC, *_LEGACY_REFERENCE_MAGICS)):
        return filepath

    source = payload["source"]
    if not os.path.exists(source):
        raise CheckpointReferenceError(
            f"{payload['owner']} was frozen when {filepath!r} was written, so its weights "
            f"were not copied there -- the file only points at {source!r}, which no longer "
            f"exists. Restore it, or re-run with the model unfrozen so its weights are "
            f"written in full. (Reference written {payload['written_at']}.)"
        )
    return source
