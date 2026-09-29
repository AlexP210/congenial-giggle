import numpy as np


class State:
    def __init__(self, state_vec: np.ndarray, state_order, state_dim_cfg):
        self._state_vec = state_vec

        # Ensure at least 2D so batching logic is consistent
        if self._state_vec.ndim == 1:
            self._state_vec = self._state_vec[None, :]
            self._single = True
        else:
            self._single = False

        idx = 0
        self._slices = {}

        for name in state_order:
            dim = state_dim_cfg[name]
            sl = slice(idx, idx + dim)
            self._slices[name] = sl

            value = self._state_vec[..., sl]
            if self._single:
                value = value[0]

            setattr(self, name, value)

            idx += dim

        self._dim = idx
        if self._state_vec.shape[-1] != self._dim:
            raise ValueError(
                f"State vector length {self._state_vec.shape[-1]} "
                f"does not match expected dimension {self._dim}"
            )

    def get(self, name):
        return getattr(self, name)

    def as_dict(self):
        return {k: getattr(self, k) for k in self._slices}

    def raw(self):
        return self._state_vec[0] if self._single else self._state_vec