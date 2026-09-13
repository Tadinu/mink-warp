"""Scalar pairwise distance task on surface slots."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TaskDefinitionError
from ..geometry.body_point_cloud import (
    BodyPointCloud,
    slot_active_mask,
    slot_weight_cost,
)
from ..interop import to_wp
from ..kernels.slots import pair_distance_residual
from .task import TargetedTask


class PairDistanceTask(TargetedTask):
    r"""Regulate distances between slot pairs.

    .. math::

        e_{ij} = \|p_i - p_j\| - d_{ij}^\star,\qquad
        J_{ij} = \hat u^\top (J_{p,i} - J_{p,j})

    With ``one_sided=True`` (default), the row is zeroed when
    :math:`\|p_i-p_j\| \ge d_{ij}^\star`. A near-zero pair uses ``guides`` if
    set, else :math:`(1,0,0)`.
    """

    def __init__(
        self,
        cloud: BodyPointCloud,
        pairs: npt.ArrayLike,
        pair_cost: npt.ArrayLike = 1.0,
        one_sided: bool = True,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        pairs = np.asarray(pairs, dtype=np.int32).reshape(-1, 2)
        if pairs.size == 0:
            raise TaskDefinitionError("PairDistanceTask needs at least one pair")
        if np.any(pairs < 0) or np.any(pairs >= cloud.n_slots):
            raise TaskDefinitionError(
                f"pair indices must be in [0, {cloud.n_slots}), got "
                f"min={int(pairs.min())} max={int(pairs.max())}"
            )
        n = int(pairs.shape[0])
        super().__init__(cost=np.zeros(n), gain=gain, lm_damping=lm_damping)
        self.k = n
        self.target_width = n
        self.cloud = cloud
        self.n_pairs = n
        self.one_sided = bool(one_sided)
        self._pairs_np = np.ascontiguousarray(pairs)
        self._guides_set = False
        self._pair_i: wp.array | None = None
        self._pair_j: wp.array | None = None
        self._guides: wp.array | None = None
        self._active_dev: wp.array | None = None
        self._active_np = np.ones(n, dtype=np.int32)
        self.set_pair_cost(pair_cost)

    def _pair_index_arrays(self, nworld: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = self._pairs_np
        if pairs.ndim == 2:
            i = np.broadcast_to(pairs[None, :, 0], (nworld, self.n_pairs))
            j = np.broadcast_to(pairs[None, :, 1], (nworld, self.n_pairs))
            return np.ascontiguousarray(i), np.ascontiguousarray(j)
        if pairs.ndim == 3:
            if pairs.shape != (nworld, self.n_pairs, 2):
                raise TaskDefinitionError(
                    f"batched pairs expected ({nworld}, {self.n_pairs}, 2), "
                    f"got {pairs.shape}"
                )
            return np.ascontiguousarray(pairs[:, :, 0]), np.ascontiguousarray(pairs[:, :, 1])
        raise TaskDefinitionError(f"pairs must be (P, 2) or (nworld, P, 2), got {pairs.shape}")

    def set_pairs(self, pairs: npt.ArrayLike) -> None:
        """Replace pair indices in-place.

        ``(n_pairs, 2)`` is broadcast to every world. ``(nworld, n_pairs, 2)``
        sets per-world contacts (pad unused rows). Length ``n_pairs`` is fixed.
        """
        arr = np.asarray(pairs, dtype=np.int32)
        n_slots = self.cloud.n_slots
        if arr.ndim == 2:
            if arr.shape != (self.n_pairs, 2):
                raise TaskDefinitionError(
                    f"set_pairs expected shape ({self.n_pairs}, 2), got {arr.shape}"
                )
        elif arr.ndim == 3:
            if arr.shape[1:] != (self.n_pairs, 2):
                raise TaskDefinitionError(
                    f"set_pairs expected shape (nworld, {self.n_pairs}, 2), "
                    f"got {arr.shape}"
                )
        else:
            raise TaskDefinitionError(
                f"set_pairs expected (n_pairs, 2) or (nworld, n_pairs, 2), got {arr.shape}"
            )
        if np.any(arr < 0) or np.any(arr >= n_slots):
            raise TaskDefinitionError(
                f"pair indices must be in [0, {n_slots}), got "
                f"min={int(arr.min())} max={int(arr.max())}"
            )
        self._pairs_np = np.ascontiguousarray(arr)
        if self._pair_i is not None:
            nworld = int(self._pair_i.shape[0])
            ii, jj = self._pair_index_arrays(nworld)
            self._pair_i.assign(ii)
            self._pair_j.assign(jj)

    def set_pair_indices_device(self, pair_i: wp.array, pair_j: wp.array) -> None:
        """Copy per-world pair indices already on device. GPU hot path."""
        if self._pair_i is None or self._pair_j is None:
            raise RuntimeError("PairDistanceTask buffers are not allocated")
        wp.copy(self._pair_i, pair_i)
        wp.copy(self._pair_j, pair_j)

    def set_pair_cost(self, pair_cost: npt.ArrayLike) -> None:
        self.cost[:] = slot_weight_cost(pair_cost, self.n_pairs, 1)
        w = np.atleast_1d(np.asarray(pair_cost, dtype=np.float64))
        if w.size == 1:
            w = np.full(self.n_pairs, float(w[0]))
        self._active_np = slot_active_mask(w, self.n_pairs)
        if self._cost_dev is not None:
            self._cost_dev.assign(self.cost.astype(np.float32))
        if self._active_dev is not None:
            self._active_dev.assign(self._active_np)

    def set_target(
        self,
        distances: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
        guides: npt.ArrayLike | wp.array | None = None,
    ) -> None:
        if not isinstance(distances, wp.array):
            arr = np.asarray(distances, dtype=np.float32)
            if arr.ndim == 0:
                arr = np.full((self.n_pairs,), float(arr), dtype=np.float32)
            elif arr.ndim == 1:
                if arr.size != self.n_pairs:
                    raise InvalidTarget(
                        f"PairDistanceTask: expected ({self.n_pairs},), got {arr.shape}"
                    )
            elif arr.ndim == 2:
                if arr.shape[1] != self.n_pairs:
                    raise InvalidTarget(
                        f"PairDistanceTask: expected (nworld, {self.n_pairs}), "
                        f"got {arr.shape}"
                    )
            else:
                raise InvalidTarget(f"PairDistanceTask: bad distance shape {arr.shape}")
            distances = arr
        self._set_pending(distances, configuration=configuration)
        if guides is not None:
            self.set_guides(guides, configuration=configuration)

    def set_guides(
        self,
        guides: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        """Unit-direction fallback when a pair collapses to zero length.

        Shape ``(n_pairs, 3)`` or ``(nworld, n_pairs, 3)``.
        """
        if isinstance(guides, wp.array):
            self._guides = guides
            self._guides_set = True
            return
        arr = np.ascontiguousarray(np.asarray(guides, dtype=np.float32))
        if arr.ndim == 2 and arr.shape == (self.n_pairs, 3):
            arr = arr.reshape(1, self.n_pairs, 3)
        if arr.ndim != 3 or arr.shape[1:] != (self.n_pairs, 3):
            raise InvalidTarget(
                f"guides expected (n_pairs, 3) or (nworld, n_pairs, 3), got {arr.shape}"
            )
        device = configuration.device if configuration is not None else None
        if (
            self._guides is not None
            and tuple(int(x) for x in self._guides.shape) == tuple(int(x) for x in arr.shape)
        ):
            self._guides.assign(arr)
        else:
            self._guides = to_wp(arr, dtype=float, device=device)
        self._guides_set = True

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        super()._alloc_extra_buffers(configuration)
        nworld = configuration.nworld
        device = configuration.device
        ii, jj = self._pair_index_arrays(nworld)
        with wp.ScopedDevice(device):
            self._pair_i = wp.array(ii, dtype=wp.int32)
            self._pair_j = wp.array(jj, dtype=wp.int32)
            self._active_dev = to_wp(self._active_np, dtype=wp.int32, device=device)
            if (
                self._guides is None
                or len(self._guides.shape) != 3
                or int(self._guides.shape[0]) != nworld
            ):
                if self._guides_set and self._guides is not None and int(self._guides.shape[0]) == 1:
                    tiled = np.broadcast_to(
                        self._guides.numpy(), (nworld, self.n_pairs, 3)
                    ).copy()
                    self._guides = wp.array(tiled, dtype=float)
                else:
                    self._guides = wp.zeros((nworld, self.n_pairs, 3), dtype=float)

    def _eval(self, configuration: Configuration) -> None:
        target = self._require_target(configuration)
        kin = self.cloud.eval(configuration)
        self._ensure_buffers(configuration)
        if self._active_dev is None:
            self._active_dev = to_wp(
                self._active_np, dtype=wp.int32, device=configuration.device
            )
        assert self._pair_i is not None
        assert self._pair_j is not None
        assert self._guides is not None
        assert self._active_dev is not None
        assert self._error is not None
        assert self._jacobian is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                pair_distance_residual,
                dim=(configuration.nworld, self.n_pairs),
                inputs=[
                    kin.xpos,
                    kin.jacp,
                    self._pair_i,
                    self._pair_j,
                    target,
                    self._guides,
                    self._active_dev,
                    1 if self.one_sided else 0,
                    configuration.nv,
                ],
                outputs=[self._error, self._jacobian],
            )
