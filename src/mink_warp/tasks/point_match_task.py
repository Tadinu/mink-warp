"""Match geom-local slot world positions to target points."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidTarget
from ..geometry.body_point_cloud import (
    BodyPointCloud,
    slot_active_mask,
    slot_weight_cost,
)
from ..interop import to_wp
from ..kernels.slots import flatten_slot_vectors, slot_vector_residual
from .task import TargetedTask


def _flatten_slot_vectors(points: npt.ArrayLike, n_slots: int, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 3 * n_slots:
            raise InvalidTarget(f"{name}: expected ({3 * n_slots},), got {arr.shape}")
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape == (n_slots, 3):
        return arr.reshape(-1)
    if arr.ndim == 2 and arr.shape[1] == 3 * n_slots:
        return arr
    if arr.ndim == 3 and arr.shape[1:] == (n_slots, 3):
        return arr.reshape(arr.shape[0], -1)
    raise InvalidTarget(
        f"{name}: expected (n_slots, 3), (3*n_slots,), (nworld, n_slots, 3) "
        f"or (nworld, 3*n_slots); got {arr.shape}"
    )


class PointMatchTask(TargetedTask):
    r"""Drive slot world positions to targets: :math:`e_i = p_i(q) - p_i^\star`.

    ``slot_cost`` is an UMR-style per-slot weight :math:`w_i` (scalar or length
    ``n_slots``). Stored mink cost is :math:`\sqrt{w_i}` on each XYZ axis.
    Zero weight zeros that slot's Jacobian row.
    """

    def __init__(
        self,
        cloud: BodyPointCloud,
        slot_cost: npt.ArrayLike = 1.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        n = cloud.n_slots
        k = 3 * n
        super().__init__(cost=np.zeros(k), gain=gain, lm_damping=lm_damping)
        self.k = k
        self.target_width = k
        self.cloud = cloud
        self.n_slots = n
        self._slot_cost_np = np.ones(n, dtype=np.float64)
        self._active_np = np.ones(n, dtype=np.int32)
        self._active_dev: wp.array | None = None
        self.set_slot_cost(slot_cost)

    def set_slot_cost(self, slot_cost: npt.ArrayLike) -> None:
        self.cost[:] = slot_weight_cost(slot_cost, self.n_slots, 3)
        self._slot_cost_np = np.atleast_1d(np.asarray(slot_cost, dtype=np.float64))
        if self._slot_cost_np.size == 1:
            self._slot_cost_np = np.full(self.n_slots, float(self._slot_cost_np[0]))
        self._active_np = slot_active_mask(self._slot_cost_np, self.n_slots)
        if self._cost_dev is not None:
            self._cost_dev.assign(self.cost.astype(np.float32))
        if self._active_dev is not None:
            self._active_dev.assign(self._active_np)

    def set_target(
        self,
        points: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        if isinstance(points, wp.array):
            self._set_pending(points, configuration=configuration)
            return
        flat = _flatten_slot_vectors(points, self.n_slots, type(self).__name__)
        self._set_pending(flat, configuration=configuration)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        kin = self.cloud.eval(configuration)
        self._ensure_buffers(configuration)
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                flatten_slot_vectors,
                dim=(configuration.nworld, self.n_slots),
                inputs=[kin.xpos, self.n_slots],
                outputs=[self._target],
            )
        self._pending = None
        self._target_set = True

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        super()._alloc_extra_buffers(configuration)
        self._active_dev = to_wp(
            self._active_np, dtype=wp.int32, device=configuration.device
        )

    def _eval(self, configuration: Configuration) -> None:
        target = self._require_target(configuration)
        kin = self.cloud.eval(configuration)
        if self._active_dev is None:
            self._active_dev = to_wp(
                self._active_np, dtype=wp.int32, device=configuration.device
            )
        assert self._error is not None
        assert self._jacobian is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                slot_vector_residual,
                dim=(configuration.nworld, self.n_slots),
                inputs=[
                    kin.xpos,
                    kin.jacp,
                    target,
                    self._active_dev,
                    self.n_slots,
                    configuration.nv,
                ],
                outputs=[self._error, self._jacobian],
            )


class PointAnchorTask(PointMatchTask):
    """Pin slots to world-space anchors (anti-slide). Same residual as point match."""
