"""Match one coordinate of each slot (typically world z / ground height)."""

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
from ..kernels.slots import copy_slot_axis, slot_axis_residual
from .task import TargetedTask


class AxisMatchTask(TargetedTask):
    r"""Scalar coordinate match :math:`e_i = p_{i,\mathrm{axis}} - z_i^\star`.

    Default ``axis=2`` is world *z* (ground height). ``slot_cost`` is UMR-style
    :math:`w_i`; mink cost stores :math:`\sqrt{w_i}`.
    """

    def __init__(
        self,
        cloud: BodyPointCloud,
        slot_cost: npt.ArrayLike = 1.0,
        axis: int = 2,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        n = cloud.n_slots
        super().__init__(cost=np.zeros(n), gain=gain, lm_damping=lm_damping)
        self.k = n
        self.target_width = n
        self.cloud = cloud
        self.n_slots = n
        self.axis = int(axis)
        self._active_np = np.ones(n, dtype=np.int32)
        self._active_dev: wp.array | None = None
        self.set_slot_cost(slot_cost)

    def set_slot_cost(self, slot_cost: npt.ArrayLike) -> None:
        self.cost[:] = slot_weight_cost(slot_cost, self.n_slots, 1)
        self._active_np = slot_active_mask(slot_cost, self.n_slots)
        if self._cost_dev is not None:
            self._cost_dev.assign(self.cost.astype(np.float32))
        if self._active_dev is not None:
            self._active_dev.assign(self._active_np)

    def set_target(
        self,
        values: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        if not isinstance(values, wp.array):
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 0:
                arr = np.full((self.n_slots,), float(arr), dtype=np.float32)
            elif arr.ndim == 1 and arr.size != self.n_slots:
                raise InvalidTarget(
                    f"AxisMatchTask: expected ({self.n_slots},), got {arr.shape}"
                )
            elif arr.ndim == 2 and arr.shape[1] != self.n_slots:
                raise InvalidTarget(
                    f"AxisMatchTask: expected (nworld, {self.n_slots}), got {arr.shape}"
                )
            values = arr
        self._set_pending(values, configuration=configuration)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        kin = self.cloud.eval(configuration)
        self._ensure_buffers(configuration)
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                copy_slot_axis,
                dim=(configuration.nworld, self.n_slots),
                inputs=[kin.xpos, self.axis],
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
                slot_axis_residual,
                dim=(configuration.nworld, self.n_slots),
                inputs=[
                    kin.xpos,
                    kin.jacp,
                    target,
                    self._active_dev,
                    self.axis,
                    configuration.nv,
                ],
                outputs=[self._error, self._jacobian],
            )
