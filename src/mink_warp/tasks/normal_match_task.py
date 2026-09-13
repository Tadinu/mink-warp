"""Match geom-local slot world normals to target unit vectors."""

from __future__ import annotations

import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ..geometry.body_point_cloud import BodyPointCloud
from ..kernels.slots import (
    flatten_slot_vectors,
    normalize_target_vectors,
    slot_vector_residual,
)
from .point_match_task import PointMatchTask


class NormalMatchTask(PointMatchTask):
    r"""Drive slot normals to unit targets: :math:`e_i = n_i(q) - \hat n_i^\star`.

    Requires :class:`~mink_warp.BodyPointCloud` constructed with ``local_normals``.
    Targets are normalized on :meth:`set_target`.
    """

    def __init__(
        self,
        cloud: BodyPointCloud,
        slot_cost: npt.ArrayLike = 1.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        if not cloud.has_normals:
            raise TaskDefinitionError(
                "NormalMatchTask requires BodyPointCloud(..., local_normals=...)"
            )
        super().__init__(cloud, slot_cost=slot_cost, gain=gain, lm_damping=lm_damping)

    def set_target(
        self,
        normals: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        super().set_target(normals, configuration=configuration)
        if configuration is not None and self._target is not None:
            self._normalize_target(configuration)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        kin = self.cloud.eval(configuration)
        self._ensure_buffers(configuration)
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                flatten_slot_vectors,
                dim=(configuration.nworld, self.n_slots),
                inputs=[kin.nrm, self.n_slots],
                outputs=[self._target],
            )
        self._pending = None
        self._target_set = True

    def _normalize_target(self, configuration: Configuration) -> None:
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                normalize_target_vectors,
                dim=(configuration.nworld, self.n_slots),
                inputs=[self._target, self.n_slots],
            )

    def _eval(self, configuration: Configuration) -> None:
        target = self._require_target(configuration)
        self._normalize_target(configuration)
        kin = self.cloud.eval(configuration)
        if self._active_dev is None:
            from ..interop import to_wp

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
                    kin.nrm,
                    kin.jacn,
                    target,
                    self._active_dev,
                    self.n_slots,
                    configuration.nv,
                ],
                outputs=[self._error, self._jacobian],
            )
