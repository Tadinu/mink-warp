"""Geom-attached rigid sample points evaluated for every world."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ..kernels.slots import slot_fk, slot_jacobians


@dataclass
class SlotKinematics:
    """Device buffers from :meth:`BodyPointCloud.eval` (overwritten next eval)."""

    xpos: wp.array
    nrm: wp.array
    jacp: wp.array
    jacn: wp.array


def slot_weight_cost(
    weights: npt.ArrayLike,
    n_slots: int,
    dim: int,
) -> np.ndarray:
    r"""Expand UMR-style per-slot weights :math:`w` to mink ``cost = \sqrt{w}``.

    Objective contribution is :math:`\tfrac12 \sum_i w_i \|e_i\|^2`, so each of
    the ``dim`` residual axes for slot ``i`` stores :math:`\sqrt{w_i}`.
    """
    w = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    if w.size == 1:
        w = np.full(n_slots, float(w[0]), dtype=np.float64)
    elif w.size != n_slots:
        raise TaskDefinitionError(
            f"slot weights must be scalar or shape ({n_slots},), got {w.shape}"
        )
    if np.any(w < 0.0):
        raise TaskDefinitionError("slot weights must be >= 0")
    return np.repeat(np.sqrt(w), int(dim))


def slot_active_mask(weights: npt.ArrayLike, n_slots: int) -> np.ndarray:
    """1 where per-slot weight is positive, else 0."""
    w = np.atleast_1d(np.asarray(weights, dtype=np.float64))
    if w.size == 1:
        w = np.full(n_slots, float(w[0]), dtype=np.float64)
    elif w.size != n_slots:
        raise TaskDefinitionError(
            f"slot weights must be scalar or shape ({n_slots},), got {w.shape}"
        )
    return (w > 0.0).astype(np.int32)


class BodyPointCloud:
    """Rigid points (and optional normals) glued to MuJoCo geoms.

    World kinematics:

    .. math::

        p_i = t_{g(i)} + R_{g(i)} p_i^{\\mathrm{loc}},\\qquad
        n_i = R_{g(i)} n_i^{\\mathrm{loc}}

    Jacobians use the MuJoCo point-jac formula on the owning body (same as
    ``mj_jac`` at the world point), then :math:`J_n = -[n]_\\times J_\\omega`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_ids: npt.ArrayLike,
        local_pos: npt.ArrayLike,
        local_normals: npt.ArrayLike | None = None,
    ):
        geom_ids = np.asarray(geom_ids, dtype=np.int32).reshape(-1)
        local_pos = np.asarray(local_pos, dtype=np.float32).reshape(-1, 3)
        if geom_ids.size == 0:
            raise ValueError("BodyPointCloud needs at least one slot")
        if local_pos.shape[0] != geom_ids.size:
            raise ValueError(
                f"local_pos has {local_pos.shape[0]} rows, geom_ids has {geom_ids.size}"
            )
        if np.any(geom_ids < 0) or np.any(geom_ids >= model.ngeom):
            raise ValueError(
                f"geom_ids must be in [0, {model.ngeom}), got "
                f"min={int(geom_ids.min())} max={int(geom_ids.max())}"
            )
        self.model = model
        self.n_slots = int(geom_ids.size)
        self.has_normals = local_normals is not None
        if local_normals is None:
            local_nrm = np.zeros((self.n_slots, 3), dtype=np.float32)
            local_nrm[:, 2] = 1.0
        else:
            local_nrm = np.asarray(local_normals, dtype=np.float32).reshape(-1, 3)
            if local_nrm.shape[0] != self.n_slots:
                raise ValueError(
                    f"local_normals has {local_nrm.shape[0]} rows, "
                    f"expected {self.n_slots}"
                )
            norms = np.linalg.norm(local_nrm, axis=1, keepdims=True)
            local_nrm = local_nrm / np.maximum(norms, 1e-12)
        body_ids = np.asarray(model.geom_bodyid[geom_ids], dtype=np.int32)
        self._geom_ids_np = geom_ids
        self._body_ids_np = body_ids
        self._local_pos_np = np.ascontiguousarray(local_pos)
        self._local_nrm_np = np.ascontiguousarray(local_nrm)
        self._dev: dict[str, tuple[wp.array, wp.array, wp.array, wp.array]] = {}
        self._kin: SlotKinematics | None = None
        self._cfg_id: int | None = None
        self._stamp: int | None = None
        self._nworld: int | None = None
        self._device: str | None = None

    @property
    def geom_ids(self) -> np.ndarray:
        return self._geom_ids_np

    @property
    def body_ids(self) -> np.ndarray:
        return self._body_ids_np

    def _ensure_dev(self, device: str):
        cached = self._dev.get(device)
        if cached is not None:
            return cached
        with wp.ScopedDevice(device):
            arrs = (
                wp.array(self._geom_ids_np, dtype=wp.int32),
                wp.array(self._body_ids_np, dtype=wp.int32),
                wp.array(self._local_pos_np, dtype=wp.vec3),
                wp.array(self._local_nrm_np, dtype=wp.vec3),
            )
        self._dev[device] = arrs
        return arrs

    def _alloc_kin(self, nworld: int, nv: int, device: str) -> SlotKinematics:
        n = self.n_slots
        with wp.ScopedDevice(device):
            return SlotKinematics(
                xpos=wp.zeros((nworld, n, 3), dtype=float),
                nrm=wp.zeros((nworld, n, 3), dtype=float),
                jacp=wp.zeros((nworld, n, 3, nv), dtype=float),
                jacn=wp.zeros((nworld, n, 3, nv), dtype=float),
            )

    def invalidate(self) -> None:
        """Force the next :meth:`eval` to recompute (e.g. after an in-place q write)."""
        self._stamp = None

    def eval(self, configuration: Configuration) -> SlotKinematics:
        """Refresh world kinematics if the configuration's FK stamp moved."""
        nworld = configuration.nworld
        device = configuration.device
        nv = configuration.nv
        stamp = configuration.kin_stamp
        reuse = (
            self._kin is not None
            and self._cfg_id == id(configuration)
            and self._stamp == stamp
            and self._nworld == nworld
            and self._device == device
        )
        if reuse:
            assert self._kin is not None
            return self._kin

        if (
            self._kin is None
            or self._nworld != nworld
            or self._device != device
            or self._kin.jacp.shape[3] != nv
        ):
            self._kin = self._alloc_kin(nworld, nv, device)

        geom_ids, body_ids, local_pos, local_nrm = self._ensure_dev(device)
        kin = self._kin
        wd = configuration.wp_data
        wm = configuration.wp_model
        has_n = 1 if self.has_normals else 0
        with wp.ScopedDevice(device):
            wp.launch(
                slot_fk,
                dim=(nworld, self.n_slots),
                inputs=[
                    wd.geom_xpos,
                    wd.geom_xmat,
                    geom_ids,
                    local_pos,
                    local_nrm,
                    has_n,
                ],
                outputs=[kin.xpos, kin.nrm],
            )
            wp.launch(
                slot_jacobians,
                dim=(nworld, self.n_slots, nv),
                inputs=[
                    wm.body_rootid,
                    wm.body_isdofancestor,
                    wd.subtree_com,
                    wd.cdof,
                    body_ids,
                    kin.xpos,
                    kin.nrm,
                    has_n,
                ],
                outputs=[kin.jacp, kin.jacn],
            )
        self._cfg_id = id(configuration)
        self._stamp = stamp
        self._nworld = nworld
        self._device = device
        return kin
