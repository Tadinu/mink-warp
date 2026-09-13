"""Hard floor / axis inequality on surface slots: linearized p_axis >= margin."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..configuration import Configuration
from ..geometry.body_point_cloud import BodyPointCloud
from ..kernels.constrained import reset_ineq_block
from ..kernels.slots import (
    fill_i32_2d,
    scatter_selected_axis_limit,
    select_lowest_slots,
)
from .limit import Limit


class SlotHeightLimit(Limit):
    r"""Keep slot coordinates above a margin.

    Linearized :math:`p_{i,\mathrm{axis}} + J_{\mathrm{axis}} \Delta q \ge
    \mathrm{margin}`, i.e. :math:`-J_{\mathrm{axis}} \Delta q \le
    p_{i,\mathrm{axis}} - \mathrm{margin}`.

    Up to ``max_points`` lowest slots with
    :math:`p_{\mathrm{axis}} < \mathrm{margin} + \mathrm{threshold}` are active
    each step. Padded rows are inert (:math:`G=0`, :math:`h=` ``BOX_INF``).

    Selection and scatter stay on device (fixed launch dims) so this limit can
    participate in CUDA graphs and in ``nworld`` batches.
    """

    box_capable = False
    supports_cuda_graph = True

    def __init__(
        self,
        cloud: BodyPointCloud,
        axis: int = 2,
        margin: float = 0.0,
        threshold: float = 0.05,
        max_points: int = 128,
        slot_ids: np.ndarray | None = None,
    ):
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        if int(max_points) < 1:
            raise ValueError(f"max_points must be >= 1, got {max_points}")
        self.cloud = cloud
        self.axis = int(axis)
        self.margin = float(margin)
        self.threshold = float(threshold)
        if slot_ids is None:
            self._slot_ids = np.arange(cloud.n_slots, dtype=np.int32)
        else:
            self._slot_ids = np.asarray(slot_ids, dtype=np.int32).reshape(-1)
        n_cand = int(self._slot_ids.size)
        self.n_inequalities = min(int(max_points), max(n_cand, 1))
        self.max_points = self.n_inequalities
        self._n_cand = n_cand
        self._slot_ids_dev: wp.array | None = None
        self._selected: wp.array | None = None
        self._nworld: int | None = None
        self._device: str | None = None

    def _ensure_buffers(self, configuration: Configuration) -> None:
        nworld = configuration.nworld
        device = configuration.device
        if (
            self._selected is not None
            and self._slot_ids_dev is not None
            and self._nworld == nworld
            and self._device == device
        ):
            return
        with wp.ScopedDevice(device):
            self._slot_ids_dev = wp.array(self._slot_ids, dtype=wp.int32)
            self._selected = wp.zeros((nworld, self.max_points), dtype=wp.int32)
        self._nworld = nworld
        self._device = device

    def scatter_inequalities(
        self,
        configuration: Configuration,
        dt: float,
        row_offset: int,
        G: wp.array,
        h: wp.array,
    ) -> None:
        del dt
        kin = self.cloud.eval(configuration)
        self._ensure_buffers(configuration)
        assert self._slot_ids_dev is not None
        assert self._selected is not None
        nworld = configuration.nworld
        nv = configuration.nv
        device = configuration.device
        with wp.ScopedDevice(device):
            wp.launch(
                reset_ineq_block,
                dim=(nworld, self.max_points),
                inputs=[row_offset],
                outputs=[G, h],
            )
            wp.launch(
                fill_i32_2d,
                dim=(nworld, self.max_points),
                inputs=[self._selected, -1],
            )
            wp.launch(
                select_lowest_slots,
                dim=(nworld, self._n_cand),
                inputs=[
                    kin.xpos,
                    self._slot_ids_dev,
                    self._n_cand,
                    self.axis,
                    self.margin,
                    self.threshold,
                    self.max_points,
                ],
                outputs=[self._selected],
            )
            wp.launch(
                scatter_selected_axis_limit,
                dim=(nworld, self.max_points, nv),
                inputs=[
                    self._selected,
                    kin.jacp,
                    kin.xpos,
                    self.axis,
                    self.margin,
                    row_offset,
                ],
                outputs=[G, h],
            )
