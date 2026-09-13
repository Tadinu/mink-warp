"""SlotHeightLimit scatters linearized p_z >= margin rows."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp
from test_point_match_task_parity import _cloud, _cpu_slot

import mink_warp as mw
from mink_warp.kernels.constrained import BOX_INF, reset_ineq_block


def test_height_limit_row_matches_jacobian(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, jp0, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    margin = float(p0[2] + 0.02)
    lim = mw.SlotHeightLimit(cloud, axis=2, margin=margin, threshold=1.0, max_points=4)
    nworld, nv = 1, cfg.nv
    with wp.ScopedDevice(cfg.device):
        G = wp.zeros((nworld, lim.n_inequalities, nv), dtype=float)
        h = wp.zeros((nworld, lim.n_inequalities), dtype=float)
        wp.launch(reset_ineq_block, dim=(nworld, lim.n_inequalities), inputs=[0], outputs=[G, h])
    lim.scatter_inequalities(cfg, dt=1.0, row_offset=0, G=G, h=h)
    G_np, h_np = G.numpy()[0], h.numpy()[0]
    # Lowest slots first; both are typically near the same z. At least one row active.
    active = np.isfinite(h_np) & (h_np < 0.5 * float(BOX_INF))
    assert np.any(active)
    row = int(np.flatnonzero(active)[0])
    # G = -J_z, h = z - margin
    kin = cloud.eval(cfg)
    z = kin.xpos.numpy()[0, :, 2]
    sid = int(np.argmin(z))
    np.testing.assert_allclose(G_np[row], -kin.jacp.numpy()[0, sid, 2], atol=1e-4)
    np.testing.assert_allclose(h_np[row], z[sid] - margin, atol=1e-5)


def test_height_limit_inactive_when_above_threshold(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, _, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    # Margin far below the arm: nothing is close to the floor.
    lim = mw.SlotHeightLimit(
        cloud, axis=2, margin=float(p0[2] - 10.0), threshold=0.01, max_points=4
    )
    with wp.ScopedDevice(cfg.device):
        G = wp.zeros((1, lim.n_inequalities, cfg.nv), dtype=float)
        h = wp.zeros((1, lim.n_inequalities), dtype=float)
    lim.scatter_inequalities(cfg, dt=1.0, row_offset=0, G=G, h=h)
    np.testing.assert_allclose(G.numpy(), 0.0, atol=1e-8)
    assert np.all(h.numpy() > 1e6)


def test_height_limit_lifts_penetrating_slot(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, _, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    # Planar arm moves in xy; constrain world-x so joints can satisfy the row.
    axis = 0
    margin = float(p0[axis] + 0.04)
    task = mw.PointMatchTask(cloud, slot_cost=0.05)
    task.set_target_from_configuration(cfg)
    lim = mw.SlotHeightLimit(
        cloud, axis=axis, margin=margin, threshold=1.0, max_points=4
    )
    solver = mw.ConstrainedSolver(cfg, limits=[lim], damping=1e-3, admm_iters=60)
    x_before = float(cloud.eval(cfg).xpos.numpy()[0, :, axis].min())
    assert x_before < margin
    solver.solve_and_integrate([task], dt=1.0, iterations=12)
    cloud.invalidate()
    x_after = float(cloud.eval(cfg).xpos.numpy()[0, :, axis].min())
    assert x_after > x_before + 5e-3
    assert (margin - x_after) < (margin - x_before)


def test_height_limit_nworld_independent(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0.copy()
    q1[0] += 0.35
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)
    p0, _, _, _ = _cpu_slot(arm_model, q0, geom_ids[0], local[0], nrm[0])
    axis = 0
    margin = float(p0[axis] + 0.02)
    lim = mw.SlotHeightLimit(cloud, axis=axis, margin=margin, threshold=1.0, max_points=4)
    with wp.ScopedDevice(cfg.device):
        G = wp.zeros((2, lim.n_inequalities, cfg.nv), dtype=float)
        h = wp.zeros((2, lim.n_inequalities), dtype=float)
    lim.scatter_inequalities(cfg, dt=1.0, row_offset=0, G=G, h=h)
    kin = cloud.eval(cfg)
    z = kin.xpos.numpy()[:, :, axis]
    G_np, h_np = G.numpy(), h.numpy()
    for w in range(2):
        active = np.isfinite(h_np[w]) & (h_np[w] < 0.5 * float(BOX_INF))
        assert np.any(active)
        row = int(np.flatnonzero(active)[0])
        sid = int(np.argmin(z[w]))
        np.testing.assert_allclose(G_np[w, row], -kin.jacp.numpy()[w, sid, axis], atol=1e-4)
        np.testing.assert_allclose(h_np[w, row], z[w, sid] - margin, atol=1e-5)
        # Worlds must not share a single host-built row list.
        assert int(lim._selected.numpy()[w, 0]) >= 0


def test_height_limit_cuda_graph_matches_eager(arm_model):
    if not wp.get_cuda_devices():
        pytest.skip("no CUDA device; graph capture is CUDA-only")
    device = "cuda:0"
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    p0, _, _, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    axis = 0
    margin = float(p0[axis] + 0.04)

    def rollout(use_graph: bool) -> np.ndarray:
        cfg = mw.Configuration(arm_model, q=q, nworld=1, device=device)
        task = mw.PointMatchTask(cloud, slot_cost=0.05)
        task.set_target_from_configuration(cfg)
        lim = mw.SlotHeightLimit(
            cloud, axis=axis, margin=margin, threshold=1.0, max_points=4
        )
        assert lim.supports_cuda_graph
        solver = mw.ConstrainedSolver(cfg, limits=[lim], damping=1e-3, admm_iters=60)
        solver.solve_and_integrate([task], dt=1.0, iterations=1, use_graph=use_graph)
        for _ in range(8):
            solver.solve_and_integrate([task], dt=1.0, iterations=1, use_graph=use_graph)
        return cfg.q.numpy()[0].copy()

    q_graph = rollout(True)
    q_eager = rollout(False)
    np.testing.assert_allclose(q_graph, q_eager, atol=5e-4)
