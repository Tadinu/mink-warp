"""PairDistanceTask one-sided residual and AxisMatchTask."""

from __future__ import annotations

import numpy as np
from test_point_match_task_parity import _cloud, _cpu_slot

import mink_warp as mw


def test_pair_distance_matches_cpu(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, jp0, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    p1, _, jp1, _ = _cpu_slot(arm_model, q, geom_ids[1], local[1], nrm[1])
    d = float(np.linalg.norm(p0 - p1))
    u = (p0 - p1) / d
    dstar = d + 0.05
    task = mw.PairDistanceTask(cloud, pairs=[[0, 1]], pair_cost=9.0, one_sided=True)
    task.set_target(np.array([dstar], dtype=np.float32))
    e = task.compute_error(cfg).numpy()[0, 0]
    J = task.compute_jacobian(cfg).numpy()[0, 0]
    np.testing.assert_allclose(e, d - dstar, atol=1e-5)
    np.testing.assert_allclose(J, u @ (jp0 - jp1), atol=1e-4)


def test_pair_distance_one_sided_inactive_when_far(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, _, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    p1, _, _, _ = _cpu_slot(arm_model, q, geom_ids[1], local[1], nrm[1])
    d = float(np.linalg.norm(p0 - p1))
    task = mw.PairDistanceTask(cloud, pairs=[[0, 1]], pair_cost=1.0, one_sided=True)
    # Inactive when current distance is already at least the target.
    task.set_target(np.array([d * 0.5], dtype=np.float32))
    e = task.compute_error(cfg).numpy()
    J = task.compute_jacobian(cfg).numpy()
    np.testing.assert_allclose(e, 0.0, atol=1e-8)
    np.testing.assert_allclose(J, 0.0, atol=1e-8)

    two_sided = mw.PairDistanceTask(cloud, pairs=[[0, 1]], pair_cost=1.0, one_sided=False)
    two_sided.set_target(np.array([d * 0.5], dtype=np.float32))
    e2 = two_sided.compute_error(cfg).numpy()[0, 0]
    assert abs(e2) > 1e-4


def test_set_pair_cost_reuploads_active_mask(arm_model):
    """set_pair_cost after the first eval must keep working (in-place device upload)."""
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, _, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    p1, _, _, _ = _cpu_slot(arm_model, q, geom_ids[1], local[1], nrm[1])
    d = float(np.linalg.norm(p0 - p1))
    task = mw.PairDistanceTask(cloud, pairs=[[0, 1]], pair_cost=1.0, one_sided=False)
    task.set_target(np.array([d], dtype=np.float32))
    task.compute_error(cfg)
    task.set_pair_cost(4.0)
    task.set_target(np.array([d + 0.01], dtype=np.float32))
    e = task.compute_error(cfg).numpy()[0, 0]
    np.testing.assert_allclose(e, d - (d + 0.01), atol=1e-5)


def test_set_pairs_per_world(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0.copy()
    q1[0] += 0.2
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)
    p10, _, _, _ = _cpu_slot(arm_model, q1, geom_ids[0], local[0], nrm[0])
    p11, _, _, _ = _cpu_slot(arm_model, q1, geom_ids[1], local[1], nrm[1])
    d1 = float(np.linalg.norm(p10 - p11))
    task = mw.PairDistanceTask(
        cloud, pairs=[[0, 0], [0, 0]], pair_cost=[0.0, 1.0], one_sided=False
    )
    task.set_pairs(np.array([[[0, 0], [0, 0]], [[0, 0], [0, 1]]], dtype=np.int32))
    task.set_target(np.array([[0.0, 0.0], [0.0, d1 + 0.05]], dtype=np.float32))
    e = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(e[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(e[1, 0], 0.0, atol=1e-8)
    np.testing.assert_allclose(e[1, 1], d1 - (d1 + 0.05), atol=1e-5)


def test_set_pairs_updates_active_pair(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    p0, _, jp0, _ = _cpu_slot(arm_model, q, geom_ids[0], local[0], nrm[0])
    p1, _, jp1, _ = _cpu_slot(arm_model, q, geom_ids[1], local[1], nrm[1])
    d = float(np.linalg.norm(p0 - p1))
    u = (p0 - p1) / d
    task = mw.PairDistanceTask(cloud, pairs=[[0, 0], [0, 0]], pair_cost=[0.0, 9.0], one_sided=False)
    task.set_pairs([[0, 0], [0, 1]])
    task.set_target(np.array([0.0, d + 0.05], dtype=np.float32))
    e = task.compute_error(cfg).numpy()[0]
    J = task.compute_jacobian(cfg).numpy()[0]
    np.testing.assert_allclose(e[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(J[0], 0.0, atol=1e-8)
    np.testing.assert_allclose(e[1], d - (d + 0.05), atol=1e-5)
    np.testing.assert_allclose(J[1], u @ (jp0 - jp1), atol=1e-4)


def test_axis_match_z(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    z_star = np.array([0.0, 0.05], dtype=np.float32)
    task = mw.AxisMatchTask(cloud, slot_cost=1.0, axis=2)
    task.set_target(z_star)
    e = task.compute_error(cfg).numpy()[0]
    J = task.compute_jacobian(cfg).numpy()[0]
    for i, gid in enumerate(geom_ids):
        p, _, jp, _ = _cpu_slot(arm_model, q, gid, local[i], nrm[i])
        np.testing.assert_allclose(e[i], p[2] - z_star[i], atol=1e-5)
        np.testing.assert_allclose(J[i], jp[2], atol=1e-4)
