"""BodyPointCloud / PointMatchTask / NormalMatchTask vs CPU mj_jac."""

from __future__ import annotations

import mujoco
import numpy as np

import mink_warp as mw


def _cpu_slot(model, q, geom_id, local_pos, local_nrm):
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    rot = data.geom_xmat[int(geom_id)].reshape(3, 3)
    pos = rot @ np.asarray(local_pos, dtype=np.float64) + data.geom_xpos[int(geom_id)]
    nrm = rot @ np.asarray(local_nrm, dtype=np.float64)
    nrm = nrm / max(float(np.linalg.norm(nrm)), 1e-12)
    body = int(model.geom_bodyid[int(geom_id)])
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    mujoco.mj_jac(model, data, jacp, jacr, pos, body)
    skew = np.array(
        [
            [0.0, -nrm[2], nrm[1]],
            [nrm[2], 0.0, -nrm[0]],
            [-nrm[1], nrm[0], 0.0],
        ]
    )
    return pos, nrm, jacp, -skew @ jacr


def _cloud(arm_model, locals=None, normals=None):
    local = np.asarray(
        locals if locals is not None else [[0.08, 0.0, 0.0], [0.22, 0.0, 0.0]],
        dtype=np.float32,
    )
    nrm = np.asarray(
        normals if normals is not None else [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    geom_ids = np.array([1, 1], dtype=np.int32)
    return mw.BodyPointCloud(arm_model, geom_ids, local, nrm), local, nrm, geom_ids


def test_slot_fk_and_jacobian_match_mj_jac(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    kin = cloud.eval(cfg)
    xpos = kin.xpos.numpy()[0]
    nrm_wp = kin.nrm.numpy()[0]
    jacp = kin.jacp.numpy()[0]
    jacn = kin.jacn.numpy()[0]
    for i, gid in enumerate(geom_ids):
        p, n, jp, jn = _cpu_slot(arm_model, q, gid, local[i], nrm[i])
        np.testing.assert_allclose(xpos[i], p, atol=1e-5)
        np.testing.assert_allclose(nrm_wp[i], n, atol=1e-5)
        np.testing.assert_allclose(jacp[i], jp, atol=1e-4)
        np.testing.assert_allclose(jacn[i], jn, atol=1e-4)


def test_point_match_error_and_dls_step(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.PointMatchTask(cloud, slot_cost=4.0)
    p_star = []
    for i, gid in enumerate(geom_ids):
        p, _, _, _ = _cpu_slot(arm_model, q, gid, local[i], nrm[i])
        p_star.append(p + np.array([0.01, -0.02, 0.0]))
    p_star = np.asarray(p_star, dtype=np.float32)
    task.set_target(p_star)

    err = task.compute_error(cfg).numpy()[0].reshape(-1, 3)
    J = task.compute_jacobian(cfg).numpy()[0]
    kin = cloud.eval(cfg)
    np.testing.assert_allclose(err, kin.xpos.numpy()[0] - p_star, atol=1e-5)

    solver = mw.DLSSolver(cfg, damping=1e-4)
    v = solver.solve([task], dt=1.0).numpy()[0]
    cost = task.cost
    W = np.diag(cost)
    WJ = W @ J
    We = cost * task.compute_error(cfg).numpy()[0]
    H = WJ.T @ WJ + 1e-4 * np.eye(cfg.nv)
    dq = np.linalg.solve(H, -(WJ.T @ We))
    np.testing.assert_allclose(v, dq, atol=2e-4)


def test_normal_match_error(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.NormalMatchTask(cloud, slot_cost=1.0)
    targets = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    task.set_target(targets, configuration=cfg)
    err = task.compute_error(cfg).numpy()[0].reshape(-1, 3)
    kin = cloud.eval(cfg)
    hat = targets / np.linalg.norm(targets, axis=1, keepdims=True)
    np.testing.assert_allclose(err, kin.nrm.numpy()[0] - hat, atol=1e-5)


def test_zero_slot_cost_zeros_jacobian(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, local, nrm, geom_ids = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.PointMatchTask(cloud, slot_cost=[1.0, 0.0])
    task.set_target_from_configuration(cfg)
    p = cloud.eval(cfg).xpos.numpy()[0].copy()
    p[1] += 0.1
    task.set_target(p)
    J = task.compute_jacobian(cfg).numpy()[0]
    np.testing.assert_allclose(J[3:6], 0.0, atol=1e-8)
    assert np.linalg.norm(J[0:3]) > 1e-6


def test_nworld_independence(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0 + np.array([0.2, -0.15])
    cloud, _, _, _ = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)
    task = mw.PointMatchTask(cloud, slot_cost=1.0)
    task.set_target_from_configuration(cfg)
    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err, 0.0, atol=1e-5)

    solo = mw.Configuration(arm_model, q=q0, nworld=1)
    t1 = mw.PointMatchTask(cloud, slot_cost=1.0)
    # Fresh cloud eval cache is per-cloud; invalidate after switching cfg.
    cloud.invalidate()
    t1.set_target_from_configuration(solo)
    p = t1.compute_error(solo).numpy()
    np.testing.assert_allclose(p, 0.0, atol=1e-5)

    shifted = cloud.eval(cfg).xpos.numpy().copy()
    shifted[1] += 0.05
    task.set_target(shifted)
    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err[0], 0.0, atol=1e-5)
    assert np.linalg.norm(err[1]) > 1e-3


def test_eval_cache_follows_kin_stamp(arm_model):
    q = arm_model.key_qpos[0].copy()
    cloud, _, _, _ = _cloud(arm_model)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    kin0 = cloud.eval(cfg)
    before = kin0.xpos.numpy().copy()
    assert cloud.eval(cfg) is kin0
    cfg.update(q=q + np.array([0.1, 0.0]))
    after = cloud.eval(cfg).xpos.numpy()
    assert not np.allclose(after, before)
