"""Shared residual weighting kernel."""

from __future__ import annotations

import warp as wp

@wp.kernel
def weighted_residual(
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
    cost: wp.array[float],
    gain: float,
    lm_damping: float,
    k: int,
    nv: int,
    weighted_jac: wp.array3d[float],
    weighted_err: wp.array2d[float],
    mu_out: wp.array[float],
):
    """Weight one residual row. Launch ``dim=(nworld, k)``; zero ``mu_out`` first."""
    worldid, i = wp.tid()
    if i >= k:
        return
    we = cost[i] * (-gain * error[worldid, i])
    weighted_err[worldid, i] = we
    for j in range(nv):
        weighted_jac[worldid, i, j] = cost[i] * jacobian[worldid, i, j]
    if lm_damping != 0.0:
        wp.atomic_add(mu_out, worldid, lm_damping * we * we)
