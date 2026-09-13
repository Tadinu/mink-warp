"""Kernels for geom-local surface slots: FK, Jacobians, tasks, height limits."""

from __future__ import annotations

import warp as wp


@wp.kernel
def slot_fk(
    geom_xpos: wp.array2d[wp.vec3],
    geom_xmat: wp.array2d[wp.mat33],
    geom_ids: wp.array[int],
    local_pos: wp.array[wp.vec3],
    local_nrm: wp.array[wp.vec3],
    has_normals: int,
    xpos_out: wp.array3d[float],
    nrm_out: wp.array3d[float],
):
    """World position (and optional normal) of each geom-local slot."""
    worldid, sid = wp.tid()
    gid = geom_ids[sid]
    R = geom_xmat[worldid, gid]
    p = geom_xpos[worldid, gid] + R * local_pos[sid]
    xpos_out[worldid, sid, 0] = p[0]
    xpos_out[worldid, sid, 1] = p[1]
    xpos_out[worldid, sid, 2] = p[2]
    if has_normals != 0:
        n = R * local_nrm[sid]
        nlen = wp.sqrt(wp.dot(n, n))
        if nlen > 1.0e-12:
            n = n / nlen
        nrm_out[worldid, sid, 0] = n[0]
        nrm_out[worldid, sid, 1] = n[1]
        nrm_out[worldid, sid, 2] = n[2]


@wp.kernel
def slot_jacobians(
    body_rootid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    body_ids: wp.array[int],
    xpos: wp.array3d[float],
    nrm: wp.array3d[float],
    has_normals: int,
    jacp: wp.array4d[float],
    jacn: wp.array4d[float],
):
    """Point (and normal) Jacobians via the mj_jac / mjwarp formula."""
    worldid, sid, dofid = wp.tid()
    body = body_ids[sid]
    jp = wp.vec3(0.0, 0.0, 0.0)
    jr = wp.vec3(0.0, 0.0, 0.0)
    if body_isdofancestor[body, dofid] != 0:
        cd = cdof[worldid, dofid]
        cang = wp.spatial_top(cd)
        clin = wp.spatial_bottom(cd)
        point = wp.vec3(xpos[worldid, sid, 0], xpos[worldid, sid, 1], xpos[worldid, sid, 2])
        off = point - subtree_com[worldid, body_rootid[body]]
        jp = clin + wp.cross(cang, off)
        jr = cang
    jacp[worldid, sid, 0, dofid] = jp[0]
    jacp[worldid, sid, 1, dofid] = jp[1]
    jacp[worldid, sid, 2, dofid] = jp[2]
    if has_normals != 0:
        n = wp.vec3(nrm[worldid, sid, 0], nrm[worldid, sid, 1], nrm[worldid, sid, 2])
        jn = -wp.cross(n, jr)
        jacn[worldid, sid, 0, dofid] = jn[0]
        jacn[worldid, sid, 1, dofid] = jn[1]
        jacn[worldid, sid, 2, dofid] = jn[2]


@wp.kernel
def slot_vector_residual(
    values: wp.array3d[float],
    jac: wp.array4d[float],
    target: wp.array2d[float],
    slot_active: wp.array[int],
    n_slots: int,
    nv: int,
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
):
    """e = value - target (3 per slot); copy J; zero inactive slots."""
    worldid, sid = wp.tid()
    base = 3 * sid
    active = slot_active[sid]
    for c in range(3):
        row = base + c
        if active != 0:
            error[worldid, row] = values[worldid, sid, c] - target[worldid, row]
        else:
            error[worldid, row] = 0.0
        for d in range(nv):
            if active != 0:
                jacobian[worldid, row, d] = jac[worldid, sid, c, d]
            else:
                jacobian[worldid, row, d] = 0.0


@wp.kernel
def slot_axis_residual(
    xpos: wp.array3d[float],
    jacp: wp.array4d[float],
    target: wp.array2d[float],
    slot_active: wp.array[int],
    axis: int,
    nv: int,
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
):
    worldid, sid = wp.tid()
    if slot_active[sid] != 0:
        error[worldid, sid] = xpos[worldid, sid, axis] - target[worldid, sid]
        for d in range(nv):
            jacobian[worldid, sid, d] = jacp[worldid, sid, axis, d]
    else:
        error[worldid, sid] = 0.0
        for d in range(nv):
            jacobian[worldid, sid, d] = 0.0


@wp.kernel
def pair_distance_residual(
    xpos: wp.array3d[float],
    jacp: wp.array4d[float],
    pair_i: wp.array2d[int],
    pair_j: wp.array2d[int],
    target_dist: wp.array2d[float],
    guides: wp.array3d[float],
    pair_active: wp.array[int],
    one_sided: int,
    nv: int,
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
):
    worldid, pid = wp.tid()
    if pair_active[pid] == 0:
        error[worldid, pid] = 0.0
        for d in range(nv):
            jacobian[worldid, pid, d] = 0.0
        return
    i = pair_i[worldid, pid]
    j = pair_j[worldid, pid]
    di = xpos[worldid, i, 0] - xpos[worldid, j, 0]
    dj = xpos[worldid, i, 1] - xpos[worldid, j, 1]
    dk = xpos[worldid, i, 2] - xpos[worldid, j, 2]
    delta = wp.vec3(di, dj, dk)
    dist = wp.sqrt(wp.dot(delta, delta))
    dstar = target_dist[worldid, pid]
    if one_sided != 0 and dist >= dstar:
        error[worldid, pid] = 0.0
        for d in range(nv):
            jacobian[worldid, pid, d] = 0.0
        return
    u = wp.vec3(1.0, 0.0, 0.0)
    if dist > 1.0e-12:
        u = delta / dist
    else:
        g = wp.vec3(guides[worldid, pid, 0], guides[worldid, pid, 1], guides[worldid, pid, 2])
        glen = wp.sqrt(wp.dot(g, g))
        if glen > 1.0e-12:
            u = g / glen
    error[worldid, pid] = dist - dstar
    for d in range(nv):
        jdiff = wp.vec3(
            jacp[worldid, i, 0, d] - jacp[worldid, j, 0, d],
            jacp[worldid, i, 1, d] - jacp[worldid, j, 1, d],
            jacp[worldid, i, 2, d] - jacp[worldid, j, 2, d],
        )
        jacobian[worldid, pid, d] = wp.dot(u, jdiff)


@wp.kernel
def normalize_target_vectors(
    target: wp.array2d[float],
    n_slots: int,
):
    """In-place unit-normalize each 3-vector in a (nworld, 3N) target."""
    worldid, sid = wp.tid()
    base = 3 * sid
    v = wp.vec3(target[worldid, base], target[worldid, base + 1], target[worldid, base + 2])
    nlen = wp.sqrt(wp.dot(v, v))
    if nlen > 1.0e-12:
        v = v / nlen
    target[worldid, base] = v[0]
    target[worldid, base + 1] = v[1]
    target[worldid, base + 2] = v[2]


@wp.kernel
def flatten_slot_vectors(
    values: wp.array3d[float],
    n_slots: int,
    out: wp.array2d[float],
):
    worldid, sid = wp.tid()
    base = 3 * sid
    out[worldid, base] = values[worldid, sid, 0]
    out[worldid, base + 1] = values[worldid, sid, 1]
    out[worldid, base + 2] = values[worldid, sid, 2]


@wp.kernel
def copy_slot_axis(
    xpos: wp.array3d[float],
    axis: int,
    out: wp.array2d[float],
):
    """out[w, i] = xpos[w, i, axis]. Launch dim=(nworld, n_slots)."""
    worldid, sid = wp.tid()
    out[worldid, sid] = xpos[worldid, sid, axis]


@wp.kernel
def fill_i32_2d(out: wp.array2d[int], value: int):
    i, j = wp.tid()
    out[i, j] = value


@wp.kernel
def select_lowest_slots(
    xpos: wp.array3d[float],
    slot_ids: wp.array[int],
    n_cand: int,
    axis: int,
    margin: float,
    threshold: float,
    max_points: int,
    selected: wp.array2d[int],
):
    """Write the ``max_points`` lowest eligible slots per world into ``selected``.

    Eligibility: all candidates if ``threshold < 0``, else
    ``xpos[axis] < margin + threshold``. Rank is ``(z, slot_id)``.
    Launch ``dim=(nworld, n_cand)``.
    """
    worldid, i = wp.tid()
    if i >= n_cand:
        return
    sid = slot_ids[i]
    zi = xpos[worldid, sid, axis]
    eligible = int(0)
    if threshold < 0.0:
        eligible = 1
    elif zi < margin + threshold:
        eligible = 1
    if eligible == 0:
        return
    rank = int(0)
    for j in range(n_cand):
        sj = slot_ids[j]
        zj = xpos[worldid, sj, axis]
        other = int(0)
        if threshold < 0.0:
            other = 1
        elif zj < margin + threshold:
            other = 1
        if other != 0:
            if zj < zi:
                rank = rank + 1
            elif zj == zi and sj < sid:
                rank = rank + 1
            elif zj == zi and sj == sid and j < i:
                rank = rank + 1
    if rank < max_points:
        selected[worldid, rank] = sid


@wp.kernel
def scatter_selected_axis_limit(
    selected: wp.array2d[int],
    jacp: wp.array4d[float],
    xpos: wp.array3d[float],
    axis: int,
    margin: float,
    row_offset: int,
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Write -J_axis Δq ≤ z - margin for selected[world, rank] (sid < 0 = inert).

    Launch ``dim=(nworld, max_points, nv)``.
    """
    worldid, rank, dofid = wp.tid()
    sid = selected[worldid, rank]
    if sid < 0:
        return
    r = row_offset + rank
    G[worldid, r, dofid] = -jacp[worldid, sid, axis, dofid]
    if dofid == 0:
        h[worldid, r] = xpos[worldid, sid, axis] - margin


@wp.kernel
def scatter_axis_limit_rows(
    worlds: wp.array[int],
    slots: wp.array[int],
    local_rows: wp.array[int],
    jacp: wp.array4d[float],
    xpos: wp.array3d[float],
    axis: int,
    margin: float,
    row_offset: int,
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Write -J_axis Δq ≤ z - margin for K selected (world, slot) contacts."""
    k, dofid = wp.tid()
    w = worlds[k]
    sid = slots[k]
    r = row_offset + local_rows[k]
    G[w, r, dofid] = -jacp[w, sid, axis, dofid]
    if dofid == 0:
        h[w, r] = xpos[w, sid, axis] - margin
