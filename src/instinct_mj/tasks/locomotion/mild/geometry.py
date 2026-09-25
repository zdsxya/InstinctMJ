"""Area quadrature of the union of G1's horizontal sole capsules."""

from __future__ import annotations

import mujoco
import numpy as np

from instinct_mj.assets.unitree_g1 import get_g1_spec

FOOT_NAMES = ("left_ankle_roll_link", "right_ankle_roll_link")


def mild_g1_spec() -> mujoco.MjSpec:
    """Preserve self/body collisions, exclude only foot/ground contacts.

    Ground retains bit 1. Other robot geoms accept bits 1 and 2, while feet
    only use bit 2. Visual geoms remain noncolliding. No shared asset is edited.
    """
    spec = get_g1_spec()
    for body in spec.bodies:
        for geom in body.geoms:
            if not (geom.contype or geom.conaffinity):
                continue
            if body.name in FOOT_NAMES:
                geom.contype = geom.conaffinity = 2
            else:
                geom.contype = 1
                geom.conaffinity = 3
    return spec


def sole_quadrature(model: mujoco.MjModel, body_id: int, cell_size: float, subdivisions: int):
    """Integrate the projected union, not the sum of overlapping capsules.

    Each occupied cell contributes its union area and area-weighted lower
    surface centroid. Fine subcells resolve capsule ends and edge fractions.
    """
    capsules = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id or not model.geom_contype[gid]:
            continue
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError("MILD G1 sole quadrature expects horizontal capsule collision geoms")
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        axis = rotation.reshape(3, 3)[:, 2]
        if abs(axis[2]) > 1.0e-4:
            raise ValueError("MILD sole capsule axis must lie in the local xy plane")
        center = model.geom_pos[gid]
        radius, half_length = model.geom_size[gid, :2]
        capsules.append((center - half_length * axis, center + half_length * axis, radius))
    if not capsules:
        raise ValueError("No sole collision capsules found")
    lower = np.min([np.minimum(a[:2], b[:2]) - r for a, b, r in capsules], axis=0)
    upper = np.max([np.maximum(a[:2], b[:2]) + r for a, b, r in capsules], axis=0)
    counts = np.ceil((upper - lower) / cell_size).astype(int)
    spacing = (upper - lower) / counts
    points, areas = [], []
    for ix in range(counts[0]):
        for iy in range(counts[1]):
            xy = np.array([
                lower + (np.array([ix, iy]) + (np.array([sx, sy]) + 0.5) / subdivisions) * spacing
                for sx in range(subdivisions) for sy in range(subdivisions)
            ])
            z = np.full(len(xy), np.inf)
            for a, b, radius in capsules:
                segment = b[:2] - a[:2]
                along = np.clip(((xy - a[:2]) @ segment) / (segment @ segment), 0, 1)
                delta = xy - (a[:2] + along[:, None] * segment)
                distance2 = np.sum(delta**2, axis=-1)
                inside = distance2 <= radius**2
                bottom = a[2] - np.sqrt(np.maximum(radius**2 - distance2, 0))
                z = np.minimum(z, np.where(inside, bottom, np.inf))
            mask = np.isfinite(z)
            if mask.any():
                points.append(np.r_[xy[mask].mean(0), z[mask].mean()])
                areas.append(float(np.prod(spacing) * mask.mean()))
    return np.asarray(points, dtype=np.float32), np.asarray(areas, dtype=np.float32)
