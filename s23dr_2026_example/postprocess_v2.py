"""Post-processing functions for segment predictions."""
import numpy as np


def snap_to_point_cloud(vertices, xyz, class_id, snap_radius=0.5,
                         target_classes=None):
    """Snap vertices to nearby point cloud clusters of specific semantic classes."""
    if target_classes is None:
        target_classes = [1, 2]  # apex, eave_end_point

    snapped = vertices.copy()
    mask = np.isin(class_id, target_classes)

    if mask.sum() < 2:
        return snapped

    target_pts = xyz[mask]

    for i, v in enumerate(vertices):
        dists = np.linalg.norm(target_pts - v, axis=-1)
        close = dists < snap_radius
        if close.sum() >= 2:
            snapped[i] = target_pts[close].mean(axis=0)

    return snapped


def snap_horizontal(vertices, edges, max_slope=0.05):
    """Snap near-horizontal edges to be exactly horizontal."""
    verts = vertices.copy()
    for a, b in edges:
        a, b = int(a), int(b)
        dy = abs(verts[a, 1] - verts[b, 1])
        dxz = np.sqrt((verts[a, 0] - verts[b, 0])**2 + (verts[a, 2] - verts[b, 2])**2)
        if dxz > 0.1 and dy / dxz < max_slope:
            avg_y = 0.5 * (verts[a, 1] + verts[b, 1])
            verts[a, 1] = avg_y
            verts[b, 1] = avg_y
    return verts
