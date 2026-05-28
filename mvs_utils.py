"""Multi-view geometry helpers around pycolmap.

This module extracts the intrinsics/extrinsics we need for triangulation,
in a shape-normalised form that doesn't depend on pycolmap's exact version.

Everything here is pure numpy + pycolmap — no torch, no kornia — so it can
run inside the HuggingFace submission container without installs.

Key data structure: ``ViewInfo`` (a plain dict) with keys:

    image_id       str   — the short sample-level id (matches entry['image_ids'])
    colmap_img     pycolmap.Image
    camera_id      int
    K              (3,3) float64 — calibration matrix
    R              (3,3) float64 — world→camera rotation
    t              (3,)  float64 — world→camera translation
    P              (3,4) float64 — K @ [R | t]  (projection matrix)
    center         (3,)  float64 — camera centre in world coords, -R^T t
    width, height  int   — image resolution at COLMAP scale

Downstream code uses ``P`` for DLT triangulation and ``K, R, t`` for epipolar
geometry. All functions here are side-effect-free.
"""

from __future__ import annotations

import numpy as np

from hoho2025.example_solutions import _cam_matrix_from_image


def get_view_info(colmap_rec, img_id_substring: str) -> dict | None:
    """Return ViewInfo for the COLMAP image whose name contains ``img_id_substring``.

    Returns None if the image is not registered in the reconstruction.
    """
    found = None
    for _, col_img in colmap_rec.images.items():
        if img_id_substring in col_img.name:
            found = col_img
            break
    if found is None:
        return None

    R, t = _cam_matrix_from_image(found)
    cam = colmap_rec.cameras[found.camera_id]
    K = np.asarray(cam.calibration_matrix(), dtype=np.float64)
    P = K @ np.hstack([R, t.reshape(3, 1)])
    center = -R.T @ t

    return {
        "image_id": img_id_substring,
        "colmap_img": found,
        "camera_id": int(found.camera_id),
        "K": K,
        "R": R,
        "t": t,
        "P": P,
        "center": center,
        "width": int(cam.width),
        "height": int(cam.height),
    }


def collect_views(colmap_rec, image_ids) -> dict[str, dict]:
    """Build a mapping ``{image_id → ViewInfo}`` for every id found in the recon.

    Skips ids that are not registered (returns fewer items than requested
    — caller must handle the missing keys).
    """
    out: dict[str, dict] = {}
    for iid in image_ids:
        info = get_view_info(colmap_rec, iid)
        if info is not None:
            out[iid] = info
    return out


def project_world_to_image(P: np.ndarray, points3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project Nx3 world points through a 3x4 projection matrix.

    Returns
    -------
    uv : (N, 2) float64 — pixel coordinates
    z  : (N,)  float64 — camera-space depth (>0 means in front of the camera)
    """
    pts = np.asarray(points3d, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    homog = np.hstack([pts, np.ones((len(pts), 1))])
    proj = homog @ P.T  # (N, 3)
    z = proj[:, 2]
    safe = np.where(np.abs(z) < 1e-12, 1e-12, z)
    uv = proj[:, :2] / safe[:, None]
    return uv, z


def relative_pose(view_a: dict, view_b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return rotation and translation from view_a's frame to view_b's frame.

    If x_a is a point in view_a's camera frame, then
        x_b = R_ab @ x_a + t_ab
    with
        R_ab = R_b @ R_a^T
        t_ab = t_b - R_ab @ t_a
    """
    R_a, t_a = view_a["R"], view_a["t"]
    R_b, t_b = view_b["R"], view_b["t"]
    R_ab = R_b @ R_a.T
    t_ab = t_b - R_ab @ t_a
    return R_ab, t_ab


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y],
                     [z,  0, -x],
                     [-y, x,  0]], dtype=np.float64)


def fundamental_matrix(view_a: dict, view_b: dict) -> np.ndarray:
    """Compute the fundamental matrix F_ab such that
        x_b^T @ F_ab @ x_a = 0
    for corresponding points (in homogeneous pixel coordinates).

    Derivation: F = K_b^{-T} · [t_ab]× · R_ab · K_a^{-1}
    """
    R_ab, t_ab = relative_pose(view_a, view_b)
    K_a_inv = np.linalg.inv(view_a["K"])
    K_b_inv_T = np.linalg.inv(view_b["K"]).T
    E = _skew(t_ab) @ R_ab  # essential matrix
    F = K_b_inv_T @ E @ K_a_inv
    return F


def epipolar_line(F: np.ndarray, point_in_a: np.ndarray) -> np.ndarray:
    """Epipolar line in view b induced by a point in view a.

    Returns ``(a, b, c)`` with ``a*u + b*v + c = 0`` in view b.
    """
    x = np.array([point_in_a[0], point_in_a[1], 1.0], dtype=np.float64)
    return F @ x


def point_to_line_distance(line: np.ndarray, point_uv: np.ndarray) -> float:
    """Perpendicular distance from a 2D point to a homogeneous line (a,b,c)."""
    a, b, c = line
    num = abs(a * point_uv[0] + b * point_uv[1] + c)
    den = np.sqrt(a * a + b * b) + 1e-12
    return float(num / den)


def triangulate_dlt(Ps, pts2d) -> np.ndarray:
    """Linear triangulation (DLT) from ``>=2`` views.

    Parameters
    ----------
    Ps : sequence of (3,4) projection matrices
    pts2d : sequence of (x, y) pixel coordinates, one per view

    Returns the 3D point as a (3,) ndarray in world coordinates.
    """
    A = []
    for P, (x, y) in zip(Ps, pts2d):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64)
    try:
        _, _, Vt = np.linalg.svd(A)
    except Exception:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return X[:3] / X[3]


def mean_reprojection_error(X: np.ndarray, Ps, pts2d) -> float:
    """Mean L2 reprojection error of ``X`` across multiple views.

    Points behind the camera (depth <= 0) contribute a large penalty so the
    caller can use this as a direct cost for track acceptance.
    """
    if np.any(~np.isfinite(X)):
        return float("inf")
    errs = []
    for P, uv in zip(Ps, pts2d):
        u, z = project_world_to_image(P, X.reshape(1, 3))
        if z[0] <= 0:
            return float("inf")
        errs.append(float(np.linalg.norm(u[0] - np.asarray(uv, dtype=np.float64))))
    if not errs:
        return float("inf")
    return float(np.mean(errs))
