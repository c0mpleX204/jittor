from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except ImportError:
    cKDTree = None
    HAS_SCIPY = False

try:
    import point_cloud_utils as pcu

    HAS_PCU = True
except ImportError:
    pcu = None
    HAS_PCU = False


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        points = np.load(path)
    else:
        points = np.loadtxt(path, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"{path}: expected an (N, 3) point cloud")
    return points[:, :3]


def load_mesh(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_PCU:
        raise RuntimeError(
            "P2S needs point-cloud-utils. Install it in your Jittor environment with: "
            "python -m pip install point-cloud-utils"
        )
    vertices, faces = pcu.load_mesh_vf(str(path))
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def unit_sphere_params(ref_points: np.ndarray) -> Tuple[np.ndarray, float]:
    center = (ref_points.max(axis=0) + ref_points.min(axis=0)) / 2.0
    centered = ref_points - center
    scale = float(np.sqrt((centered * centered).sum(axis=1)).max())
    if scale < 1e-12:
        scale = 1.0
    return center, scale


def apply_unit_sphere(points: np.ndarray, center: np.ndarray, scale: float) -> np.ndarray:
    return (points - center) / scale


def maybe_normalize(
    points: np.ndarray,
    mesh_vertices: Optional[np.ndarray] = None,
    normalize_ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if normalize_ref is None:
        return points, mesh_vertices
    center, scale = unit_sphere_params(normalize_ref)
    points_n = apply_unit_sphere(points, center, scale)
    mesh_n = None if mesh_vertices is None else apply_unit_sphere(mesh_vertices, center, scale)
    return points_n, mesh_n


def p2s_distances(points: np.ndarray, mesh_vertices: np.ndarray, mesh_faces: np.ndarray) -> np.ndarray:
    if not HAS_PCU:
        raise RuntimeError(
            "P2S needs point-cloud-utils. Install it in your Jittor environment with: "
            "python -m pip install point-cloud-utils"
        )
    distances, _, _ = pcu.closest_points_on_mesh(
        points.astype(np.float32),
        mesh_vertices.astype(np.float32),
        mesh_faces.astype(np.int32),
    )
    return np.asarray(distances, dtype=np.float64)


def nearest_distances(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    if HAS_SCIPY:
        tree = cKDTree(ref)
        distances, _ = tree.query(src, k=1)
        return np.asarray(distances, dtype=np.float64)

    if len(src) * len(ref) > 4_000_000:
        raise RuntimeError(
            "CD needs scipy for large point clouds. Install scipy or use the Jittor environment."
        )

    out = np.empty(len(src), dtype=np.float64)
    chunk = 512
    for start in range(0, len(src), chunk):
        end = min(start + chunk, len(src))
        diff = src[start:end, None, :] - ref[None, :, :]
        out[start:end] = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
    return out


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    a_to_b = nearest_distances(a, b)
    b_to_a = nearest_distances(b, a)
    return float(np.mean(a_to_b * a_to_b) + np.mean(b_to_a * b_to_a)), a_to_b, b_to_a


def official_score(pred_metric: Optional[float], noisy_metric: Optional[float]) -> Optional[float]:
    if pred_metric is None or noisy_metric is None:
        return None
    if noisy_metric < 1e-15:
        return 100.0 if pred_metric < 1e-15 else 0.0
    return float(np.clip(100.0 * (1.0 - pred_metric / noisy_metric), 0.0, 100.0))


def metric_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_mean_sq": float(np.mean(values * values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_p99": float(np.percentile(values, 99)),
        f"{prefix}_max": float(np.max(values)),
    }


def heatmap_colors(
    values: np.ndarray,
    clip_percentile: float = 99.0,
    color_min: Optional[float] = None,
    color_max: Optional[float] = None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros((len(values), 3), dtype=np.uint8)
    vmin = float(np.min(finite)) if color_min is None else float(color_min)
    vmax = float(np.percentile(finite, clip_percentile)) if color_max is None else float(color_max)
    if vmax <= vmin:
        vmax = float(np.max(finite))
    if vmax <= vmin:
        vmax = vmin + 1e-12
    t = np.clip((values - vmin) / (vmax - vmin), 0.0, 1.0)

    stops = np.array(
        [
            [35, 80, 220],
            [40, 210, 240],
            [245, 230, 60],
            [220, 40, 35],
        ],
        dtype=np.float64,
    )
    scaled = t * (len(stops) - 1)
    lo = np.floor(scaled).astype(int)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    frac = (scaled - lo)[:, None]
    colors = stops[lo] * (1.0 - frac) + stops[hi] * frac
    return np.clip(colors, 0, 255).astype(np.uint8)


def improvement_colors(
    improvement: np.ndarray,
    neutral_threshold: float = 0.0,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    improvement = np.asarray(improvement, dtype=np.float64)
    colors = np.full((len(improvement), 3), 170, dtype=np.uint8)
    finite_abs = np.abs(improvement[np.isfinite(improvement)])
    if finite_abs.size == 0:
        return colors
    scale = float(np.percentile(finite_abs, clip_percentile))
    if scale <= 1e-12:
        scale = float(np.max(finite_abs)) if finite_abs.size else 1.0
    if scale <= 1e-12:
        scale = 1.0

    good = improvement > neutral_threshold
    bad = improvement < -neutral_threshold
    g = np.clip(np.abs(improvement) / scale, 0.0, 1.0)

    # More intense color means larger change. Green is closer to the surface;
    # red is farther from the surface.
    colors[good, 0] = (130 * (1.0 - g[good]) + 25 * g[good]).astype(np.uint8)
    colors[good, 1] = (190 * (1.0 - g[good]) + 230 * g[good]).astype(np.uint8)
    colors[good, 2] = (130 * (1.0 - g[good]) + 70 * g[good]).astype(np.uint8)
    colors[bad, 0] = (190 * (1.0 - g[bad]) + 235 * g[bad]).astype(np.uint8)
    colors[bad, 1] = (130 * (1.0 - g[bad]) + 40 * g[bad]).astype(np.uint8)
    colors[bad, 2] = (130 * (1.0 - g[bad]) + 40 * g[bad]).astype(np.uint8)
    return colors


def write_ply(
    path: Path,
    points: np.ndarray,
    scalar_fields: Optional[Dict[str, np.ndarray]] = None,
    colors: Optional[np.ndarray] = None,
) -> None:
    scalar_fields = scalar_fields or {}
    n = len(points)
    for name, values in scalar_fields.items():
        if len(values) != n:
            raise ValueError(f"{name}: scalar length {len(values)} does not match {n} points")
    if colors is not None and len(colors) != n:
        raise ValueError(f"color length {len(colors)} does not match {n} points")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        for name in scalar_fields:
            f.write(f"property float {name}\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")

        for i in range(n):
            row = [f"{points[i, 0]:.8f}", f"{points[i, 1]:.8f}", f"{points[i, 2]:.8f}"]
            for values in scalar_fields.values():
                value = float(values[i])
                row.append("nan" if math.isnan(value) else f"{value:.8e}")
            if colors is not None:
                row.extend(str(int(c)) for c in colors[i])
            f.write(" ".join(row) + "\n")


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def common_keys(*sample_maps: Dict[str, Path]) -> Iterable[str]:
    keys = set(sample_maps[0].keys())
    for sample_map in sample_maps[1:]:
        keys &= set(sample_map.keys())
    return sorted(keys)
