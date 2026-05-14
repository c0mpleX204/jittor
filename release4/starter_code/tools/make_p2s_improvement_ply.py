#!/usr/bin/env python
"""Generate a CloudCompare-ready P2S improvement PLY.

For each corresponding point:
    improvement = distance(noisy, mesh) - distance(denoised, mesh)

The output is located at the denoised point positions. Green means the point
became closer to the clean mesh surface; red means it became farther; gray
means almost unchanged. Scalar fields are included for inspection in
CloudCompare.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pointcloud_eval_utils import (
    improvement_colors,
    load_mesh,
    load_points,
    maybe_normalize,
    metric_stats,
    official_score,
    p2s_distances,
    write_json,
    write_ply,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", required=True, type=Path, help="Noisy input .npy")
    parser.add_argument("--pred", required=True, type=Path, help="Denoised prediction .npy")
    parser.add_argument("--mesh", required=True, type=Path, help="Clean mesh .obj")
    parser.add_argument("--out", required=True, type=Path, help="Output improvement .ply")
    parser.add_argument(
        "--normalize_ref",
        type=Path,
        default=None,
        help="Optional clean.npy used to match the official unit-sphere normalization.",
    )
    parser.add_argument(
        "--neutral_threshold",
        type=float,
        default=0.0,
        help="Absolute improvement below this value is colored gray.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Color clipping percentile. Lower values make small changes more visible.",
    )
    args = parser.parse_args()

    noisy_raw = load_points(args.noisy)
    pred_raw = load_points(args.pred)
    if noisy_raw.shape != pred_raw.shape:
        raise ValueError(
            f"Point count/shape mismatch: noisy={noisy_raw.shape}, pred={pred_raw.shape}. "
            "The competition requires the denoised point cloud to keep the same point count."
        )

    mesh_vertices_raw, mesh_faces = load_mesh(args.mesh)
    normalize_ref = load_points(args.normalize_ref) if args.normalize_ref else None

    noisy_eval, mesh_vertices_eval = maybe_normalize(noisy_raw, mesh_vertices_raw, normalize_ref)
    pred_eval, _ = maybe_normalize(pred_raw, mesh_vertices_raw, normalize_ref)
    assert mesh_vertices_eval is not None

    noisy_dist = p2s_distances(noisy_eval, mesh_vertices_eval, mesh_faces)
    pred_dist = p2s_distances(pred_eval, mesh_vertices_eval, mesh_faces)
    improvement = noisy_dist - pred_dist
    movement = np.linalg.norm(pred_raw - noisy_raw, axis=1)

    colors = improvement_colors(
        improvement,
        neutral_threshold=args.neutral_threshold,
        clip_percentile=args.clip_percentile,
    )

    write_ply(
        args.out,
        pred_raw,
        scalar_fields={
            "p2s_noisy": noisy_dist,
            "p2s_pred": pred_dist,
            "p2s_improvement": improvement,
            "movement": movement,
        },
        colors=colors,
    )

    p2s_noisy = float(np.mean(noisy_dist * noisy_dist))
    p2s_pred = float(np.mean(pred_dist * pred_dist))
    report = {
        "noisy": str(args.noisy),
        "pred": str(args.pred),
        "mesh": str(args.mesh),
        "normalize_ref": str(args.normalize_ref) if args.normalize_ref else None,
        "out": str(args.out),
        "point_count": int(len(pred_raw)),
        "p2s_noisy_mean_squared": p2s_noisy,
        "p2s_pred_mean_squared": p2s_pred,
        "p2s_score": official_score(p2s_pred, p2s_noisy),
        "improved_point_ratio": float(np.mean(improvement > args.neutral_threshold)),
        "worsened_point_ratio": float(np.mean(improvement < -args.neutral_threshold)),
        "unchanged_point_ratio": float(np.mean(np.abs(improvement) <= args.neutral_threshold)),
        **metric_stats("p2s_noisy_distance", noisy_dist),
        **metric_stats("p2s_pred_distance", pred_dist),
        **metric_stats("p2s_improvement", improvement),
        **metric_stats("movement", movement),
    }
    report_path = args.out.with_suffix(".json")
    write_json(report_path, report)
    print(f"Wrote P2S improvement PLY: {args.out.resolve()}")
    print(f"Wrote P2S improvement report: {report_path.resolve()}")
    print(f"P2S score: {report['p2s_score']:.4f}")


if __name__ == "__main__":
    main()
