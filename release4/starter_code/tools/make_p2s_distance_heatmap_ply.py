#!/usr/bin/env python
"""Generate a CloudCompare-ready P2S distance heatmap PLY.

The output PLY keeps the original point coordinates, includes scalar fields
`p2s_distance` and `p2s_distance_squared`, and also bakes RGB heatmap colors:
blue/cyan = close to the mesh surface, yellow/red = far from the mesh surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pointcloud_eval_utils import (
    heatmap_colors,
    load_mesh,
    load_points,
    maybe_normalize,
    metric_stats,
    p2s_distances,
    write_json,
    write_ply,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", required=True, type=Path, help="Point cloud .npy, e.g. denoised.npy")
    parser.add_argument("--mesh", required=True, type=Path, help="Clean mesh .obj")
    parser.add_argument("--out", required=True, type=Path, help="Output heatmap .ply")
    parser.add_argument(
        "--normalize_ref",
        type=Path,
        default=None,
        help="Optional clean.npy used to match the official unit-sphere normalization.",
    )
    parser.add_argument(
        "--clip_percentile",
        type=float,
        default=99.0,
        help="Color clipping percentile. Lower values make small errors more visible.",
    )
    parser.add_argument(
        "--color_min",
        type=float,
        default=None,
        help="Fixed heatmap minimum. Use with --color_max to keep noisy/pred colors comparable.",
    )
    parser.add_argument(
        "--color_max",
        type=float,
        default=None,
        help="Fixed heatmap maximum. Use the same value for noisy and denoised heatmaps.",
    )
    args = parser.parse_args()

    points_raw = load_points(args.points)
    mesh_vertices_raw, mesh_faces = load_mesh(args.mesh)
    normalize_ref = load_points(args.normalize_ref) if args.normalize_ref else None

    points_eval, mesh_vertices_eval = maybe_normalize(points_raw, mesh_vertices_raw, normalize_ref)
    assert mesh_vertices_eval is not None

    distances = p2s_distances(points_eval, mesh_vertices_eval, mesh_faces)
    squared = distances * distances
    colors = heatmap_colors(
        distances,
        clip_percentile=args.clip_percentile,
        color_min=args.color_min,
        color_max=args.color_max,
    )

    write_ply(
        args.out,
        points_raw,
        scalar_fields={
            "p2s_distance": distances,
            "p2s_distance_squared": squared,
        },
        colors=colors,
    )

    report = {
        "points": str(args.points),
        "mesh": str(args.mesh),
        "normalize_ref": str(args.normalize_ref) if args.normalize_ref else None,
        "out": str(args.out),
        "color_min": args.color_min,
        "color_max": args.color_max,
        "clip_percentile": args.clip_percentile,
        "point_count": int(len(points_raw)),
        "p2s_mean_squared": float(squared.mean()),
        **metric_stats("p2s_distance", distances),
    }
    report_path = args.out.with_suffix(".json")
    write_json(report_path, report)
    print(f"Wrote P2S heatmap PLY: {args.out.resolve()}")
    print(f"Wrote P2S report:      {report_path.resolve()}")


if __name__ == "__main__":
    main()
