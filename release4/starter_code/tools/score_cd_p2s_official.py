#!/usr/bin/env python
"""Compute non-visual CD/P2S metrics and the competition-style 100-point score.

Directory mode matches the official submission layout:
    pred_dir/<...>/denoised.npy
    noisy_dir/<...>/noisy.npy
    gt_dir/<...>/clean.npy
    mesh_dir/<...>/models/model_normalized.obj

Single-sample mode is also supported with --pred/--noisy/--gt/--mesh.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from pointcloud_eval_utils import (
    apply_unit_sphere,
    chamfer_distance,
    common_keys,
    load_mesh,
    load_points,
    metric_stats,
    official_score,
    p2s_distances,
    unit_sphere_params,
    write_json,
)


def find_samples(base_dir: Path, filename: str) -> Dict[str, Path]:
    samples: Dict[str, Path] = {}
    pattern = str(base_dir / "**" / filename)
    for path_str in sorted(glob.glob(pattern, recursive=True)):
        path = Path(path_str)
        key = os.path.relpath(path.parent, base_dir).replace("\\", "/")
        samples[key] = path
    return samples


def find_meshes(mesh_dir: Path, mesh_data_name: str) -> Dict[str, Path]:
    meshes: Dict[str, Path] = {}
    pattern = str(mesh_dir / "**" / mesh_data_name)
    depth = len(Path(mesh_data_name).parts)
    for path_str in sorted(glob.glob(pattern, recursive=True)):
        model_dir = Path(path_str)
        for _ in range(depth):
            model_dir = model_dir.parent
        key = os.path.relpath(model_dir, mesh_dir).replace("\\", "/")
        meshes[key] = Path(path_str)
    return meshes


def score_one(
    key: str,
    pred_path: Path,
    noisy_path: Path,
    gt_path: Path,
    mesh_path: Optional[Path],
) -> Dict[str, object]:
    pred_raw = load_points(pred_path)
    noisy_raw = load_points(noisy_path)
    gt_raw = load_points(gt_path)

    if pred_raw.shape != noisy_raw.shape:
        raise ValueError(
            f"{key}: pred shape {pred_raw.shape} does not match noisy shape {noisy_raw.shape}"
        )

    center, scale = unit_sphere_params(gt_raw)
    pred = apply_unit_sphere(pred_raw, center, scale)
    noisy = apply_unit_sphere(noisy_raw, center, scale)
    gt = apply_unit_sphere(gt_raw, center, scale)

    cd_noisy, noisy_to_gt, gt_to_noisy = chamfer_distance(noisy, gt)
    cd_pred, pred_to_gt, gt_to_pred = chamfer_distance(pred, gt)
    cd_score = official_score(cd_pred, cd_noisy)

    row: Dict[str, object] = {
        "key": key,
        "pred_path": str(pred_path),
        "noisy_path": str(noisy_path),
        "gt_path": str(gt_path),
        "mesh_path": str(mesh_path) if mesh_path else "",
        "pred_points": int(len(pred_raw)),
        "noisy_points": int(len(noisy_raw)),
        "gt_points": int(len(gt_raw)),
        "cd_noisy": cd_noisy,
        "cd_pred": cd_pred,
        "cd_score": cd_score,
        **metric_stats("noisy_to_gt", noisy_to_gt),
        **metric_stats("pred_to_gt", pred_to_gt),
        **metric_stats("gt_to_noisy", gt_to_noisy),
        **metric_stats("gt_to_pred", gt_to_pred),
    }

    if mesh_path:
        mesh_vertices_raw, mesh_faces = load_mesh(mesh_path)
        mesh_vertices = apply_unit_sphere(mesh_vertices_raw, center, scale)
        noisy_p2s_dist = p2s_distances(noisy, mesh_vertices, mesh_faces)
        pred_p2s_dist = p2s_distances(pred, mesh_vertices, mesh_faces)
        p2s_noisy = float(np.mean(noisy_p2s_dist * noisy_p2s_dist))
        p2s_pred = float(np.mean(pred_p2s_dist * pred_p2s_dist))
        p2s_score = official_score(p2s_pred, p2s_noisy)
        row.update(
            {
                "p2s_noisy": p2s_noisy,
                "p2s_pred": p2s_pred,
                "p2s_score": p2s_score,
                "final_score": 0.5 * float(cd_score) + 0.5 * float(p2s_score),
                **metric_stats("noisy_p2s_distance", noisy_p2s_dist),
                **metric_stats("pred_p2s_distance", pred_p2s_dist),
            }
        )
    else:
        row.update(
            {
                "p2s_noisy": "",
                "p2s_pred": "",
                "p2s_score": "",
                "final_score": cd_score,
            }
        )

    return row


def score_one_star(args: Tuple[str, Path, Path, Path, Optional[Path]]) -> Dict[str, object]:
    return score_one(*args)


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_numeric(rows: Iterable[Dict[str, object]], key: str) -> Optional[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if value == "" or value is None:
            continue
        values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    summary = {
        "num_samples": len(rows),
        "mean_cd_noisy": mean_numeric(rows, "cd_noisy"),
        "mean_cd_pred": mean_numeric(rows, "cd_pred"),
        "mean_cd_score": mean_numeric(rows, "cd_score"),
        "mean_p2s_noisy": mean_numeric(rows, "p2s_noisy"),
        "mean_p2s_pred": mean_numeric(rows, "p2s_pred"),
        "mean_p2s_score": mean_numeric(rows, "p2s_score"),
        "mean_final_score": mean_numeric(rows, "final_score"),
    }
    return summary


def build_tasks(args: argparse.Namespace) -> List[Tuple[str, Path, Path, Path, Optional[Path]]]:
    if args.pred and args.noisy and args.gt:
        return [
            (
                args.key or Path(args.pred).stem,
                Path(args.pred),
                Path(args.noisy),
                Path(args.gt),
                Path(args.mesh) if args.mesh else None,
            )
        ]

    if not (args.pred_dir and args.noisy_dir and args.gt_dir):
        raise ValueError(
            "Use either single-sample mode (--pred --noisy --gt) or directory mode "
            "(--pred_dir --noisy_dir --gt_dir)."
        )

    pred_samples = find_samples(Path(args.pred_dir), args.pred_filename)
    noisy_samples = find_samples(Path(args.noisy_dir), args.noisy_filename)
    gt_samples = find_samples(Path(args.gt_dir), args.gt_filename)
    mesh_samples = (
        find_meshes(Path(args.mesh_dir), args.mesh_data_name) if args.mesh_dir else {}
    )

    keys = list(common_keys(pred_samples, noisy_samples, gt_samples))
    missing_pred = sorted(set(gt_samples) - set(pred_samples))
    if missing_pred:
        print(f"WARNING: {len(missing_pred)} GT samples have no prediction and will be absent.")
    if not keys:
        raise ValueError(
            f"No matching samples. pred={len(pred_samples)}, noisy={len(noisy_samples)}, gt={len(gt_samples)}"
        )

    tasks = []
    for key in keys:
        tasks.append(
            (
                key,
                pred_samples[key],
                noisy_samples[key],
                gt_samples[key],
                mesh_samples.get(key) if args.mesh_dir else None,
            )
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--pred", type=Path, default=None, help="Single-sample denoised.npy")
    parser.add_argument("--noisy", type=Path, default=None, help="Single-sample noisy.npy")
    parser.add_argument("--gt", type=Path, default=None, help="Single-sample clean.npy")
    parser.add_argument("--mesh", type=Path, default=None, help="Single-sample model_normalized.obj")
    parser.add_argument("--key", type=str, default="", help="Single-sample name")

    parser.add_argument("--pred_dir", type=Path, default=None)
    parser.add_argument("--noisy_dir", type=Path, default=None)
    parser.add_argument("--gt_dir", type=Path, default=None)
    parser.add_argument("--mesh_dir", type=Path, default=None)
    parser.add_argument("--pred_filename", type=str, default="denoised.npy")
    parser.add_argument("--noisy_filename", type=str, default="noisy.npy")
    parser.add_argument("--gt_filename", type=str, default="clean.npy")
    parser.add_argument("--mesh_data_name", type=str, default="models/model_normalized.obj")

    parser.add_argument("--out_dir", type=Path, default=Path("score_cd_p2s_report"))
    parser.add_argument("--workers", type=int, default=0, help="0 = auto, 1 = no multiprocessing")
    args = parser.parse_args()

    tasks = build_tasks(args)
    workers = args.workers if args.workers > 0 else min(cpu_count(), 16)

    if workers > 1 and len(tasks) > 1:
        with Pool(processes=workers) as pool:
            rows = pool.map(score_one_star, tasks)
    else:
        rows = [score_one_star(task) for task in tasks]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "per_sample_cd_p2s_scores.csv"
    summary_path = args.out_dir / "summary_cd_p2s_scores.json"
    write_csv(csv_path, rows)
    summary = summarize(rows)
    write_json(summary_path, summary)

    print(f"Wrote per-sample scores: {csv_path.resolve()}")
    print(f"Wrote score summary:     {summary_path.resolve()}")
    print(f"Samples: {summary['num_samples']}")
    print(f"Mean CD score: {summary['mean_cd_score']:.4f}")
    if summary["mean_p2s_score"] is not None:
        print(f"Mean P2S score: {summary['mean_p2s_score']:.4f}")
    print(f"Mean final score: {summary['mean_final_score']:.4f}")


if __name__ == "__main__":
    main()
