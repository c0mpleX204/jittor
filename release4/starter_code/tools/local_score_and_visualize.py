#!/usr/bin/env python
"""Run local CD/P2S scoring and generate selected P2S visualization PLY files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _is_allowed_delete_target(target: Path) -> bool:
    roots = [Path.cwd().resolve(), Path("/root/autodl-tmp").resolve()]
    return any(target != root and root in target.parents for root in roots)


def _clear_dir(path: Path) -> None:
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def safe_rmtree(path: Path) -> None:
    if path.is_symlink():
        target = path.resolve()
        if not _is_allowed_delete_target(target):
            raise ValueError(f"Refuse to clear symlink target: {target}")
        if target.exists():
            _clear_dir(target)
        return

    target = path.resolve()
    if not _is_allowed_delete_target(target):
        raise ValueError(f"Refuse to delete outside allowed workspace/data disk: {target}")
    if target.exists():
        shutil.rmtree(target)


def read_list(path: Optional[Path]) -> List[str]:
    if path is None or not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_samples(base_dir: Path, filename: str) -> Dict[str, Path]:
    return {
        path.parent.relative_to(base_dir).as_posix(): path
        for path in sorted(base_dir.rglob(filename))
    }


def mesh_path(mesh_dir: Path, key: str, mesh_name: str) -> Path:
    return mesh_dir / key / mesh_name


def run_cmd(cmd: List[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(" ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.write_text(proc.stdout, encoding="utf-8")
    print(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed, see {log_path}")


def choose_visual_keys(keys: List[str], sample_list: List[str], count: int) -> List[str]:
    ordered = [key for key in sample_list if key in set(keys)] if sample_list else sorted(keys)
    if count < 0:
        return ordered
    return ordered[:count]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=Path("true_test"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("noisy_test"))
    parser.add_argument("--pred-dir", type=Path, default=Path("denoisy_test"))
    parser.add_argument("--mesh-dir", type=Path, default=Path("dataset_train"))
    parser.add_argument("--sample-list", type=Path, default=Path("datalist/noisy_test.txt"))
    parser.add_argument("--result-dir", type=Path, default=Path("result/local_eval"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--visualize-count", type=int, default=8, help="0 skips visualization, -1 means all")
    parser.add_argument("--mesh-name", type=str, default="models/model_normalized.obj")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    score_dir = args.result_dir / "scores"
    vis_dir = args.result_dir / "visualization"
    if args.overwrite:
        safe_rmtree(score_dir)
        safe_rmtree(vis_dir)
    args.result_dir.mkdir(parents=True, exist_ok=True)
    score_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    gt_samples = find_samples(args.gt_dir, "clean.npy")
    noisy_samples = find_samples(args.noisy_dir, "noisy.npy")
    pred_samples = find_samples(args.pred_dir, "denoised.npy")
    common_keys = sorted(set(gt_samples) & set(noisy_samples) & set(pred_samples))
    if not common_keys:
        raise RuntimeError(
            "No matching samples before scoring. "
            f"pred={len(pred_samples)}, noisy={len(noisy_samples)}, gt={len(gt_samples)}. "
            f"Check --pred-dir: {args.pred_dir}"
        )
    print(
        "Pre-score sample counts: "
        f"pred={len(pred_samples)}, noisy={len(noisy_samples)}, gt={len(gt_samples)}, "
        f"common={len(common_keys)}"
    )

    score_cmd = [
        sys.executable,
        str(TOOLS / "score_cd_p2s_official.py"),
        "--pred_dir",
        str(args.pred_dir),
        "--noisy_dir",
        str(args.noisy_dir),
        "--gt_dir",
        str(args.gt_dir),
        "--mesh_dir",
        str(args.mesh_dir),
        "--out_dir",
        str(score_dir),
        "--workers",
        str(args.workers),
    ]
    run_cmd(score_cmd, score_dir / "score_stdout.txt")

    if args.visualize_count == 0:
        print("Visualization skipped.")
        return

    sample_list = read_list(args.sample_list)
    visual_keys = choose_visual_keys(common_keys, sample_list, args.visualize_count)

    for key in visual_keys:
        mesh = mesh_path(args.mesh_dir, key, args.mesh_name)
        if not mesh.exists():
            print(f"Skip visualization without mesh: {key}")
            continue
        sample_out = vis_dir / key.replace("/", "__")
        sample_out.mkdir(parents=True, exist_ok=True)

        common = ["--mesh", str(mesh), "--normalize_ref", str(gt_samples[key])]
        run_cmd(
            [
                sys.executable,
                str(TOOLS / "make_p2s_distance_heatmap_ply.py"),
                "--points",
                str(noisy_samples[key]),
                "--out",
                str(sample_out / "noisy_p2s_heatmap.ply"),
                *common,
            ],
            sample_out / "noisy_p2s_heatmap.log",
        )
        run_cmd(
            [
                sys.executable,
                str(TOOLS / "make_p2s_distance_heatmap_ply.py"),
                "--points",
                str(pred_samples[key]),
                "--out",
                str(sample_out / "denoised_p2s_heatmap.ply"),
                *common,
            ],
            sample_out / "denoised_p2s_heatmap.log",
        )
        run_cmd(
            [
                sys.executable,
                str(TOOLS / "make_p2s_improvement_ply.py"),
                "--noisy",
                str(noisy_samples[key]),
                "--pred",
                str(pred_samples[key]),
                "--out",
                str(sample_out / "p2s_improvement.ply"),
                *common,
            ],
            sample_out / "p2s_improvement.log",
        )

    print(f"Visualization samples: {len(visual_keys)}")
    print(f"Result dir: {args.result_dir.resolve()}")


if __name__ == "__main__":
    main()
