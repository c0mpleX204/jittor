#!/usr/bin/env python
"""Add fixed synthetic noise to true_test and build noisy_test."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np


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


def read_list(path: Optional[Path], gt_dir: Path) -> List[str]:
    if path is not None and path.exists():
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples = []
    for clean_path in sorted(gt_dir.rglob("clean.npy")):
        samples.append(clean_path.parent.relative_to(gt_dir).as_posix())
    return samples


def make_noise(rng: np.random.Generator, shape, noise_type: str, std: float) -> np.ndarray:
    if noise_type == "laplace":
        return rng.laplace(0.0, std, size=shape)
    if noise_type == "gaussian":
        return rng.normal(0.0, std, size=shape)
    raise ValueError(f"Unsupported noise type: {noise_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-dir", type=Path, default=Path("true_test"))
    parser.add_argument("--gt-list", type=Path, default=Path("datalist/true_test.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("noisy_test"))
    parser.add_argument("--list-out", type=Path, default=Path("datalist/noisy_test.txt"))
    parser.add_argument("--noise-type", choices=["laplace", "gaussian"], default="laplace")
    parser.add_argument("--std-min", type=float, default=0.005)
    parser.add_argument("--std-max", type=float, default=0.020)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.std_min < 0 or args.std_max < args.std_min:
        raise ValueError("Require 0 <= std-min <= std-max")

    if args.overwrite:
        safe_rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.list_out.parent.mkdir(parents=True, exist_ok=True)

    keys = read_list(args.gt_list, args.gt_dir)
    rng = np.random.default_rng(args.seed)
    rows = []

    for i, key in enumerate(keys):
        clean_path = args.gt_dir / key / "clean.npy"
        if not clean_path.exists():
            raise FileNotFoundError(clean_path)
        clean = np.load(clean_path).astype(np.float32)
        if clean.ndim != 2 or clean.shape[1] != 3:
            raise ValueError(f"{clean_path} must have shape (N, 3), got {clean.shape}")

        std = float(rng.uniform(args.std_min, args.std_max))
        noise = make_noise(rng, clean.shape, args.noise_type, std).astype(np.float32)
        noisy = clean + noise

        out_path = args.out_dir / key / "noisy.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, noisy.astype(np.float32))
        rows.append(
            {
                "key": key,
                "points": int(clean.shape[0]),
                "noise_type": args.noise_type,
                "std": std,
                "noise_l2_mean": float(np.linalg.norm(noise, axis=1).mean()),
            }
        )

        if (i + 1) % 20 == 0 or i + 1 == len(keys):
            print(f"[noisy_test] {i + 1}/{len(keys)} processed")

    args.list_out.write_text("\n".join(keys) + ("\n" if keys else ""), encoding="utf-8")

    csv_path = args.out_dir / "noise_meta.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "points", "noise_type", "std", "noise_l2_mean"])
        writer.writeheader()
        writer.writerows(rows)

    meta = {
        "gt_dir": str(args.gt_dir),
        "out_dir": str(args.out_dir),
        "list_out": str(args.list_out),
        "noise_type": args.noise_type,
        "std_min": args.std_min,
        "std_max": args.std_max,
        "seed": args.seed,
        "num_samples": len(keys),
    }
    (args.out_dir / "noise_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote noisy samples: {args.out_dir.resolve()}")
    print(f"Wrote datalist:      {args.list_out.resolve()}")
    print(f"Samples: {len(keys)}")


if __name__ == "__main__":
    main()
