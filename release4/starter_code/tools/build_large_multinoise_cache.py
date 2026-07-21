#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from multivm_pipeline import paths as path_utils
from multivm_pipeline.cache import build_patch_cache


PROTECTED_CACHE_NAMES = {
    "patch_cache_n12000_seed123",
    "cache_surface_straight_nf12000_seed123",
}


def _parse_csv_strings(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_csv_floats(value: str) -> Optional[List[float]]:
    if not value.strip():
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _default_cache_name(cache_items: int, seed: int, noise_types: List[str], noise_probs: Optional[List[float]]) -> str:
    if noise_probs is None:
        prob_label = "_".join(noise_types)
    else:
        parts = []
        for noise_type, prob in zip(noise_types, noise_probs):
            parts.append(f"{noise_type}{int(round(prob * 100)):02d}")
        prob_label = "_".join(parts)
    return f"patch_cache_n{cache_items}_seed{seed}_multinoise_{prob_label}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a larger multi-noise patch cache without overwriting the "
            "existing 12000 single-noise cache."
        )
    )
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)

    parser.add_argument("--cache-items", type=int, default=60000)
    parser.add_argument("--seed", type=int, default=456)
    parser.add_argument("--sample-points", type=int, default=32768)
    parser.add_argument("--patch-size", type=int, default=1000)
    parser.add_argument("--mesh-name", type=str, default="models/model_normalized.obj")

    parser.add_argument("--noise-types", type=str, default="laplace,gaussian")
    parser.add_argument("--noise-probs", type=str, default="0.7,0.3")
    parser.add_argument("--noise-std-min", type=float, default=0.005)
    parser.add_argument("--noise-std-max", type=float, default=0.020)
    parser.add_argument("--l2-noise-std", type=float, default=0.020)
    parser.add_argument("--no-l2-noisy", action="store_true")
    parser.add_argument("--paired-target", action="store_true")
    parser.add_argument("--no-linear-aug", action="store_true")

    parser.add_argument("--force", action="store_true", help="Overwrite only the selected multi-noise cache dir.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    workspace = path_utils.workspace_root(args.workspace_root)
    output_root = path_utils.output_root(workspace, args.output_root)
    dataset_dir = path_utils.dataset_train_root(workspace, args.dataset_dir)

    noise_types = _parse_csv_strings(args.noise_types)
    noise_probs = _parse_csv_floats(args.noise_probs)
    if not noise_types:
        raise ValueError("--noise-types cannot be empty")
    unsupported = sorted(set(noise_types) - {"laplace", "gaussian"})
    if unsupported:
        raise ValueError(f"Unsupported noise types: {unsupported}")
    if noise_probs is not None:
        if len(noise_probs) != len(noise_types):
            raise ValueError("--noise-probs length must match --noise-types length")
        total_prob = sum(noise_probs)
        if total_prob <= 0 or any(prob < 0 for prob in noise_probs):
            raise ValueError("--noise-probs must be non-negative and have positive sum")
        noise_probs = [prob / total_prob for prob in noise_probs]

    if args.cache_items <= 12000:
        print(
            "warning: this script is meant for larger caches. "
            "Use --cache-items above 12000 for the long-run cache."
        )

    if args.cache_dir is None:
        cache_name = _default_cache_name(args.cache_items, args.seed, noise_types, noise_probs)
        cache_dir = (output_root / "cache" / cache_name).resolve()
    else:
        cache_dir = args.cache_dir.expanduser().resolve()

    if cache_dir.name in PROTECTED_CACHE_NAMES:
        raise ValueError(f"Refuse to write protected single-noise cache directory: {cache_dir}")

    print("Large multi-noise cache plan:")
    print(f"  workspace_root: {workspace}")
    print(f"  dataset_dir:    {dataset_dir}")
    print(f"  output_root:    {output_root}")
    print(f"  cache_dir:      {cache_dir}")
    print(f"  cache_items:    {args.cache_items}")
    print(f"  seed:           {args.seed}")
    print(f"  noise_types:    {noise_types}")
    print(f"  noise_probs:    {noise_probs if noise_probs is not None else 'uniform'}")
    print(f"  force:          {args.force}")

    if (cache_dir / "train_cache.txt").exists() and not args.force:
        print(f"Cache already exists and will be reused: {cache_dir / 'train_cache.txt'}")
        return 0

    if args.dry_run:
        return 0

    build_patch_cache(
        dataset_dir=dataset_dir,
        cache_dir=cache_dir,
        cache_items=args.cache_items,
        sample_points=args.sample_points,
        patch_size=args.patch_size,
        seed=args.seed,
        mesh_name=args.mesh_name,
        noise_type=noise_types[0],
        noise_types=noise_types,
        noise_probs=noise_probs,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        l2_noise_std=args.l2_noise_std,
        use_l2_as_noisy=not args.no_l2_noisy,
        surface_target=not args.paired_target,
        augment_linear=not args.no_linear_aug,
        force=args.force,
    )

    run_dir = output_root / "runs" / f"multivm_multinoise_n{args.cache_items}_seed{args.seed}_m2"
    print()
    print("Suggested training command:")
    print(
        "python multivm_main.py train "
        f"--cache-dir {cache_dir.as_posix()} "
        f"--cache-items {args.cache_items} "
        f"--epochs 5 "
        f"--run-dir {run_dir.as_posix()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
