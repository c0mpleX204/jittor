#!/usr/bin/env python
"""Convert an existing patch cache to official-style StraightPCF supervision.

Some exploratory caches store both:

- pc_clean: nearest sampled surface endpoint
- pc_clean_corr: clean counterpart selected by the same noisy KNN indices

The official StraightPCF patch builder uses the counterpart target. It also
centers each patch at the interpolation point between the noisy seed and the
counterpart clean seed. This tool rebuilds those local tensors from cached
fields, avoiding another expensive OBJ sampling pass.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np


def read_cache_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_patch_path(cache_root: Path, entry: str, data_name: str) -> Path:
    path = Path(entry)
    if path.is_absolute():
        return path / data_name
    return cache_root / path / data_name


def clear_out_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"{path} is not empty. Use --overwrite to replace it.")
        for child in path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def load_patch(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key].astype(np.float32) for key in data.files}


def convert_patch(
    patch: Dict[str, np.ndarray],
    target_field: str,
    keep_optional: bool,
) -> Dict[str, np.ndarray]:
    missing = [key for key in ("pc_noisy", target_field, "pc_time") if key not in patch]
    if missing:
        raise KeyError(f"missing required fields: {missing}")

    pc_noisy = patch["pc_noisy"].astype(np.float32)
    pc_clean = patch[target_field].astype(np.float32)
    pc_time = patch["pc_time"].astype(np.float32).reshape(-1)

    if pc_noisy.ndim != 3 or pc_clean.ndim != 3:
        raise ValueError(f"expected pc arrays with shape (P,N,3), got {pc_noisy.shape}, {pc_clean.shape}")
    if pc_noisy.shape != pc_clean.shape:
        raise ValueError(f"pc_noisy and {target_field} shape mismatch: {pc_noisy.shape} vs {pc_clean.shape}")
    if pc_time.shape[0] != pc_noisy.shape[0]:
        raise ValueError(f"pc_time length {pc_time.shape[0]} does not match patches {pc_noisy.shape[0]}")

    t = pc_time[:, None, None]
    # cKDTree KNN includes the seed itself as the first point. The source cache
    # may be centered at a nearest-surface seed; shift it to the counterpart
    # interpolation seed used by StraightPCF.
    seed_shift = t * pc_clean[:, 0:1, :] + (1.0 - t) * pc_noisy[:, 0:1, :]
    pc_noisy_out = pc_noisy - seed_shift
    pc_clean_out = pc_clean - seed_shift
    pc_mix_out = t * pc_clean + (1.0 - t) * pc_noisy - seed_shift

    payload = {
        "pc_noisy": pc_noisy_out.astype(np.float32),
        "pc_clean": pc_clean_out.astype(np.float32),
        "pc_mix": pc_mix_out.astype(np.float32),
        "pc_time": pc_time.astype(np.float32),
        "pc_clean_corr": pc_clean_out.astype(np.float32),
    }
    if keep_optional:
        for key in ("pc_edge_risk", "pc_normal"):
            if key in patch:
                payload[key] = patch[key].astype(np.float32)
        if "pc_center" in patch:
            payload["pc_center"] = (patch["pc_center"].astype(np.float32) + seed_shift[:, 0, :]).astype(np.float32)
    return payload


def save_patch(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--list-name", type=str, default="train_cache.txt")
    parser.add_argument("--data-name", type=str, default="patch.npz")
    parser.add_argument("--target-field", type=str, default="pc_clean_corr")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--drop-optional", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    list_path = args.source_cache / args.list_name
    entries = read_cache_list(list_path)
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError(f"No cache entries found in {list_path}")

    clear_out_dir(args.out_dir, args.overwrite)

    out_lines = []
    written = 0
    for idx, entry in enumerate(entries):
        patch = load_patch(resolve_patch_path(args.source_cache, entry, args.data_name))
        payload = convert_patch(
            patch=patch,
            target_field=args.target_field,
            keep_optional=not args.drop_optional,
        )
        rel_dir = Path("patches") / f"{idx:08d}"
        save_patch(args.out_dir / rel_dir / args.data_name, payload)
        out_lines.append(rel_dir.as_posix())
        written += int(payload["pc_noisy"].shape[0])
        if written % args.log_every == 0:
            print(f"converted {written}/{len(entries)} patches", flush=True)

    (args.out_dir / args.list_name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    metadata = {
        "source_cache": str(args.source_cache),
        "list_name": args.list_name,
        "data_name": args.data_name,
        "target_field": args.target_field,
        "num_entries": len(entries),
        "num_patches": written,
        "kept_optional": not args.drop_optional,
        "cache_list": str(args.out_dir / args.list_name),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"converted patches: {written}")
    print(f"cache list: {args.out_dir / args.list_name}")


if __name__ == "__main__":
    main()
