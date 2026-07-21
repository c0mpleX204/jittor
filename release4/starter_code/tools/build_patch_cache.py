#!/usr/bin/env python
"""Build cached training patches for Surface-Straight style VM training.

The normal training dataloader repeatedly loads OBJ meshes, samples surface
points, adds noise, and builds KDTree patches. This tool performs those CPU
steps once and writes small npz files that can be loaded directly during
training with the `npz_patch` datapath loader.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datapath import ObjLazyAsset  # noqa: E402
from src.data.transform import Transform  # noqa: E402


def read_keys(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shuffled_cycle(keys: List[str], count: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    pool = keys[:]
    out = []
    while len(out) < count:
        rng.shuffle(pool)
        out.extend(pool)
    return out[:count]


def load_train_transform(path: Path) -> Transform:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    train_cfg = cfg.get("train_transform", None)  # type: ignore[union-attr]
    if train_cfg is None:
        raise ValueError(f"{path} does not contain train_transform")
    return Transform.parse(**train_cfg)


def save_patch(path: Path, meta: Dict[str, np.ndarray], patch_idx: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pc_noisy": meta["pc_noisy"][patch_idx:patch_idx + 1].astype(np.float32),
        "pc_clean": meta["pc_clean"][patch_idx:patch_idx + 1].astype(np.float32),
        "pc_mix": meta["pc_mix"][patch_idx:patch_idx + 1].astype(np.float32),
    }
    for optional_key in ("pc_time", "pc_clean_corr", "pc_normal"):
        if optional_key in meta:
            payload[optional_key] = np.asarray(
                meta[optional_key][patch_idx:patch_idx + 1],
                dtype=np.float32,
            )
    np.savez_compressed(path, **payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--list", type=Path, default=Path("datalist/train.txt"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--transform", type=Path, default=Path("configs/transform/surface_straight.yaml"))
    parser.add_argument("--data-name", type=str, default="models/model_normalized.obj")
    parser.add_argument("--num-items", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.out_dir} is not empty. Use --overwrite to replace it.")
    if args.overwrite and args.out_dir.exists():
        for child in args.out_dir.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    random.seed(args.seed)

    keys = read_keys(args.list)
    if not keys:
        raise ValueError(f"No keys found in {args.list}")
    selected_keys = shuffled_cycle(keys, args.num_items, args.seed)
    transform = load_train_transform(args.transform)

    list_lines = []
    written = 0
    for item_idx, key in enumerate(selected_keys):
        mesh_path = args.dataset_dir / key / args.data_name
        if not mesh_path.exists():
            print(f"missing mesh: {mesh_path}", file=sys.stderr)
            continue

        asset = ObjLazyAsset(path=str(mesh_path), cls=None).load()
        transform.apply(asset)
        if asset.meta is None:
            raise RuntimeError(f"transform did not produce meta for {mesh_path}")

        num_patches = int(asset.meta["pc_noisy"].shape[0])
        for patch_idx in range(num_patches):
            rel_dir = Path("patches") / f"{written:08d}"
            save_patch(args.out_dir / rel_dir / "patch.npz", asset.meta, patch_idx)
            list_lines.append(rel_dir.as_posix())
            written += 1

        if (item_idx + 1) % 100 == 0:
            print(f"processed {item_idx + 1}/{len(selected_keys)} shapes, wrote {written} patches", flush=True)

    list_path = args.out_dir / "train_cache.txt"
    list_path.write_text("\n".join(list_lines) + "\n", encoding="utf-8")

    meta = {
        "dataset_dir": str(args.dataset_dir),
        "list": str(args.list),
        "transform": str(args.transform),
        "data_name": args.data_name,
        "num_items": args.num_items,
        "num_patches": written,
        "seed": args.seed,
        "cache_list": str(list_path),
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote patches: {written}")
    print(f"cache list: {list_path}")


if __name__ == "__main__":
    main()
