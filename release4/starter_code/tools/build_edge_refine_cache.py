#!/usr/bin/env python
"""Build cached CDRefine outputs for EdgeRefine training.

The input cache is a refine cache containing pc_stage1 and clean supervision
fields. This script runs a frozen CDRefine checkpoint on each cached patch and
writes a second cache with pc_cd, the base CDRefine output. EdgeRefine can then
train as a cheap stage after CDRefine.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import jittor as jt
import numpy as np
from omegaconf import OmegaConf

jt.flags.use_cuda = 1

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model.parse import get_model  # noqa: E402


def read_cache_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_yaml(path: Path) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def load_model(args) -> jt.nn.Module:
    model_config = load_yaml(args.model_config)
    transform_config = (
        load_yaml(args.transform_config)
        if args.transform_config
        else {"train_transform": {}, "validate_transform": {}, "predict_transform": {}}
    )
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(str(args.checkpoint))
    model.eval()
    for param in model.parameters():
        if hasattr(param, "stop_grad"):
            param.stop_grad()
        if hasattr(param, "requires_grad"):
            param.requires_grad = False
    return model


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
        if "pc_stage1" not in data:
            raise KeyError(f"{path} missing pc_stage1")
        return {key: data[key].astype(np.float32) for key in data.files}


def save_patch(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def sync_jittor() -> None:
    if hasattr(jt, "sync_all"):
        jt.sync_all()


@jt.no_grad()
def predict_cd(model, patch_batch: Dict[str, np.ndarray], stage1_field: str, base_field: str) -> np.ndarray:
    pc_stage1 = jt.array(np.asarray(patch_batch[stage1_field], dtype=np.float32))
    pc_edge_risk = None
    pc_normal = None
    if "pc_edge_risk" in patch_batch:
        pc_edge_risk = jt.array(np.asarray(patch_batch["pc_edge_risk"], dtype=np.float32))
    if "pc_normal" in patch_batch:
        pc_normal = jt.array(np.asarray(patch_batch["pc_normal"], dtype=np.float32))

    if hasattr(model, "refine"):
        pc_cd, _ = model.refine(
            pc_stage1,
            pc_edge_risk=pc_edge_risk,
            pc_normal_proxy=pc_normal,
        )
    elif hasattr(model, "denoise_langevin_dynamics"):
        pc_cd, _ = model.denoise_langevin_dynamics(pc_stage1)
    else:
        raise TypeError(f"Model does not support CD prediction for {base_field}")
    return pc_cd.detach().numpy().astype(np.float32)


def flush_batch(args, model, rel_batch: List[str], patch_batch: List[Dict[str, np.ndarray]]) -> int:
    merged = {args.stage1_field: np.concatenate([item[args.stage1_field] for item in patch_batch], axis=0)}
    for optional_key in ("pc_edge_risk", "pc_normal"):
        if all(optional_key in item for item in patch_batch):
            merged[optional_key] = np.concatenate([item[optional_key] for item in patch_batch], axis=0)
    pc_cd_batch = predict_cd(model, merged, args.stage1_field, args.base_field)
    sync_jittor()

    written = 0
    offset = 0
    for rel_dir, patch in zip(rel_batch, patch_batch):
        count = int(patch[args.stage1_field].shape[0])
        payload = dict(patch)
        payload[args.base_field] = pc_cd_batch[offset:offset + count].astype(np.float32)
        save_patch(args.out_dir / rel_dir / args.data_name, payload)
        offset += count
        written += count
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--transform-config", type=Path, default=None)
    parser.add_argument("--list-name", type=str, default="train_cache.txt")
    parser.add_argument("--data-name", type=str, default="patch.npz")
    parser.add_argument("--stage1-field", type=str, default="pc_stage1")
    parser.add_argument("--base-field", type=str, default="pc_cd")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.model_config.exists():
        raise FileNotFoundError(args.model_config)

    list_path = args.source_cache / args.list_name
    rel_paths = read_cache_list(list_path)
    if args.limit is not None:
        rel_paths = rel_paths[:args.limit]
    if not rel_paths:
        raise ValueError(f"No cache entries found in {list_path}")

    clear_out_dir(args.out_dir, args.overwrite)
    model = load_model(args)

    out_lines = []
    rel_batch: List[str] = []
    patch_batch: List[Dict[str, np.ndarray]] = []
    written = 0
    for rel_dir in rel_paths:
        patch = load_patch(args.source_cache / rel_dir / args.data_name)
        if args.stage1_field not in patch:
            raise KeyError(f"{rel_dir}/{args.data_name} missing {args.stage1_field}")
        rel_batch.append(rel_dir)
        patch_batch.append(patch)
        out_lines.append(rel_dir)

        if len(patch_batch) >= args.batch_size:
            written += flush_batch(args, model, rel_batch, patch_batch)
            rel_batch = []
            patch_batch = []
            if written % args.log_every == 0:
                print(f"wrote {written}/{len(rel_paths)} edge-refine patches", flush=True)

    if patch_batch:
        written += flush_batch(args, model, rel_batch, patch_batch)

    (args.out_dir / args.list_name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    metadata = {
        "source_cache": str(args.source_cache),
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "transform_config": str(args.transform_config) if args.transform_config else "empty_passthrough",
        "list_name": args.list_name,
        "data_name": args.data_name,
        "stage1_field": args.stage1_field,
        "base_field": args.base_field,
        "batch_size": args.batch_size,
        "num_entries": len(rel_paths),
        "num_patches": written,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote patches: {written}")
    print(f"edge-refine cache list: {args.out_dir / args.list_name}")


if __name__ == "__main__":
    main()
