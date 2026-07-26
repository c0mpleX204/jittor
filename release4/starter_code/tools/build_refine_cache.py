#!/usr/bin/env python
"""Build cached stage-1 outputs for CDRefine training.

The input cache is the Surface-Straight patch cache produced by
tools/build_patch_cache.py. This script runs a frozen CVM+DM model on each
cached patch and writes a second cache with an extra pc_stage1 field. Training
CDRefine can then read pc_stage1 directly instead of running CVM+DM online.
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


DEFAULT_CHECKPOINT = (
    "/root/autodl-tmp/experiments/"
    "v1.3b_dm/checkpoint_0.pkl"
)


def default_cvm_dm_config() -> Dict:
    return {
        "__target__": "StraightPCFVelocityDistanceModule",
        "num_modules": 2,
        "tot_its": 4,
        "dsm_sigma": 0.01,
        "frame_knn": 16,
        "feat_embedding_dim": 256,
        "decoder_hidden_dim": 64,
        "ratio_min": 0.0,
        "ratio_max": 1.0,
        "target_ratio_min": 0.0,
        "target_ratio_max": 1.0,
        "ratio_loss_weight": 1.0,
        "finetune_loss_weight": 200.0,
        "velocity_model": {
            "frame_knn": 16,
            "num_train_points": 128,
            "feat_embedding_dim": 256,
            "decoder_hidden_dim": 64,
            "dsm_sigma": 0.01,
            "denoise_steps": 4,
        },
        "coupled_model": {
            "num_modules": 2,
            "tot_its": 4,
            "num_train_points": 128,
            "dsm_sigma": 0.01,
            "consistency_loss_weight": 10.0,
            "velocity_model": {
                "frame_knn": 16,
                "num_train_points": 128,
                "feat_embedding_dim": 256,
                "decoder_hidden_dim": 64,
                "dsm_sigma": 0.01,
                "denoise_steps": 4,
            },
        },
    }


def read_cache_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve_patch_path(source_cache: Path, entry: str, data_name: str) -> Path:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        return entry_path / data_name
    return source_cache / entry_path / data_name


def output_entry(entry: str, idx: int, sequential_output: bool) -> str:
    if sequential_output or Path(entry).is_absolute():
        return (Path("patches") / f"{idx:08d}").as_posix()
    return Path(entry).as_posix()


def load_yaml(path: Path) -> Dict:
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def load_model(args) -> jt.nn.Module:
    model_config = load_yaml(args.model_config) if args.model_config else default_cvm_dm_config()
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
        required = ("pc_noisy", "pc_clean", "pc_mix")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path} missing required fields: {missing}")
        return {key: data[key].astype(np.float32) for key in data.files}


def save_patch(path: Path, payload: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def sync_jittor() -> None:
    if hasattr(jt, "sync_all"):
        jt.sync_all()


@jt.no_grad()
def predict_stage1(model, pc_noisy_batch: np.ndarray) -> np.ndarray:
    pc_noisy = jt.array(np.asarray(pc_noisy_batch, dtype=np.float32))
    pc_stage1, _ = model.denoise_langevin_dynamics(pc_noisy)
    return pc_stage1.detach().numpy().astype(np.float32)


def flush_batch(args, model, out_batch: List[str], patch_batch: List[Dict[str, np.ndarray]]) -> int:
    pc_noisy_batch = np.concatenate([item[args.input_field] for item in patch_batch], axis=0)
    pc_stage1_batch = predict_stage1(model, pc_noisy_batch)
    sync_jittor()

    written = 0
    offset = 0
    for rel_dir, patch in zip(out_batch, patch_batch):
        count = int(patch[args.input_field].shape[0])
        payload = dict(patch)
        payload[args.stage1_field] = pc_stage1_batch[offset:offset + count].astype(np.float32)
        save_patch(args.out_dir / Path(rel_dir) / args.data_name, payload)
        offset += count
        written += count
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--transform-config", type=Path, default=None)
    parser.add_argument("--list-name", type=str, default="train_cache.txt")
    parser.add_argument("--data-name", type=str, default="patch.npz")
    parser.add_argument("--input-field", type=str, default="pc_noisy")
    parser.add_argument("--stage1-field", type=str, default="pc_stage1")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--sequential-output",
        action="store_true",
        help="Write output patches as patches/00000000... instead of preserving source list entries.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    list_path = args.source_cache / args.list_name
    rel_paths = read_cache_list(list_path)
    if args.limit is not None:
        rel_paths = rel_paths[:args.limit]
    if not rel_paths:
        raise ValueError(f"No cache entries found in {list_path}")
    sequential_output = args.sequential_output or any(Path(entry).is_absolute() for entry in rel_paths)

    clear_out_dir(args.out_dir, args.overwrite)
    model = load_model(args)

    out_lines = []
    out_batch: List[str] = []
    patch_batch: List[Dict[str, np.ndarray]] = []
    written = 0
    for idx, source_entry in enumerate(rel_paths):
        patch = load_patch(resolve_patch_path(args.source_cache, source_entry, args.data_name))
        if args.input_field not in patch:
            raise KeyError(f"{source_entry}/{args.data_name} missing {args.input_field}")
        out_dir = output_entry(source_entry, idx, sequential_output)
        out_batch.append(out_dir)
        patch_batch.append(patch)
        out_lines.append(out_dir)

        if len(patch_batch) >= args.batch_size:
            written += flush_batch(args, model, out_batch, patch_batch)
            out_batch = []
            patch_batch = []
            if written % args.log_every == 0:
                print(f"wrote {written}/{len(rel_paths)} stage1 patches", flush=True)

    if patch_batch:
        written += flush_batch(args, model, out_batch, patch_batch)

    (args.out_dir / args.list_name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    metadata = {
        "source_cache": str(args.source_cache),
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config) if args.model_config else "default_cvm_dm_config",
        "transform_config": str(args.transform_config) if args.transform_config else "empty_passthrough",
        "list_name": args.list_name,
        "data_name": args.data_name,
        "input_field": args.input_field,
        "stage1_field": args.stage1_field,
        "batch_size": args.batch_size,
        "source_entries_are_absolute": any(Path(entry).is_absolute() for entry in rel_paths),
        "sequential_output": sequential_output,
        "num_entries": len(rel_paths),
        "num_patches": written,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote patches: {written}")
    print(f"refine cache list: {args.out_dir / args.list_name}")


if __name__ == "__main__":
    main()
