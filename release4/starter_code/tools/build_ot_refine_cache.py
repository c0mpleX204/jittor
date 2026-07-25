#!/usr/bin/env python
"""Build a third-stage OT-target refine cache.

This reads an existing CDRefine cache, runs a frozen CDRefine checkpoint to
produce stage-2 patches, then computes an approximate one-to-one matching from
stage-2 points to clean-correspondence points. The resulting cache can train a
small final CDRefine stage with ``target_field: pc_ot_target``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import jittor as jt
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching

    HAS_SPARSE_MATCHING = True
except Exception:  # pragma: no cover - optional SciPy submodules differ by build
    linear_sum_assignment = None
    coo_matrix = None
    min_weight_full_bipartite_matching = None
    HAS_SPARSE_MATCHING = False

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


def resolve_entry(source_cache: Path, entry: str, data_name: str, idx: int) -> Tuple[Path, str]:
    entry_path = Path(entry)
    if entry_path.is_absolute():
        source_patch = entry_path / data_name
        try:
            rel_dir = entry_path.relative_to(source_cache).as_posix()
        except ValueError:
            rel_dir = f"patches/{idx:08d}"
        return source_patch, rel_dir
    return source_cache / entry / data_name, entry


def load_patch(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        required = ("pc_stage1", "pc_clean", "pc_clean_corr")
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
def predict_stage2(model, patch_batch: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    pc_stage1 = jt.array(np.concatenate([p["pc_stage1"] for p in patch_batch], axis=0))
    pc_edge_risk = None
    if all("pc_edge_risk" in p for p in patch_batch):
        pc_edge_risk = jt.array(np.concatenate([p["pc_edge_risk"] for p in patch_batch], axis=0))
    pc_normal = None
    if all("pc_normal" in p for p in patch_batch):
        pc_normal = jt.array(np.concatenate([p["pc_normal"] for p in patch_batch], axis=0))
    pc_stage2, _ = model.refine(
        pc_stage1,
        pc_edge_risk=pc_edge_risk,
        pc_normal_proxy=pc_normal,
    )
    return pc_stage2.detach().numpy().astype(np.float32)


def _greedy_match(pred: np.ndarray, target: np.ndarray, k: int) -> np.ndarray:
    n = len(pred)
    k = min(k, n)
    dist, idx = cKDTree(target).query(pred, k=k)
    if k == 1:
        dist = dist[:, None]
        idx = idx[:, None]

    edges = []
    for row in range(n):
        for rank in range(k):
            edges.append((float(dist[row, rank]), row, int(idx[row, rank])))
    edges.sort(key=lambda x: x[0])

    row_to_col = np.full(n, -1, dtype=np.int64)
    used_cols = np.zeros(n, dtype=bool)
    for _, row, col in edges:
        if row_to_col[row] < 0 and not used_cols[col]:
            row_to_col[row] = col
            used_cols[col] = True

    unused_cols = [i for i in range(n) if not used_cols[i]]
    for row in np.where(row_to_col < 0)[0]:
        row_to_col[row] = unused_cols.pop()
    return target[row_to_col]


def _dense_match(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    if linear_sum_assignment is None:
        return _greedy_match(pred, target, k=min(64, len(pred)))
    diff = pred[:, None, :] - target[None, :, :]
    cost = np.sum(diff * diff, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    order = np.argsort(row_ind)
    return target[col_ind[order]]


def _sparse_match(pred: np.ndarray, target: np.ndarray, k: int, max_k: int) -> Tuple[np.ndarray, int, str]:
    n = len(pred)
    if n != len(target):
        raise ValueError(f"OT matching expects equal point counts, got pred={n}, target={len(target)}")
    if not HAS_SPARSE_MATCHING:
        return _greedy_match(pred, target, k=min(k, n)), min(k, n), "greedy"

    tree = cKDTree(target)
    cur_k = min(max(k, 1), n)
    max_k = min(max(max_k, cur_k), n)
    while True:
        dist, idx = tree.query(pred, k=cur_k)
        if cur_k == 1:
            dist = dist[:, None]
            idx = idx[:, None]
        rows = np.repeat(np.arange(n), cur_k)
        cols = idx.reshape(-1).astype(np.int64)
        costs = dist.reshape(-1) ** 2 + 1e-12
        graph = coo_matrix((costs, (rows, cols)), shape=(n, n)).tocsr()
        try:
            row_ind, col_ind = min_weight_full_bipartite_matching(graph)
            order = np.argsort(row_ind)
            return target[col_ind[order]], cur_k, "sparse"
        except Exception:
            if cur_k >= max_k:
                return _dense_match(pred, target), n, "dense"
            cur_k = min(cur_k * 2, max_k)


def ot_targets(
    stage2_batch: np.ndarray,
    target_batch: np.ndarray,
    k: int,
    max_k: int,
    method: str,
) -> Tuple[np.ndarray, Dict[str, int]]:
    out = np.empty_like(stage2_batch, dtype=np.float32)
    stats = {"sparse": 0, "dense": 0, "greedy": 0, "max_k_used": 0}
    for i in range(stage2_batch.shape[0]):
        pred = stage2_batch[i].astype(np.float64)
        target = target_batch[i].astype(np.float64)
        if method == "greedy":
            matched = _greedy_match(pred, target, k=min(k, len(pred)))
            used_k = min(k, len(pred))
            used_method = "greedy"
        elif method == "dense":
            matched = _dense_match(pred, target)
            used_k = len(pred)
            used_method = "dense"
        else:
            matched, used_k, used_method = _sparse_match(
                pred,
                target,
                k=k,
                max_k=max_k,
            )
        out[i] = matched.astype(np.float32)
        stats[used_method] += 1
        stats["max_k_used"] = max(stats["max_k_used"], int(used_k))
    return out, stats


def flush_batch(args, model, batch_entries, patch_batch) -> Tuple[int, Dict[str, int]]:
    pc_stage2_batch = predict_stage2(model, patch_batch)
    sync_jittor()
    pc_target_batch = np.concatenate([p[args.target_field] for p in patch_batch], axis=0)
    pc_ot_target_batch, stats = ot_targets(
        pc_stage2_batch,
        pc_target_batch.astype(np.float32),
        k=args.ot_knn,
        max_k=args.ot_max_knn,
        method=args.match_method,
    )

    written = 0
    offset = 0
    for rel_dir, patch in batch_entries:
        count = int(patch["pc_stage1"].shape[0])
        payload = dict(patch)
        payload["pc_dm_stage1"] = patch["pc_stage1"].astype(np.float32)
        payload["pc_stage1"] = pc_stage2_batch[offset:offset + count].astype(np.float32)
        payload["pc_stage2"] = pc_stage2_batch[offset:offset + count].astype(np.float32)
        payload["pc_ot_target"] = pc_ot_target_batch[offset:offset + count].astype(np.float32)
        save_patch(args.out_dir / rel_dir / args.data_name, payload)
        offset += count
        written += count
    return written, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--transform-config", type=Path, default=None)
    parser.add_argument("--list-name", type=str, default="train_cache.txt")
    parser.add_argument("--data-name", type=str, default="patch.npz")
    parser.add_argument("--target-field", type=str, default="pc_clean_corr")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ot-knn", type=int, default=64)
    parser.add_argument("--ot-max-knn", type=int, default=256)
    parser.add_argument(
        "--match-method",
        choices=("sparse", "greedy", "dense"),
        default="sparse",
        help="sparse is closer to OT but slower; greedy is a fast one-to-one approximation.",
    )
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)

    entries = read_cache_list(args.source_cache / args.list_name)
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise ValueError(f"No cache entries found in {args.source_cache / args.list_name}")

    clear_out_dir(args.out_dir, args.overwrite)
    model = load_model(args)

    out_lines: List[str] = []
    batch_entries = []
    written = 0
    match_stats = {"sparse": 0, "dense": 0, "greedy": 0, "max_k_used": 0}
    for idx, entry in enumerate(entries):
        source_patch, rel_dir = resolve_entry(args.source_cache, entry, args.data_name, idx)
        patch = load_patch(source_patch)
        if args.target_field not in patch:
            raise KeyError(f"{source_patch} missing {args.target_field}")
        out_lines.append(rel_dir)
        batch_entries.append((rel_dir, patch))

        if len(batch_entries) >= args.batch_size:
            n, stats = flush_batch(args, model, batch_entries, [p for _, p in batch_entries])
            written += n
            for key in ("sparse", "dense", "greedy"):
                match_stats[key] += stats[key]
            match_stats["max_k_used"] = max(match_stats["max_k_used"], stats["max_k_used"])
            batch_entries = []
            if written % args.log_every == 0:
                print(f"wrote {written}/{len(entries)} OT refine patches", flush=True)

    if batch_entries:
        n, stats = flush_batch(args, model, batch_entries, [p for _, p in batch_entries])
        written += n
        for key in ("sparse", "dense", "greedy"):
            match_stats[key] += stats[key]
        match_stats["max_k_used"] = max(match_stats["max_k_used"], stats["max_k_used"])

    (args.out_dir / args.list_name).write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    metadata = {
        "source_cache": str(args.source_cache),
        "checkpoint": str(args.checkpoint),
        "model_config": str(args.model_config),
        "transform_config": str(args.transform_config) if args.transform_config else "empty_passthrough",
        "list_name": args.list_name,
        "data_name": args.data_name,
        "input_field": "pc_stage1",
        "stage2_output_field": "pc_stage1",
        "debug_stage2_field": "pc_stage2",
        "target_field": args.target_field,
        "ot_target_field": "pc_ot_target",
        "batch_size": args.batch_size,
        "ot_knn": args.ot_knn,
        "ot_max_knn": args.ot_max_knn,
        "match_method": args.match_method,
        "num_entries": len(entries),
        "num_patches": written,
        "matching": match_stats,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote patches: {written}")
    print(f"matching stats: {match_stats}")
    print(f"OT refine cache list: {args.out_dir / args.list_name}")


if __name__ == "__main__":
    main()
