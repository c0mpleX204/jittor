#!/usr/bin/env python
"""Build a fixed local clean test set from dataset_train OBJ meshes."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import trimesh


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


def read_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def class_key(rel: str) -> str:
    parts = rel.replace("\\", "/").split("/")
    return parts[1] if len(parts) >= 2 and parts[0] == "shapenet" else "unknown"


def choose_entries(entries: List[str], num_models: int, seed: int, stratified: bool) -> List[str]:
    if num_models <= 0 or num_models >= len(entries):
        selected = list(entries)
        random.Random(seed).shuffle(selected)
        return selected

    rng = random.Random(seed)
    if not stratified:
        selected = list(entries)
        rng.shuffle(selected)
        return selected[:num_models]

    by_cls: Dict[str, List[str]] = defaultdict(list)
    for rel in entries:
        by_cls[class_key(rel)].append(rel)
    for items in by_cls.values():
        rng.shuffle(items)

    total = sum(len(items) for items in by_cls.values())
    quotas = {}
    fractional = []
    for cls, items in by_cls.items():
        raw = num_models * len(items) / total
        quota = int(raw)
        if num_models >= len(by_cls) and quota == 0:
            quota = 1
        quota = min(quota, len(items))
        quotas[cls] = quota
        fractional.append((raw - int(raw), cls))

    while sum(quotas.values()) < num_models:
        added = False
        for _, cls in sorted(fractional, reverse=True):
            if quotas[cls] < len(by_cls[cls]):
                quotas[cls] += 1
                added = True
                break
        if not added:
            break

    selected: List[str] = []
    for cls, items in by_cls.items():
        selected.extend(items[: quotas[cls]])
    rng.shuffle(selected)
    return selected[:num_models]


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type: {path}")
    return mesh


def sample_points(mesh: trimesh.Trimesh, num_points: int, seed: int) -> np.ndarray:
    np.random.seed(seed)
    if mesh.faces is not None and len(mesh.faces) > 0:
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
    else:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        if len(vertices) == 0:
            raise ValueError("Mesh has no vertices")
        idx = np.random.choice(len(vertices), size=num_points, replace=len(vertices) < num_points)
        points = vertices[idx]
    return np.asarray(points, dtype=np.float32)


def normalize_unit_sphere(points: np.ndarray) -> np.ndarray:
    p_max = points.max(axis=0)
    p_min = points.min(axis=0)
    center = (p_max + p_min) / 2.0
    centered = points - center
    scale = np.sqrt((centered * centered).sum(axis=1)).max()
    if scale < 1e-12:
        return centered.astype(np.float32)
    return (centered / scale).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_train"))
    parser.add_argument("--datalist", type=Path, default=Path("datalist/train.txt"))
    parser.add_argument("--out-dir", type=Path, default=Path("true_test"))
    parser.add_argument("--list-out", type=Path, default=Path("datalist/true_test.txt"))
    parser.add_argument("--mesh-name", type=str, default="models/model_normalized.obj")
    parser.add_argument("--num-models", type=int, default=200, help="<=0 means all models in datalist")
    parser.add_argument("--points", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-stratified", action="store_true")
    parser.add_argument("--normalize-unit-sphere", action="store_true")
    parser.add_argument("--ignore-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        safe_rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.list_out.parent.mkdir(parents=True, exist_ok=True)

    entries = read_list(args.datalist)
    selected = choose_entries(
        entries,
        num_models=args.num_models,
        seed=args.seed,
        stratified=not args.no_stratified,
    )

    written: List[str] = []
    skipped: List[str] = []
    for i, rel in enumerate(selected):
        mesh_path = args.dataset_dir / rel / args.mesh_name
        if not mesh_path.exists():
            if args.ignore_missing:
                skipped.append(rel)
                continue
            raise FileNotFoundError(mesh_path)

        mesh = load_mesh(mesh_path)
        points = sample_points(mesh, args.points, seed=args.seed + i)
        if args.normalize_unit_sphere:
            points = normalize_unit_sphere(points)

        out_path = args.out_dir / rel / "clean.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path, points.astype(np.float32))
        written.append(rel)

        if (i + 1) % 20 == 0 or i + 1 == len(selected):
            print(f"[true_test] {i + 1}/{len(selected)} processed")

    args.list_out.write_text("\n".join(written) + ("\n" if written else ""), encoding="utf-8")
    meta = {
        "dataset_dir": str(args.dataset_dir),
        "datalist": str(args.datalist),
        "out_dir": str(args.out_dir),
        "list_out": str(args.list_out),
        "mesh_name": args.mesh_name,
        "num_requested": args.num_models,
        "num_written": len(written),
        "num_skipped": len(skipped),
        "points": args.points,
        "seed": args.seed,
        "normalize_unit_sphere": args.normalize_unit_sphere,
        "skipped": skipped,
    }
    (args.out_dir / "true_test_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote clean samples: {args.out_dir.resolve()}")
    print(f"Wrote datalist:      {args.list_out.resolve()}")
    print(f"Samples: {len(written)}")


if __name__ == "__main__":
    main()
