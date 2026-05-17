#!/usr/bin/env python
"""Collect VMWriter prediction files into denoisy_test/<key>/denoised.npy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional


def safe_rmtree(path: Path) -> None:
    target = path.resolve()
    root = Path.cwd().resolve()
    if target == root or root not in target.parents:
        raise ValueError(f"Refuse to delete outside current workspace: {target}")
    if target.exists():
        shutil.rmtree(target)


def read_list(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def find_prediction(raw_dir: Path, key: str, filename: str, input_prefix: str) -> Path:
    direct_candidates = [
        raw_dir / input_prefix / key / filename,
        raw_dir / key / filename,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    normalized_suffixes = [
        f"{input_prefix.rstrip('/')}/{key}".replace("\\", "/"),
        key.replace("\\", "/"),
    ]
    matches = []
    for path in raw_dir.rglob(filename):
        parent = path.parent.as_posix()
        if any(parent.endswith(suffix) for suffix in normalized_suffixes):
            matches.append(path)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No prediction found for key={key} under {raw_dir}")
    raise RuntimeError(f"Multiple predictions found for key={key}: {matches}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("denoisy_test_raw"))
    parser.add_argument("--out-dir", type=Path, default=Path("denoisy_test"))
    parser.add_argument("--list", type=Path, default=Path("datalist/noisy_test.txt"))
    parser.add_argument("--input-prefix", type=str, default="noisy_test")
    parser.add_argument("--filename", type=str, default="denoised.npy")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.overwrite:
        safe_rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    keys = read_list(args.list)
    if not keys:
        raise ValueError(f"No keys found in {args.list}")

    copied = 0
    for key in keys:
        src = find_prediction(args.raw_dir, key, args.filename, args.input_prefix)
        dst = args.out_dir / key / args.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Collected predictions: {copied}")
    print(f"Output dir: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
