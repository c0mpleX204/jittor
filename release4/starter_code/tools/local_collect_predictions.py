#!/usr/bin/env python
"""Collect VMWriter prediction files into denoisy_test/<key>/denoised.npy."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Optional


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
    missing = []
    multiple = []
    for key in keys:
        try:
            src = find_prediction(args.raw_dir, key, args.filename, args.input_prefix)
        except FileNotFoundError:
            missing.append(key)
            continue
        except RuntimeError as exc:
            multiple.append(str(exc))
            continue
        dst = args.out_dir / key / args.filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"Collected predictions: {copied}")
    print(f"Output dir: {args.out_dir.resolve()}")
    if missing or multiple:
        if missing:
            print(f"Missing predictions: {len(missing)}")
            for key in missing[:20]:
                print(f"  missing: {key}")
        if multiple:
            print(f"Ambiguous predictions: {len(multiple)}")
            for msg in multiple[:20]:
                print(f"  {msg}")
        raw_count = len(list(args.raw_dir.rglob(args.filename))) if args.raw_dir.exists() else 0
        raise SystemExit(
            f"Collect failed: copied={copied}, expected={len(keys)}, raw_files={raw_count}, "
            f"raw_dir={args.raw_dir}"
        )


if __name__ == "__main__":
    main()
