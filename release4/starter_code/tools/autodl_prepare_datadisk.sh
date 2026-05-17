#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${1:-/root/autodl-tmp/jittor_pcd}"

mkdir -p "${DATA_ROOT}"

move_and_link_dir() {
  local name="$1"
  local create_if_missing="${2:-1}"
  local target="${DATA_ROOT}/${name}"

  mkdir -p "${target}"

  if [ -L "${name}" ]; then
    ln -sfn "${target}" "${name}"
    echo "[ok] ${name} -> ${target}"
    return
  fi

  if [ -e "${name}" ]; then
    echo "[move] ${name} -> ${target}"
    shopt -s dotglob nullglob
    local items=("${name}"/*)
    if [ ${#items[@]} -gt 0 ]; then
      mv "${items[@]}" "${target}/"
    fi
    rmdir "${name}"
  elif [ "${create_if_missing}" != "1" ]; then
    echo "[skip] ${name} does not exist"
    return
  fi

  ln -s "${target}" "${name}"
  echo "[link] ${name} -> ${target}"
}

move_and_link_dir "experiments"
move_and_link_dir "result"
move_and_link_dir "true_test"
move_and_link_dir "noisy_test"
move_and_link_dir "denoisy_test"
move_and_link_dir "denoisy_test_raw"
move_and_link_dir "tmp_predict"

# Only move datasets when they already exist in starter_code. Do not create empty
# dataset symlinks, because that would hide data stored elsewhere.
move_and_link_dir "dataset_train" 0
move_and_link_dir "dataset_test_noisy" 0
move_and_link_dir "dataset_clean" 0

echo "Data root is ready: ${DATA_ROOT}"
