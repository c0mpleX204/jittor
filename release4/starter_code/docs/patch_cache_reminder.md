# Patch Cache Reminder For Teammate AI

Read this before helping with training speed or dataloader issues.

## Directory Rule

The cloud runtime directory is:

```text
/root/src/starter_code
```

The full repository has this structure:

```text
release4/starter_code/...
```

So after pulling or updating the temporary full repo, code must be copied from:

```text
$TMP/release4/starter_code/
```

to:

```text
/root/src/starter_code/
```

Do not assume `/root/src/starter_code` itself is a full git repo.

## Why Patch Cache Exists

Normal training builds every patch online:

```text
load OBJ mesh
sample 32768 surface points
normalize
add noise
build cKDTree patch
return pc_noisy / pc_mix / pc_clean
```

This is CPU-heavy and can make Jittor `num_workers > 0` crash. Patch cache
precomputes those patch tensors once and stores them as `patch.npz`, so training
can use `num_workers: 0` while avoiding most online CPU work.

## Sync Latest Code

```bash
TMP=/root/src/jittor_surface_straight_branch

cd "$TMP"
git fetch origin exp/surface-straight-vm
git checkout exp/surface-straight-vm
git reset --hard origin/exp/surface-straight-vm

rsync -av "$TMP/release4/starter_code/" /root/src/starter_code/

cd /root/src/starter_code
python -m py_compile tools/build_patch_cache.py src/data/datapath.py
```

If the remote branch is not named `origin/exp/surface-straight-vm`, run
`git remote -v` and use the actual remote name.

## Build Surface-Straight Cache

Run in the cloud runtime directory:

```bash
cd /root/src/starter_code

python tools/build_patch_cache.py \
  --dataset-dir /root/autodl-tmp/dataset_train \
  --list datalist/train.txt \
  --transform configs/transform/surface_straight.yaml \
  --out-dir /root/autodl-tmp/cache_surface_straight_nf3000_seed123 \
  --num-items 3000 \
  --seed 123 \
  --overwrite
```

Expected output:

```text
/root/autodl-tmp/cache_surface_straight_nf3000_seed123/train_cache.txt
/root/autodl-tmp/cache_surface_straight_nf3000_seed123/patches/00000000/patch.npz
...
```

Each `patch.npz` contains:

```text
pc_noisy
pc_mix
pc_clean
pc_time, optional
```

The shapes should be `(1, 1000, 3)` for the current Surface-Straight transform.

## Temporary YAML For Cached Training

Do not commit these experiment YAML files unless the experiment is confirmed
useful. Write them directly on cloud.

```bash
cd /root/src/starter_code

cat > configs/transform/cache_passthrough_tmp.yaml <<'YAML'
train_transform: {}
validate_transform: {}
predict_transform: {}
YAML

cat > configs/data/train_cache_surface_tmp.yaml <<'YAML'
train_dataset:
  shuffle: True
  batch_size: 2
  num_workers: 0
  datapath:
    input_dataset_dir: /root/autodl-tmp/cache_surface_straight_nf3000_seed123
    use_prob: False
    loader: npz_patch
    data_name: patch.npz
    ignore_check: True
    data_path:
      cache: [
        [/root/autodl-tmp/cache_surface_straight_nf3000_seed123/train_cache.txt, 1.0],
      ]
YAML
```

Important details:

- `loader: npz_patch` is required.
- `transform: cache_passthrough_tmp` is required because cached patches already
  contain transformed tensors.
- Keep `num_workers: 0`; the cache is meant to avoid unstable multi-worker CPU
  preprocessing.
- `input_dataset_dir` must be the cache directory.
- `train_cache.txt` contains relative directories like `patches/00000000`.

## Example Coupled VM Cached Training Task

This assumes `configs/model/coupled_vm_tmp.yaml` and
`configs/system/coupled_surface_tmp.yaml` already exist.

```bash
cat > configs/task/train_coupled_surface_cache_tmp.yaml <<'YAML'
mode: train
debug: False

components:
  data: train_cache_surface_tmp
  transform: cache_passthrough_tmp
  system: coupled_surface_tmp
  model: coupled_vm_tmp

loss:
  loss: 1.0

optimizer:
  __target__: adam
  lr: 0.00005

trainer:
  epochs: 30
YAML

python run.py --task configs/task/train_coupled_surface_cache_tmp.yaml \
  2>&1 | tee /root/autodl-tmp/train_coupled_surface_cache_30ep.log
```

## Quick Sanity Checks

Check cache files:

```bash
wc -l /root/autodl-tmp/cache_surface_straight_nf3000_seed123/train_cache.txt
find /root/autodl-tmp/cache_surface_straight_nf3000_seed123/patches -name patch.npz | head
```

Check one cached patch:

```bash
python - <<'PY'
import numpy as np
p = "/root/autodl-tmp/cache_surface_straight_nf3000_seed123/patches/00000000/patch.npz"
d = np.load(p)
print({k: d[k].shape for k in d.files})
PY
```

Expected:

```text
{'pc_noisy': (1, 1000, 3), 'pc_clean': (1, 1000, 3), 'pc_mix': (1, 1000, 3), ...}
```

If training crashes with missing `pc_mix` or `pc_clean`, the wrong loader or
wrong transform was used. Use `loader: npz_patch` and
`transform: cache_passthrough_tmp`.
