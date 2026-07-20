# Patch Cache Training

This project normally builds each training sample online:

1. load an OBJ mesh
2. sample surface points
3. normalize and augment
4. add high-noise endpoint points
5. build KDTree patches

That CPU pipeline can dominate training time and can make Jittor multi-worker
dataloading unstable. The patch cache stores the final patch tensors on disk so
training only loads small `patch.npz` files.

## Cache Format

Each cache item is:

```text
patches/00000000/patch.npz
```

The file contains:

- `pc_noisy`: high-noise endpoint patch, shape `(1, patch_size, 3)`
- `pc_mix`: interpolated current patch, shape `(1, patch_size, 3)`
- `pc_clean`: target patch, shape `(1, patch_size, 3)`
- `pc_time`: optional interpolation time

The generated `train_cache.txt` stores relative patch directories for the
existing `Datapath` class.

## Build Cache On Cloud

Run from the actual training directory:

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

`num-items=3000` matches the current low-memory Surface-Straight training
setting where one online patch is produced per sampled shape.

## Temporary YAML For Cached Training

Do not commit experiment YAML until the experiment is confirmed useful. Write it
on the cloud with `cat > ...`.

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

Use this data component with the usual model/system/task YAML. Keep
`num_workers=0`; the point of the cache is to avoid online CPU work without
relying on fragile multi-worker loading.

## Sync Reminder

The cloud runtime directory is:

```text
/root/src/starter_code
```

The full repository layout is:

```text
release4/starter_code/...
```

After pulling the branch into the temporary full repo directory, sync with:

```bash
rsync -av "$TMP/release4/starter_code/" /root/src/starter_code/
```
