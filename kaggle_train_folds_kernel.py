#!/usr/bin/env python
"""Kaggle Standalone Training Kernel for Biohub Cell Tracking Cross-Validation Folds.

Run this script or notebook on Kaggle with GPU accelerator (GPU T4 x2 recommended)
to train folds 1, 2, 3, and 4 in parallel.

When finished, all checkpoints are packaged into `/kaggle/working/biohub_cv_weights.zip`.
"""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

WORKING_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
REPO_DIR = WORKING_DIR / "tracking_repo"

# 1. Locate competition datasets
def find_train_dir() -> Path:
    candidates = [
        INPUT_DIR / "competitions" / "biohub-cell-tracking-during-development" / "train",
        INPUT_DIR / "biohub-cell-tracking-during-development" / "train",
    ]
    for c in candidates:
        if c.exists() and any(c.glob("*.zarr")):
            return c
    # Fallback search
    for zarr_path in INPUT_DIR.rglob("*.zarr"):
        if "train" in str(zarr_path):
            return zarr_path.parent
    raise FileNotFoundError("Could not find competition train directory under /kaggle/input")


# 2. Locate support pack repo
def find_support_pack_dir() -> Path:
    for candidate in [
        INPUT_DIR / "datasets" / "pilkwang" / "biohub-tracking-support-pack-50ep-v1",
        INPUT_DIR / "biohub-tracking-support-pack-50ep-v1",
    ]:
        if candidate.exists():
            return candidate
    for r in INPUT_DIR.rglob("scripts"):
        if (r / "train_unet_transformer.py").exists():
            return r.parent
    raise FileNotFoundError("Could not find biohub support pack repo under /kaggle/input")


def main() -> None:
    print("=" * 70)
    print("BIOHUB 5-FOLD CV ENSEMBLE TRAINING KERNEL")
    print("=" * 70)

    train_dir = find_train_dir()
    support_dir = find_support_pack_dir()
    print(f"Train directory: {train_dir}")
    print(f"Support package: {support_dir}")

    # Materialize repo to /kaggle/working/tracking_repo
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    REPO_DIR.mkdir(parents=True, exist_ok=True)

    repo_src = support_dir / "repo"
    repo_zip = support_dir / "repo.zip"
    if repo_src.exists():
        shutil.copytree(repo_src, REPO_DIR, dirs_exist_ok=True)
    elif repo_zip.exists():
        with zipfile.ZipFile(repo_zip) as zf:
            zf.extractall(REPO_DIR)
    else:
        # Check if scripts is directly under support_dir
        if (support_dir / "scripts").exists():
            shutil.copytree(support_dir, REPO_DIR, dirs_exist_ok=True)

    print(f"Materialized tracking repo to: {REPO_DIR}")

    # Copy fold 0 weights if available so output has all 5 folds
    weights_src = support_dir / "weights"
    if weights_src.exists():
        shutil.copytree(weights_src, REPO_DIR / "weights", dirs_exist_ok=True)

    # Copy dataset_splits.json
    splits_file = REPO_DIR / "dataset_splits.json"
    splits_candidates = [
        support_dir / "dataset_splits.json",
        INPUT_DIR / "dataset_splits.json",
        WORKING_DIR / "dataset_splits.json",
    ]
    splits_found = False
    for sc in splits_candidates:
        if sc.exists():
            shutil.copy(sc, splits_file)
            splits_found = True
            break

    if not splits_found:
        print("Warning: dataset_splits.json not found in input; generating from train datasets...")
        # Create minimal 5-fold split from available zarrs
        zarrs = sorted([p.name for p in train_dir.glob("*.zarr")])
        folds = []
        n_val = len(zarrs) // 5
        for i in range(5):
            val_start = i * n_val
            val_end = val_start + n_val if i < 4 else len(zarrs)
            val_set = zarrs[val_start:val_end]
            train_set = [z for z in zarrs if z not in val_set]
            folds.append({"split": i, "train": train_set, "test": val_set})
        splits_file.write_text(json.dumps(folds, indent=2))
        print(f"Generated 5-fold splits for {len(zarrs)} datasets.")

    # Configure training parameters
    import torch
    n_gpus = torch.cuda.device_count()
    print(f"Visible CUDA GPUs: {n_gpus}")
    use_multi_gpu = n_gpus > 1

    folds_to_train = [1, 2, 3, 4]  # Train folds 1-4 (fold 0 already exists in support pack)
    epochs = int(os.environ.get("BIOHUB_TRAIN_EPOCHS", "50"))
    batch_size = int(os.environ.get("BIOHUB_TRAIN_BATCH_SIZE", "8" if n_gpus > 1 else "4"))
    use_pretrained = os.environ.get("BIOHUB_USE_PRETRAINED", "1").lower() in ("1", "true", "yes")
    freeze_unet = os.environ.get("BIOHUB_FREEZE_UNET", "0").lower() in ("1", "true", "yes")
    mitosis_weight = float(os.environ.get("BIOHUB_MITOSIS_LOSS_WEIGHT", "1.0"))

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_DIR / 'src'}:{REPO_DIR / 'scripts'}:{env.get('PYTHONPATH', '')}"

    train_script = REPO_DIR / "scripts" / "train_unet_transformer.py"
    pretrained_weights = REPO_DIR / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"

    for fold in folds_to_train:
        print(f"\n{'='*30} TRAINING FOLD {fold} ({epochs} epochs) {'='*30}")
        cmd = [
            sys.executable,
            str(train_script),
            "--data-dir", str(train_dir),
            "--splits", str(splits_file),
            "--split", str(fold),
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--num-workers", "4",
            "--lr", "1e-4",
            "--pool-kernel-um", "5.0",
            "--det-loss-weight", "1.0",
            "--det-neg-weight", "0.01",
            "--mitosis-loss-weight", str(mitosis_weight),
        ]
        if use_pretrained and pretrained_weights.exists():
            cmd.extend(["--pretrained-weights", str(pretrained_weights)])
        if freeze_unet:
            cmd.append("--freeze-unet")
        if not use_multi_gpu:
            cmd.append("--single-gpu")

        ret = subprocess.run(cmd, env=env, cwd=str(REPO_DIR))
        if ret.returncode != 0:
            print(f"Error training fold {fold}! Exit code {ret.returncode}")
        else:
            print(f"Fold {fold} training completed successfully!")

    # Package all trained weights
    weights_out = WORKING_DIR / "weights"
    if (REPO_DIR / "weights").exists():
        shutil.copytree(REPO_DIR / "weights", weights_out, dirs_exist_ok=True)

    zip_path = WORKING_DIR / "biohub_cv_ensemble_weights.zip"
    print(f"\nCompressing all weights and ensemble scripts into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in weights_out.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(WORKING_DIR))
        # Include updated predict_unet_transformer.py and train_unet_transformer.py
        predict_script = REPO_DIR / "scripts" / "predict_unet_transformer.py"
        if predict_script.exists():
            zf.write(predict_script, "scripts/predict_unet_transformer.py")
        train_script = REPO_DIR / "scripts" / "train_unet_transformer.py"
        if train_script.exists():
            zf.write(train_script, "scripts/train_unet_transformer.py")

    print(f"Done! Created archive: {zip_path} ({zip_path.stat().st_size / (1024*1024):.1f} MB)")
    print("This zip archive contains all trained folds and updated scripts ready for ensemble inference!")


if __name__ == "__main__":
    main()
