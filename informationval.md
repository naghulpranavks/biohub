# Biohub Cell Tracking: Master Project Knowledge Base & Technical Blueprint

This document compiles all valuable technical knowledge, architectural breakdowns, debugging discoveries, hyperparameter sweeps, and codebase solutions accumulated across the entire pair-programming engagement for the **Kaggle Biohub - Cell Tracking During Development** competition.

---

## 1. Executive Summary & Competition Fundamentals

### The Challenge
- **Competition**: Biohub - Cell Tracking During Development (Kaggle)
- **Objective**: Accurately detect cell centroids, reconstruct continuous 3D+time cell migration trajectories (lineages), and correctly identify cell division events (mitosis) in dense, developing zebrafish and Drosophila embryos imaged with volumetric light-sheet microscopy.
- **Data Format**: 3D+time volumes stored in Zarr format (`(t, z, y, x)`), spanning dozens to hundreds of frames per movie, with variable voxel anisotropic spacing (typically $1 \mu m$ in $z$ vs $0.25 - 0.5 \mu m$ in $y, x$).
- **Output Submission**: `submission.csv` specifying:
  - Node rows: `id, dataset, row_type='node', node_id, t, z, y, x, -1, -1`
  - Edge rows: `id, dataset, row_type='edge', -1, -1, -1, -1, -1, source_id, target_id`

### The Evaluation Metric
The evaluation uses a graph-matching tracking metric (akin to AOGM / TRA tracking metrics):
- **False Negative Nodes (FN)**: Missing a cell in a frame.
- **False Positive Nodes (FP)**: Hallucinating a cell where none exists.
- **False Negative Edges (FE)**: Broken track continuity between $t$ and $t+1$.
- **False Positive Edges (EE)**: Spurious linking between different cells.
- **Division Score (Splits)**: Heavily rewards detecting true cell divisions (1 parent $\to$ 2 daughters) while severely penalizing false bifurcations (spurious splits).

---

## 2. Hardware Environment & Compute Architecture

### Local Setup (Windows 11)
- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **CPU**: Intel Core processor, 16+ GB RAM
- **Environment**: Python 3.12 64-bit
- **VRAM Constraints**: 
  - 6 GB VRAM is insufficient to run 8-view volumetric test-time augmentation (TTA) or dual-seed 3D UNet inference across full-frame stacks without Out-Of-Memory (OOM) crashes.
  - Used locally for debugging, Optuna orchestration, script development, and inspection.

### Cloud Engine (Kaggle Environment)
- **Accelerators**: 2x NVIDIA Tesla T4 GPUs (or 1x Tesla P100)
- **GPU Sharding**:
  - Full production runs shard datasets across both GPUs concurrently:
    - GPU 0: `CUDA_VISIBLE_DEVICES=0` processing slices `0::2`
    - GPU 1: `CUDA_VISIBLE_DEVICES=1` processing slices `1::2`
  - Prediction outputs are saved into independent split folders and merged before post-processing.
- **Volatile Storage & Offline Requirements**:
  - Kaggle interactive/commit environments wipe `/kaggle/working` on session reset.
  - Internet access is disabled during competition submission commits.
  - Dependencies (`tracksdata`, `zarr>=3.0.10,<4`, `pyscipopt`, `geff`, `ilpy`, `polars`, `imagecodecs`, `rustworkx`, `numcodecs`, `donfig`, `bidict`) must be installed from offline wheels bundled in attached support datasets (`--no-index --no-deps`).

---

## 3. Analysis & Lineage of Public Baselines

Throughout the project, several community notebooks and baselines were analyzed:

1. **`cell-tracking-getting-started-w-nearest-neighbor.ipynb`**:
   - Simplistic spatial tracker using nearest-neighbor greedy pairing across adjacent frames.
   - Prone to massive edge crossover errors, track breakage, and zero division detection capability (baseline score ~0.3 - 0.5).

2. **`kimi-notebook-v18.ipynb`**:
   - Introduced the core deep learning architecture: 3D Temporal UNet + Node Transformer + SCIP Integer Linear Programming (ILP).
   - Introduced geometric gates on divisions: Sister separation caps and a divergence gate (`SAFE_DIV_DIVERGE_UM=2.25 - 4.5\mu m`) to reject parallel non-separating tracks.

3. **`biohub-0-942-lb-one-knob-past-the-public-line.ipynb`**:
   - Pushed the public benchmark to LB ~0.939 - 0.942.
   - Core discovery (`sdm9` arm): Relaxed the parent-daughter distance limit from $7\mu m \to 9\mu m$, and daughter-sister distance from $12\mu m \to 14\mu m$, recovering ~25-29% of real biological divisions that were being rejected by overly strict heuristics.

4. **`biohub-lf-dctta-v020.ipynb` & `biohub-942tta.ipynb`**:
   - Introduced D4 dihedral test-time augmentation (8 spatial views) for UNet cell detection.
   - Introduced the DeepCenter neural network veto gate (`best.pt`).

5. **`biohub-final-notebook.ipynb`**:
   - The master notebook combining all top community advancements.
   - Rather than clean imports, this notebook executes inline Python string-replacement patches on `predict_unet_transformer.py` to inject multi-model ensembling, TTA, and bidirectional tracking.

---

## 4. End-to-End Pipeline Architecture (`unet_transformer`)

The state-of-the-art inference engine consists of five distinct, sequential phases:

```
[4D Zarr Movie]
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 1: 3D Temporal UNet (Sliding Window + 8-Way TTA) │
│ - Predicts detection heatmaps (det_logits)             │
│ - Accumulates D4 spatial features (Edge TTA)           │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Dual-Seed Ensemble & Peak Finding             │
│ - Blends Primary 400ep model + Secondary Seed 314159   │
│ - Detects cell coordinate peaks (_detect_cells_pooled) │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Transformer Pairwise Edge Prediction          │
│ - Forward pass (t -> t+1)                              │
│ - Reverse-time pass (t+1 -> t)                         │
│ - Harmonic probability association fusion              │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Global Integer Linear Programming (ILP)       │
│ - Minimum-cost flow / binary decision variables        │
│ - Solves optimal global graph with SCIP / ilpy         │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Post-Processing & Geometry Verification       │
│ - Velocity motion relinking                            │
│ - Density-adaptive 1-frame & 2-frame gap recovery      │
│ - Safe division repair & DeepCenter gate checks        │
│ - Short track pruning & adaptive rescue                │
│ - Linefit coordinate smoothing                         │
└────────────────────────────────────────────────────────┘
      │
      ▼
[submission.csv (Nodes + Edges)]
```

### Detailed Phase Breakdown

#### Phase 1: Temporal 3D UNet with 8-View D4 TTA
- Encodes volumetric temporal windows ($W=2$).
- **D4 Spatial TTA**: Generates 8 geometric views per window:
  1. Canonical orientation
  2. Flips: `(-1,)`, `(-2,)`, `(-2, -1)` (horizontal, vertical, 180° rotation)
  3. Rotations: 90° and 270° via `torch.rot90`
  4. Transpose: `transpose(-1, -2)`
  5. Anti-transpose: 90° rotation + transpose
- **Edge-Feature TTA (`BIOHUB_EDGE_FEATURE_TTA=1`)**: Instead of discarding augmented feature maps after cell detection, the feature maps are inverted back to canonical coordinates and averaged:
  $$\text{UNet}_{\text{accum}} = \frac{1}{8} \sum_{v=1}^{8} T_v^{-1}(\text{UNet}_v)$$
  This provides rotation-invariant feature embeddings for downstream edge prediction.

#### Phase 2: Dual-Seed Model Ensembling
- **Primary Model**: 400-epoch snapshot (`weights/unet_transformer/split_0/edge_predictor_best.pth`).
- **Secondary Model**: Independent seed 314159 (`biohub-temporal-unet3d-seed314159-v1`).
- **Detection Logit Alignment**: Secondary detection logits are mean- and variance-aligned to the primary model scale before weighted blending:
  $$\text{Det}_{\text{blended}} = (1 - w_{\text{det}}) \cdot \text{Det}_{\text{primary}} + w_{\text{det}} \cdot \text{Det}_{\text{secondary\_aligned}}$$
  Optimal detection weight: $w_{\text{det}} = 0.80$.

#### Phase 3: Bidirectional Harmonic Association Fusion
- Candidate edges are constructed between cell nodes in frame $t$ and frame $t+1$ within distance threshold.
- The Transformer predicts edge probabilities in **forward time** ($t \to t+1$).
- A secondary **reverse-time pass** ($t+1 \to t$) is executed by passing target features as sources and source features as targets.
- Forward and reverse probabilities are merged via **Harmonic Mean Fusion**:
  $$P_{\text{harmonic}} = \frac{1}{\frac{1 - w_{\text{bi}}}{P_{\text{forward}}} + \frac{w_{\text{bi}}}{P_{\text{reverse}}}}$$
  *Rationale*: If a candidate link is biologically impossible in reverse time ($P_{\text{reverse}} \approx 0$), the harmonic mean severely suppresses it, eliminating temporal tracking artifacts.

#### Phase 4: Global Integer Linear Programming (ILP) Solver
Tracks are formulated as an integer optimization problem using SCIP:
- Decision variables for each node:
  - $x_{\text{app}}$: Cell appears / enters frame (cost: `ilp_appearance_weight`)
  - $x_{\text{disapp}}$: Cell disappears / leaves frame (cost: `ilp_disappearance_weight`)
  - $x_{\text{div}}$: Cell undergoes mitosis into two daughters (cost: `ilp_division_weight`)
- Decision variables for each candidate edge $(u, v)$:
  - $x_{uv} \in \{0, 1\}$: Edge selected (cost: `ilp_edge_weight` + logit costs)
- Strict conservation constraints:
  - Inflow to node $v$: $x_{\text{app}} + \sum_u x_{uv} = 1$
  - Outflow from node $u$: $x_{\text{disapp}} + \sum_v x_{uv} + x_{\text{div}} = 1$

#### Phase 5: Post-Processing & Geometry Verification
1. **Motion Relinking**: Velocity vectors of existing tracks are computed. Tracks terminating near initiating tracks with matching trajectories are relinked (`BIOHUB_OUTPUT_MOTION_RELINK=1`).
2. **Density-Adaptive Gap Closing**: Closes 1-frame and 2-frame skips caused by photobleaching or focal drift. Step distance scales dynamically with local neighbor density (`BIOHUB_GAP_DENSITY_ADAPTIVE=1`).
3. **Safe Division Repair**:
   - Maximum parent-to-daughter distance: $9.0\mu m$ (`BIOHUB_SAFE_DIV_MAX_UM`).
   - Maximum daughter-to-daughter separation: $14.0\mu m$ (`BIOHUB_SAFE_DIV_SISTER_MAX_UM`).
   - Sister symmetry gate: Reject divisions if daughter distances from parent differ by $>60\%$ (`SAFE_DIV_SISTER_SYMMETRY_TAU=0.6`).
   - Forward divergence gate: Sisters must actively separate over subsequent frames (`SAFE_DIV_DIVERGE_UM=2.25\mu m`).
4. **Short Track Pruning & Rescue**:
   - Prunes spurious isolated tracklets with length $<6$ frames (`BIOHUB_OUTPUT_MIN_TRACK_LEN=6`).
   - Adaptive Rescue: Rescues tracklets with length $\ge 4$ if mean edge confidence $>0.88$ (`BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE=1`).

---

## 5. Optuna Sweeps & Parameter Tuning Insights

### The "Frozen Score" Bug & Discovery
During initial Optuna sweeps on Kaggle, trial scores were completely identical across different parameters.
- **Root Cause**: The Python wrapper set environment variables like `os.environ["BIOHUB_ILP_EDGE_WEIGHT"] = ...`, but `predict_unet_transformer.py` relied on direct command-line arguments via `argparse` (`--ilp-edge-weight`, `--det-threshold`). The environment variables were ignored by the ILP branch of the codebase.
- **The Fix**: Updated the Optuna subprocess command call to explicitly inject the swept parameters as CLI flags:
  ```python
  cmd_predict = [
      sys.executable, "scripts/predict_unet_transformer.py",
      "--det-threshold", str(det_threshold),
      "--ilp-edge-weight", str(ilp_edge_weight),
      "--ilp-appearance-weight", str(ilp_appearance_weight),
      "--ilp-disappearance-weight", str(ilp_disappearance_weight),
      "--ilp-division-weight", str(ilp_division_weight),
      # ...
  ]
  ```

### Tuned Parameter Matrix

| Parameter | Baseline Value | Tuned / Optimal Value | Purpose & Rationale |
| :--- | :--- | :--- | :--- |
| `BIOHUB_DET_THRESHOLD` | `0.990` | `0.9710` | Slightly lower detector cutoff; captures faint cell bodies. |
| `BIOHUB_ILP_EDGE_WEIGHT` | `-0.6348` | `-1.2810` | Stronger negative cost incentive; encourages continuing valid edges. |
| `BIOHUB_ILP_APPEARANCE_WEIGHT` | `0.2946` | `0.3753` | Penalizes spontaneous cell appearances without track history. |
| `BIOHUB_ILP_DISAPPEARANCE_WEIGHT`| `0.7523` | `1.9440` | Heavy penalty on cell vanishing; forces ILP to find gap-closures. |
| `BIOHUB_ILP_DIVISION_WEIGHT` | `1.000` | `1.2109` – `1.7883` | Calibrated division incentive to balance true mitosis vs false splits. |
| `BIOHUB_SAFE_DIV_MAX_UM` | `4.7 \mu m` | `9.0 \mu m` | Recovers wide biological parent-daughter links (up to $10.4\mu m$ in GT). |
| `BIOHUB_SAFE_DIV_SISTER_MAX_UM` | `7.2 \mu m` | `14.0 \mu m` | Accommodates rapid post-mitotic daughter displacement. |
| `BIOHUB_SECONDARY_DETECTION_WEIGHT`| `0.475` | `0.800` | Stronger weight to secondary model consensus on cell locations. |
| `BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT` | `0.000` | `0.150` – `0.200` | Incorporates reverse-time tracking consensus. |

---

## 6. The Major Breakthrough: The Silent Bugs That Capped the Score at 0.946

Despite running massive ensembles and sophisticated post-processing, the Kaggle submission score was strictly plateaued at **0.946**. An in-depth code audit of `biohub-final-notebook.ipynb` revealed the exact underlying causes.

### Bug 1: The Frame Retention Guard (`BIOHUB_RETENTION_GUARD`)
- **The Code**:
  ```python
  candidate_retention = blended_candidates / primary_candidates if primary_candidates else 1.0
  use_primary_detection = bool(primary_candidates > 0 and candidate_retention < minimum_retention)
  det_logits[f] = primary_det if use_primary_detection else blended_det
  ```
- **The Defect**:
  `minimum_retention` was hardcoded to `0.90` (90%).
  The author assumed an ensemble would only drop $<10\%$ of cell candidates. However, the primary model generated a substantial number of noisy false positives. When the secondary model ensembled with it, the consensus legitimately pruned 15–30% of these noisy candidates.
  Because retention routinely dropped to 0.80–0.85 (and as low as 0.40 on dense frames), the guard panicked and set:
  `"use_primary": true`
- **The Consequence**:
  On nearly every single frame, the notebook **silently discarded the dual-seed ensemble** and reverted 100% to the single primary model! The user was running a multi-GPU ensemble pipeline, but receiving the score of a single, noisy model.
- **The Solution**:
  Lowered `BIOHUB_DUAL_SEED_MIN_CANDIDATE_RETENTION` from `0.90` to `0.50` across all three locations:
  1. Main configuration header (Line 60)
  2. Configuration guard dictionary `_EXPECTED_TEXT` (Line 90)
  3. Pre-patch variable reset (Line 1125)

### Bug 2: The Post-Run Diagnostic Assertion Crash
- **The Code**:
  ```python
  for _guard_record in _guard_records:
      if (
          float(_guard_record["minimum_retention"]) != 0.9
          or int(_guard_record["primary_candidates"]) < 0
          or int(_guard_record["blended_candidates"]) < 0
      ):
          raise RuntimeError("Frame-retention diagnostic contract changed")
      _guard_expected_use_primary = bool(
          int(_guard_record["primary_candidates"]) > 0
          and float(_guard_record["retention"]) < 0.9
      )
  ```
- **The Defect**:
  After the tracking completed (9.73 minutes) and successfully wrote `submission.csv` (233,831 rows), the notebook ran an internal post-run diagnostic cell. It checked the logs and asserted that `minimum_retention == 0.9`. Because we updated the threshold to `0.5`, it threw:
  `RuntimeError: Frame-retention diagnostic contract changed`
  On Kaggle, any unhandled exception in an interactive or background commit marks the submission as "Failed".
- **The Solution**:
  Patched lines 3221 and 3229 of the notebook to validate against `0.5` instead of `0.9`.

### Bug 3: The DeepCenter Veto Suppression Trap
- **The Observation**:
  During post-processing logs:
  `[44b6_0113de3b] after safe-division repair: ... deepcenter_rejected=110, post_veto_candidates=63`
  `[44b6_0b24845f] ... deepcenter_rejected=168, post_veto_candidates=10`
  `[6bba_05db0fb1] ... deepcenter_rejected=281, post_veto_candidates=33`
- **The Defect**:
  The `DeepCenter` model was only trained for 2 epochs (`best.pt`, epoch 2, `best_score: -0.045`). It is an under-trained network being given absolute veto power over the 400-epoch UNet + Transformer ensemble. It vetoed hundreds of valid division pairs per movie.
- **Future Improvement Opportunity**:
  Relaxing `BIOHUB_REQUIRE_DEEPCENTER_VETO=0` or lowering `BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD` from `0.25` to `0.10` prevents this under-trained model from destroying genuine biological divisions.

---

## 7. Artifacts & Reference Checksums

The following artifacts and cryptographic hashes govern the verified inference pipeline:

| Component | Path / Identifier | Expected SHA256 Checksum |
| :--- | :--- | :--- |
| **Primary Model Weights** | `weights/unet_transformer/split_0/edge_predictor_best.pth` | `12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771` |
| **Secondary Seed Weights**| `biohub-temporal-unet3d-seed314159-v1` | `9bac2fa0dadc4a6fc1899e0caf187f4b553e0a7cd90ba1261a68b35ffe9e305f` |
| **DeepCenter Veto Model** | `weights/full_frame_center/best.pt` | `8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0` |
| **Support Repo Manifest**| `tracking_repo` Python scripts | `978b626d1fd1e7397435a437dfe68691defe1572fc3c20e61012d7c9b52ed029` |

---

## 8. Summary of Actionable Files on Local Disk

- **Master Submission Notebook**: `D:\Downloads_2\biohub-final-notebook.ipynb` (Fully patched with retention limit `0.50` and verified diagnostic checks).
- **Optuna Sweep Runner**: `C:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\optuna_sweep.py` (CLI flag injection enabled).
- **Extracted Source Code**: `C:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking\nb_source.py` (Full 4,172-line unrolled Python script).
- **Diagnostic Patch Scripts**:
  - `patch_notebook.py`: Patches retention threshold to `0.50`.
  - `patch_diagnostics.py`: Patches post-run diagnostic assertions.
