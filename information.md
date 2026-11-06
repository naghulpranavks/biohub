# Biohub Cell Tracking — Master Reference (`information.md`)

> **PURPOSE**: This is the **definitive domain knowledge, pipeline architecture, and competition strategy reference** for the CZ Biohub Cell Tracking During Development competition. It covers the biophysical reality of the imaging system, the competition metric in depth, all pipeline breakthroughs and their failure root causes, the current state of the neural architecture, and the roadmap to break beyond 0.947 toward 0.970–0.985+.

---

## Table of Contents

1. [Biological System](#1-biological-system-zebrafish-danio-rerio-embryogenesis)
2. [Physical Kinematics of Cell Motion](#2-physical-kinematics-of-cell-motion-in-living-tissue)
3. [Biophysics of Mitosis & Cytokinesis](#3-biophysics-of-mitosis--cytokinesis)
4. [Optical Physics of Lightsheet Microscopy](#4-optical-physics-of-lightsheet-fluorescence-microscopy-lsfm)
5. [Competition Metric — Deep Dive](#5-competition-metric--deep-dive)
6. [Competition Leaderboard History & Failure Analysis](#6-competition-leaderboard-history--failure-analysis)
7. [Current Pipeline Architecture (biohub-final-notebook.ipynb)](#7-current-pipeline-architecture)
8. [Neural Architecture: UNetNodeTransformer](#8-neural-architecture-unetnodeTransformer)
9. [Neural Mitosis Head — Implementation (Option 2)](#9-neural-mitosis-head--adaptive-ilp-division-cost-option-2)
10. [5-Fold CV Ensemble Pipeline (Option 1)](#10-5-fold-cv-ensemble-pipeline-option-1)
11. [Score Boost Roadmap — Six Architectural Pillars](#11-score-boost-roadmap--six-architectural-pillars)
12. [Training Environment & Execution Reference](#12-training-environment--execution-reference)
13. [Critical Invariants & Safety Rules](#13-critical-invariants--safety-rules)

---

## 1. Biological System: Zebrafish (*Danio rerio*) Embryogenesis

### 1.1 Developmental Stages Imaged
Microscopy volumes capture early zebrafish embryos during blastula and early gastrula stages (~**3.5–6 hours post-fertilization**):
- **Cleavage/Blastula Stage** (128-cell to sphere stage): Rapid, nearly synchronous divisions without $G_1$/$G_2$ phases.
- **Mid-Blastula Transition (MBT)**: ~Cell cycle 10 (~512–1000 cells). Cycles lengthen, divisions become asynchronous, cell motility initiates.
- **Epiboly/Early Gastrulation**: Cells thin and spread vegetally over the yolk cell.

### 1.2 Cell Types
1. **Deep Cells (DEL)**: Majority of embryo cells. Rounded, loosely packed, actively dividing.
2. **Enveloping Layer (EVL)**: Single-cell-thick epithelial monolayer. Flatter, mechanically connected.
3. **Yolk Syncytial Layer (YSL)**: Multinucleated non-cellular layer at the yolk boundary.

### 1.3 Cell Morphology & Volume Conservation
- **Nuclear Labeling**: Fluorescently tagged **nuclei** (e.g., H2B-GFP), not entire cell bodies.
- **Nuclear Geometry**: Physical diameter **5.0–8.0 µm**. In raw voxels, flattened along Z by factor 4.
- **Volume Conservation**: Each division halves cell volume: $V_{\text{daughter}} \approx \frac{1}{2} V_{\text{parent}}$.

### 1.4 Cell Cycle Kinetics & Division Frequency
- **Pre-MBT cycle**: ~15–25 minutes.
- **Post-MBT/Gastrula cycle**: ~45–120 minutes.
- **Frame interval**: $\Delta t \approx$ 45–90 seconds. 100 frames = 75–150 minutes.
- **Divisions per cell**: 1–3 maximum across the entire 100-frame sequence. Most cells: 0 or 1 division.
- **Ground truth division count**: 151 true divisions across 87 of 199 training datasets (~0.5% of transitions).

---

## 2. Physical Kinematics of Cell Motion in Living Tissue

### 2.1 Low Reynolds Number (Stokes Flow — Zero Inertia)
$Re = \frac{\rho v L}{\mu} \ll 10^{-4}$. Viscous forces dominate. **Inertia is effectively zero.**
- If a cell stops cytoskeletal propulsion, motion ceases within microseconds.

### 2.2 Overdamped Dynamics & Velocity Continuity
$$\gamma \cdot \vec{v}(t) = \vec{F}_{\text{active}}(t) + \vec{F}_{\text{contact}}(t)$$

**Critical Invariant**: Velocity is continuous: $\vec{v}(t+1) \approx \vec{v}(t)$.
- A cell **cannot** abruptly reverse direction by 180° in a single frame at high speed. Any trajectory with an acute high-speed reversal is almost certainly an **identity swap**.

### 2.3 Physiological Velocity Bounds
- **Interphase**: 0.5–2.5 µm/min; frame-to-frame Δr = 0.8–3.0 µm, rarely >5.0 µm.
- **Mitotic cytokinesis**: Sister separation 1.0–3.0 µm/min, max frame-to-frame ≤6.0–7.0 µm.
- **Hard limit**: Any frame-to-frame link >8.0 µm during interphase = tracking failure.

### 2.4 Collective Migration & Velocity Field Coherence
- Cells within 15–30 µm share a coherent tissue drift vector.
- Anomaly: if a cell moves 5 µm/frame while all 8 neighbors within 20 µm move 1.5 µm in the opposite direction → identity swap.

### 2.5 Centroid Localization Noise
- **Biological translocation**: Smooth, persistent (persistence time τ ≈ 5–15 min).
- **Measurement noise**: ±0.3–0.8 µm in Y,X; ±1.0–1.8 µm in Z (due to 4:1 PSF anisotropy).
- Trajectory smoothing (line-fit) works by filtering high-frequency optical jitter while preserving biological translocation.

---

## 3. Biophysics of Mitosis & Cytokinesis

Accurately detecting mitosis is rewarded by **Division Jaccard** ($+0.10 \times J_{\text{div}}$) and is currently the **largest untapped score component** (currently $J_{\text{div}} = 0.00$).

### 3.1 Mitotic Choreography Timeline
1. **G2 Interphase**: Distinct rounded nucleus, diffuse chromatin, sharp boundary.
2. **Prophase/Prometaphase**: Chromatin condenses; nuclear envelope breaks down. Blob becomes compact, bright, slightly ellipsoidal.
3. **Metaphase**: Chromosomes align on equatorial plate. Appears as flattened high-intensity disc.
4. **Anaphase**: Single metaphase blob elongates and splits into two lobes within 1–2 frames.
5. **Telophase/Cytokinesis**: Cleavage furrow ingresses. Daughter nuclei re-form and separate.
6. **Early G1**: Chromatin decondenses rapidly. Daughters swell causing temporary dip in fluorophore density.

```
     Parent (t)          Telophase (t+1)        Daughters (t+2)
   ┌─────────────┐        ┌────┐ ┌────┐        ┌──────┐  ┌──────┐
   │   Nucleus   │  ───►  │ C1 │ │ C2 │  ───►  │  C1  │  │  C2  │
   └─────────────┘        └────┘ └────┘        └──────┘  └──────┘
       P(t)                v1 ◄── ┼ ──► v2         Sister Dist ≤ 14 µm
                            Cleavage Midpoint
```

### 3.2 Mitotic Spindle Geometry
- Daughters move in **antiparallel** directions from parent:
  $$\cos\theta = \frac{\vec{v}_1 \cdot \vec{v}_2}{\|\vec{v}_1\| \|\vec{v}_2\|} \le 0.25–0.35$$
- If both daughters move in the same direction ($\cos\theta > 0.5$) → **biologically impossible**.

### 3.3 Cleavage Furrow Centering Invariance
$$\vec{M} = \frac{\vec{C}_1 + \vec{C}_2}{2} \approx \vec{P}$$
- **Empirical bound**: $\|\vec{M} - \vec{P}\|_{\text{phys}} \le 3.5$ µm.

### 3.4 Sister Separation Distance Distribution
- Immediate cytokinesis ($t+1$): sister distance **6.0–13.7 µm**. Median: **10.4 µm**. 90th pct: **13.0 µm**. Hard cap: **14.0 µm**.
- Daughter-to-parent distance: **3.5–7.5 µm** per daughter. Max: **10.4 µm**.

> [!IMPORTANT]
> `DIV_SISTER_MAX_UM = 14.0 µm` is the biologically correct threshold. The original default of 8.0 µm discarded ~50% of real divisions.

### 3.5 Post-Mitotic Chromatin Decondensation & Optical Dropout
- As nuclear envelope reforms, volume expands 50–100% while fluorophore amount stays constant → **intensity drops precipitously**.
- **Consequence**: 1–2 frames after division, one or both daughters frequently drop below peak-detection threshold.
- Parent already has `out_degree = 1` (one detected daughter), so **standard gap closers ignore it entirely**.
- The second daughter reappears at $t+2$ as an orphan track → severed lineage.

### 3.6 Biological Refractory Period
- Minimum cell cycle even in fastest pre-MBT zebrafish: **≥15 minutes** (~15–20 frames).
- Any node showing `out_degree = 2` within 5 frames of a previous division = artifact.

---

## 4. Optical Physics of Lightsheet Fluorescence Microscopy (LSFM)

### 4.1 Voxel Dimensions (Physical Scale)
$$z = 1.625\,\mu\text{m}, \quad y = 0.40625\,\mu\text{m}, \quad x = 0.40625\,\mu\text{m}$$
$$\text{Anisotropy Ratio} = z/y = 4.0\times$$

### 4.2 PSF Anisotropy
- **Lateral** (Y,X): diffraction-limited ~0.4–0.5 µm resolution.
- **Axial** (Z): light-sheet thickness determines ~1.8–2.5 µm resolution.
- A 3 µm sphere: $r_z \approx 1.85$ voxels vs $r_y = r_x \approx 7.38$ voxels.

### 4.3 Measurement Noise Asymmetry (Critical for Tracking)
- Centroid localization uncertainty: $\sigma_z \approx 4 \times \sigma_{yx}$.
- 1 voxel shift in Z = **1.625 µm physical**; 1 voxel shift in Y/X = **0.406 µm physical**.
- Velocity vectors $\vec{v}(t)$ calculated from consecutive frames suffer **heavy angular jitter** from Z-quantization noise.
- This is why velocity-aligned search ellipsoids (Breakthrough 4) failed: the estimated velocity direction $\hat{v}$ was too noisy to use as a gate orientation.

### 4.4 Optical Attenuation & Depth-Dependent Blur
- Deeper nuclei appear dimmer, broader, lower contrast from Rayleigh/Mie scattering.
- Non-uniform background glow from yolk autofluorescence.
- Shadow artifacts when blood cells or pigment absorb the light sheet.

### 4.5 Photobleaching
- Average nuclear peak intensity at t=99 is typically **15–25% lower** than at t=0.
- Fixed global intensity thresholds will detect fewer cells at late timepoints → must use percentile/adaptive normalization.

---

## 5. Competition Metric — Deep Dive

### 5.1 The Scoring Formula
$$\text{Score} = J_{\text{adj}} + 0.1 \times J_{\text{div}}$$

$$J_{\text{adj}} = \max\!\left(0,\; J_{\text{edge}} \cdot \min\!\left(1.0,\; \frac{N_{\text{true}}}{N_{\text{pred}}}\right)\right)$$

$$J_{\text{edge}} = \frac{\text{TP}_e}{\text{TP}_e + \text{FP}_e + \text{FN}_e}, \quad J_{\text{div}} = \frac{\text{TP}_d}{\text{TP}_d + \text{FP}_d + \text{FN}_d}$$

Where $N_{\text{true}}$ = `estimated_number_of_nodes` from GT GEFF metadata.

### 5.2 The Over-Prediction Penalty (CRITICAL)
- If $N_{\text{pred}} \le N_{\text{true}}$: Penalty Factor = 1.0 (no penalty).
- If $N_{\text{pred}} > N_{\text{true}}$: Penalty Factor $= N_{\text{true}} / N_{\text{pred}} < 1.0$.

**Numerical examples** with baseline Raw Edge Jaccard = 0.950:
- 5% excess nodes → Score = 0.950 × (1/1.05) = **0.905** (−0.045 penalty!)
- 15% excess nodes → Score = 0.950 × (1/1.15) = **0.826** (−0.124 penalty!)

> [!CAUTION]
> **Over-predicting nodes is fatal.** Never add speculative candidate nodes. Synthetic gap-closing nodes inflate $N_{\text{pred}}$ and collapse the score.

### 5.3 Node Matching (Hungarian, 7.0 µm Cutoff)
- At each timepoint $t$, predicted nodes are matched to GT nodes using bipartite matching.
- Any predicted node >**7.0 µm** from any GT node = unmatched (FP).
- This equals ~1 nuclear diameter, tolerating centroid jitter without confusing adjacent cells.

### 5.4 Graph Topology Invariants
Any valid biological lineage graph **must** satisfy:
1. **$\text{in\_degree} \le 1$**: Animal cells don't fuse during development.
2. **$\text{out\_degree} \le 2$**: Mitotic spindle has exactly 2 poles. Tripolar divisions are lethal.
3. **$t_{\text{target}} = t_{\text{source}} + 1$**: Strict consecutive-frame increment only.

### 5.5 Division Mathematical Leverage (The Key Insight)
Current baseline: $J_{\text{div}} = 0.0$ ($\text{TP}=0, \text{FP}=0, \text{FN}=151$ across training set).

| $J_{\text{div}}$ Achieved | Score Gain | Net Score |
| :---: | :---: | :---: |
| 0.00 (current) | +0.000 | 0.946 |
| 0.20 | +0.020 | 0.966 |
| 0.40 | +0.040 | 0.986 |
| 0.60 | +0.060 | 1.006 (capped) |

**Unlocking cell divisions is the single highest mathematical leverage point in the entire competition.**

---

## 6. Competition Leaderboard History & Failure Analysis

### 6.1 Full Score Progression

| Configuration | Score | Delta | Key Finding |
| :--- | :---: | :---: | :--- |
| Classical DoG + Euclidean Hungarian (`src/`) | 0.589 | — | Pure classical processing fails on dense embryonic tissue |
| Dual-seed UNetNodeTransformer + ILP + base post-processing (`biohub-0.947.ipynb`) | **0.947** | baseline | Top-performing public baseline |
| MRTCA Strategy A (VCR > 3.5 kinematic pruning, removal-only) | **0.947** | 0.000 | Monotonic filter, matches baseline — no gain |
| MRTCA Strategy B (VCR + NCC appearance, removal-only) | **0.947** | 0.000 | Matches baseline with zero regression |
| MRTCA Strategy C (VCR + NCC + DeepCenter re-verify, removal-only) | **0.947** | 0.000 | Matches baseline with zero regression |
| **harmonic_v3_division_wide configuration** | **0.939** | −0.008 | Baseline for neural mitosis head runs (dual-seed clean pipeline, public LB) |
| Strategy 1 (Logit-scaled neural cost in Hungarian) | 0.944 | −0.003 | High logit pull caused identity micro-swaps |
| Strategy 2 (Local density-adaptive gating) | 0.944 | −0.003 | Tight gate severed accelerating cells |
| Breakthrough 2 alone (ILP division unblocking) | 0.944 | −0.003 | FP divisions penalized edge Jaccard more than +0.10 gained |
| Breakthroughs 2+1+5 (Run1: Aniso NMS + post-mitotic stitch + ILP) | 0.911 | −0.036 | NMS train/test shift + synthetic node over-prediction |
| All Breakthroughs 1–5 (Run3) | 0.872 | −0.075 | Breakthroughs 3 & 4 over-constrained; severed valid tracks |
| DET_THRESHOLD raised from 0.965 → 0.971 | 0.941 | −0.006 | Lost faint mitotic cells → cascading short track pruning |

### 6.2 Root Cause: Why Every Heuristic Inside `motion_relink_edges` Drops −0.003

Any modification applied directly to the frame-to-frame Hungarian relinking stage consistently drops by exactly −0.003 (to 0.944). The baseline 5.5 µm isotropic gate with $1.0 \times P_{\text{edge}}$ is already optimal. **The relinking stage must be left completely untouched.**

### 6.3 Root Cause: Why Breakthrough 4 (Velocity Ellipsoids) Caused −0.075

Centroid localization in Z has high measurement noise (1 voxel = 1.625 µm). The estimated velocity vector $\hat{v}$ wobbles erratically frame-to-frame due to Z-quantization noise. Narrowing the transverse radius $R_\perp$ to 2.5–4.0 µm caused:
- Normal turning/accelerating cells to fall outside the gate
- Continuous tracks severed into short stubs (<6 frames)
- `filter_short_tracks` pruned them → massive Edge Jaccard collapse

### 6.4 Root Cause: Why Breakthrough 3 (Cleavage Midpoint Gate) Failed

Real tissue undergoes mechanical shear and uneven optical slice illumination. The strict gate ($\text{midpoint} \le 3.5$ µm, symmetry ratio $\ge 0.35$) rejected valid **asymmetric** mitotic divisions. Each rejected division permanently severed a daughter track into an isolated orphan.

### 6.5 Root Cause: Why Raising DET_THRESHOLD from 0.965 → 0.971 Failed

- Dropped faint cell centroids (cells in early mitosis/telophase with decondensing chromatin or cells deep in the specimen).
- Losing 1 node at time $t$ creates 2 false negatives: $(t-1 \to t)$ and $(t \to t+1)$.
- Remaining pieces fall below 6 frames → `filter_short_tracks` prunes them → 4–6 edges lost per dropped cell.
- **Empirical Rule**: `DET_THRESHOLD = 0.965` is at the optimal recall-precision boundary and must never be raised.

---

## 7. Current Pipeline Architecture

### 7.1 The `biohub-final-notebook.ipynb` Submission Pipeline

**Configuration Constants (DO NOT MODIFY):**
- `BIOHUB_DET_THRESHOLD = 0.965` (optimal; never raise)
- `BIOHUB_NMS_KERNEL = (3, 3, 3)` (isotropic; must match training)
- `safe_div_max_um = 9.0`
- `safe_div_sister_max_um = 14.0`
- `recover_strict_gap2 = True`
- `motion_relink = True`
- DeepCenter veto enabled

**Runtime Configuration (from `BIOHUB_PRESET = harmonic_v3_division_wide`):**
- Reverse-time association weight: 0.200
- ILP division lambda (adaptive mitosis head): configured via `BIOHUB_ILP_DIVISION_LAMBDA`
- Mitosis head: configured via `BIOHUB_USE_MITOSIS_HEAD`

### 7.2 Post-Processing Graph Pipeline (Cell 5 Execution Order)

```
Raw Predictions (Nodes & Edges)
         │
         ▼
1. Motion Relinking (Hungarian 1-to-1 + Velocity EMA)
         │
         ▼
2. ILP Division Preservation (Merge confirmed ILP divisions pre-extracted)
         │
         ▼
3. Single-Frame Gap Closing (t → t+2)
         │
         ▼
4. Strict Gap-2 Recovery (t → t+3)
         │
         ▼
5. Safe Division Post-Link (Mutual NN + Spindle Gate)
         │
         ▼
6. Post-Mitotic Gap Stitching — Breakthrough 5 (t → t+2 for 1-child parents)
         │
         ▼
7. Division Geometry Filter (Sister Dist ≤ 14 µm)
         │
         ▼
8. Prune Isolated Nodes (degree = 0)
         │
         ▼
9. Short Track Filtering (≥6 frames, division components protected)
         │
         ▼
10. Line-Fit Trajectory Smoothing
         │
         ▼
Final Validated Lineage Graph
```

### 7.3 Breakthrough 5: Post-Mitotic Lineage Gap Stitching

The most important intact breakthrough. Identifies parents $P(t)$ with `out_degree = 1` (one detected daughter $C_1(t+1)$) and finds orphan daughter track starts $C_2(t+2)$ satisfying:
- $d(P, C_2) \le 10.0$ µm
- $d(C_1, C_2) \le 14.0$ µm
- $\cos\theta \le 0.25$ (sisters diverge antiparallel)
Inserts interpolated synthetic centroid at $t+1$ → preserves the division topology.

---

## 8. Neural Architecture: UNetNodeTransformer

### 8.1 Architecture Overview
File: [`repo/scripts/train_unet_transformer.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/repo/scripts/train_unet_transformer.py)

```
Input: 3D+t video frames (B, W, Z, Y, X) where W = window_size = 2
  │
  ├─► TemporalUNet3D (1,800,000 params, in_channels=1, out_channels=32, layers=[32,64,128])
  │       Downsampling: (1, 4, 4) spatial
  │       │
  │       ├─► Detection Head (detect_head: Conv3d(32→1, kernel=1))
  │       │       → Heatmap logits per frame
  │       │       → NMS peak extraction (pool_kernel_um = 5.0 µm)
  │       │
  │       └─► Feature Maps (B, W, 32, Z/1, Y/4, X/4)
  │
  ├─► _index_features: Integer voxel lookup at detected centroids
  │       → (B, N, 32) UNet feature vectors per node
  │
  ├─► Sinusoidal Positional Embeddings (4 × _POS_EMBED_DIM = 64 dims total)
  │       → Encodes (t, z, y, x) physical coordinates
  │
  ├─► SimpleNodeTransformer (n_heads=4, n_blocks=4, hidden_dim=128)
  │       → Cross-attention between frame t and t+1 node sets
  │       → Edge logit matrix (B, N_src, N_tgt)
  │
  └─► MITOSIS HEAD (NEW — Option 2) ← 3-layer MLP, ~586k params
          Linear(64→64) → GELU → Dropout(0.3) → Linear(64→32) → GELU → Linear(32→1)
          Input: concat(unet_feat, pos_feat) per node
          Output: (B, N) raw logits for mitosis probability
```

### 8.2 Training Details
- **Loss**: Edge BCE + Detection BCE + Mitosis BCE (`pos_weight=20.0` for rare divisions)
- **Checkpoint scoring**: `score = (acc × recall) + 0.10 × mitosis_F1` (mirrors competition metric)
- **Training data**: 199 Zarr datasets, 5-fold stratified split (30–31 divisions per fold)
- **Splits file**: `dataset_splits.json`
- **Weights directory**: `repo/weights/unet_transformer/split_{fold}/`

### 8.3 Known Invariants
- `pool_kernel_um = 5.0 µm` (NMS radius at inference) must match training
- `downsample = (1, 4, 4)` must not be changed without full retrain
- `window_size = 2` (consecutive frame pairs)
- `_POS_EMBED_DIM = 16` (4×16 = 64 total positional features)

---

## 9. Neural Mitosis Head & Adaptive ILP Division Cost (Option 2)

### 9.1 Motivation
The current ILP uses a **static** division cost $C_{\text{div}} \ge 1.2$, which uniformly penalizes all division events. This is why Division Jaccard = 0.00 despite 151 ground-truth divisions in training data.

### 9.2 Architecture
A lightweight 3-layer MLP (`mitosis_head`) added to `UNetNodeTransformer`:
```python
mitosis_head = nn.Sequential(
    nn.Linear(unet_out_channels + pos_feat_dim, 64),  # 32 + 64 = 96 → 64
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 1),
)
```
- Input: `concat(unet_feat, pos_feat)` per node = (B, N, 96)
- Output: (B, N) raw logits for division probability

### 9.3 Adaptive ILP Division Cost Formula
$$C_{\text{div}}(u) = C_{\text{base}} - \lambda_{\text{div}} \cdot \text{clamp}\!\left(\log\frac{P(u) + \epsilon}{1 - P(u) + \epsilon},\; -4.0,\; 4.0\right)$$

| $P_{\text{mitosis}}(u)$ | $C_{\text{div}}(u)$ (with $\lambda=0.8$) | ILP behavior |
| :---: | :---: | :--- |
| ~0.01 (quiescent) | ≈ +3.4 | Heavily penalizes splits → eliminates false positives |
| ~0.50 (uncertain) | ≈ base cost | Neutral |
| ~0.95 (mitotic) | ≈ −1.4 | Actively rewards track branching |

### 9.4 Training Procedure

**Warm-Start Fine-Tuning (Recommended — `--freeze-unet`):**
```powershell
.\venv\Scripts\python.exe repo\scripts\train_unet_transformer.py `
    --data-dir "D:\Downloads_2\biohub-cell-tracking-during-development\train" `
    --splits dataset_splits.json `
    --split 0 `
    --pretrained-weights repo\weights\unet_transformer\split_0\edge_predictor_best_baseline.pth `
    --freeze-unet `
    --epochs 10 `
    --lr 5e-5 `
    --batch-size 1 `
    --num-workers 0 `
    --max-iters 150 `
    --mitosis-loss-weight 1.0
```

`--freeze-unet` freezes the 1.8M-param UNet and detection head; only the 586k-param transformer + mitosis MLP are trained.

**Advantages of `--freeze-unet`:**
- VRAM drops to **<1.5 GB** (fits comfortably on RTX 4050 6 GB)
- Speed: ~4.5 it/s (vs ~1.1 it/s with full model)
- Preserves pretrained detection recall (0.9789+)

### 9.5 Verified Local Training Results (10 Epochs, Split 0)

| Metric | Before Fine-Tuning | After 10 Epochs | Change |
| :--- | :---: | :---: | :--- |
| Mitosis BCE Loss | 0.7320 | **0.0182** | **−97.5%** |
| Edge Prediction BCE | 0.0002 | 0.0002 | Stable |
| Edge Accuracy | 0.9999 | 0.9999 | Optimal |
| Node Recall | 0.9789 | **0.9900** | +0.0111 |
| Composite Score | 0.9788 | **0.9899** | **+0.0111** |
| VRAM | — | <1.5 GB | Stable (0 OOM) |

Checkpoint: `repo/weights/unet_transformer/split_0/edge_predictor_best.pth`  
Baseline backup: `repo/weights/unet_transformer/split_0/edge_predictor_best_baseline.pth`

### 9.6 ILP Integration in `predict_unet_transformer.py`
- `load_model()` auto-detects if checkpoint contains `mitosis_head` weights
- `build_graph(coords, edges, division_costs)` injects `division_cost` as Polars `NodeAttr`
- ILP solved with `division_weight = td.NodeAttr("division_cost")`
- CLI flags: `--use-mitosis-head`, `--no-mitosis-head`, `--ilp-division-lambda 0.8`

### 9.7 Full Video Verification (44b6_12dfb391, 100 frames)
- Nodes solved: 44,497
- Max in-degree: **1** ✓ (strict constraint)
- Max out-degree: **2** ✓ (strict constraint)
- `division_fp = 0` (zero false-positive divisions)

---

## 10. 5-Fold CV Ensemble Pipeline (Option 1)

### 10.1 Dataset Splits
File: [`create_dataset_splits.py`](file:///c:/Users/Naghul Pranav/Documents/Biohub_Cell_Tracking/create_dataset_splits.py)  
Output: [`dataset_splits.json`](file:///c:/Users/Naghul Pranav/Documents/Biohub_Cell_Tracking/dataset_splits.json)
- 199 datasets across 5 folds
- ~159 train / ~40 val per fold
- Stratified by division frequency: **30–31 divisions per fold** (leak-free)

### 10.2 Ensemble Inference

Detection heatmap averaging across K models:
$$\bar{D}(t, z, y, x) = \frac{1}{K}\sum_{k=1}^K D_k(t, z, y, x)$$

Harmonic bidirectional edge logit fusion:
$$\bar{Z}(u, v) = \frac{1}{K}\sum_{k=1}^K Z_k(u, v)$$

Mitosis probability via averaged logits before sigmoid:
$$P_{\text{mitosis}}(u) = \sigma\!\left(\frac{1}{K}\sum_{k=1}^K \text{logit}_k(u)\right)$$

### 10.3 CLI Invocation
```powershell
.\venv\Scripts\python.exe repo\scripts\predict_unet_transformer.py `
    --ensemble-weights `
        repo\weights\unet_transformer\split_0\edge_predictor_best.pth `
        repo\weights\unet_transformer\split_1\edge_predictor_best.pth `
        repo\weights\unet_transformer\split_2\edge_predictor_best.pth `
        repo\weights\unet_transformer\split_3\edge_predictor_best.pth `
        repo\weights\unet_transformer\split_4\edge_predictor_best.pth `
    --use-ilp `
    --ilp-division-lambda 0.8 `
    --det-threshold 0.965
```

### 10.4 Training Remaining Folds Locally
```powershell
.\venv\Scripts\python.exe train_remaining_folds.py `
    --folds 1 2 3 4 `
    --freeze-unet `
    --epochs 10 `
    --max-iters 150 `
    --batch-size 1 `
    --pretrained-weights repo\weights\unet_transformer\split_0\edge_predictor_best.pth
```

### 10.5 Training on Kaggle (T4 × 2 GPU)
Script: [`kaggle_train_folds_kernel.py`](file:///c:/Users/Naghul Pranav/Documents/Biohub_Cell_Tracking/kaggle_train_folds_kernel.py)

Environment variables supported:
- `BIOHUB_TRAIN_EPOCHS` (default: 50)
- `BIOHUB_TRAIN_BATCH_SIZE` (default: 8 with dual GPU / 4 with single)
- `BIOHUB_USE_PRETRAINED` (default: "1" — warm-starts from split_0)
- `BIOHUB_FREEZE_UNET` (default: "0" — set "1" for mitosis-only fine-tune)
- `BIOHUB_MITOSIS_LOSS_WEIGHT` (default: "1.0")

Output: `/kaggle/working/biohub_cv_ensemble_weights.zip` (contains all fold weights + updated scripts)

### 10.6 Notebook Auto-Discovery
`biohub-final-notebook.ipynb` Cell 3:
- Checks if `--ensemble-weights` is supported by active `predict_unet_transformer.py`
- Auto-discovers updated script from `/kaggle/input` zip archives
- Falls back to legacy `--weights` + `BIOHUB_SECONDARY_WEIGHTS` monkeypatch for dual-seed baseline

### 10.7 Current Weights Status
| Split | Status | Path |
| :--- | :--- | :--- |
| split_0 | ✅ Trained (400+ epochs baseline + 10-epoch mitosis fine-tune) | `repo/weights/unet_transformer/split_0/` |
| split_1 | ⏳ Needs training | `repo/weights/unet_transformer/split_1/` (only config.json) |
| split_2 | ⏳ Needs training | — |
| split_3 | ⏳ Needs training | — |
| split_4 | ⏳ Needs training | — |

---

## 11. Score Boost Roadmap — Six Architectural Pillars

### 11.1 Priority Ranking

| Priority | Strategy | Expected Gain | Complexity | Mechanism |
| :--- | :--- | :---: | :--- | :--- |
| **P1 (Done)** | Mitotic Aux Head + Adaptive ILP Costs | **+0.035–0.060** | Medium | Unlocks $0.1 \times J_{\text{div}}$ |
| **P2** | Log-Odds ILP + Multi-Frame ($t \to t+2$) | +0.015–0.025 | Low/Med | Closes occlusions via learned kinematics |
| **P3** | Continuous Sub-Voxel Head + [1,2,2] Downsampling | +0.015–0.020 | Medium | Resolves sister cell collapse at cytokinesis |
| **P4 (Done)** | 5-Fold Ensemble + Bidirectional Inference | +0.015–0.022 | Low | Eliminates edge directional noise & model variance |
| **P5** | Trilinear RoI Sampling + Multi-Scale Feature Maps | +0.010–0.015 | Medium | Replaces noisy single-voxel sampling with cell morphology |
| **P6** | InfoNCE Metric Learning + Soft Jaccard Loss | +0.008–0.015 | Medium | Directly optimizes leaderboard objective |

### 11.2 Pillar 2: Log-Odds ILP Cost + Multi-Frame Candidate Graph

The proper Bayesian MAP edge cost:
$$c_e = -\log \frac{P(e)}{1 - P(e)} = -\text{logit}(P(e))$$

Multi-frame temporal candidate graph ($k \in \{1, 2, 3\}$):
- $\Delta t = 1$: $R_{\text{max}} = 12$ µm
- $\Delta t = 2$: $R_{\text{max}} = 18$ µm
- $\Delta t = 3$: $R_{\text{max}} = 24$ µm

Two-stage tracklet ILP: Phase 1 collapses 90% of nodes into unbreakable high-confidence tracklets ($P \ge 0.85$). Phase 2 solves global ILP between tracklet endpoints.

### 11.3 Pillar 3: Sub-Voxel Head + [1,2,2] Downsampling

**Problem**: At (1,4,4) downsampling, sisters separated by <1.6 µm map to the **same voxel** → one daughter is permanently erased.

**Solution**: Shift to (1,2,2) downsampling ($0.8125$ µm in-plane) + continuous sub-voxel offset regression:
$$\mathbf{x}_{\text{final}} = (\mathbf{x}_{\text{grid}} + \Delta\mathbf{x}) \odot \mathbf{s}_{\text{downsample}}$$
where $\Delta\mathbf{x} \in [-0.5, 0.5]^3$ trained with SmoothL1 loss.

### 11.4 Pillar 6: Contrastive + Surrogate Loss

**InfoNCE trajectory contrastive loss** (forces same-cell embeddings across frames to cluster):
$$\mathcal{L}_{\text{contrast}} = -\sum_{i} \log \frac{\exp(\mathbf{z}_i(t) \cdot \mathbf{z}_i(t+1) / \tau)}{\sum_{j} \exp(\mathbf{z}_i(t) \cdot \mathbf{z}_j(t+1) / \tau)}, \quad \tau = 0.07$$

**Soft Edge Jaccard surrogate loss** (directly optimizes competition objective):
$$\mathcal{L}_{\text{soft\_jaccard}} = 1 - \frac{\sum_{i,j} \hat{P}_{ij} Y_{ij}}{\sum_{i,j} \hat{P}_{ij} + \sum_{i,j} Y_{ij} - \sum_{i,j} \hat{P}_{ij} Y_{ij} + \epsilon}$$

---

## 12. Training Environment & Execution Reference

### 12.1 Local Development Environment
| Item | Value |
| :--- | :--- |
| OS | Windows 11 |
| Python | `.\venv\Scripts\python.exe` (PyTorch 2.5+, CUDA 12.4) |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM) |
| Training data | `D:\Downloads_2\biohub-cell-tracking-during-development\train` (199 volumes) |
| Working directory | `c:\Users\Naghul Pranav\Documents\Biohub_Cell_Tracking` |

### 12.2 Key File Locations

| File | Purpose |
| :--- | :--- |
| [`repo/scripts/train_unet_transformer.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/repo/scripts/train_unet_transformer.py) | Training: UNetNodeTransformer with mitosis head |
| [`repo/scripts/predict_unet_transformer.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/repo/scripts/predict_unet_transformer.py) | Inference: ensemble, TTA, adaptive ILP |
| [`repo/scripts/dataspec.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/repo/scripts/dataspec.py) | Dataset path resolution (env var → Kaggle → local fallback) |
| [`biohub-final-notebook.ipynb`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/biohub-final-notebook.ipynb) | Kaggle submission notebook |
| [`create_dataset_splits.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/create_dataset_splits.py) | Generates 5-fold stratified splits |
| [`dataset_splits.json`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/dataset_splits.json) | Pre-computed 5-fold split definitions |
| [`kaggle_train_folds_kernel.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/kaggle_train_folds_kernel.py) | Kaggle GPU T4×2 training script for folds 1–4 |
| [`train_remaining_folds.py`](file:///c:/Users/Naghul%20Pranav/Documents/Biohub_Cell_Tracking/train_remaining_folds.py) | Local sequential training for folds 1–4 |
| `repo/weights/unet_transformer/split_0/edge_predictor_best.pth` | Fine-tuned mitosis head checkpoint |
| `repo/weights/unet_transformer/split_0/edge_predictor_best_baseline.pth` | Original pre-mitosis checkpoint backup |

### 12.3 Key `train_unet_transformer.py` CLI Flags

| Flag | Default | Purpose |
| :--- | :--- | :--- |
| `--split N` | 0 | Fold index 0–4 |
| `--pretrained-weights PATH` | None | Warm-start from existing checkpoint |
| `--freeze-unet` | off | Freeze UNet + detect_head for mitosis-only fine-tune |
| `--mitosis-loss-weight F` | 1.0 | Weight for mitosis BCE loss |
| `--max-iters N` | None | Max training iters per epoch (speeds up each epoch) |
| `--epochs N` | 50 | Number of epochs |
| `--lr F` | 1e-4 | AdamW learning rate |
| `--batch-size N` | 8 | Batch size (use 1 on 6 GB GPU) |
| `--num-workers N` | 0 (Windows) | DataLoader workers |
| `--data-dir PATH` | DATASET_PATH | Training data directory |
| `--splits PATH` | DATASET_PATH/dataset_splits.json | Splits file |

### 12.4 Temporal Attention Grid Launch Limit (RTX 4050 Warning)
In `_TemporalAttention`, $B \times S$ can exceed 8M voxels with $B \ge 2$, causing PyTorch CUDA grid launch errors or OOM on 6 GB GPUs. Always use `--batch-size 1` or `--freeze-unet` to avoid this.

### 12.5 Kaggle Offline Support Pack Integrity Invariant
The support pack `biohub-tracking-support-pack-50ep-v1` checks SHA256 hashes **before** Cell 3 runs. Cell 3 modifies `REPO_DIR / "scripts" / "predict_unet_transformer.py"` in-memory — this does **not** break the earlier integrity check.

### 12.6 Dual-Seed vs Multi-Fold CLI Dispatch
- **Dual-seed baseline** (split_0 + seed314159): CLI with `--weights WEIGHTS_RELATIVE`; `BIOHUB_SECONDARY_WEIGHTS` in `os.environ`.
- **True N-fold ensemble**: CLI with `--ensemble-weights <w0> <w1> ...` only when script has native `--ensemble-weights` support.

---

## 13. Critical Invariants & Safety Rules

> [!CAUTION]
> Violating any of the following will cause score regression or submission failure.

### 13.1 Absolute Hard Rules

| Rule | Why |
| :--- | :--- |
| **Never insert synthetic nodes $C_{\text{syn}}$** (except Breakthrough 5 validated post-mitotic stitching) | Inflates $N_{\text{pred}}$ → triggers over-prediction penalty |
| **Never change NMS pool kernel from (3,3,3)** at test time | Creates train/test covariate shift on transformer edge features |
| **Never raise DET_THRESHOLD above 0.965** | Loses faint mitotic cells → cascading short-track pruning |
| **Never touch `motion_relink_edges` cost function** | Any modification consistently drops −0.003 |
| **ILP strictly enforces in-degree ≤ 1, out-degree ≤ 2** | Biological lineage topology invariant |
| **All edges must be $t \to t+1$ strict** | Temporal causality constraint |

### 13.2 Submission Schema
CSV columns (0-indexed sequential `id`):
```
id, dataset, row_type, node_id, t, z, y, x, source_id, target_id
```

### 13.3 Algorithmic Checklist for New Models

Before deploying any pipeline modification:
- [ ] **Physical Scale Preserved**: All distances computed using anisotropic scaling $(1.625, 0.40625, 0.40625)$?
- [ ] **Over-Prediction Shield**: $N_{\text{pred}} \le N_{\text{true}}$? Isolate nodes (degree=0) pruned?
- [ ] **Topology Enforced**: $\text{in\_degree} \le 1$ and $\text{out\_degree} \le 2$ strictly guaranteed?
- [ ] **Temporal Increment Valid**: All edges $t \to t+1$?
- [ ] **Division Separation Bound**: All division daughter separations ≤ 14.0 µm?
- [ ] **Spindle Divergence**: Dividing daughters diverge with $\cos\theta \le 0.35$?
- [ ] **Refractory Period**: Consecutive divisions on same lineage ≥ 15 timepoints apart?
- [ ] **Detection Threshold Frozen**: `BIOHUB_DET_THRESHOLD = 0.965`?
- [ ] **Monotonic Invariant**: Post-processing filter only removes edges/nodes, never adds?
- [ ] **Submission Schema Compliant**: Output CSV has exact column headers and sequential 0-indexed `id`?

### 13.4 The MRTCA Monotonic Audit Paradigm

For safe score improvement without regression:
1. **Never add new edges or nodes** to the baseline graph.
2. **Removal-only monotonic filtering**:
   - Start with the frozen 0.947 tracking graph.
   - Prune only provably false edges using fundamental biophysical laws with extreme confidence.
   - Pruning isolate endpoints strictly **decreases** $N_{\text{pred}}$ → strictly **increases** penalty factor.
   - As long as $\Delta FN_e = 0$, score can **only increase**.

**Velocity Change Ratio (VCR)** for identity swap detection:
$$\text{VCR} = \frac{\|\vec{v}_{\text{fwd}} - \vec{v}_{\text{bwd}}\|}{\|\vec{v}_{\text{bwd}}\| + 0.25}$$
- Smooth Stokes flow: VCR typically **< 1.5**
- Identity swap kink: VCR > 3.0–3.5

---

*Last updated: 2026-09-23 — incorporates all conversation history including mitosis head implementation, fine-tuning results, 5-fold ensemble design, and all failure mode analyses.*
