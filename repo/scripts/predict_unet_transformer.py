#!/usr/bin/env python
"""Run UNet + transformer edge prediction on datasets and export to .geff.

Usage:
    uv run scripts/predict_unet_transformer.py --split 0
"""

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
import zarr
from tqdm import tqdm

import tracksdata as td

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from biohub_tracking.io import open_dataset, save_graph

# Import model and helpers from companion training script.
sys.path.insert(0, str(Path(__file__).parent))
from train_unet_transformer import (
    DEFAULT_METHOD,
    UNetNodeTransformer,
    extract_pos_features,
    _POS_EMBED_DIM,
)
from biohub_tracking.models import TemporalUNet3D

from dataspec import USERNAME, INTERACTIVE, WEIGHTS_PATH
from evaluate import evaluate_run
from biohub_tracking.metrics import summarise


# =============================================================================
# Prediction config
# =============================================================================

@dataclass
class PredictConfig:
    """All hyperparameters that can affect prediction quality / score.

    Detection
    ---------
    det_threshold : float
        Minimum sigmoid probability for a local-max peak to be kept.

    Edge filtering
    --------------
    edge_activation : str
        Activation applied to raw edge logits: ``"sigmoid"`` (independent
        per-edge scores) or ``"softmax"`` (row-normalised over t+1 nodes).
    threshold : float
        Minimum edge probability to consider a link at all.
    max_parents_per_node : int
        Maximum number of incoming edges per node (typically 1).
    max_children_per_node : int
        Maximum number of outgoing edges per node (1 = no divisions, 2 = divisions allowed).
    """
    # Detection
    det_threshold: float = 0.5
    det_tta: bool = True  # flip-xy TTA for detection logits
    pool_kernel_um: float = 3.0  # max-pool kernel size in µm for detection peak extraction
    # Edge filtering
    edge_activation: str = "softmax"  # "sigmoid" or "softmax"
    threshold: float = 0.5

    # ILP post-processing
    use_ilp: bool = False
    ilp_edge_weight: float = -1.0
    ilp_appearance_weight: float = 0.1
    ilp_disappearance_weight: float = 0.1
    ilp_division_weight: float = 1.0
    # Neural mitosis head adaptive division cost
    use_mitosis_head: bool = True
    ilp_division_lambda: float = 0.8

    max_parents_per_node: int | None = None
    max_children_per_node: int | None = None

    def __post_init__(self) -> None:
        # When ILP is enabled it handles parent/children constraints itself,
        # so greedy limits are left unconstrained (None).  When ILP is
        # disabled, default to 1/1 to avoid unconstrained edge assignment.
        if not self.use_ilp:
            if self.max_parents_per_node is None:
                self.max_parents_per_node = 1
            if self.max_children_per_node is None:
                self.max_children_per_node = 2



# =============================================================================
# Helpers
# =============================================================================


@contextlib.contextmanager
def suppress_output():
    """Context manager to suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


# =============================================================================
# Graph building
# =============================================================================

def build_graph(
    coords: np.ndarray,
    edges: list[tuple[int, int, float, float]],
    division_costs: np.ndarray | None = None,
) -> td.graph.InMemoryGraph:
    """Build a tracksdata graph from detection coords, predicted edges, and optional division costs.

    Avoids ``add_node_attr_key`` to sidestep a tracksdata/Polars compatibility
    issue where the float default value is mistakenly used as a dtype.
    Probabilities are passed as-is (softmax output, already in [0, 1]).
    """
    graph = td.graph.InMemoryGraph()

    for key in ["z", "y", "x"]:
        graph.add_node_attr_key(key, pl.Float64, -999999.0)

    if division_costs is not None:
        graph.add_node_attr_key("division_cost", pl.Float64, 10.0)

    node_dicts = []
    for idx, (t, z, y, x) in enumerate(coords):
        d = {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        if division_costs is not None:
            d["division_cost"] = float(division_costs[idx])
        node_dicts.append(d)

    node_ids = graph.bulk_add_nodes(node_dicts)

    if edges:
        graph.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
        graph.add_edge_attr_key("edge_dist", pl.Float64, 0.0)
        graph.bulk_add_edges([
            {
                "source_id": node_ids[src],
                "target_id": node_ids[tgt],
                "edge_prob": prob,
                "edge_dist": dist,
            }
            for src, tgt, prob, dist in edges
        ])

    return graph


# =============================================================================
# Model loading
# =============================================================================

_DEFAULT_CONFIG = {
    "unet_out_channels": 32,
    "unet_layers": [32, 64, 128],
    "downsample": [1, 4, 4],
    "window_size": 2,
}


def load_model(
    weights_path: Path, device: torch.device,
) -> tuple[UNetNodeTransformer, int, tuple[int, ...]]:
    """Reconstruct UNetNodeTransformer from saved config + weights.

    Reads ``config.json`` from the same directory as the weights file.
    Falls back to ``_DEFAULT_CONFIG`` if the file is missing.

    Returns ``(model, window_size, downsample)``.
    """
    config_path = weights_path.parent / "config.json"
    if config_path.exists():
        config = {**_DEFAULT_CONFIG, **json.loads(config_path.read_text())}
    else:
        print(f"Warning: config.json not found at {config_path}, using defaults.", flush=True)
        config = _DEFAULT_CONFIG

    # Support legacy configs that used "downsample_factor" (scalar).
    if "downsample_factor" in config and "downsample" not in config:
        df = config["downsample_factor"]
        config["downsample"] = [df, df, df]

    downsample = tuple(config["downsample"])

    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=config["unet_out_channels"],
        layers=config["unet_layers"],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=config["unet_out_channels"],
        pos_feat_dim=4 * _POS_EMBED_DIM,
    )
    state = torch.load(weights_path, map_location=device, weights_only=True)
    state = {k.replace("unet.module.", "unet.").replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    has_mitosis = any("mitosis_head" in k for k in state.keys())
    model.has_mitosis_head = has_mitosis
    if has_mitosis:
        print(f"  Loaded model with neural mitosis head from {weights_path.name}", flush=True)
    model.to(device)
    model.eval()
    return model, config["window_size"], downsample


# =============================================================================
# Per-frame loading
# =============================================================================

def _load_frame(
    zarr_arr,
    t: int,
    target_shape: list[int],
    downsample: tuple[int, ...] = (1, 1, 1),
) -> torch.Tensor:
    """Load one frame from zarr with strided spatial downsample (no normalisation)."""
    dz, dy, dx = downsample
    raw = zarr_arr[t, ::dz, ::dy, ::dx].astype(np.float32)
    frame = torch.from_numpy(raw)
    if list(frame.shape) != target_shape:
        frame = F.interpolate(
            frame[None, None], size=target_shape,
            mode="trilinear", align_corners=False,
        )[0, 0]
    return frame


# =============================================================================
# Inference
# =============================================================================

def pool_kernel_from_um(
    um: float,
    voxel_size: tuple[float, ...],
) -> tuple[int, ...]:
    """Convert a physical suppression distance (microns) to a per-axis voxel kernel.

    Each axis gets ``round(um / voxel_size_axis)`` voxels, forced to odd
    (for symmetric padding) and at least 1.

    Parameters
    ----------
    um : float
        Desired suppression distance in microns.
    voxel_size : tuple[float, ...]
        Per-axis voxel sizes in microns, e.g. ``(1.625, 0.40625, 0.40625)``.
    """
    kernel = []
    for s in voxel_size:
        k = max(1, round(um / s))
        if k % 2 == 0:
            k += 1
        kernel.append(k)
    return tuple(kernel)


def _detect_cells_pooled(
    det_logits: torch.Tensor,
    t: int,
    det_threshold: float = 0.5,
    pool_kernel: tuple[int, ...] = (3, 3, 3),
) -> np.ndarray:
    """Extract cell coordinates via max-pool local-max (same as training).

    Coordinates are returned in the downsampled grid.  The caller is
    responsible for scaling back to original resolution if needed.

    Parameters
    ----------
    det_logits : torch.Tensor
        (1, Z, Y, X) raw logits.
    t : int
        Time index to prepend as the first column.
    det_threshold : float
        Minimum sigmoid probability for a peak to be considered (default 0.5).
    pool_kernel : tuple[int, ...]
        Per-axis kernel size for local-max pooling,
        e.g. ``(3, 11, 11)`` for anisotropic data.

    Returns
    -------
    np.ndarray
        (N, 4) int16 array with columns [t, z, y, x] in downsampled space.
    """
    logits = det_logits.unsqueeze(0)  # (1, 1, Z, Y, X)
    pad = tuple(k // 2 for k in pool_kernel)
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
    peak_idx = torch.nonzero(is_peak[0, 0])  # (N, 3)

    if peak_idx.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)

    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)


def _encode_model_tta(
    m: UNetNodeTransformer,
    imgs: torch.Tensor,
    det_tta: bool = True,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Run UNet encode with 8-view planar D4 TTA."""
    unet_out, det_logits = m.encode(imgs)
    if not det_tta:
        return unet_out, det_logits

    W = imgs.shape[1]
    _nv = 1
    # 3 axis flips in XY
    for dims in [(-1,), (-2,), (-2, -1)]:
        imgs_flip = imgs.flip(dims)
        _, det_flip = m.encode(imgs_flip)
        for f in range(W):
            det_logits[f] = det_logits[f] + det_flip[f].flip(dims)
        del imgs_flip, det_flip
        _nv += 1
    # 90 and 270 degree planar rotations
    for _k in (1, 3):
        imgs_rot = torch.rot90(imgs, _k, dims=(-2, -1))
        _, det_rot = m.encode(imgs_rot)
        for f in range(W):
            det_logits[f] = det_logits[f] + torch.rot90(det_rot[f], -_k, dims=(-2, -1))
        del imgs_rot, det_rot
        _nv += 1
    # Transpose and rot90-transpose
    imgs_t = imgs.transpose(-1, -2)
    _, det_t = m.encode(imgs_t)
    for f in range(W):
        det_logits[f] = det_logits[f] + det_t[f].transpose(-1, -2)
    del imgs_t, det_t
    _nv += 1
    imgs_at = torch.rot90(imgs, 1, dims=(-2, -1)).transpose(-1, -2)
    _, det_at = m.encode(imgs_at)
    for f in range(W):
        det_logits[f] = det_logits[f] + torch.rot90(det_at[f].transpose(-1, -2), -1, dims=(-2, -1))
    del imgs_at, det_at
    _nv += 1

    for f in range(W):
        det_logits[f] = det_logits[f] / _nv

    return unet_out, det_logits


@torch.no_grad()
def predict_video(
    model: UNetNodeTransformer | list[UNetNodeTransformer],
    ds_path: Path,
    device: torch.device,
    cfg: PredictConfig,
    window_size: int = 2,
    max_frames: int | None = None,
    unet_batch_size: int = 4,
    downsample: tuple[int, ...] = (1, 4, 4),
) -> tuple[np.ndarray, list[tuple[int, int, float, float]], np.ndarray | None]:
    """Run inference on a single video using sliding windows of W frames.

    Supports single model or multi-model ensemble (averaging detection
    and edge logits across all models).

    Returns
    -------
    coords : np.ndarray
        Shape (N, 4) — columns [t, z, y, x] in original resolution.
    edges : list of (src_idx, tgt_idx, prob, distance) tuples
    division_costs : np.ndarray or None
        Per-node adaptive division costs computed by neural mitosis head.
    """
    models: list[UNetNodeTransformer] = (
        list(model) if isinstance(model, (list, tuple)) else [model]
    )
    K = len(models)

    ds = open_dataset(ds_path, normalize=False, load_image=False, downsample=downsample)
    if "0.001" not in ds.quantiles or "0.999" not in ds.quantiles:
        raise ValueError(f"Zarr attrs missing image_statistics.quantiles for {ds_path}")
    zarr_arr = zarr.open_group(str(ds.zarr_path), mode="r")["0"]
    q_low = float(ds.quantiles["0.001"])
    q_high = float(ds.quantiles["0.999"])

    T = ds.image_shape[0] if max_frames is None else min(ds.image_shape[0], max_frames)
    image_shape = (T,) + ds.image_shape[1:]
    target_shape = list(image_shape[1:])

    ds_arr = np.array(downsample, dtype=np.float32)  # for coord rescaling at the end
    ds_arr_t = torch.from_numpy(ds_arr).to(device)   # for predict_edges (original-space coords)
    pos_feat_dim = 4 * _POS_EMBED_DIM
    W = window_size
    voxel_size = tuple(s * d for s, d in zip(ds.scale, downsample))
    pool_k = pool_kernel_from_um(cfg.pool_kernel_um, voxel_size)

    # Running node registry — each entry records the frame-t detections.
    # coord_offset[t] = (start, end) half-open range into the stacked array.
    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    coord_offset: dict[int, tuple[int, int]] = {}
    global_node_count: int = 0
    all_edges: list[tuple[int, int, float, float]] = []
    node_division_costs: dict[int, float] = {}

    # Sliding windows with stride W-1 cover every consecutive pair exactly once.
    stride = max(W - 1, 1)
    window_starts = list(range(0, T - W + 1, stride))
    # Ensure the very last pair (T-2 → T-1) is covered.
    if not window_starts or window_starts[-1] + W < T:
        last = max(T - W, 0)
        if not window_starts or last != window_starts[-1]:
            window_starts.append(last)

    for ws in tqdm(
        window_starts,
        desc="  windows",
        leave=False,
        disable=not INTERACTIVE,
    ):
        frame_indices = list(range(ws, ws + W))

        # --- UNet encode (single window, batch_size=1) ---
        imgs = torch.stack([
            _load_frame(zarr_arr, t, target_shape, downsample)
            for t in frame_indices
        ])  # (W, *spatial)
        # Quantile normalisation (0.1%–99.9%) to match training pipeline.
        imgs = ((imgs - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        imgs = imgs.unsqueeze(0).to(device)   # (1, W, *spatial)

        # --- UNet encode across ensemble models ---
        all_unet_outs: list[torch.Tensor] = []
        ensemble_det_logits: list[torch.Tensor] | None = None

        for m in models:
            u_out, d_logits = _encode_model_tta(m, imgs, cfg.det_tta)
            all_unet_outs.append(u_out)
            if ensemble_det_logits is None:
                ensemble_det_logits = [d.clone() for d in d_logits]
            else:
                for f in range(W):
                    ensemble_det_logits[f] = ensemble_det_logits[f] + d_logits[f]
            del d_logits

        del imgs
        if K > 1 and ensemble_det_logits is not None:
            for f in range(W):
                ensemble_det_logits[f] = ensemble_det_logits[f] / K

        # --- Detect cells in each frame (dedup across windows) ---
        for f_idx, t in enumerate(frame_indices):
            if t not in seen_frames:
                arr = _detect_cells_pooled(
                    ensemble_det_logits[f_idx][0], t, cfg.det_threshold, pool_k,
                )
                coord_offset[t] = (global_node_count, global_node_count + len(arr))
                global_node_count += len(arr)
                coord_lists.append(arr)
                seen_frames.add(t)

        coords_so_far = (
            np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
        )

        # --- Edge prediction for each consecutive pair in the window ---
        for f_idx in range(W - 1):
            t_src, t_tgt = frame_indices[f_idx], frame_indices[f_idx + 1]
            if (t_src, t_tgt) in seen_pairs:
                continue
            seen_pairs.add((t_src, t_tgt))

            if t_src not in coord_offset or t_tgt not in coord_offset:
                continue
            s_src, e_src = coord_offset[t_src]
            s_tgt, e_tgt = coord_offset[t_tgt]
            if e_src == s_src or e_tgt == s_tgt:
                continue

            c_src = coords_so_far[s_src:e_src]
            c_tgt = coords_so_far[s_tgt:e_tgt]
            n_src, n_tgt = len(c_src), len(c_tgt)
            idx_src = np.arange(s_src, e_src, dtype=np.int64)
            idx_tgt = np.arange(s_tgt, e_tgt, dtype=np.int64)

            # Build tensors (batch_size=1).
            p_coords_src = torch.from_numpy(c_src[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            p_coords_tgt = torch.from_numpy(c_tgt[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            # Use window-relative time (f_idx, f_idx+1) normalised by W, not absolute frame index.
            window_shape = (W,) + image_shape[1:]
            c_src_rel = c_src.copy()
            c_src_rel[:, 0] = f_idx
            c_tgt_rel = c_tgt.copy()
            c_tgt_rel[:, 0] = f_idx + 1
            p_pos_src = torch.from_numpy(extract_pos_features(c_src_rel, window_shape)).unsqueeze(0).to(device)
            p_pos_tgt = torch.from_numpy(extract_pos_features(c_tgt_rel, window_shape)).unsqueeze(0).to(device)
            p_mask_src = torch.ones(1, n_src, dtype=torch.bool, device=device)
            p_mask_tgt = torch.ones(1, n_tgt, dtype=torch.bool, device=device)

            accum_edge_logits: torch.Tensor | None = None
            accum_mitosis_logits: torch.Tensor | None = None
            num_mitosis_models: int = 0
            bidir_weight = float(os.environ.get("BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT", "0.15"))

            for k, m in enumerate(models):
                u_out = all_unet_outs[k]
                unet_feat_src = m._index_features(
                    u_out[:, f_idx], p_coords_src, p_mask_src,
                )
                unet_feat_tgt = m._index_features(
                    u_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
                )

                if getattr(m, "has_mitosis_head", False) and cfg.use_mitosis_head:
                    m_logits = m.predict_mitosis(unet_feat_src, p_pos_src)
                    accum_mitosis_logits = (
                        m_logits if accum_mitosis_logits is None else accum_mitosis_logits + m_logits
                    )
                    num_mitosis_models += 1

                edge_logits_k = m.predict_edges(
                    unet_feat_src, unet_feat_tgt,
                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    p_pos_src, p_pos_tgt,
                    p_mask_src, p_mask_tgt,
                )  # (1, n_src, n_tgt)

                if bidir_weight > 0.0:
                    reverse_logits_native = m.predict_edges(
                        unet_feat_tgt, unet_feat_src,
                        p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,
                        p_pos_tgt, p_pos_src,
                        p_mask_tgt, p_mask_src,
                    )
                    reverse_logits_pair = reverse_logits_native.transpose(1, 2)
                    forward_center = edge_logits_k.mean(dim=1, keepdim=True)
                    forward_scale = edge_logits_k.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    reverse_center = reverse_logits_pair.mean(dim=1, keepdim=True)
                    reverse_scale = reverse_logits_pair.float().std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    reverse_scale_ratio = (forward_scale / reverse_scale).clamp(0.5, 2.0).to(reverse_logits_pair.dtype)
                    reverse_aligned = (reverse_logits_pair - reverse_center) * reverse_scale_ratio + forward_center
                    forward_prob = torch.softmax(edge_logits_k.float(), dim=1).clamp_min(1e-8)
                    reverse_prob = torch.softmax(reverse_aligned.float(), dim=1).clamp_min(1e-8)
                    harmonic_prob = 1.0 / (
                        (1.0 - bidir_weight) / forward_prob + bidir_weight / reverse_prob
                    )
                    harmonic_prob = harmonic_prob / harmonic_prob.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    harmonic_logits = torch.log(harmonic_prob.clamp_min(1e-8))
                    harmonic_center = harmonic_logits.mean(dim=1, keepdim=True)
                    harmonic_scale = harmonic_logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
                    harmonic_scale_ratio = (forward_scale / harmonic_scale).clamp(0.5, 2.0)
                    edge_logits_k = ((harmonic_logits - harmonic_center) * harmonic_scale_ratio + forward_center).to(edge_logits_k.dtype)

                if accum_edge_logits is None:
                    accum_edge_logits = edge_logits_k
                else:
                    accum_edge_logits = accum_edge_logits + edge_logits_k

            if num_mitosis_models > 0 and accum_mitosis_logits is not None:
                avg_m_logits = (accum_mitosis_logits / num_mitosis_models)[0]  # (n_src,)
                clamped_logits = avg_m_logits.clamp(-4.0, 4.0).cpu().numpy()
                div_costs = cfg.ilp_division_weight - cfg.ilp_division_lambda * clamped_logits
                for i in range(n_src):
                    gi = int(idx_src[i])
                    node_division_costs[gi] = float(div_costs[i])

            ensemble_edge_logits = accum_edge_logits / K
            raw = ensemble_edge_logits[0]
            if cfg.edge_activation == "softmax":
                probs = torch.softmax(raw, dim=0).cpu().numpy()
            else:
                probs = torch.sigmoid(raw).cpu().numpy()

            candidates = sorted(
                [
                    (probs[i, j], i, j)
                    for i in range(n_src)
                    for j in range(n_tgt)
                    if probs[i, j] > cfg.threshold
                ],
                reverse=True,
            )

            children_count: dict[int, int] = {}
            parents_count: dict[int, int] = {}

            for prob, i, j in candidates:
                n_ch = children_count.get(i, 0)
                n_pa = parents_count.get(j, 0)
                if cfg.max_children_per_node is not None and n_ch >= cfg.max_children_per_node:
                    continue
                if cfg.max_parents_per_node is not None and n_pa >= cfg.max_parents_per_node:
                    continue

                gi, gj = int(idx_src[i]), int(idx_tgt[j])
                dist = float(np.linalg.norm(
                    coords_so_far[gi, 1:].astype(np.float32)
                    - coords_so_far[gj, 1:].astype(np.float32)
                ))
                all_edges.append((gi, gj, float(prob), dist))
                children_count[i] = n_ch + 1
                parents_count[j] = n_pa + 1

        del all_unet_outs

    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    total_nodes = len(coords)
    division_costs = None
    if node_division_costs:
        division_costs = np.full(total_nodes, cfg.ilp_division_weight, dtype=np.float32)
        for gi, cost in node_division_costs.items():
            if gi < total_nodes:
                division_costs[gi] = cost

    # Scale spatial coords back to original resolution.
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)
    return coords, all_edges, division_costs


# =============================================================================
# Prediction loop
# =============================================================================

def predict(
    data_dir: Path,
    fold: int,
    splits_file: Path,
    weights_path: Path | None,
    cfg: PredictConfig,
    method: str = DEFAULT_METHOD,
    debug_video: Path | None = None,
    unet_batch_size: int = 4,
    video_slice: slice | None = None,
    evaluate: bool = False,
    ensemble_weights: list[Path] | None = None,
) -> None:
    """Run inference on the test split and save predictions as .geff files."""
    if debug_video is not None:
        test_names = [debug_video.name]
        data_dir = debug_video.parent
    else:
        folds = json.loads(splits_file.read_text())
        test_names = folds[fold]["test"]
        if video_slice is not None:
            test_names = test_names[video_slice]

    from dataspec import PREDICTIONS_PATH
    output_dir = PREDICTIONS_PATH / USERNAME / method / f"split_{fold}"
    if output_dir.exists():
        import shutil
        for old in output_dir.glob("*.geff"):
            if old.is_dir():
                shutil.rmtree(old)
            else:
                old.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_paths = ensemble_weights if ensemble_weights else ([weights_path] if weights_path else [])
    if not weights_paths:
        raise ValueError("Either weights_path or ensemble_weights must be provided.")

    models: list[UNetNodeTransformer] = []
    window_size = None
    downsample = None
    for wp in weights_paths:
        m, ws, ds = load_model(wp, device)
        models.append(m)
        if window_size is None:
            window_size = ws
            downsample = ds
        else:
            if ws != window_size or ds != downsample:
                raise ValueError(
                    f"Incompatible model config for {wp}: "
                    f"expected window_size={window_size}, downsample={downsample}, "
                    f"got window_size={ws}, downsample={ds}"
                )

    print(
        f"Fold {fold}: {len(test_names)} datasets | "
        f"{len(models)} model(s) loaded | device={device} | "
        f"window_size={window_size} | pool_kernel_um={cfg.pool_kernel_um}",
        flush=True,
    )
    for i, wp in enumerate(weights_paths):
        print(f"  Model {i}: {wp}", flush=True)

    for name in tqdm(test_names, desc="Predicting", disable=not INTERACTIVE):
        ds_path = data_dir / name
        coords, edges, division_costs = predict_video(
            models, ds_path, device,
            cfg=cfg,
            window_size=window_size,
            unet_batch_size=unet_batch_size,
            downsample=downsample,
        )
        graph = build_graph(coords, edges, division_costs)
        if cfg.use_ilp and graph.num_edges() > 0:
            div_weight = td.NodeAttr("division_cost") if division_costs is not None else cfg.ilp_division_weight
            solver = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=div_weight,
            )
            with suppress_output():
                graph = solver.solve(graph)
        save_graph(graph, output_dir / f"{name}.geff")

    print(f"Saved {len(test_names)} predictions to {output_dir}", flush=True)

    if evaluate:
        run = {
            "username": USERNAME,
            "method": method,
            "split": f"split_{fold}",
            "dir": output_dir,
            "geffs": sorted(output_dir.glob("*.geff")),
        }
        results = evaluate_run(run)
        s = summarise(results)
        print(
            f"Evaluation ({len(results)} videos): "
            f"score={s['score']:.4f}  "
            f"edge_jaccard={s['edge_jaccard']:.4f}  "
            f"adj_edge_jaccard={s['adj_edge_jaccard']:.4f} (n_adj={s['n_adj']})  "
            f"division_jaccard={s['division_jaccard']:.4f} "
            f"(TP={s['division_tp']} FP={s['division_fp']} FN={s['division_fn']})  "
            f"node_recall={s['node_recall']:.4f}  (n={s['n']})",
            flush=True,
        )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run UNet + transformer edge prediction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--method", type=str, default=DEFAULT_METHOD)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Default: DATASET_PATH")
    parser.add_argument("--splits", type=str, default=None,
                        help="Default: DATASET_PATH/dataset_splits.json")
    parser.add_argument("--split", type=str, default="0",
                        help="Split index (0-4) or 'all'.")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to weights file. "
                             "Default: weights/{method}/split_{split}/edge_predictor_best.pth")
    parser.add_argument("--ensemble-weights", type=str, nargs="+", default=None,
                        help="List of checkpoint paths for multi-model ensemble")
    parser.add_argument("--debug-video", type=str, default=None,
                        help="Path to a single dataset. Ignores fold/splits.")
    parser.add_argument("--slice", type=str, default=None,
                        help="Python slice of the test list, e.g. ':1' for first video, "
                             "'2:5' for videos 2-4.")
    parser.add_argument("--unet-batch-size", type=int, default=4,
                        help="Number of frame pairs per UNet forward pass (default: 4).")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation against GT after saving predictions.")
    parser.add_argument("--det-threshold", type=float, default=0.99,
                        help="Min sigmoid probability for a detection peak to be kept. "
                             "Default 0.99: the detector is poorly calibrated because the "
                             "ground truth is sparse (only some cells annotated), so a high "
                             "threshold keeps precision up. Sweep it for your model.")
    parser.add_argument("--use-ilp", action="store_true",
                        help="Post-process the predicted graph with the tracksdata ILP "
                             "solver (global, flow-consistent linking) instead of greedy "
                             "assignment. Needs pyscipopt; produces cleaner tracks.")
    parser.add_argument("--ilp-edge-weight", type=float, default=-1.0,
                        help="ILP: weight on edge_prob (default -1.0).")
    parser.add_argument("--ilp-appearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track appearing (default 0.1).")
    parser.add_argument("--ilp-disappearance-weight", type=float, default=0.1,
                        help="ILP: cost of a track disappearing (default 0.1).")
    parser.add_argument("--ilp-division-weight", type=float, default=1.0,
                        help="ILP: cost of a division; lower to allow more splits (default 1.0).")
    parser.add_argument("--use-mitosis-head", dest="use_mitosis_head", action="store_true", default=True,
                        help="Enable neural mitosis head for adaptive ILP division cost (default: on).")
    parser.add_argument("--no-mitosis-head", dest="use_mitosis_head", action="store_false",
                        help="Disable neural mitosis head.")
    parser.add_argument("--ilp-division-lambda", type=float, default=0.8,
                        help="Multiplier for mitosis log-odds in adaptive ILP division cost (default: 0.8).")

    args = parser.parse_args()

    from dataspec import DATASET_PATH
    data_dir = Path(args.data_dir) if args.data_dir else Path(DATASET_PATH)
    splits_file = Path(args.splits) if args.splits else data_dir / "dataset_splits.json"
    debug_video = Path(args.debug_video) if args.debug_video else None
    video_slice = (
        slice(*[int(x) if x else None for x in args.slice.split(":")])
        if args.slice else None
    )
    cfg = PredictConfig(
        det_threshold=args.det_threshold,
        use_ilp=args.use_ilp,
        ilp_edge_weight=args.ilp_edge_weight,
        ilp_appearance_weight=args.ilp_appearance_weight,
        ilp_disappearance_weight=args.ilp_disappearance_weight,
        ilp_division_weight=args.ilp_division_weight,
        use_mitosis_head=args.use_mitosis_head,
        ilp_division_lambda=args.ilp_division_lambda,
    )

    ensemble_weights = (
        [Path(p) for p in args.ensemble_weights] if args.ensemble_weights else None
    )

    folds = range(5) if args.split == "all" else [int(args.split)]

    for fold in folds:
        weights_path = (
            Path(args.weights) if args.weights
            else WEIGHTS_PATH / args.method / f"split_{fold}" / "edge_predictor_best.pth"
        )
        predict(
            data_dir=data_dir,
            fold=fold,
            splits_file=splits_file,
            weights_path=weights_path,
            cfg=cfg,
            method=args.method,
            debug_video=debug_video,
            unet_batch_size=args.unet_batch_size,
            video_slice=video_slice,
            evaluate=args.evaluate,
            ensemble_weights=ensemble_weights,
        )


if __name__ == "__main__":
    main()
