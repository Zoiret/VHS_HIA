from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from validate_centerhead import (
    _connected_components,
    _geometry_topo_u8,
    _watershed,
    compute_instance_metrics_from_masks,
)


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
DEFAULT_MANIFEST_PATH = REPO_ROOT / "training" / "manifests" / "center_full_val_manifest.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "leaflet_oracle_count_geometric_split_audit"
DEFAULT_SEMANTIC_CONFIG = (
    REPO_ROOT / "training" / "configs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep.yaml"
)
DEFAULT_SEMANTIC_CHECKPOINT = (
    REPO_ROOT
    / "training"
    / "runs"
    / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep"
    / "best_mean_fg.pth"
)
DEFAULT_INSTANCE_ROOT = REPO_ROOT / "datasets" / "converted_leaflet_instances"
DEFAULT_SEMANTIC_ROOT = REPO_ROOT / "datasets" / "converted_full_multiclass"
LEAFLET_CLASS_ID = 1
RING_CLASS_ID = 2


@dataclass(frozen=True)
class SeedMethodSpec:
    key: str
    method_family: str
    variant: str
    params: dict[str, Any]


SEED_METHOD_SPECS: list[SeedMethodSpec] = [
    SeedMethodSpec(
        key="baseline_connected_components",
        method_family="connected_components",
        variant="raw_cc",
        params={},
    ),
    SeedMethodSpec(
        key="global_distance_maxima_r09",
        method_family="global_distance_maxima",
        variant="radius9_rel0p08",
        params={"min_distance_px": 9, "suppression_scale": 1.00, "peak_floor_rel": 0.08},
    ),
    SeedMethodSpec(
        key="global_distance_maxima_r15",
        method_family="global_distance_maxima",
        variant="radius15_rel0p12",
        params={"min_distance_px": 15, "suppression_scale": 1.10, "peak_floor_rel": 0.12},
    ),
    SeedMethodSpec(
        key="prominent_maxima_rel20",
        method_family="prominent_maxima",
        variant="prominence0p20_radius12",
        params={"min_distance_px": 12, "suppression_scale": 1.00, "peak_floor_rel": 0.20},
    ),
    SeedMethodSpec(
        key="prominent_maxima_rel30",
        method_family="prominent_maxima",
        variant="prominence0p30_radius15",
        params={"min_distance_px": 15, "suppression_scale": 1.10, "peak_floor_rel": 0.30},
    ),
    SeedMethodSpec(
        key="component_aware_maxima",
        method_family="component_aware_maxima",
        variant="per_component_greedy",
        params={"min_distance_px": 12, "suppression_scale": 1.00, "peak_floor_rel": 0.08},
    ),
]


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("PyYAML is required. Install training dependencies first.") from e
    obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected dict config root in {path}")
    return obj


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"Expected dict JSON in {path}")
    return obj


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError(f"Invalid JSONL row in {path}")
        rows.append(obj)
    return rows


def _resolve_path(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


def _load_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _load_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _center_crop_like_validation(image: np.ndarray, target_h: int, target_w: int, *, is_mask: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h < target_h or w < target_w:
        new_h = max(h, target_h)
        new_w = max(w, target_w)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        h, w = image.shape[:2]
    y0 = (h - target_h) // 2 if h > target_h else 0
    x0 = (w - target_w) // 2 if w > target_w else 0
    return np.ascontiguousarray(image[y0 : y0 + target_h, x0 : x0 + target_w])


def _count_positive_labels(labels_u8: np.ndarray) -> int:
    return int(len([int(v) for v in np.unique(labels_u8) if int(v) > 0]))


def extract_gt_count_from_metadata(metadata: dict[str, Any]) -> int:
    if "leaflet_selected_instances" in metadata:
        return int(metadata["leaflet_selected_instances"])
    if "instance_count" in metadata:
        return int(metadata["instance_count"])
    stats = metadata.get("instance_mask_stats", [])
    if isinstance(stats, list) and stats:
        return int(sum(1 for item in stats if isinstance(item, dict) and bool(item.get("present", False))))
    src_ids = metadata.get("source_instance_ids", [])
    if isinstance(src_ids, list) and src_ids:
        return int(len(src_ids))
    selected = metadata.get("leaflet_selected", [])
    if isinstance(selected, list) and selected:
        return int(len(selected))
    instances = metadata.get("instances", [])
    if isinstance(instances, list):
        return int(len(instances))
    raise ValueError("Metadata does not contain instance count information")


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _build_semantic_model(cfg: dict[str, Any]):
    import segmentation_models_pytorch as smp
    import torch

    model_cfg = cfg.get("model") or {}
    encoder = model_cfg.get("encoder") or model_cfg.get("encoder_name")
    if not encoder:
        raise SystemExit("Config is missing model.encoder_name")
    model = smp.UnetPlusPlus(
        encoder_name=str(encoder),
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=int(model_cfg.get("in_channels", 3)),
        classes=int(model_cfg.get("classes", 3)),
    )
    return model


def _load_semantic_checkpoint(model, checkpoint_path: Path, device):
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()


def _predict_semantic_mask(
    model,
    image_rgb_u8: np.ndarray,
    *,
    target_hw: tuple[int, int],
    device,
    use_amp: bool,
) -> np.ndarray:
    import torch

    crop = _center_crop_like_validation(image_rgb_u8, target_hw[0], target_hw[1], is_mask=False)
    img_f32 = _simple_preprocess_uint8_rgb(crop)
    image_t = torch.from_numpy(img_f32.transpose(2, 0, 1)[None, ...]).to(device=device, dtype=torch.float32)
    with torch.no_grad():
        if str(device).startswith("cuda"):
            with torch.amp.autocast("cuda", enabled=bool(use_amp)):
                logits = model(image_t)
        else:
            logits = model(image_t)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.shape[-2:] != tuple(target_hw):
            logits = torch.nn.functional.interpolate(
                logits,
                size=tuple(target_hw),
                mode="bilinear",
                align_corners=False,
            )
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred


def _distance_transform(mask01: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(mask01.astype(np.uint8), cv2.DIST_L2, 5).astype(np.float32)


def _peak_candidates(dt: np.ndarray, mask01: np.ndarray, *, peak_floor_rel: float) -> list[dict[str, Any]]:
    if int(mask01.sum()) <= 0:
        return []
    dt = dt.astype(np.float32)
    max_val = float(dt.max())
    if max_val <= 0.0:
        ys, xs = np.where(mask01.astype(bool))
        if ys.size == 0:
            return []
        return [{"y": int(ys[0]), "x": int(xs[0]), "score": 0.0}]
    peak_floor = max(max_val * float(peak_floor_rel), 1e-6)
    kernel = np.ones((3, 3), dtype=np.uint8)
    dt_dil = cv2.dilate(dt, kernel)
    peak_mask = (mask01.astype(bool)) & (dt >= (dt_dil - 1e-6)) & (dt >= peak_floor)
    ys, xs = np.where(peak_mask)
    out: list[dict[str, Any]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        out.append({"y": int(y), "x": int(x), "score": float(dt[int(y), int(x)])})
    if not out:
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        out.append({"y": int(y), "x": int(x), "score": float(dt[int(y), int(x)])})
    out.sort(key=lambda item: (-float(item["score"]), int(item["y"]), int(item["x"])))
    return out


def _nms_radius(score: float, *, min_distance_px: int, suppression_scale: float) -> float:
    return float(max(float(min_distance_px), float(score) * float(suppression_scale)))


def _greedy_select_candidates(
    candidates: list[dict[str, Any]],
    k: int,
    *,
    min_distance_px: int,
    suppression_scale: float,
    reserved: list[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = list(reserved or [])
    for cand in candidates:
        if len(chosen) >= int(k):
            break
        y = int(cand["y"])
        x = int(cand["x"])
        rad = _nms_radius(float(cand["score"]), min_distance_px=min_distance_px, suppression_scale=suppression_scale)
        keep = True
        for py, px in chosen:
            if float((y - py) ** 2 + (x - px) ** 2) < (rad * rad):
                keep = False
                break
        if keep:
            chosen.append((y, x))
    return chosen[: int(k)]


def _component_peak_lists(mask01: np.ndarray, *, peak_floor_rel: float) -> list[dict[str, Any]]:
    labels_cc, cc_k = _connected_components(mask01.astype(np.uint8))
    out: list[dict[str, Any]] = []
    for comp_id in range(1, int(cc_k) + 1):
        comp01 = labels_cc == comp_id
        dt = _distance_transform(comp01.astype(np.uint8))
        candidates = _peak_candidates(dt, comp01.astype(np.uint8), peak_floor_rel=peak_floor_rel)
        out.append(
            {
                "component_id": int(comp_id),
                "mask01": comp01.astype(np.uint8),
                "dt": dt,
                "area": int(np.sum(comp01)),
                "max_dt": float(dt.max()) if dt.size else 0.0,
                "candidates": candidates,
            }
        )
    return out


def select_exact_k_seeds(mask01: np.ndarray, k: int, spec: SeedMethodSpec) -> dict[str, Any]:
    if int(k) <= 0:
        return {"seeds": [], "component_count": 0, "components": [], "seed_target": 0}
    comps = _component_peak_lists(mask01, peak_floor_rel=float(spec.params.get("peak_floor_rel", 0.08)))
    comp_count = int(len(comps))
    if comp_count == 0:
        return {"seeds": [], "component_count": 0, "components": [], "seed_target": int(k)}
    if comp_count > int(k):
        return {
            "seeds": [],
            "component_count": comp_count,
            "components": [{"component_id": int(c["component_id"]), "allocated": 1} for c in comps],
            "seed_target": int(k),
            "impossible_reason": "connected_components_exceed_k",
        }
    min_distance_px = int(spec.params.get("min_distance_px", 12))
    suppression_scale = float(spec.params.get("suppression_scale", 1.0))

    if spec.method_family == "component_aware_maxima":
        allocations = {int(c["component_id"]): 1 for c in comps}
        extra = int(k) - comp_count
        ranked = sorted(comps, key=lambda c: (-float(c["max_dt"]), -int(c["area"]), int(c["component_id"])))
        rank_idx = 0
        while extra > 0 and ranked:
            comp = ranked[rank_idx % len(ranked)]
            allocations[int(comp["component_id"])] += 1
            extra -= 1
            rank_idx += 1
        seeds: list[tuple[int, int]] = []
        details: list[dict[str, Any]] = []
        for comp in comps:
            alloc = int(allocations[int(comp["component_id"])])
            chosen = _greedy_select_candidates(
                comp["candidates"],
                alloc,
                min_distance_px=min_distance_px,
                suppression_scale=suppression_scale,
            )
            if len(chosen) < alloc and comp["candidates"]:
                top = comp["candidates"][0]
                while len(chosen) < alloc:
                    chosen.append((int(top["y"]), int(top["x"])))
            seeds.extend(chosen)
            details.append({"component_id": int(comp["component_id"]), "allocated": alloc, "selected": len(chosen)})
        seeds = seeds[: int(k)]
        return {"seeds": seeds, "component_count": comp_count, "components": details, "seed_target": int(k)}

    reserved: list[tuple[int, int]] = []
    details = []
    for comp in comps:
        if comp["candidates"]:
            top = comp["candidates"][0]
            reserved.append((int(top["y"]), int(top["x"])))
            details.append({"component_id": int(comp["component_id"]), "allocated": 1, "selected": 1})
    candidates: list[dict[str, Any]] = []
    for comp in comps:
        for cand in comp["candidates"]:
            row = dict(cand)
            row["component_id"] = int(comp["component_id"])
            candidates.append(row)
    candidates.sort(key=lambda item: (-float(item["score"]), int(item["y"]), int(item["x"])))
    chosen = _greedy_select_candidates(
        candidates,
        int(k),
        min_distance_px=min_distance_px,
        suppression_scale=suppression_scale,
        reserved=reserved,
    )
    while len(chosen) < int(k) and candidates:
        top = candidates[0]
        chosen.append((int(top["y"]), int(top["x"])))
    return {"seeds": chosen[: int(k)], "component_count": comp_count, "components": details, "seed_target": int(k)}


def _labels_from_connected_components(mask01: np.ndarray) -> np.ndarray:
    labels_cc, cc_k = _connected_components(mask01.astype(np.uint8))
    out = np.zeros(mask01.shape, dtype=np.uint8)
    next_lab = 1
    for comp_id in range(1, int(cc_k) + 1):
        out[labels_cc == comp_id] = np.uint8(next_lab)
        next_lab += 1
    return out


def split_mask_with_oracle_k(mask01: np.ndarray, k: int, spec: SeedMethodSpec) -> dict[str, Any]:
    mask01 = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    if spec.method_family == "connected_components":
        labels = _labels_from_connected_components(mask01)
        dt = _distance_transform(mask01)
        return {
            "labels": labels,
            "pred_count": int(labels.max()),
            "distance_transform": dt,
            "seeds": [],
            "seed_trace": {"seed_target": int(k), "component_count": int(_count_positive_labels(labels)), "components": []},
        }

    dt = _distance_transform(mask01)
    seed_trace = select_exact_k_seeds(mask01, int(k), spec)
    labels_cc, cc_k = _connected_components(mask01.astype(np.uint8))
    if int(seed_trace.get("component_count", 0)) > int(k):
        labels = _labels_from_connected_components(mask01)
        return {
            "labels": labels,
            "pred_count": int(labels.max()),
            "distance_transform": dt,
            "seeds": [],
            "seed_trace": seed_trace,
        }

    labels = np.zeros(mask01.shape, dtype=np.uint8)
    next_lab = 1
    seeds = [(int(y), int(x)) for y, x in seed_trace.get("seeds", [])]
    for comp_id in range(1, int(cc_k) + 1):
        comp01 = labels_cc == comp_id
        in_comp = [(y, x) for (y, x) in seeds if bool(comp01[int(y), int(x)])]
        if len(in_comp) <= 1:
            labels[comp01] = np.uint8(next_lab)
            next_lab += 1
            continue
        topo = _geometry_topo_u8(comp01.astype(np.uint8))
        seg = _watershed(comp01.astype(np.uint8), in_comp, topo)
        seg_k = int(seg.max())
        if seg_k <= 1:
            labels[comp01] = np.uint8(next_lab)
            next_lab += 1
            continue
        for local_id in range(1, int(seg_k) + 1):
            labels[seg == local_id] = np.uint8(next_lab)
            next_lab += 1

    return {
        "labels": labels,
        "pred_count": int(labels.max()),
        "distance_transform": dt,
        "seeds": seeds,
        "seed_trace": seed_trace,
    }


def _best_assignment(iou_mat: np.ndarray) -> dict[str, Any]:
    gt_k = int(iou_mat.shape[0])
    pred_k = int(iou_mat.shape[1])
    if gt_k == 0 or pred_k == 0:
        return {"sum_iou": 0.0, "pairs": []}
    best_sum = -1.0
    best_pairs: list[tuple[int, int]] = []
    k = min(gt_k, pred_k)
    for gt_idxs in itertools.combinations(range(gt_k), k):
        for pred_idxs in itertools.permutations(range(pred_k), k):
            score = 0.0
            pairs: list[tuple[int, int]] = []
            for gi, pi in zip(gt_idxs, pred_idxs):
                score += float(iou_mat[gi, pi])
                pairs.append((int(gi), int(pi)))
            if score > best_sum:
                best_sum = score
                best_pairs = pairs
    return {"sum_iou": float(max(best_sum, 0.0)), "pairs": best_pairs}


def compute_detailed_instance_metrics(gt_inst_u8: np.ndarray, pred_inst_u8: np.ndarray, *, gt_k: int, pred_k: int) -> dict[str, Any]:
    base = compute_instance_metrics_from_masks(gt_inst_u8, pred_inst_u8, gt_k=gt_k, pred_k=pred_k)
    iou_mat = np.asarray(base["iou_matrix"], dtype=np.float64)
    assign = _best_assignment(iou_mat)
    matched_ious = [0.0 for _ in range(int(gt_k))]
    matched_pred_ids: set[int] = set()
    for gi, pi in assign["pairs"]:
        val = float(iou_mat[int(gi), int(pi)])
        matched_ious[int(gi)] = val
        if val > 0.0:
            matched_pred_ids.add(int(pi) + 1)
    unmatched_gt = int(sum(1 for v in matched_ious if float(v) <= 0.0))
    unmatched_pred = int(len([pi for pi in range(1, int(pred_k) + 1) if pi not in matched_pred_ids]))
    thresholds = {
        "all_iou_ge_0.50": float(all(float(v) >= 0.50 for v in matched_ious)) if gt_k > 0 else 0.0,
        "all_iou_ge_0.70": float(all(float(v) >= 0.70 for v in matched_ious)) if gt_k > 0 else 0.0,
        "all_iou_ge_0.80": float(all(float(v) >= 0.80 for v in matched_ious)) if gt_k > 0 else 0.0,
    }
    base.update(
        {
            "matched_iou_per_gt": [float(v) for v in matched_ious],
            "median_matched_iou": float(np.median(np.asarray(matched_ious, dtype=np.float64))) if matched_ious else 0.0,
            "unmatched_gt_instances": unmatched_gt,
            "unmatched_pred_instances": unmatched_pred,
            **thresholds,
        }
    )
    return base


def _leaflet_union_from_instance_mask(gt_inst_u8: np.ndarray) -> np.ndarray:
    return (gt_inst_u8 > 0).astype(np.uint8)


def _semantic_union_from_prediction(pred_sem_u8: np.ndarray) -> np.ndarray:
    return (pred_sem_u8 == LEAFLET_CLASS_ID).astype(np.uint8)


def _instance_rgb(labels_u8: np.ndarray) -> np.ndarray:
    palette = np.array(
        [
            [0, 0, 0],
            [230, 57, 70],
            [29, 153, 243],
            [38, 166, 154],
            [255, 183, 3],
            [171, 71, 188],
            [240, 98, 146],
        ],
        dtype=np.uint8,
    )
    out = np.zeros(labels_u8.shape + (3,), dtype=np.uint8)
    for lab in np.unique(labels_u8):
        lab_i = int(lab)
        if lab_i <= 0:
            continue
        out[labels_u8 == lab_i] = palette[lab_i % len(palette)]
    return out


def _binary_rgb(mask01: np.ndarray) -> np.ndarray:
    out = np.zeros(mask01.shape + (3,), dtype=np.uint8)
    out[mask01.astype(bool)] = np.array([255, 255, 255], dtype=np.uint8)
    return out


def _distance_rgb(dt: np.ndarray) -> np.ndarray:
    if float(dt.max()) <= 0.0:
        return np.zeros(dt.shape + (3,), dtype=np.uint8)
    norm = cv2.normalize(dt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)[:, :, ::-1]


def _draw_seeds(mask01: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    canvas = _binary_rgb(mask01)
    for idx, (y, x) in enumerate(seeds, start=1):
        cv2.circle(canvas, (int(x), int(y)), 6, (255, 0, 0), thickness=-1)
        cv2.putText(
            canvas,
            str(idx),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            lineType=cv2.LINE_AA,
        )
    return canvas


def _overlay_comparison(image_rgb_u8: np.ndarray, pred_inst_u8: np.ndarray, gt_inst_u8: np.ndarray) -> np.ndarray:
    overlay = image_rgb_u8.copy()
    pred_rgb = _instance_rgb(pred_inst_u8)
    gt_rgb = _instance_rgb(gt_inst_u8)
    overlay = cv2.addWeighted(overlay, 0.55, pred_rgb, 0.30, 0.0)
    overlay = cv2.addWeighted(overlay, 0.75, gt_rgb, 0.25, 0.0)
    return overlay


def _panel_with_title(image_rgb_u8: np.ndarray, title: str) -> np.ndarray:
    canvas = np.full((image_rgb_u8.shape[0] + 36, image_rgb_u8.shape[1], 3), 18, dtype=np.uint8)
    canvas[36:, :, :] = image_rgb_u8
    cv2.putText(
        canvas,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return canvas


def _make_visual_grid(
    image_rgb_u8: np.ndarray,
    semantic_union01: np.ndarray,
    dt: np.ndarray,
    seeds: list[tuple[int, int]],
    pred_inst_u8: np.ndarray,
    gt_inst_u8: np.ndarray,
) -> np.ndarray:
    panels = [
        _panel_with_title(image_rgb_u8, "RGB"),
        _panel_with_title(_binary_rgb(semantic_union01), "Leaflet Union"),
        _panel_with_title(_distance_rgb(dt), "Distance Transform"),
        _panel_with_title(_draw_seeds(semantic_union01, seeds), "Selected Seeds"),
        _panel_with_title(_instance_rgb(pred_inst_u8), "Watershed Instances"),
        _panel_with_title(_instance_rgb(gt_inst_u8), "GT Instances"),
        _panel_with_title(_overlay_comparison(image_rgb_u8, pred_inst_u8, gt_inst_u8), "Overlay Comparison"),
    ]
    rows: list[np.ndarray] = []
    width = int(panels[0].shape[1])
    filler = np.full_like(panels[0], 18)
    row1 = np.concatenate(panels[0:3], axis=1)
    row2 = np.concatenate(panels[3:6], axis=1)
    row3 = np.concatenate([panels[6], filler, filler], axis=1)
    for row in (row1, row2, row3):
        rows.append(row)
    return np.concatenate(rows, axis=0)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "n": 0,
            "exact_instance_count": 0.0,
            "mean_matched_iou": 0.0,
            "median_matched_iou": 0.0,
            "all_iou_ge_0.50": 0.0,
            "all_iou_ge_0.70": 0.0,
            "all_iou_ge_0.80": 0.0,
            "merge_rate": 0.0,
            "fragmentation_rate": 0.0,
            "perfect_rate": 0.0,
        }
    return {
        "n": int(len(rows)),
        "exact_instance_count": _mean([float(r["instance_exact_count_acc"]) for r in rows]),
        "mean_matched_iou": _mean([float(r["instance_mean_matched_iou"]) for r in rows]),
        "median_matched_iou": _mean([float(r["median_matched_iou"]) for r in rows]),
        "all_iou_ge_0.50": _mean([float(r["all_iou_ge_0.50"]) for r in rows]),
        "all_iou_ge_0.70": _mean([float(r["all_iou_ge_0.70"]) for r in rows]),
        "all_iou_ge_0.80": _mean([float(r["all_iou_ge_0.80"]) for r in rows]),
        "merge_rate": _mean([float(r["instance_merged_rate"]) for r in rows]),
        "fragmentation_rate": _mean([float(r["instance_fragmented_rate"]) for r in rows]),
        "perfect_rate": _mean([float(r["instance_perfect_rate"]) for r in rows]),
    }


def _attribution_category(
    *,
    gt_union01: np.ndarray,
    sem_union01: np.ndarray,
    pred_inst_u8: np.ndarray,
    gt_k: int,
    pred_k: int,
    seeds: list[tuple[int, int]],
    matched_ious: list[float],
    is_predicted_semantic: bool,
) -> str:
    if is_predicted_semantic:
        gt_sum = max(int(np.sum(gt_union01)), 1)
        recall = float(np.sum((gt_union01 > 0) & (sem_union01 > 0))) / float(gt_sum)
        extra = int(np.sum((sem_union01 > 0) & (gt_union01 == 0)))
        if recall < 0.85:
            return "missing_semantic_tissue"
        if extra > max(1500, int(0.05 * gt_sum)) and _count_positive_labels(sem_union01.astype(np.uint8)) < gt_k:
            return "false_semantic_bridges"
    cc_k = _count_positive_labels(_labels_from_connected_components(sem_union01.astype(np.uint8)))
    if gt_k >= 2 and cc_k < gt_k:
        eroded = sem_union01.astype(np.uint8)
        narrow = False
        for iters in (1, 2):
            eroded = cv2.erode(eroded, np.ones((3, 3), dtype=np.uint8), iterations=1)
            if int(np.max(_labels_from_connected_components(eroded))) >= gt_k:
                narrow = True
                break
        if narrow:
            return "narrow_bridges"
        return "broad_bridges"
    if len(seeds) < gt_k:
        return "seed_failures"
    if pred_k > gt_k:
        return "seed_failures"
    if any(float(v) < 0.5 for v in matched_ious):
        return "watershed_boundary_failures"
    return "other"


def _classification(pred_summary: dict[str, Any], pred_gt2: dict[str, Any], gt_summary: dict[str, Any]) -> str:
    if (
        float(pred_summary["exact_instance_count"]) >= 0.85
        and float(pred_summary["all_iou_ge_0.50"]) >= 0.75
        and float(pred_gt2["all_iou_ge_0.50"]) >= 0.70
    ):
        return "STRONG_GEOMETRIC_SIGNAL"
    if (
        float(pred_summary["exact_instance_count"]) >= 0.70
        and float(pred_summary["all_iou_ge_0.50"]) >= 0.60
    ):
        return "PROMISING_GEOMETRIC_SIGNAL"
    if (
        float(gt_summary["exact_instance_count"]) >= 0.70
        and float(gt_summary["all_iou_ge_0.50"]) >= 0.60
        and float(pred_summary["all_iou_ge_0.50"]) + 0.15 < float(gt_summary["all_iou_ge_0.50"])
    ):
        return "WEAK_GEOMETRIC_SIGNAL"
    return "GEOMETRY_INSUFFICIENT"


def _next_step_decision(classification: str, geometric_summary: dict[str, Any], center_reference: dict[str, float]) -> tuple[str, str]:
    if (
        float(geometric_summary["exact_instance_count"]) < float(center_reference["instance_exact_count"])
        and float(geometric_summary["mean_matched_iou"]) < float(center_reference["instance_mean_matched_iou"])
    ):
        return ("D. KEEP_CENTER_APPROACH", "Oracle-K geometry underperformed the current center reference on both count reconstruction and matched IoU.")
    if classification in {"STRONG_GEOMETRIC_SIGNAL", "PROMISING_GEOMETRIC_SIGNAL"}:
        return ("A. BUILD_COUNT_CLASSIFIER", "Predicted semantic masks plus oracle K were strong enough that the remaining learning problem is plausibly image-level leaflet count.")
    if classification == "WEAK_GEOMETRIC_SIGNAL":
        return ("B. IMPROVE_SEMANTIC_MASK_FOR_SPLITTING", "GT-semantic geometry was materially stronger than predicted-semantic geometry, so semantic mask quality is the limiting factor.")
    return ("C. BUILD_BOUNDARY_OR_KEYPOINT_HEAD", "Even oracle-K geometry was not reliable enough, so explicit boundary or keypoint supervision is the safer next direction.")


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Unsupported JSON type: {type(obj)!r}")


def _best_method_row(method_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(
        method_rows,
        key=lambda row: (
            -float(row["all_iou_ge_0.50"]),
            -float(row["exact_instance_count"]),
            -float(row["mean_matched_iou"]),
            str(row["method_key"]),
        ),
    )
    return ranked[0]


def _semantic_contract_summary(
    manifest_rows: list[dict[str, Any]],
    semantic_cfg: dict[str, Any],
    instance_root: Path,
    semantic_root: Path,
) -> dict[str, Any]:
    first = manifest_rows[0]
    sem_mask = _load_u8(_resolve_path(instance_root, str(first["semantic_mask_rel"])))
    inst_mask_full = _load_u8(_resolve_path(instance_root, str(first["instance_mask_rel"])))
    inst_mask_crop = _center_crop_like_validation(inst_mask_full, int(first["image_height"]), int(first["image_width"]), is_mask=True)
    meta = _read_json(_resolve_path(instance_root, str(first["metadata_rel"])))
    crop_visible_mismatch_count = 0
    metadata_fullimage_mismatch_count = 0
    for row in manifest_rows:
        inst_full = _load_u8(_resolve_path(instance_root, str(row["instance_mask_rel"])))
        inst_crop = _center_crop_like_validation(inst_full, int(row["image_height"]), int(row["image_width"]), is_mask=True)
        meta_row = _read_json(_resolve_path(instance_root, str(row["metadata_rel"])))
        crop_count = _count_positive_labels(inst_crop)
        full_meta_count = extract_gt_count_from_metadata(meta_row)
        if crop_count != int(row["gt_instance_count"]):
            crop_visible_mismatch_count += 1
        if full_meta_count != int(row["gt_instance_count"]):
            metadata_fullimage_mismatch_count += 1
    return {
        "semantic_class_ids": [int(v) for v in np.unique(sem_mask)],
        "leaflet_class": int(LEAFLET_CLASS_ID),
        "ring_class": int(RING_CLASS_ID),
        "gt_instance_format": "uint8 PNG label map with 0=background and positive integer IDs for individual leaflets",
        "gt_count_source": {
            "oracle_field_used_for_audit": "manifest.gt_instance_count",
            "metadata_full_image_fields_observed": [
                "leaflet_selected_instances",
                "instance_mask_stats.present",
                "leaflet_selected",
            ],
            "instance_mask_positive_ids": _count_positive_labels(inst_mask_crop),
            "metadata_instance_count": extract_gt_count_from_metadata(meta),
            "visible_crop_contract": "manifest gt_instance_count matches positive IDs in the 768x768 center-cropped instance mask",
            "crop_visible_mismatch_count": int(crop_visible_mismatch_count),
            "metadata_fullimage_mismatch_count": int(metadata_fullimage_mismatch_count),
        },
        "instance_ids_are_individual_leaflets": True,
        "image_resolution": [int(first["image_height"]), int(first["image_width"])],
        "gt_instance_native_resolution": [int(inst_mask_full.shape[0]), int(inst_mask_full.shape[1])],
        "semantic_prediction_resolution": [int(semantic_cfg["model"]["input_size"]), int(semantic_cfg["model"]["input_size"])],
        "validation_preprocessing": {
            "spatial": f"deterministic center crop to {int(semantic_cfg['model']['input_size'])}x{int(semantic_cfg['model']['input_size'])}",
            "image_scale": "uint8 RGB -> float32 / 255.0",
            "semantic_argmax": True,
            "semantic_threshold_tuning": False,
        },
        "alignment": "GT instance masks are center-cropped to 768x768 to match semantic validation coordinates; predicted semantic and GT masks are then aligned in the same 768x768 image space.",
        "dataset_roots": {
            "instance_root": str(instance_root),
            "semantic_root": str(semantic_root),
        },
    }


def run_audit(
    *,
    manifest_path: Path,
    output_dir: Path,
    semantic_config_path: Path,
    semantic_checkpoint_path: Path,
    instance_root: Path,
    semantic_root: Path,
    limit: int | None,
) -> dict[str, Any]:
    import torch

    manifest_rows = _read_jsonl(manifest_path)
    if limit is not None:
        manifest_rows = manifest_rows[: int(limit)]
    if not manifest_rows:
        raise SystemExit("Manifest is empty")
    if any(bool(row.get("present_in_authoritative_106_holdout", False)) for row in manifest_rows):
        raise SystemExit("Manifest unexpectedly references authoritative holdout samples")

    semantic_cfg = _read_yaml(semantic_config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = output_dir / "visual_review"
    visual_dir.mkdir(parents=True, exist_ok=True)

    contract = _semantic_contract_summary(manifest_rows, semantic_cfg, instance_root, semantic_root)
    (output_dir / "semantic_contract.json").write_text(json.dumps(contract, indent=2, default=_json_default), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool((semantic_cfg.get("train") or {}).get("amp", False)) and device.type == "cuda"
    model = _build_semantic_model(semantic_cfg).to(device)
    _load_semantic_checkpoint(model, semantic_checkpoint_path, device)

    per_sample_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []
    gt_count_rows: list[dict[str, Any]] = []
    patient_rows: list[dict[str, Any]] = []

    sample_debug: dict[tuple[str, str, str], dict[str, Any]] = {}

    for idx, row in enumerate(manifest_rows, start=1):
        sample_id = str(row["sample"])
        patient_id = str(row["patient_id"])
        target_hw = (int(row["image_height"]), int(row["image_width"]))
        rgb = _load_rgb(_resolve_path(instance_root, str(row["image_rel"])))
        rgb = _center_crop_like_validation(rgb, target_hw[0], target_hw[1], is_mask=False)
        gt_inst_full = _load_u8(_resolve_path(instance_root, str(row["instance_mask_rel"])))
        gt_inst = _center_crop_like_validation(gt_inst_full, target_hw[0], target_hw[1], is_mask=True)
        meta = _read_json(_resolve_path(instance_root, str(row["metadata_rel"])))
        gt_count_meta = extract_gt_count_from_metadata(meta)
        gt_count_mask = _count_positive_labels(gt_inst)
        gt_k = int(row["gt_instance_count"])
        if gt_k != gt_count_mask:
            raise SystemExit(f"GT count mismatch for {sample_id}: manifest={gt_k}, cropped_mask={gt_count_mask}")
        if gt_k not in (1, 2, 3):
            raise SystemExit(f"Unexpected GT count {gt_k} for {sample_id}")
        gt_union = _leaflet_union_from_instance_mask(gt_inst)

        pred_sem = _predict_semantic_mask(model, rgb, target_hw=target_hw, device=device, use_amp=use_amp)
        pred_union = _semantic_union_from_prediction(pred_sem)

        mask_conditions = {
            "GT_SEMANTIC": gt_union,
            "PREDICTED_SEMANTIC": pred_union,
        }

        for condition_key, sem_union in mask_conditions.items():
            for spec in SEED_METHOD_SPECS:
                split = split_mask_with_oracle_k(sem_union, gt_k, spec)
                labels = split["labels"].astype(np.uint8)
                pred_k = int(split["pred_count"])
                metrics = compute_detailed_instance_metrics(gt_inst, labels, gt_k=gt_k, pred_k=pred_k)
                attribution = _attribution_category(
                    gt_union01=gt_union,
                    sem_union01=sem_union,
                    pred_inst_u8=labels,
                    gt_k=gt_k,
                    pred_k=pred_k,
                    seeds=split["seeds"],
                    matched_ious=[float(v) for v in metrics["matched_iou_per_gt"]],
                    is_predicted_semantic=bool(condition_key == "PREDICTED_SEMANTIC"),
                )
                per_row = {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "gt_count": int(gt_k),
                    "mask_condition": condition_key,
                    "method_key": spec.key,
                    "method_family": spec.method_family,
                    "method_variant": spec.variant,
                    "pred_instance_count": int(pred_k),
                    "instance_exact_count_acc": float(metrics["instance_exact_count_acc"]),
                    "instance_mean_matched_iou": float(metrics["instance_mean_matched_iou"]),
                    "median_matched_iou": float(metrics["median_matched_iou"]),
                    "all_iou_ge_0.50": float(metrics["all_iou_ge_0.50"]),
                    "all_iou_ge_0.70": float(metrics["all_iou_ge_0.70"]),
                    "all_iou_ge_0.80": float(metrics["all_iou_ge_0.80"]),
                    "instance_merged_rate": float(metrics["instance_merged_rate"]),
                    "instance_fragmented_rate": float(metrics["instance_fragmented_rate"]),
                    "instance_perfect_rate": float(metrics["instance_perfect_rate"]),
                    "unmatched_gt_instances": int(metrics["unmatched_gt_instances"]),
                    "unmatched_pred_instances": int(metrics["unmatched_pred_instances"]),
                    "matched_iou_per_gt": json.dumps([float(v) for v in metrics["matched_iou_per_gt"]]),
                    "failure_attribution": attribution,
                    "metadata_fullimage_count": int(gt_count_meta),
                }
                per_sample_rows.append(per_row)
                sample_debug[(sample_id, condition_key, spec.key)] = {
                    "rgb": rgb,
                    "sem_union": sem_union,
                    "dt": split["distance_transform"],
                    "seeds": split["seeds"],
                    "pred_inst": labels,
                    "gt_inst": gt_inst,
                    "row": per_row,
                    "seed_trace": split["seed_trace"],
                }
        print(f"[{idx}/{len(manifest_rows)}] audited {sample_id}")

    method_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_sample_rows:
        method_groups.setdefault((str(row["mask_condition"]), str(row["method_key"])), []).append(row)
    for (mask_condition, method_key), rows in sorted(method_groups.items()):
        spec = next(spec for spec in SEED_METHOD_SPECS if spec.key == method_key)
        agg = _aggregate_rows(rows)
        method_rows.append(
            {
                "mask_condition": mask_condition,
                "method_key": method_key,
                "method_family": spec.method_family,
                "method_variant": spec.variant,
                **agg,
            }
        )

    gt_groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in per_sample_rows:
        gt_groups.setdefault((str(row["mask_condition"]), str(row["method_key"]), int(row["gt_count"])), []).append(row)
    for (mask_condition, method_key, gt_count), rows in sorted(gt_groups.items()):
        agg = _aggregate_rows(rows)
        gt_count_rows.append(
            {
                "mask_condition": mask_condition,
                "method_key": method_key,
                "gt_count": int(gt_count),
                **agg,
            }
        )

    patient_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in per_sample_rows:
        patient_groups.setdefault((str(row["mask_condition"]), str(row["method_key"]), str(row["patient_id"])), []).append(row)
    for (mask_condition, method_key, patient_id), rows in sorted(patient_groups.items()):
        agg = _aggregate_rows(rows)
        patient_rows.append(
            {
                "mask_condition": mask_condition,
                "method_key": method_key,
                "patient_id": patient_id,
                **agg,
            }
        )

    pred_method_rows = [row for row in method_rows if str(row["mask_condition"]) == "PREDICTED_SEMANTIC"]
    gt_method_rows = [row for row in method_rows if str(row["mask_condition"]) == "GT_SEMANTIC"]
    best_pred_method = _best_method_row(pred_method_rows)
    best_gt_method = _best_method_row(gt_method_rows)
    best_pred_gt2 = _best_method_row(
        [row for row in gt_count_rows if str(row["mask_condition"]) == "PREDICTED_SEMANTIC" and int(row["gt_count"]) == 2 and str(row["method_key"]) == str(best_pred_method["method_key"])]
    )
    classification = _classification(best_pred_method, best_pred_gt2, best_gt_method)
    center_reference = {
        "strict_marker_contract": 0.3655913978494624,
        "instance_exact_count": 0.5913978494623656,
        "instance_mean_matched_iou": 0.572104696693949,
        "instance_score": 0.5043627612100781,
    }
    next_step, next_reason = _next_step_decision(classification, best_pred_method, center_reference)

    best_pred_failure_rows = [
        row
        for row in per_sample_rows
        if str(row["mask_condition"]) == "PREDICTED_SEMANTIC"
        and str(row["method_key"]) == str(best_pred_method["method_key"])
        and not bool(float(row["all_iou_ge_0.50"]) >= 1.0 and float(row["instance_exact_count_acc"]) >= 1.0)
    ]
    failure_counts: dict[str, int] = {}
    for row in best_pred_failure_rows:
        failure_counts[str(row["failure_attribution"])] = failure_counts.get(str(row["failure_attribution"]), 0) + 1

    failure_csv_rows = []
    for row in best_pred_failure_rows:
        failure_csv_rows.append(
            {
                "sample_id": row["sample_id"],
                "patient_id": row["patient_id"],
                "gt_count": row["gt_count"],
                "mask_condition": row["mask_condition"],
                "method_key": row["method_key"],
                "failure_attribution": row["failure_attribution"],
                "matched_iou_per_gt": row["matched_iou_per_gt"],
            }
        )

    selected_method_config = {
        "method_key": str(best_pred_method["method_key"]),
        "mask_condition": "PREDICTED_SEMANTIC",
        "selection_rule": "maximize all_iou_ge_0.50, then exact_instance_count, then mean_matched_iou",
        "method_variant": str(best_pred_method["method_variant"]),
        "method_family": str(best_pred_method["method_family"]),
        "params": next(spec.params for spec in SEED_METHOD_SPECS if spec.key == str(best_pred_method["method_key"])),
    }

    visual_manifest: list[dict[str, Any]] = []
    requested_categories = [
        ("good_gt1", lambda row: int(row["gt_count"]) == 1 and float(row["all_iou_ge_0.70"]) >= 1.0),
        ("good_gt2", lambda row: int(row["gt_count"]) == 2 and float(row["all_iou_ge_0.70"]) >= 1.0),
        ("good_gt3", lambda row: int(row["gt_count"]) == 3 and float(row["all_iou_ge_0.70"]) >= 1.0),
        ("gt_semantic_success_predicted_failure", lambda row: False),
        ("watershed_merge_failure", lambda row: float(row["instance_merged_rate"]) >= 1.0),
        ("watershed_oversplit_failure", lambda row: float(row["instance_fragmented_rate"]) >= 1.0),
    ]

    best_pred_rows = [
        row
        for row in per_sample_rows
        if str(row["mask_condition"]) == "PREDICTED_SEMANTIC" and str(row["method_key"]) == str(best_pred_method["method_key"])
    ]
    best_gt_rows = {
        str(row["sample_id"]): row
        for row in per_sample_rows
        if str(row["mask_condition"]) == "GT_SEMANTIC" and str(row["method_key"]) == str(best_gt_method["method_key"])
    }
    gt_success_pred_failure = []
    for row in best_pred_rows:
        gt_row = best_gt_rows.get(str(row["sample_id"]))
        if gt_row is None:
            continue
        if float(gt_row["all_iou_ge_0.70"]) >= 1.0 and float(row["all_iou_ge_0.50"]) < 1.0:
            gt_success_pred_failure.append(row)
    if gt_success_pred_failure:
        requested_categories[3] = ("gt_semantic_success_predicted_failure", lambda row: row in gt_success_pred_failure)

    used_samples: set[str] = set()
    for category, predicate in requested_categories:
        candidates = [row for row in best_pred_rows if predicate(row) and str(row["sample_id"]) not in used_samples]
        if not candidates:
            continue
        candidates.sort(
            key=lambda row: (
                -float(row["all_iou_ge_0.70"]),
                -float(row["all_iou_ge_0.50"]),
                -float(row["instance_exact_count_acc"]),
                str(row["sample_id"]),
            )
        )
        chosen = candidates[0]
        used_samples.add(str(chosen["sample_id"]))
        dbg = sample_debug[(str(chosen["sample_id"]), str(chosen["mask_condition"]), str(chosen["method_key"]))]
        grid = _make_visual_grid(dbg["rgb"], dbg["sem_union"], dbg["dt"], dbg["seeds"], dbg["pred_inst"], dbg["gt_inst"])
        out_path = visual_dir / f"{category}_{chosen['sample_id']}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        visual_manifest.append({"category": category, "sample_id": chosen["sample_id"], "file": str(out_path.resolve())})

    audit_summary = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "semantic_checkpoint_path": str(semantic_checkpoint_path.resolve()),
        "semantic_checkpoint_sha256": _sha256_file(semantic_checkpoint_path),
        "semantic_contract": contract,
        "methods": [
            {"method_key": spec.key, "method_family": spec.method_family, "method_variant": spec.variant, "params": spec.params}
            for spec in SEED_METHOD_SPECS
        ],
        "best_gt_semantic_method": best_gt_method,
        "best_predicted_semantic_method": best_pred_method,
        "classification": classification,
        "center_reference": center_reference,
        "vs_center_interpretation": (
            "Compare reconstruction metrics directly; strict marker contract is reported only as reference because geometric outputs do not emit center markers."
        ),
        "failure_attribution_best_predicted_method": failure_counts,
        "next_step": {"decision": next_step, "reason": next_reason},
        "visual_review": visual_manifest,
        "center_visual_availability": {
            "retry1_checkpoint_present_locally": bool(
                (
                    REPO_ROOT
                    / "training"
                    / "runs"
                    / "unetpp_effb3_centerhead_multiscale_x2_2_x1_1_full_dataset_aug_100ep_retry1"
                    / "best_strict_marker_contract.pth"
                ).exists()
            ),
            "note": "The requested retry1 center checkpoint is not present locally, so center-failure exemplar categories could not be rendered from per-sample center predictions in this audit.",
        },
    }

    _write_csv(
        output_dir / "per_sample_results.csv",
        per_sample_rows,
        fieldnames=[
            "sample_id",
            "patient_id",
            "gt_count",
            "mask_condition",
            "method_key",
            "method_family",
            "method_variant",
            "metadata_fullimage_count",
            "pred_instance_count",
            "instance_exact_count_acc",
            "instance_mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50",
            "all_iou_ge_0.70",
            "all_iou_ge_0.80",
            "instance_merged_rate",
            "instance_fragmented_rate",
            "instance_perfect_rate",
            "unmatched_gt_instances",
            "unmatched_pred_instances",
            "matched_iou_per_gt",
            "failure_attribution",
        ],
    )
    _write_csv(
        output_dir / "method_comparison.csv",
        method_rows,
        fieldnames=[
            "mask_condition",
            "method_key",
            "method_family",
            "method_variant",
            "n",
            "exact_instance_count",
            "mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50",
            "all_iou_ge_0.70",
            "all_iou_ge_0.80",
            "merge_rate",
            "fragmentation_rate",
            "perfect_rate",
        ],
    )
    _write_csv(
        output_dir / "gt_count_comparison.csv",
        gt_count_rows,
        fieldnames=[
            "mask_condition",
            "method_key",
            "gt_count",
            "n",
            "exact_instance_count",
            "mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50",
            "all_iou_ge_0.70",
            "all_iou_ge_0.80",
            "merge_rate",
            "fragmentation_rate",
            "perfect_rate",
        ],
    )
    _write_csv(
        output_dir / "patient_comparison.csv",
        patient_rows,
        fieldnames=[
            "mask_condition",
            "method_key",
            "patient_id",
            "n",
            "exact_instance_count",
            "mean_matched_iou",
            "median_matched_iou",
            "all_iou_ge_0.50",
            "all_iou_ge_0.70",
            "all_iou_ge_0.80",
            "merge_rate",
            "fragmentation_rate",
            "perfect_rate",
        ],
    )
    _write_csv(
        output_dir / "failure_attribution.csv",
        failure_csv_rows,
        fieldnames=[
            "sample_id",
            "patient_id",
            "gt_count",
            "mask_condition",
            "method_key",
            "failure_attribution",
            "matched_iou_per_gt",
        ],
    )
    (output_dir / "selected_method_config.json").write_text(
        json.dumps(selected_method_config, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "audit_summary.json").write_text(json.dumps(audit_summary, indent=2, default=_json_default), encoding="utf-8")
    return audit_summary


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--semantic-config", type=Path, default=DEFAULT_SEMANTIC_CONFIG)
    ap.add_argument("--semantic-checkpoint", type=Path, default=DEFAULT_SEMANTIC_CHECKPOINT)
    ap.add_argument("--instance-root", type=Path, default=DEFAULT_INSTANCE_ROOT)
    ap.add_argument("--semantic-root", type=Path, default=DEFAULT_SEMANTIC_ROOT)
    ap.add_argument("--limit", type=int, default=None)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    run_audit(
        manifest_path=args.manifest.resolve(),
        output_dir=args.output_dir.resolve(),
        semantic_config_path=args.semantic_config.resolve(),
        semantic_checkpoint_path=args.semantic_checkpoint.resolve(),
        instance_root=args.instance_root.resolve(),
        semantic_root=args.semantic_root.resolve(),
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
