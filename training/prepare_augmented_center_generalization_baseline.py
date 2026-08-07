from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from augmentations import (
    apply_exact_geometric_transform,
    sample_train_augmentation_params,
    transform_points_row_col_yx,
)
from dataset_centerhead import _read_image_rgb, _read_mask_u8, _read_u16
from prepare_full_dataset_center_training import INPUT_SIZE, REPO_ROOT
from validate_centerhead import _extract_metadata_centers


BASELINE_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_baseline_100ep.yaml"
NEW_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_baseline_100ep.yaml"
OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "center_full_dataset_augmentation_readiness"
TRAIN_MANIFEST_PATH = REPO_ROOT / "training" / "manifests" / "center_full_train_manifest.jsonl"
NEW_SAVE_DIR = "training/runs/unetpp_effb3_centerhead_x2_2_adapter_full_dataset_aug_baseline_100ep"
COORDINATE_CONVENTION = "row_col_yx"
AUDIT_SEED = 1337
VISUAL_SAMPLE_LIMIT = 6


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _align_instance_mask(path: Path, target_hw: tuple[int, int]) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read instance mask: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.int32)
    th, tw = [int(v) for v in target_hw]
    if arr.shape[:2] != (th, tw):
        gh, gw = arr.shape[:2]
        if gh < th or gw < tw:
            raise ValueError(f"Unexpected instance mask shape for augmentation audit: {path} shape={arr.shape} target={target_hw}")
        y0 = (gh - th) // 2
        x0 = (gw - tw) // 2
        arr = arr[y0 : y0 + th, x0 : x0 + tw]
    return arr


def _peak_points(center_map: np.ndarray) -> list[tuple[int, int]]:
    max_val = int(center_map.max())
    if max_val <= 0:
        return []
    ys, xs = np.where(center_map == max_val)
    return sorted((int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist()))


def _labels_to_bgr(labels: np.ndarray) -> np.ndarray:
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    palette = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    ]
    for lab in sorted(int(v) for v in np.unique(labels) if int(v) > 0):
        out[labels == lab] = palette[lab % len(palette)]
    return out


def _draw_points(base_bgr: np.ndarray, points_yx: list[tuple[int, int]], *, color: tuple[int, int, int]) -> np.ndarray:
    out = base_bgr.copy()
    for y, x in points_yx:
        cv2.circle(out, (int(x), int(y)), 5, color, thickness=2, lineType=cv2.LINE_AA)
    return out


def _config_diff_paths(base: Any, other: Any, prefix: str = "") -> list[str]:
    if isinstance(base, dict) and isinstance(other, dict):
        keys = sorted(set(base.keys()) | set(other.keys()))
        out: list[str] = []
        for key in keys:
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in base or key not in other:
                out.append(path)
                continue
            out.extend(_config_diff_paths(base[key], other[key], path))
        return out
    if isinstance(base, list) and isinstance(other, list):
        if len(base) != len(other):
            return [prefix]
        out: list[str] = []
        for idx, (b, o) in enumerate(zip(base, other)):
            out.extend(_config_diff_paths(b, o, f"{prefix}[{idx}]"))
        return out
    if base != other:
        return [prefix]
    return []


def _build_augmented_config(baseline_cfg: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cfg = copy.deepcopy(baseline_cfg)
    cfg["augment"] = {
        "rotate90": True,
        "hflip": True,
        "vflip": True,
        "brightness_contrast": True,
        "brightness_limit": 12.0,
        "contrast_limit": 0.10,
        "gamma": False,
        "random_crop": False,
    }
    cfg["train"]["save_dir"] = NEW_SAVE_DIR
    diff_paths = _config_diff_paths(baseline_cfg, cfg)
    return cfg, diff_paths


def _transform_name(params: dict[str, Any]) -> str:
    parts = [f"rot90_k{int(params['rot90_k'])}"]
    if bool(params["hflip"]):
        parts.append("hflip")
    if bool(params["vflip"]):
        parts.append("vflip")
    return "__".join(parts)


def _make_visual_panel(
    *,
    sample: str,
    transform_name: str,
    image: np.ndarray,
    gt_inst: np.ndarray,
    gt_points: list[tuple[int, int]],
    aug_image: np.ndarray,
    aug_inst: np.ndarray,
    aug_points: list[tuple[int, int]],
    out_path: Path,
) -> None:
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    aug_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
    gt_panel = _draw_points(_labels_to_bgr(gt_inst), gt_points, color=(0, 255, 255))
    aug_panel = _draw_points(_labels_to_bgr(aug_inst), aug_points, color=(0, 255, 255))
    img_panel = _draw_points(image_bgr, gt_points, color=(0, 255, 255))
    aug_img_panel = _draw_points(aug_bgr, aug_points, color=(0, 255, 255))
    header = np.full((72, image.shape[1] * 2, 3), 255, dtype=np.uint8)
    cv2.putText(header, f"{sample} / {transform_name}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(header, COORDINATE_CONVENTION, (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
    top = np.concatenate([img_panel, aug_img_panel], axis=1)
    bottom = np.concatenate([gt_panel, aug_panel], axis=1)
    panel = np.concatenate([header, top, bottom], axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def run(*, baseline_config_path: Path, output_dir: Path, config_out_path: Path, manifest_path: Path) -> dict[str, Any]:
    baseline_cfg = _read_yaml(baseline_config_path)
    aug_cfg, diff_paths = _build_augmented_config(baseline_cfg)
    _write_yaml(config_out_path, aug_cfg)

    manifest_rows = sorted(_read_jsonl(manifest_path), key=lambda row: int(row["sample_index"]))
    dataset_root = (REPO_ROOT / str(aug_cfg["dataset"]["root"])).resolve()
    instance_root = (REPO_ROOT / str(aug_cfg["dataset"]["instance_root"])).resolve()

    contract_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    visual_records: list[dict[str, Any]] = []
    visual_transform_specs = [
        {"rot90_k": 0, "hflip": False, "vflip": False},
        {"rot90_k": 0, "hflip": True, "vflip": False},
        {"rot90_k": 0, "hflip": False, "vflip": True},
        {"rot90_k": 1, "hflip": False, "vflip": False},
        {"rot90_k": 2, "hflip": False, "vflip": False},
        {"rot90_k": 3, "hflip": False, "vflip": False},
    ]

    for row in manifest_rows:
        sample = str(row["sample"])
        image = _read_image_rgb((dataset_root / str(row["image_rel"])).resolve())
        semantic = _read_mask_u8((dataset_root / str(row["semantic_mask_rel"])).resolve())
        center_u16 = _read_u16((dataset_root / str(row["center_target_rel"])).resolve())
        instance = _align_instance_mask((instance_root / str(row["instance_mask_rel"])).resolve(), target_hw=image.shape[:2])
        gt_points = [(int(y), int(x)) for y, x in _extract_metadata_centers(str((dataset_root / str(row["metadata_rel"])).resolve()))]

        rng = random.Random(int(AUDIT_SEED) + int(row["sample_index"]))
        params = sample_train_augmentation_params(aug_cfg["augment"], rng=rng)
        aug_image, aug_semantic, aug_center = apply_exact_geometric_transform(
            image,
            semantic,
            center=center_u16,
            hflip=bool(params["hflip"]),
            vflip=bool(params["vflip"]),
            rot90_k=int(params["rot90_k"]),
        )
        _aug_image_for_instance, aug_instance = apply_exact_geometric_transform(
            image,
            instance,
            hflip=bool(params["hflip"]),
            vflip=bool(params["vflip"]),
            rot90_k=int(params["rot90_k"]),
        )
        aug_points = transform_points_row_col_yx(
            gt_points,
            image.shape[:2],
            hflip=bool(params["hflip"]),
            vflip=bool(params["vflip"]),
            rot90_k=int(params["rot90_k"]),
        )
        peak_points = _peak_points(aug_center)
        touched_instance_ids = [int(aug_instance[int(y), int(x)]) if 0 <= int(y) < aug_instance.shape[0] and 0 <= int(x) < aug_instance.shape[1] else 0 for y, x in aug_points]
        unique_touched_ids = sorted(int(v) for v in touched_instance_ids if int(v) > 0)
        valid = True
        reasons: list[str] = []
        if image.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            valid = False
            reasons.append("raw_image_not_768")
        if semantic.shape[:2] != (INPUT_SIZE, INPUT_SIZE) or center_u16.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            valid = False
            reasons.append("raw_target_not_768")
        if aug_image.shape[:2] != (INPUT_SIZE, INPUT_SIZE) or aug_semantic.shape[:2] != (INPUT_SIZE, INPUT_SIZE) or aug_center.shape[:2] != (INPUT_SIZE, INPUT_SIZE):
            valid = False
            reasons.append("augmented_shape_not_768")
        if len(aug_points) != int(row["gt_instance_count"]):
            valid = False
            reasons.append("center_count_changed")
        if len(unique_touched_ids) != int(row["gt_instance_count"]):
            valid = False
            reasons.append("center_outside_or_duplicate_instance")
        if sorted(peak_points) != sorted(aug_points):
            valid = False
            reasons.append("target_peaks_do_not_match_transformed_points")
        contract_row = {
            "sample": sample,
            "patient_id": str(row["patient_id"]),
            "sample_index": int(row["sample_index"]),
            "transform_name": _transform_name(params),
            "hflip": bool(params["hflip"]),
            "vflip": bool(params["vflip"]),
            "rot90_k": int(params["rot90_k"]),
            "gt_instance_count": int(row["gt_instance_count"]),
            "transformed_point_count": int(len(aug_points)),
            "peak_point_count": int(len(peak_points)),
            "unique_instance_ids_touched": int(len(unique_touched_ids)),
            "image_height": int(aug_image.shape[0]),
            "image_width": int(aug_image.shape[1]),
            "coordinate_convention": COORDINATE_CONVENTION,
            "centers_inside_instances": bool(len(unique_touched_ids) == int(row["gt_instance_count"])),
            "peak_matches_transformed_target": bool(sorted(peak_points) == sorted(aug_points)),
            "valid": bool(valid),
            "invalid_reasons": "|".join(reasons),
        }
        contract_rows.append(contract_row)
        if not valid:
            invalid_rows.append(contract_row)

    first_row = manifest_rows[0]
    base_image = _read_image_rgb((dataset_root / str(first_row["image_rel"])).resolve())
    base_instance = _align_instance_mask((instance_root / str(first_row["instance_mask_rel"])).resolve(), target_hw=base_image.shape[:2])
    base_points = [(int(y), int(x)) for y, x in _extract_metadata_centers(str((dataset_root / str(first_row["metadata_rel"])).resolve()))]
    for spec in visual_transform_specs[:VISUAL_SAMPLE_LIMIT]:
        aug_image, _aug_sem, _aug_center = apply_exact_geometric_transform(
            base_image,
            _read_mask_u8((dataset_root / str(first_row["semantic_mask_rel"])).resolve()),
            center=_read_u16((dataset_root / str(first_row["center_target_rel"])).resolve()),
            hflip=bool(spec["hflip"]),
            vflip=bool(spec["vflip"]),
            rot90_k=int(spec["rot90_k"]),
        )
        _img_dummy, aug_instance = apply_exact_geometric_transform(
            base_image,
            base_instance,
            hflip=bool(spec["hflip"]),
            vflip=bool(spec["vflip"]),
            rot90_k=int(spec["rot90_k"]),
        )
        aug_points = transform_points_row_col_yx(
            base_points,
            base_image.shape[:2],
            hflip=bool(spec["hflip"]),
            vflip=bool(spec["vflip"]),
            rot90_k=int(spec["rot90_k"]),
        )
        transform_name = _transform_name(spec)
        out_path = (output_dir / "visual_review" / f"{transform_name}.png").resolve()
        _make_visual_panel(
            sample=str(first_row["sample"]),
            transform_name=transform_name,
            image=base_image,
            gt_inst=base_instance,
            gt_points=base_points,
            aug_image=aug_image,
            aug_inst=aug_instance,
            aug_points=aug_points,
            out_path=out_path,
        )
        visual_records.append({"transform_name": transform_name, "path": str(out_path)})

    summary = {
        "baseline_config": str(baseline_config_path.resolve()),
        "new_config": str(config_out_path.resolve()),
        "samples_transforms_checked": int(len(contract_rows)),
        "invalid": int(len(invalid_rows)),
        "lost_centers": int(sum(1 for row in contract_rows if "center_count_changed" in str(row["invalid_reasons"]))),
        "outside_centers": int(sum(1 for row in contract_rows if "center_outside_or_duplicate_instance" in str(row["invalid_reasons"]))),
        "coordinate_convention": COORDINATE_CONVENTION,
        "visual_review_count": int(len(visual_records)),
        "changed_fields_vs_baseline": diff_paths,
        "status": "ready_for_training" if not invalid_rows else "blocked",
    }
    _write_json((output_dir / "augmentation_summary.json").resolve(), summary)
    _write_csv(
        (output_dir / "transform_contract.csv").resolve(),
        contract_rows,
        [
            "sample",
            "patient_id",
            "sample_index",
            "transform_name",
            "hflip",
            "vflip",
            "rot90_k",
            "gt_instance_count",
            "transformed_point_count",
            "peak_point_count",
            "unique_instance_ids_touched",
            "image_height",
            "image_width",
            "coordinate_convention",
            "centers_inside_instances",
            "peak_matches_transformed_target",
            "valid",
            "invalid_reasons",
        ],
    )
    _write_csv(
        (output_dir / "invalid_transforms.csv").resolve(),
        invalid_rows,
        [
            "sample",
            "patient_id",
            "sample_index",
            "transform_name",
            "hflip",
            "vflip",
            "rot90_k",
            "gt_instance_count",
            "transformed_point_count",
            "peak_point_count",
            "unique_instance_ids_touched",
            "image_height",
            "image_width",
            "coordinate_convention",
            "centers_inside_instances",
            "peak_matches_transformed_target",
            "valid",
            "invalid_reasons",
        ],
    )
    _write_text((output_dir / "files_to_review.txt").resolve(), "\n".join([str((output_dir / "augmentation_summary.json").resolve()), str((output_dir / "transform_contract.csv").resolve()), str((output_dir / "invalid_transforms.csv").resolve()), str((output_dir / "visual_review").resolve())]) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-config", type=str, default=str(BASELINE_CONFIG_PATH))
    ap.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    ap.add_argument("--config-out", type=str, default=str(NEW_CONFIG_PATH))
    ap.add_argument("--manifest", type=str, default=str(TRAIN_MANIFEST_PATH))
    args = ap.parse_args()
    summary = run(
        baseline_config_path=Path(args.baseline_config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        config_out_path=Path(args.config_out).resolve(),
        manifest_path=Path(args.manifest).resolve(),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
