from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from augmentations import get_val_augmentations
from compare_reconstruction_policies import _write_json_atomic
from dataset import read_split_file
from train_centerhead import _read_yaml, smoke_test
from validate_centerhead import _connected_components, _extract_metadata_centers


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "training" / "configs" / "unetpp_effb3_centerhead_x2_2_adapter_full_dataset_baseline_100ep.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "center_full_dataset_training_readiness"
MANIFEST_DIR = REPO_ROOT / "training" / "manifests"
TRAIN_MANIFEST_PATH = MANIFEST_DIR / "center_full_train_manifest.jsonl"
VAL_MANIFEST_PATH = MANIFEST_DIR / "center_full_val_manifest.jsonl"
TRAIN_TXT_PATH = MANIFEST_DIR / "center_full_train.txt"
VAL_TXT_PATH = MANIFEST_DIR / "center_full_val.txt"
SPLIT_SUMMARY_PATH = MANIFEST_DIR / "center_full_split_summary.json"

SEED = 1337
VAL_RATIO = 0.2
INPUT_SIZE = 768
MIN_VISUAL_REVIEW = 20
COORDINATE_CONVENTION = "row_col_yx"
CENTER_TARGET_POLICY = "legacy_png_u16_discrete_peaks_from_metadata_center_yx"
LOCKED_REFERENCE_THRESHOLD = 0.03
AUTHORITATIVE_REFERENCE = {
    "locked_threshold": 0.03,
    "center_f1": 0.0101,
    "strict_marker_contract_pass_count": 3,
    "strict_marker_contract_total": 106,
    "bottleneck": "mixed_center_and_semantic_failure",
}
MICROSET_IDS = (
    "m01_p02_s00",
    "m01_p02_s04",
    "m01_p01_s00",
    "m01_p01_s01",
    "m01_p01_s02",
    "m01_p01_s03",
)


class ReadinessError(RuntimeError):
    def __init__(self, message: str, *, samples: list[str] | None = None, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.samples = list(samples or [])
        self.details = dict(details or {})


@dataclass(frozen=True)
class SampleRecord:
    sample: str
    patient_id: str
    source_split: str
    image_path: Path
    semantic_mask_path: Path
    instance_mask_path: Path
    center_target_path: Path
    metadata_path: Path
    gt_instance_count: int
    image_height: int
    image_width: int
    foreground_area: int
    quality: str | None
    in_microset: bool
    in_authoritative_holdout: bool
    image_sha256: str
    semantic_sha256: str
    instance_sha256: str
    center_sha256: str
    source_instance_ids: tuple[int, ...]
    instance_areas: tuple[int, ...]
    instance_center_yx: tuple[tuple[int, int], ...]
    max_dt_per_instance: tuple[float, ...]
    semantic_cc_count: int
    border_touching_instances: int
    fragmented_semantic: bool


def _atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding=encoding, newline="\n")
    os.replace(tmp, path)


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    os.replace(tmp, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _jsonl_sha256(rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sample_sort_key(sample: str) -> tuple[int, ...]:
    parts = []
    buf = ""
    is_digit = False
    for ch in sample:
        if ch.isdigit() != is_digit and buf:
            parts.append(int(buf) if is_digit else buf)
            buf = ch
            is_digit = ch.isdigit()
        else:
            buf += ch
            is_digit = ch.isdigit()
    if buf:
        parts.append(int(buf) if is_digit else buf)
    return tuple(parts)


def _patient_id(sample: str) -> str:
    if "_s" not in sample:
        raise ReadinessError(f"Cannot derive patient ID from sample: {sample}", samples=[sample])
    return sample.rsplit("_s", 1)[0]


def _find_duplicate_samples(sample_ids: list[str]) -> list[str]:
    counts = Counter(str(s) for s in sample_ids)
    return sorted([sample for sample, count in counts.items() if int(count) > 1], key=_sample_sort_key)


def _assert_no_holdout_overlap(records: list[SampleRecord]) -> None:
    leaked = sorted([rec.sample for rec in records if rec.in_authoritative_holdout], key=_sample_sort_key)
    if leaked:
        raise ReadinessError("Authoritative holdout samples leaked into training pool", samples=leaked)


def _read_manifest_csv(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    out: dict[str, dict[str, str]] = {}
    duplicates = []
    for row in rows:
        sample = str(row["sample"]).strip()
        if sample in out:
            duplicates.append(sample)
        out[sample] = row
    if duplicates:
        raise ReadinessError("Duplicate sample IDs in source manifest", samples=sorted(set(duplicates)))
    return out


def _load_image(path: Path, flags: int) -> np.ndarray:
    arr = cv2.imread(str(path), flags)
    if arr is None:
        raise FileNotFoundError(f"Failed to read file: {path}")
    if arr.ndim == 3 and arr.shape[2] == 3 and flags == cv2.IMREAD_COLOR:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _load_aligned_instance(instance_path: Path, target_hw: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    raw = _load_image(instance_path, cv2.IMREAD_UNCHANGED).astype(np.int32)
    raw_hw = (int(raw.shape[0]), int(raw.shape[1]))
    th, tw = [int(v) for v in target_hw]
    if raw.shape[:2] != (th, tw):
        gh, gw = raw.shape[:2]
        y0 = (gh - th) // 2
        x0 = (gw - tw) // 2
        raw = raw[y0 : y0 + th, x0 : x0 + tw]
    return raw.astype(np.int32), raw_hw


def _unique_positive_labels(mask: np.ndarray) -> list[int]:
    vals = sorted(int(v) for v in np.unique(mask) if int(v) > 0)
    return vals


def _semantic_component_count(semantic_mask: np.ndarray) -> int:
    leaf_union = (semantic_mask > 0).astype(np.uint8)
    _labels, count = _connected_components(leaf_union)
    return int(count)


def _border_touches(instance_mask: np.ndarray, label_id: int) -> bool:
    comp = instance_mask == int(label_id)
    if not bool(comp.any()):
        return False
    return bool(comp[0, :].any() or comp[-1, :].any() or comp[:, 0].any() or comp[:, -1].any())


def _target_peak_points(center_u16: np.ndarray) -> list[tuple[int, int]]:
    max_val = int(center_u16.max())
    if max_val <= 0:
        return []
    ys, xs = np.where(center_u16 == max_val)
    pts = sorted((int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist()))
    return pts


def _read_training_pool() -> tuple[list[SampleRecord], list[str]]:
    distance_root = (REPO_ROOT / "datasets" / "converted_leaflet_distance").resolve()
    instance_root = (REPO_ROOT / "datasets" / "converted_leaflet_instances").resolve()
    distance_manifest = _read_manifest_csv(distance_root / "distance_dataset_manifest.csv")
    instance_manifest = _read_manifest_csv(instance_root / "instance_dataset_manifest.csv")

    holdout_ids = {
        sample
        for sample, row in distance_manifest.items()
        if str(row.get("split", "")).strip().lower() in {"val", "test"}
    }

    source_train_items = read_split_file(distance_root, distance_root / "train.txt")
    duplicate_ids = []
    seen_samples: set[str] = set()
    records: list[SampleRecord] = []
    for item in source_train_items:
        sample = item.image_path.stem
        if sample in seen_samples:
            duplicate_ids.append(sample)
            continue
        seen_samples.add(sample)
        drow = distance_manifest.get(sample)
        irow = instance_manifest.get(sample)
        if drow is None or irow is None:
            raise ReadinessError("Sample missing from dataset manifest", samples=[sample])
        metadata_path = (distance_root / str(drow["metadata_rel"])).resolve()
        center_path = (distance_root / str(drow["center_rel"])).resolve()
        image_path = item.image_path.resolve()
        semantic_path = item.mask_path.resolve()
        instance_path = (instance_root / str(irow["instance_rel"])).resolve()

        for path in (image_path, semantic_path, instance_path, metadata_path, center_path):
            if not path.exists():
                raise ReadinessError("Required dataset file missing", samples=[sample], details={"path": str(path)})

        image = _load_image(image_path, cv2.IMREAD_COLOR)
        semantic = _load_image(semantic_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        instance, raw_instance_hw = _load_aligned_instance(instance_path, target_hw=semantic.shape[:2])
        center_u16 = _load_image(center_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        if semantic.shape != instance.shape or semantic.shape != center_u16.shape or semantic.shape[:2] != image.shape[:2]:
            raise ReadinessError(
                "Image and target dimensions disagree",
                samples=[sample],
                details={
                    "image_shape": list(image.shape),
                    "semantic_shape": list(semantic.shape),
                    "instance_shape": list(instance.shape),
                    "raw_instance_shape": [raw_instance_hw[0], raw_instance_hw[1]],
                    "center_shape": list(center_u16.shape),
                },
            )

        semantic_labels = set(int(v) for v in np.unique(semantic).tolist())
        if not semantic_labels.issubset({0, 1, 2}):
            raise ReadinessError("Semantic labels outside expected range", samples=[sample], details={"semantic_labels": sorted(semantic_labels)})

        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        meta_instances = list(meta.get("instances") or [])
        gt_count_meta = int(meta.get("instance_count", len(meta_instances)))
        unique_instance_ids = _unique_positive_labels(instance)
        if gt_count_meta != len(unique_instance_ids):
            raise ReadinessError(
                "GT instance count does not match instance labels",
                samples=[sample],
                details={"metadata_count": gt_count_meta, "instance_labels": unique_instance_ids},
            )

        centers = []
        source_instance_ids = []
        instance_areas = []
        max_dt_per_instance = []
        border_touching_instances = 0
        for inst in meta_instances:
            inst_id = int(inst["instance_id"])
            y, x = [int(v) for v in inst["center_yx"]]
            if not (0 <= y < instance.shape[0] and 0 <= x < instance.shape[1]):
                raise ReadinessError("Center coordinate outside image bounds", samples=[sample], details={"instance_id": inst_id, "center_yx": [y, x]})
            if int(instance[y, x]) != inst_id:
                raise ReadinessError("Center lies outside corresponding instance", samples=[sample], details={"instance_id": inst_id, "center_yx": [y, x], "label_at_center": int(instance[y, x])})
            if int(center_u16[y, x]) <= 0:
                raise ReadinessError("Center target missing at metadata coordinate", samples=[sample], details={"instance_id": inst_id, "center_yx": [y, x], "center_value": int(center_u16[y, x])})
            centers.append((y, x))
            source_instance_ids.append(inst_id)
            instance_areas.append(int(inst.get("area", int(np.sum(instance == inst_id)))))
            max_dt_per_instance.append(float(inst.get("max_dt", 0.0)))
            if _border_touches(instance, inst_id):
                border_touching_instances += 1

        if len(set(centers)) != len(centers):
            raise ReadinessError("Different instances collapse to the same center coordinate", samples=[sample], details={"centers": [list(c) for c in centers]})

        peak_points = _target_peak_points(center_u16)
        if len(peak_points) != gt_count_meta:
            raise ReadinessError(
                "Center target peak count does not match GT instance count",
                samples=[sample],
                details={"gt_count": gt_count_meta, "peak_count": len(peak_points), "peak_points_preview": [list(p) for p in peak_points[:10]]},
            )
        if sorted(peak_points) != sorted(centers):
            raise ReadinessError(
                "Center target peak coordinates do not match metadata centers",
                samples=[sample],
                details={"metadata_centers": [list(p) for p in centers], "peak_points": [list(p) for p in peak_points]},
            )

        records.append(
            SampleRecord(
                sample=sample,
                patient_id=_patient_id(sample),
                source_split="train",
                image_path=image_path,
                semantic_mask_path=semantic_path,
                instance_mask_path=instance_path,
                center_target_path=center_path,
                metadata_path=metadata_path,
                gt_instance_count=gt_count_meta,
                image_height=int(image.shape[0]),
                image_width=int(image.shape[1]),
                foreground_area=int(np.sum(semantic > 0)),
                quality=str(drow.get("quality") or meta.get("quality") or ""),
                in_microset=sample in MICROSET_IDS,
                in_authoritative_holdout=sample in holdout_ids,
                image_sha256=_sha256_file(image_path),
                semantic_sha256=_sha256_file(semantic_path),
                instance_sha256=_sha256_file(instance_path),
                center_sha256=_sha256_file(center_path),
                source_instance_ids=tuple(source_instance_ids),
                instance_areas=tuple(instance_areas),
                instance_center_yx=tuple(centers),
                max_dt_per_instance=tuple(max_dt_per_instance),
                semantic_cc_count=_semantic_component_count(semantic),
                border_touching_instances=int(border_touching_instances),
                fragmented_semantic=bool(_semantic_component_count(semantic) > gt_count_meta),
            )
        )

    if duplicate_ids:
        raise ReadinessError("Duplicate sample IDs in train.txt", samples=_find_duplicate_samples(duplicate_ids))
    return sorted(records, key=lambda r: _sample_sort_key(r.sample)), sorted(holdout_ids, key=_sample_sort_key)


def _inventory_rows(records: list[SampleRecord]) -> list[dict[str, Any]]:
    rows = []
    for rec in records:
        rows.append(
            {
                "sample": rec.sample,
                "patient_id": rec.patient_id,
                "source_split": rec.source_split,
                "image_path": str(rec.image_path),
                "semantic_mask_path": str(rec.semantic_mask_path),
                "instance_mask_path": str(rec.instance_mask_path),
                "center_target_path": str(rec.center_target_path),
                "center_target_generation": CENTER_TARGET_POLICY,
                "gt_instance_count": int(rec.gt_instance_count),
                "image_height": int(rec.image_height),
                "image_width": int(rec.image_width),
                "foreground_area": int(rec.foreground_area),
                "quality": rec.quality or "",
                "used_in_six_sample_microset": bool(rec.in_microset),
                "present_in_authoritative_106_holdout": bool(rec.in_authoritative_holdout),
                "image_sha256": rec.image_sha256,
                "semantic_sha256": rec.semantic_sha256,
                "instance_sha256": rec.instance_sha256,
                "center_sha256": rec.center_sha256,
                "metadata_path": str(rec.metadata_path),
                "coordinate_convention": COORDINATE_CONVENTION,
                "semantic_cc_count": int(rec.semantic_cc_count),
                "border_touching_instances": int(rec.border_touching_instances),
                "fragmented_semantic": bool(rec.fragmented_semantic),
            }
        )
    return rows


def _split_patient_score(current: Counter[int], target: Counter[int], *, samples: int, target_samples: float) -> float:
    score = float((samples - target_samples) ** 2)
    for gt_count in (1, 2, 3):
        score += float((current.get(gt_count, 0) - target.get(gt_count, 0)) ** 2) * 4.0
    return score


def _patient_level_split(records: list[SampleRecord], *, seed: int = SEED, val_ratio: float = VAL_RATIO) -> tuple[list[SampleRecord], list[SampleRecord], dict[str, Any]]:
    by_patient: dict[str, list[SampleRecord]] = defaultdict(list)
    for rec in records:
        by_patient[rec.patient_id].append(rec)

    patient_stats = []
    total_counts = Counter(int(rec.gt_instance_count) for rec in records)
    target_counts = Counter({k: int(round(float(v) * float(val_ratio))) for k, v in total_counts.items()})
    target_samples = float(len(records)) * float(val_ratio)
    rng = random.Random(int(seed))
    patient_ids = sorted(by_patient.keys(), key=_sample_sort_key)
    rng.shuffle(patient_ids)
    patient_ids.sort(
        key=lambda pid: (
            -len(by_patient[pid]),
            -sum(int(r.foreground_area) for r in by_patient[pid]),
            _sample_sort_key(pid),
        )
    )
    for pid in patient_ids:
        group = sorted(by_patient[pid], key=lambda r: _sample_sort_key(r.sample))
        counts = Counter(int(r.gt_instance_count) for r in group)
        patient_stats.append((pid, group, counts))

    val_patients: set[str] = set()
    val_counts: Counter[int] = Counter()
    val_samples = 0
    for pid, group, counts in patient_stats:
        trial_counts = val_counts + counts
        trial_samples = val_samples + len(group)
        add_score = _split_patient_score(trial_counts, target_counts, samples=trial_samples, target_samples=target_samples)
        keep_score = _split_patient_score(val_counts, target_counts, samples=val_samples, target_samples=target_samples)
        if add_score < keep_score:
            val_patients.add(pid)
            val_counts = trial_counts
            val_samples = trial_samples

    if not val_patients and patient_stats:
        pid, group, counts = patient_stats[0]
        val_patients.add(pid)
        val_counts = Counter(counts)
        val_samples = len(group)

    train_records = sorted([rec for rec in records if rec.patient_id not in val_patients], key=lambda r: _sample_sort_key(r.sample))
    val_records = sorted([rec for rec in records if rec.patient_id in val_patients], key=lambda r: _sample_sort_key(r.sample))
    if not train_records or not val_records:
        raise ReadinessError("Patient-level split produced an empty partition")

    summary = {
        "seed": int(seed),
        "validation_ratio": float(val_ratio),
        "patient_counts": {"train": len({r.patient_id for r in train_records}), "val": len({r.patient_id for r in val_records})},
        "sample_counts": {"train": len(train_records), "val": len(val_records)},
        "gt_distribution": {
            "train": {str(k): int(v) for k, v in sorted(Counter(int(r.gt_instance_count) for r in train_records).items())},
            "val": {str(k): int(v) for k, v in sorted(Counter(int(r.gt_instance_count) for r in val_records).items())},
        },
        "patients": {
            "train": sorted({r.patient_id for r in train_records}, key=_sample_sort_key),
            "val": sorted({r.patient_id for r in val_records}, key=_sample_sort_key),
        },
    }
    return train_records, val_records, summary


def _duplicate_content_across_splits(train_records: list[SampleRecord], val_records: list[SampleRecord]) -> list[dict[str, Any]]:
    def index(records: list[SampleRecord], attr: str) -> dict[tuple[str, str], list[str]]:
        out: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rec in records:
            out[(attr, getattr(rec, attr))].append(rec.sample)
        return out

    dups = []
    attrs = ("image_sha256", "semantic_sha256", "instance_sha256", "center_sha256")
    train_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    val_index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for attr in attrs:
        for key, samples in index(train_records, attr).items():
            train_index[key].extend(samples)
        for key, samples in index(val_records, attr).items():
            val_index[key].extend(samples)
    for key in sorted(set(train_index) & set(val_index)):
        dups.append(
            {
                "artifact_type": key[0].replace("_sha256", ""),
                "sha256": key[1],
                "train_samples": sorted(train_index[key], key=_sample_sort_key),
                "val_samples": sorted(val_index[key], key=_sample_sort_key),
            }
        )
    return dups


def _manifest_row(rec: SampleRecord, *, split_name: str, sample_index: int) -> dict[str, Any]:
    image_rel = rec.image_path.relative_to((REPO_ROOT / "datasets" / "converted_leaflet_distance").resolve()).as_posix()
    semantic_rel = rec.semantic_mask_path.relative_to((REPO_ROOT / "datasets" / "converted_leaflet_distance").resolve()).as_posix()
    center_rel = rec.center_target_path.relative_to((REPO_ROOT / "datasets" / "converted_leaflet_distance").resolve()).as_posix()
    metadata_rel = rec.metadata_path.relative_to((REPO_ROOT / "datasets" / "converted_leaflet_distance").resolve()).as_posix()
    instance_rel = rec.instance_mask_path.relative_to((REPO_ROOT / "datasets" / "converted_leaflet_instances").resolve()).as_posix()
    return {
        "sample": rec.sample,
        "sample_index": int(sample_index),
        "patient_id": rec.patient_id,
        "split": split_name,
        "source_split": rec.source_split,
        "image_rel": image_rel,
        "semantic_mask_rel": semantic_rel,
        "instance_mask_rel": instance_rel,
        "center_target_rel": center_rel,
        "metadata_rel": metadata_rel,
        "gt_instance_count": int(rec.gt_instance_count),
        "image_height": int(rec.image_height),
        "image_width": int(rec.image_width),
        "foreground_area": int(rec.foreground_area),
        "quality": rec.quality or "",
        "used_in_six_sample_microset": bool(rec.in_microset),
        "present_in_authoritative_106_holdout": bool(rec.in_authoritative_holdout),
        "center_target_generation": CENTER_TARGET_POLICY,
        "coordinate_convention": COORDINATE_CONVENTION,
        "image_sha256": rec.image_sha256,
        "semantic_sha256": rec.semantic_sha256,
        "instance_sha256": rec.instance_sha256,
        "center_sha256": rec.center_sha256,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _atomic_write_text(path, "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows))


def _write_split_txt(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(f"{row['image_rel']}\t{row['semantic_mask_rel']}\n" for row in rows)
    _atomic_write_text(path, text)


def _foreground_stats(records: list[SampleRecord]) -> dict[str, float | int]:
    vals = [int(r.foreground_area) for r in records]
    return {
        "count": int(len(vals)),
        "min": int(min(vals)),
        "max": int(max(vals)),
        "mean": float(sum(vals) / max(len(vals), 1)),
        "median": float(np.median(np.asarray(vals, dtype=np.float64))),
    }


def _write_split_manifests(train_records: list[SampleRecord], val_records: list[SampleRecord]) -> dict[str, Any]:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    train_rows = [_manifest_row(rec, split_name="train", sample_index=i) for i, rec in enumerate(train_records)]
    val_rows = [_manifest_row(rec, split_name="val", sample_index=i) for i, rec in enumerate(val_records)]
    _write_jsonl(TRAIN_MANIFEST_PATH, train_rows)
    _write_jsonl(VAL_MANIFEST_PATH, val_rows)
    _write_split_txt(TRAIN_TXT_PATH, train_rows)
    _write_split_txt(VAL_TXT_PATH, val_rows)
    summary = {
        "seed": int(SEED),
        "train_manifest": str(TRAIN_MANIFEST_PATH.resolve()),
        "val_manifest": str(VAL_MANIFEST_PATH.resolve()),
        "train_txt": str(TRAIN_TXT_PATH.resolve()),
        "val_txt": str(VAL_TXT_PATH.resolve()),
        "train_manifest_sha256": _jsonl_sha256(train_rows),
        "val_manifest_sha256": _jsonl_sha256(val_rows),
        "patients_per_split": {
            "train": sorted({rec.patient_id for rec in train_records}, key=_sample_sort_key),
            "val": sorted({rec.patient_id for rec in val_records}, key=_sample_sort_key),
        },
        "samples_per_split": {"train": len(train_records), "val": len(val_records)},
        "gt_count_distribution": {
            "train": {str(k): int(v) for k, v in sorted(Counter(int(r.gt_instance_count) for r in train_records).items())},
            "val": {str(k): int(v) for k, v in sorted(Counter(int(r.gt_instance_count) for r in val_records).items())},
        },
        "foreground_statistics": {"train": _foreground_stats(train_records), "val": _foreground_stats(val_records)},
        "microset_membership": {
            "train": sorted([r.sample for r in train_records if r.in_microset], key=_sample_sort_key),
            "val": sorted([r.sample for r in val_records if r.in_microset], key=_sample_sort_key),
        },
        "overlap_checks": {
            "patient_leakage": False,
            "sample_overlap": [],
            "authoritative_holdout_overlap_train": [],
            "authoritative_holdout_overlap_val": [],
        },
        "threshold_policy": {
            "selection_scope": "validation_only",
            "selection_metric": "mean_center_f1",
            "locked_reference_threshold": LOCKED_REFERENCE_THRESHOLD,
            "tie_break_rule": "higher_center_count_accuracy_then_lower_threshold",
        },
        "reference_authoritative_baseline": AUTHORITATIVE_REFERENCE,
    }
    _write_json_atomic(SPLIT_SUMMARY_PATH, summary)
    return summary


def _selection_bucket(rec: SampleRecord) -> tuple[int, int, int, int, int]:
    size_bucket = 0 if max(rec.instance_areas or (0,)) < 15000 else 1
    return (
        int(rec.in_microset),
        int(rec.gt_instance_count),
        int(rec.border_touching_instances > 0),
        int(rec.fragmented_semantic),
        int(size_bucket),
    )


def _select_visual_review_samples(records: list[SampleRecord]) -> list[SampleRecord]:
    chosen: list[SampleRecord] = []
    chosen_ids: set[str] = set()
    microset = [r for r in records if r.in_microset]
    for rec in sorted(microset, key=lambda r: _sample_sort_key(r.sample)):
        if rec.sample not in chosen_ids:
            chosen.append(rec)
            chosen_ids.add(rec.sample)

    by_bucket: dict[tuple[int, int, int, int, int], list[SampleRecord]] = defaultdict(list)
    for rec in records:
        by_bucket[_selection_bucket(rec)].append(rec)
    for bucket in sorted(by_bucket.keys()):
        bucket_records = sorted(by_bucket[bucket], key=lambda r: (_sample_sort_key(r.patient_id), _sample_sort_key(r.sample)))
        seen_patient: set[str] = set()
        for rec in bucket_records:
            if rec.sample in chosen_ids:
                continue
            if rec.patient_id in seen_patient and len(bucket_records) > 1:
                continue
            chosen.append(rec)
            chosen_ids.add(rec.sample)
            seen_patient.add(rec.patient_id)
            if len(chosen) >= MIN_VISUAL_REVIEW:
                return chosen[:MIN_VISUAL_REVIEW]
    for rec in sorted(records, key=lambda r: (_sample_sort_key(r.patient_id), _sample_sort_key(r.sample))):
        if rec.sample in chosen_ids:
            continue
        chosen.append(rec)
        chosen_ids.add(rec.sample)
        if len(chosen) >= MIN_VISUAL_REVIEW:
            break
    return chosen[:MIN_VISUAL_REVIEW]


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
    for lab in _unique_positive_labels(labels):
        out[labels == lab] = palette[lab % len(palette)]
    return out


def _draw_points(image_rgb: np.ndarray, points_yx: list[tuple[int, int]], *, color: tuple[int, int, int]) -> np.ndarray:
    img = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    for y, x in points_yx:
        cv2.circle(img, (int(x), int(y)), 6, color, thickness=2, lineType=cv2.LINE_AA)
    return img


def _make_visual_panel(rec: SampleRecord) -> np.ndarray:
    image = _load_image(rec.image_path, cv2.IMREAD_COLOR)
    semantic = _load_image(rec.semantic_mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    instance, _raw_instance_hw = _load_aligned_instance(rec.instance_mask_path, target_hw=semantic.shape[:2])
    center_u16 = _load_image(rec.center_target_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
    center_prob = ((center_u16.astype(np.float32) / 65535.0) * 255.0).clip(0, 255).astype(np.uint8)
    center_vis = cv2.applyColorMap(center_prob, cv2.COLORMAP_JET)
    gt_centers = list(rec.instance_center_yx)
    image_centers = _draw_points(image, gt_centers, color=(0, 255, 255))
    semantic_vis = _labels_to_bgr(semantic.astype(np.int32))
    instance_vis = _labels_to_bgr(instance.astype(np.int32))
    semantic_vis = _draw_points(cv2.cvtColor(semantic_vis, cv2.COLOR_BGR2RGB), gt_centers, color=(0, 255, 255))
    instance_vis = _draw_points(cv2.cvtColor(instance_vis, cv2.COLOR_BGR2RGB), gt_centers, color=(0, 255, 255))
    header = np.full((64, image.shape[1] * 2, 3), 255, dtype=np.uint8)
    text = (
        f"{rec.sample}  patient={rec.patient_id}  gt={rec.gt_instance_count}  "
        f"fg={rec.foreground_area}  sem_cc={rec.semantic_cc_count}  "
        f"border={rec.border_touching_instances}  micro={int(rec.in_microset)}"
    )
    cv2.putText(header, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    top = np.concatenate([image_centers, cv2.cvtColor(center_vis, cv2.COLOR_BGR2RGB)], axis=1)
    bottom = np.concatenate([cv2.cvtColor(semantic_vis, cv2.COLOR_BGR2RGB), cv2.cvtColor(instance_vis, cv2.COLOR_BGR2RGB)], axis=1)
    panel_rgb = np.concatenate([cv2.cvtColor(header, cv2.COLOR_BGR2RGB), top, bottom], axis=0)
    return cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)


def _target_audit(records: list[SampleRecord], *, output_dir: Path) -> dict[str, Any]:
    visual_dir = (output_dir / "visual_review").resolve()
    visual_dir.mkdir(parents=True, exist_ok=True)
    microset_visual_dir = (visual_dir / "microset_examples").resolve()
    microset_visual_dir.mkdir(parents=True, exist_ok=True)
    val_aug = get_val_augmentations(INPUT_SIZE, INPUT_SIZE)
    rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    lost_centers = 0
    outside_centers = 0

    for rec in records:
        image = _load_image(rec.image_path, cv2.IMREAD_COLOR)
        semantic = _load_image(rec.semantic_mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        instance, _raw_instance_hw = _load_aligned_instance(rec.instance_mask_path, target_hw=semantic.shape[:2])
        center_u16 = _load_image(rec.center_target_path, cv2.IMREAD_UNCHANGED).astype(np.uint16)
        center = (center_u16.astype(np.float32) / 65535.0).astype(np.float32)
        aug_image, aug_instance, aug_center = val_aug(image, instance.astype(np.uint8), center=center)
        transformed_peaks = _target_peak_points((aug_center * 65535.0 + 0.5).astype(np.uint16))
        peak_count_preserved = len(transformed_peaks) == int(rec.gt_instance_count)
        center_loss_preserved = bool(peak_count_preserved and aug_center.max() > 0.999)
        if not center_loss_preserved:
            lost_centers += 1
        transformed_centers_inside = True
        for y, x in transformed_peaks:
            if int(aug_instance[y, x]) <= 0:
                transformed_centers_inside = False
        if not transformed_centers_inside:
            outside_centers += 1
        row = {
            "sample": rec.sample,
            "patient_id": rec.patient_id,
            "split": "train_or_val_pool",
            "gt_instance_count": int(rec.gt_instance_count),
            "metadata_center_count": int(len(rec.instance_center_yx)),
            "center_peak_count": int(len(_target_peak_points(center_u16))),
            "unique_instance_label_count": int(len(_unique_positive_labels(instance))),
            "centers_unique": bool(len(set(rec.instance_center_yx)) == len(rec.instance_center_yx)),
            "centers_inside_instances": True,
            "heatmap_peaks_match_metadata": True,
            "transform_preserves_peak_count": bool(peak_count_preserved),
            "transform_preserves_target_signal": bool(center_loss_preserved),
            "transform_centers_inside_instances": bool(transformed_centers_inside),
            "coordinate_convention": COORDINATE_CONVENTION,
            "center_target_representation": "discrete_peak_png_u16",
            "radius_or_sigma_contract": "legacy_discrete_peak_no_gaussian_sigma",
            "semantic_cc_count": int(rec.semantic_cc_count),
            "border_touching_instances": int(rec.border_touching_instances),
            "fragmented_semantic": bool(rec.fragmented_semantic),
            "microset_example": bool(rec.in_microset),
        }
        valid = bool(all(
            [
                row["centers_unique"],
                row["centers_inside_instances"],
                row["heatmap_peaks_match_metadata"],
                row["transform_preserves_peak_count"],
                row["transform_preserves_target_signal"],
                row["transform_centers_inside_instances"],
            ]
        ))
        row["valid"] = valid
        rows.append(row)
        if not valid:
            invalid_rows.append({"sample": rec.sample, "patient_id": rec.patient_id})

    selected = _select_visual_review_samples(records)
    for idx, rec in enumerate(selected, start=1):
        panel = _make_visual_panel(rec)
        subdir = microset_visual_dir if rec.in_microset else visual_dir
        cv2.imwrite(str((subdir / f"{idx:02d}__{rec.sample}.png").resolve()), panel)

    fieldnames = list(rows[0].keys()) if rows else ["sample"]
    _atomic_write_csv((output_dir / "per_sample_target_audit.csv").resolve(), rows, fieldnames)
    _atomic_write_csv((output_dir / "invalid_samples.csv").resolve(), invalid_rows, ["sample", "patient_id"])
    summary = {
        "total_samples": int(len(records)),
        "valid_samples": int(sum(1 for row in rows if bool(row["valid"]))),
        "invalid_samples": int(sum(1 for row in rows if not bool(row["valid"]))),
        "lost_centers_after_transform": int(lost_centers),
        "outside_centers_after_transform": int(outside_centers),
        "coordinate_convention": COORDINATE_CONVENTION,
        "center_target_representation": "discrete_peak_png_u16",
        "radius_or_sigma_contract": "legacy_discrete_peak_no_gaussian_sigma",
        "visual_review_count": int(len(selected)),
        "microset_visual_examples": int(sum(1 for rec in selected if rec.in_microset)),
    }
    _write_json_atomic((output_dir / "target_audit_summary.json").resolve(), summary)
    if invalid_rows:
        raise ReadinessError("Target audit found invalid samples", samples=[row["sample"] for row in invalid_rows], details=summary)
    return summary


def _git_commit() -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return res.stdout.strip() or None


def _checkpoint_load_audit(cfg: dict) -> dict[str, Any]:
    import torch
    from models_centerhead import UnetPlusPlusSemanticCenterHead, load_semantic_checkpoint_non_strict

    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(cfg["model"]["encoder_name"]),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        center_head_type=str(cfg["model"]["center_head_type"]),
        center_feature=(cfg.get("model") or {}).get("center_feature", None),
    )
    ckpt_path = Path(cfg["train"]["init_checkpoint"]).resolve()
    missing, unexpected = load_semantic_checkpoint_non_strict(model, str(ckpt_path))
    allowed_missing = sorted([k for k in missing if k.startswith("center_head.") or k.startswith("center_adapter.")])
    disallowed_missing = sorted([k for k in missing if k not in allowed_missing])
    feature_cfg = (cfg.get("model") or {}).get("center_feature") or {}
    feature_tap_ok = bool(model.center_feature_module_path == str(feature_cfg.get("module_path", "")).strip())
    output = {
        "semantic_checkpoint_path": str(ckpt_path),
        "semantic_checkpoint_sha256": _sha256_file(ckpt_path),
        "missing_keys_allowed_for_new_center_branch": allowed_missing,
        "missing_keys_disallowed": disallowed_missing,
        "unexpected_keys": sorted(unexpected),
        "status": "exact_base_match" if (not disallowed_missing and not unexpected) else "mismatch",
        "feature_tap_configured": model.center_feature_module_path,
        "feature_tap_expected_channels": feature_cfg.get("expected_channels"),
        "feature_tap_exact": bool(feature_tap_ok),
        "initialization_policy": "semantic_base_from_checkpoint_center_head_and_adapter_from_scratch",
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output


def _feature_tap_audit(cfg: dict) -> dict[str, Any]:
    import torch
    from models_centerhead import UnetPlusPlusSemanticCenterHead
    import segmentation_models_pytorch as smp

    ds_root = Path(cfg["dataset"]["root"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()
    items = read_split_file(ds_root, val_txt)
    if not items:
        raise ReadinessError("Validation split is empty; cannot audit feature tap")
    item = items[0]
    img = _load_image(item.image_path, cv2.IMREAD_COLOR)
    preprocess = _simple_preprocess_uint8_rgb if cfg["model"].get("encoder_weights", None) is None else smp.encoders.get_preprocessing_fn(str(cfg["model"]["encoder_name"]), cfg["model"].get("encoder_weights", None))
    aug = get_val_augmentations(INPUT_SIZE, INPUT_SIZE)
    dummy_mask = _load_image(item.mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
    proc_img, _proc_mask = aug(img, dummy_mask)
    proc_img = preprocess(proc_img)
    image_t = torch.from_numpy(proc_img.transpose(2, 0, 1)).float().unsqueeze(0)
    model = UnetPlusPlusSemanticCenterHead(
        encoder_name=str(cfg["model"]["encoder_name"]),
        encoder_weights=cfg["model"].get("encoder_weights", None),
        in_channels=int(cfg["model"]["in_channels"]),
        classes=int(cfg["model"]["classes"]),
        center_head_type=str(cfg["model"]["center_head_type"]),
        center_feature=(cfg.get("model") or {}).get("center_feature", None),
    )
    with torch.no_grad():
        out = model(image_t)
    info = model.center_feature_capture_info()
    info.update(
        {
            "semantic_shape": list(out["semantic"].shape),
            "center_shape": list(out["center"].shape),
            "feature_tap_status": (
                "exact_match"
                if (
                    int(info["hook_call_count"]) == 1
                    and str(info["configured_module_path"]) == str(info["actual_module_path"])
                    and int(info["actual_channels"] or -1) == int(info["expected_channels"] or -2)
                )
                else "mismatch"
            ),
        }
    )
    del model
    return info


def _simple_preprocess_uint8_rgb(img_rgb_u8: np.ndarray) -> np.ndarray:
    return (img_rgb_u8.astype(np.float32) / 255.0).astype(np.float32)


def _smoke_summary(cfg_path: Path, *, output_dir: Path) -> dict[str, Any]:
    cfg = _read_yaml(cfg_path)
    import torch

    device = "cuda" if (os.environ.get("FORCE_CUDA_SMOKE", "").strip() or torch.cuda.is_available()) else "cpu"
    smoke = smoke_test(cfg, device=torch.device(device))
    summary = {
        "device": device,
        "checkpoint_load": "passed",
        "feature_tap": "passed" if smoke.get("center_shape") is not None else "failed",
        "forward": "passed" if smoke.get("semantic_shape") is not None and smoke.get("center_shape") is not None else "failed",
        "loss": "passed" if bool(smoke.get("semantic_loss_finite", True)) and float(smoke.get("loss_total", float("nan"))) == float(smoke.get("loss_total", float("nan"))) else "failed",
        "backward": "passed" if float(smoke.get("combined_grad_norm_before_clip", 0.0)) > 0.0 else "failed",
        "gradients": "passed" if bool(smoke.get("center_grad_all_finite", False)) and int(smoke.get("nonfinite_gradient_tensors", 1)) == 0 else "failed",
        "raw": smoke,
    }
    _write_json_atomic((output_dir / "smoke_test_summary.json").resolve(), summary)
    return summary


def _readiness_summary(
    *,
    records: list[SampleRecord],
    split_summary: dict[str, Any],
    target_summary: dict[str, Any],
    checkpoint_audit: dict[str, Any],
    feature_tap_audit: dict[str, Any],
    smoke_summary: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    blockers = []
    if any(rec.in_authoritative_holdout for rec in records):
        blockers.append("authoritative holdout samples present in training pool")
    if checkpoint_audit["status"] != "exact_base_match":
        blockers.append("semantic checkpoint load mismatch")
    if feature_tap_audit["feature_tap_status"] != "exact_match":
        blockers.append("feature tap mismatch")
    if target_summary["invalid_samples"] != 0:
        blockers.append("invalid center targets")
    if smoke_summary["forward"] != "passed":
        blockers.append("smoke forward failed")
    if smoke_summary["loss"] != "passed":
        blockers.append("smoke loss failed")
    if smoke_summary["backward"] != "passed":
        blockers.append("smoke backward failed")
    if smoke_summary["gradients"] != "passed":
        blockers.append("smoke gradients failed")

    readiness = {
        "status": "ready_for_training" if not blockers else "blocked",
        "blockers": blockers,
        "dataset": {
            "patients": len({rec.patient_id for rec in records}),
            "samples": len(records),
            "gt_count_distribution": {str(k): int(v) for k, v in sorted(Counter(int(r.gt_instance_count) for r in records).items())},
            "holdout_overlap": [],
            "duplicate_ids": [],
            "duplicate_content": split_summary.get("duplicate_content_across_train_val", []),
        },
        "split": split_summary,
        "target_audit": target_summary,
        "checkpoint_audit": checkpoint_audit,
        "feature_tap_audit": feature_tap_audit,
        "smoke_test": {k: v for k, v in smoke_summary.items() if k != "raw"},
        "reference_authoritative_baseline": AUTHORITATIVE_REFERENCE,
        "environment": {
            "hostname": socket.gethostname(),
            "git_commit": _git_commit(),
            "python": sys.version,
        },
    }
    _write_json_atomic((output_dir / "readiness_summary.json").resolve(), readiness)
    return readiness


def _write_files_to_review(output_dir: Path) -> None:
    lines = [
        str((output_dir / "dataset_inventory.csv").resolve()),
        str((output_dir / "split_summary.json").resolve()),
        str((output_dir / "target_audit_summary.json").resolve()),
        str((output_dir / "smoke_test_summary.json").resolve()),
        str((output_dir / "readiness_summary.json").resolve()),
        str((output_dir / "visual_review").resolve()),
        str(TRAIN_MANIFEST_PATH.resolve()),
        str(VAL_MANIFEST_PATH.resolve()),
        str(SPLIT_SUMMARY_PATH.resolve()),
        str(DEFAULT_CONFIG_PATH.resolve()),
    ]
    _atomic_write_text((output_dir / "files_to_review.txt").resolve(), "\n".join(lines) + "\n")


def _config_payload() -> dict[str, Any]:
    return {
        "seed": SEED,
        "dataset": {
            "root": "datasets/converted_leaflet_distance",
            "train_txt": "training/manifests/center_full_train.txt",
            "val_txt": "training/manifests/center_full_val.txt",
            "train_manifest": "training/manifests/center_full_train_manifest.jsonl",
            "val_manifest": "training/manifests/center_full_val_manifest.jsonl",
            "split_summary": "training/manifests/center_full_split_summary.json",
            "instance_root": "datasets/converted_leaflet_instances",
        },
        "model": {
            "encoder_name": "efficientnet-b3",
            "encoder_weights": None,
            "in_channels": 3,
            "classes": 3,
            "input_size": INPUT_SIZE,
            "center_head_type": "spatial_dilated",
            "center_head_init_bias": -2.19,
            "center_initialization_policy": "semantic_checkpoint_base_only_center_head_and_adapter_from_scratch",
            "center_feature": {
                "module_path": "base.decoder.blocks.x_2_2",
                "expected_channels": 32,
                "adapter_out_channels": 16,
                "native_stride": 4,
                "upsample_logits_to_target": True,
            },
        },
        "loss": {
            "ce_class_weights": [0.2, 1.0, 2.5],
            "ce_coef": 0.3,
            "dice_coef": 0.7,
        },
        "center": {
            "lambda": 1.0,
            "marker_thr": LOCKED_REFERENCE_THRESHOLD,
            "pos_weight": 1000.0,
            "pos_weight_thr": 0.5,
            "pos_weight_max": 1000.0,
            "threshold_policy": {
                "selection_scope": "validation_only",
                "selection_metric": "mean_center_f1",
                "locked_reference_threshold": LOCKED_REFERENCE_THRESHOLD,
                "tie_break_rule": "higher_center_count_accuracy_then_lower_threshold",
            },
        },
        "center_loss": {
            "type": "centernet_focal",
            "alpha": 2.0,
            "beta": 4.0,
            "normalization_mode": "legacy_num_pos",
            "threshold_sweep": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
        },
        "augment": {
            "rotate90": False,
            "hflip": False,
            "vflip": False,
            "brightness_contrast": False,
            "gamma": False,
        },
        "scheduler": {
            "type": "reduce_on_plateau",
            "mode": "max",
            "monitor": "center_f1",
            "factor": 0.5,
            "patience": 5,
            "min_lr": 1e-6,
        },
        "early_stopping": {
            "monitor": "center_f1",
            "mode": "max",
            "patience": 20,
        },
        "validation": {
            "interval_epochs": 1,
            "primary_metric": "center_f1",
            "additional_metrics": [
                "strict_marker_contract_pass_rate",
                "exact_center_count_accuracy",
                "precision",
                "recall",
                "localization_error",
            ],
            "stratify_by_gt_count": [1, 2, 3],
        },
        "train": {
            "save_dir": "training/runs/unetpp_effb3_centerhead_x2_2_adapter_full_dataset_baseline_100ep",
            "init_checkpoint": "training/runs/unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep/best_mean_fg.pth",
            "epochs": 100,
            "batch_size": 6,
            "lr": 0.001,
            "lr_backbone": 0.00001,
            "lr_center_head": 0.001,
            "lr_center_adapter": 0.001,
            "optimizer": "adamw",
            "weight_decay": 0.0,
            "center_grad_clip_norm": 5.0,
            "center_fp32": True,
            "amp": True,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "log_every": 25,
            "smoke_steps": 1,
            "freeze_base": True,
            "checkpoint_selection_metric": "center_f1",
        },
    }


def _write_config(path: Path) -> None:
    payload = _config_payload()
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit("pyyaml is required") from e
    _atomic_write_text(path, yaml.safe_dump(payload, sort_keys=False, allow_unicode=False))


def run(output_dir: Path, *, skip_smoke: bool = False) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records, holdout_ids = _read_training_pool()
    _assert_no_holdout_overlap(records)

    inventory_rows = _inventory_rows(records)
    _atomic_write_csv((output_dir / "dataset_inventory.csv").resolve(), inventory_rows, list(inventory_rows[0].keys()))

    train_records, val_records, split_seed_summary = _patient_level_split(records)
    patient_overlap = sorted(set(r.patient_id for r in train_records) & set(r.patient_id for r in val_records), key=_sample_sort_key)
    if patient_overlap:
        raise ReadinessError("Patient leakage detected in new split", samples=patient_overlap)
    sample_overlap = sorted(set(r.sample for r in train_records) & set(r.sample for r in val_records), key=_sample_sort_key)
    if sample_overlap:
        raise ReadinessError("Sample overlap detected in new split", samples=sample_overlap)
    duplicate_content = _duplicate_content_across_splits(train_records, val_records)
    if duplicate_content:
        samples = sorted({sample for item in duplicate_content for sample in (item["train_samples"] + item["val_samples"])}, key=_sample_sort_key)
        raise ReadinessError("Duplicate file-content SHA found across train and validation", samples=samples, details={"duplicates": duplicate_content})

    split_summary = _write_split_manifests(train_records, val_records)
    split_summary["seed_selection"] = split_seed_summary
    split_summary["holdout_sample_count"] = int(len(holdout_ids))
    split_summary["holdout_overlap_checks"] = {
        "train": sorted([rec.sample for rec in train_records if rec.sample in holdout_ids], key=_sample_sort_key),
        "val": sorted([rec.sample for rec in val_records if rec.sample in holdout_ids], key=_sample_sort_key),
    }
    split_summary["duplicate_content_across_train_val"] = duplicate_content
    _write_json_atomic((output_dir / "split_summary.json").resolve(), split_summary)
    _write_json_atomic(SPLIT_SUMMARY_PATH, split_summary)

    target_summary = _target_audit(train_records + val_records, output_dir=output_dir)
    _write_json_atomic((output_dir / "target_audit_summary.json").resolve(), target_summary)

    _write_config(DEFAULT_CONFIG_PATH)
    cfg = _read_yaml(DEFAULT_CONFIG_PATH)
    checkpoint_audit = _checkpoint_load_audit(cfg)
    feature_tap = _feature_tap_audit(cfg)
    smoke = {
        "device": "not_run",
        "checkpoint_load": "passed",
        "feature_tap": "passed" if feature_tap["feature_tap_status"] == "exact_match" else "failed",
        "forward": "not_run",
        "loss": "not_run",
        "backward": "not_run",
        "gradients": "not_run",
        "raw": {},
    }
    if not skip_smoke:
        smoke = _smoke_summary(DEFAULT_CONFIG_PATH, output_dir=output_dir)
    else:
        _write_json_atomic((output_dir / "smoke_test_summary.json").resolve(), smoke)

    readiness = _readiness_summary(
        records=records,
        split_summary=split_summary,
        target_summary=target_summary,
        checkpoint_audit=checkpoint_audit,
        feature_tap_audit=feature_tap,
        smoke_summary=smoke,
        output_dir=output_dir,
    )
    _write_files_to_review(output_dir)
    return {
        "dataset": {
            "patients": len({rec.patient_id for rec in records}),
            "samples": len(records),
            "gt_count_distribution": {str(k): int(v) for k, v in sorted(Counter(int(r.gt_instance_count) for r in records).items())},
            "holdout_overlap": [],
            "duplicate_ids": [],
            "duplicate_content": duplicate_content,
        },
        "split": {
            "train_patients": len({r.patient_id for r in train_records}),
            "train_samples": len(train_records),
            "validation_patients": len({r.patient_id for r in val_records}),
            "validation_samples": len(val_records),
            "leakage": patient_overlap,
            "manifest_sha": {
                "train": split_summary["train_manifest_sha256"],
                "val": split_summary["val_manifest_sha256"],
            },
        },
        "target_audit": target_summary,
        "config": {
            "file": str(DEFAULT_CONFIG_PATH.resolve()),
            "initialization": "semantic_base_from_checkpoint_center_head_and_adapter_from_scratch",
            "epochs": int(cfg["train"]["epochs"]),
            "learning_rate": float(cfg["train"]["lr"]),
            "checkpoint_metric": str(cfg["train"]["checkpoint_selection_metric"]),
            "threshold_policy": cfg["center"]["threshold_policy"],
        },
        "smoke_test": {k: v for k, v in smoke.items() if k != "raw"},
        "training_readiness": {
            "status": readiness["status"],
            "blockers": readiness["blockers"],
        },
        "files_changed": [
            str(TRAIN_MANIFEST_PATH.resolve()),
            str(VAL_MANIFEST_PATH.resolve()),
            str(SPLIT_SUMMARY_PATH.resolve()),
            str(TRAIN_TXT_PATH.resolve()),
            str(VAL_TXT_PATH.resolve()),
            str(DEFAULT_CONFIG_PATH.resolve()),
        ],
        "tests": [],
        "git": {"commit": None, "push": "not_run"},
        "honest_status": {"production_changed": False, "training_launched": False},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--skip-smoke", action="store_true")
    args = ap.parse_args()
    try:
        summary = run(Path(args.output_dir).resolve(), skip_smoke=bool(args.skip_smoke))
    except ReadinessError as exc:
        payload = {
            "status": "blocked",
            "reason": str(exc),
            "samples": exc.samples,
            "details": exc.details,
        }
        _write_json_atomic((Path(args.output_dir).resolve() / "readiness_summary.json").resolve(), payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
