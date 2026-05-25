from __future__ import annotations

import csv
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e
from tqdm import tqdm

from metrics import compute_per_class_metrics_from_logits


def _components_areas(mask01) -> list[int]:
    import cv2
    import numpy as np

    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return []
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
    areas.sort(reverse=True)
    return areas


def _count_holes(mask01) -> int:
    import cv2
    import numpy as np

    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return 0
    contours, hierarchy = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None or len(contours) == 0:
        return 0
    holes = 0
    for i in range(int(len(contours))):
        parent = int(hierarchy[0][i][3])
        if parent != -1:
            holes += 1
    return int(holes)


def _dice_iou_binary(gt01, pred01) -> tuple[float, float]:
    import numpy as np

    gt = gt01.astype(bool)
    pr = pred01.astype(bool)
    inter = int(np.logical_and(gt, pr).sum())
    gt_sum = int(gt.sum())
    pr_sum = int(pr.sum())
    denom_dice = gt_sum + pr_sum
    if denom_dice == 0:
        return 1.0, 1.0
    dice = float(2.0 * inter / denom_dice)
    union = gt_sum + pr_sum - inter
    iou = float(inter / union) if union > 0 else 1.0
    return dice, iou


def _tiny_components_count(areas_desc: list[int], *, rel_thresh: float = 0.1) -> int:
    if not areas_desc:
        return 0
    largest = float(areas_desc[0])
    if largest <= 0:
        return 0
    thresh = float(rel_thresh) * largest
    return int(sum(1 for a in areas_desc[1:] if float(a) < thresh))


def _ensure_csv_header(path: Path, header: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader,
    num_classes: int,
    device: torch.device,
    loss_fn: torch.nn.Module,
    shape_diagnostics: bool = False,
    shape_out_dir: Path | None = None,
    epoch: int | None = None,
) -> dict:
    model.eval()

    total_loss = 0.0
    n_batches = 0
    dice_sum = [0.0 for _ in range(num_classes)]
    iou_sum = [0.0 for _ in range(num_classes)]

    shape_rows: list[list] = []
    merged_suspect_count = 0
    extra_fragments_count = 0
    pred_components_sum = 0.0
    gt_components_sum = 0.0
    shape_samples = 0
    gt_leaflet_components_sum = 0.0
    pred_leaflet_components_sum = 0.0
    merged_leaflet_suspect_count = 0
    extra_leaflet_fragments_count = 0
    gt_ring_components_sum = 0.0
    pred_ring_components_sum = 0.0
    disconnected_ring_suspect_count = 0
    extra_ring_fragments_count = 0

    for batch in tqdm(loader, desc="Validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        boundary = batch.get("boundary", None)
        if boundary is not None:
            boundary = boundary.to(device, non_blocking=True)

        logits = model(images)
        loss = loss_fn(logits, masks, boundary_target=boundary)
        total_loss += float(loss.item())
        n_batches += 1

        logits_main = logits[0] if isinstance(logits, (list, tuple)) else logits

        m = compute_per_class_metrics_from_logits(logits_main, masks, num_classes=num_classes)
        for i in range(num_classes):
            dice_sum[i] += float(m.dice[i])
            iou_sum[i] += float(m.iou[i])

        if shape_diagnostics and int(num_classes) == 2:
            import numpy as np

            pred = torch.argmax(logits_main, dim=1).detach().cpu().numpy().astype(np.uint8)
            gt = masks.detach().cpu().numpy().astype(np.uint8)
            image_paths = batch.get("image_path", None)
            if not isinstance(image_paths, list):
                image_paths = [None for _ in range(int(pred.shape[0]))]

            for i in range(int(pred.shape[0])):
                sample_path = image_paths[i] if i < len(image_paths) else None
                sample_id = Path(sample_path).stem if isinstance(sample_path, str) else f"sample_{shape_samples}"

                gt01 = (gt[i] == 1).astype(np.uint8)
                pr01 = (pred[i] == 1).astype(np.uint8)

                gt_areas = _components_areas(gt01)
                pr_areas = _components_areas(pr01)
                gt_components = len(gt_areas)
                pred_components = len(pr_areas)

                gt_largest = int(gt_areas[0]) if gt_areas else 0
                pred_largest = int(pr_areas[0]) if pr_areas else 0
                pred_second = int(pr_areas[1]) if len(pr_areas) > 1 else 0
                ratio = (float(pred_largest) / float(pred_second)) if pred_second > 0 else None

                small_count = 0
                if pr_areas and pred_largest > 0:
                    thresh = 0.1 * float(pred_largest)
                    small_count = sum(1 for a in pr_areas[1:] if float(a) < thresh)

                merged_suspect = False
                if pred_components < gt_components:
                    merged_suspect = True
                if pred_components == 1 and gt_components >= 2:
                    merged_suspect = True
                if gt_components >= 2 and ratio is not None and ratio > 2.0:
                    merged_suspect = True

                extra_fragments = False
                if pred_components > 3:
                    extra_fragments = True
                if small_count > 0:
                    extra_fragments = True

                dice_leaflet, iou_leaflet = _dice_iou_binary(gt01, pr01)

                shape_rows.append(
                    [
                        int(epoch) if epoch is not None else None,
                        sample_id,
                        gt_components,
                        pred_components,
                        gt_largest,
                        pred_largest,
                        pred_second,
                        ratio,
                        small_count,
                        int(merged_suspect),
                        int(extra_fragments),
                        float(dice_leaflet),
                        float(iou_leaflet),
                    ]
                )

                merged_suspect_count += int(merged_suspect)
                extra_fragments_count += int(extra_fragments)
                pred_components_sum += float(pred_components)
                gt_components_sum += float(gt_components)
                shape_samples += 1
        elif shape_diagnostics and int(num_classes) == 3:
            import numpy as np

            pred = torch.argmax(logits_main, dim=1).detach().cpu().numpy().astype(np.uint8)
            gt = masks.detach().cpu().numpy().astype(np.uint8)
            image_paths = batch.get("image_path", None)
            if not isinstance(image_paths, list):
                image_paths = [None for _ in range(int(pred.shape[0]))]

            for i in range(int(pred.shape[0])):
                sample_path = image_paths[i] if i < len(image_paths) else None
                sample_id = Path(sample_path).stem if isinstance(sample_path, str) else f"sample_{shape_samples}"

                gt_leaflet01 = (gt[i] == 1).astype(np.uint8)
                pr_leaflet01 = (pred[i] == 1).astype(np.uint8)
                gt_ring01 = (gt[i] == 2).astype(np.uint8)
                pr_ring01 = (pred[i] == 2).astype(np.uint8)

                gt_leaf_areas = _components_areas(gt_leaflet01)
                pr_leaf_areas = _components_areas(pr_leaflet01)
                gt_leaf_c = int(len(gt_leaf_areas))
                pr_leaf_c = int(len(pr_leaf_areas))

                gt_ring_areas = _components_areas(gt_ring01)
                pr_ring_areas = _components_areas(pr_ring01)
                gt_ring_c = int(len(gt_ring_areas))
                pr_ring_c = int(len(pr_ring_areas))

                tiny_leaflet = _tiny_components_count(pr_leaf_areas, rel_thresh=0.1)
                tiny_ring = _tiny_components_count(pr_ring_areas, rel_thresh=0.1)

                merged_leaflet_suspect = False
                if pr_leaf_c < gt_leaf_c:
                    merged_leaflet_suspect = True
                if pr_leaf_c == 1 and gt_leaf_c >= 2:
                    merged_leaflet_suspect = True
                if gt_leaf_c >= 2 and len(pr_leaf_areas) > 1:
                    largest = float(pr_leaf_areas[0])
                    second = float(pr_leaf_areas[1])
                    if second > 0 and (largest / second) > 2.0:
                        merged_leaflet_suspect = True

                extra_leaflet_fragments = False
                if pr_leaf_c > gt_leaf_c + 1:
                    extra_leaflet_fragments = True
                if pr_leaf_c > 3:
                    extra_leaflet_fragments = True
                if tiny_leaflet > 0:
                    extra_leaflet_fragments = True

                disconnected_ring_suspect = pr_ring_c > 1

                extra_ring_fragments = False
                if pr_ring_c > gt_ring_c + 1:
                    extra_ring_fragments = True
                if pr_ring_c > 2:
                    extra_ring_fragments = True
                if tiny_ring > 0:
                    extra_ring_fragments = True

                shape_rows.append(
                    [
                        int(epoch) if epoch is not None else None,
                        sample_id,
                        gt_leaf_c,
                        pr_leaf_c,
                        int(tiny_leaflet),
                        int(merged_leaflet_suspect),
                        int(extra_leaflet_fragments),
                        gt_ring_c,
                        pr_ring_c,
                        int(tiny_ring),
                        int(disconnected_ring_suspect),
                        int(extra_ring_fragments),
                    ]
                )

                gt_leaflet_components_sum += float(gt_leaf_c)
                pred_leaflet_components_sum += float(pr_leaf_c)
                merged_leaflet_suspect_count += int(merged_leaflet_suspect)
                extra_leaflet_fragments_count += int(extra_leaflet_fragments)

                gt_ring_components_sum += float(gt_ring_c)
                pred_ring_components_sum += float(pr_ring_c)
                disconnected_ring_suspect_count += int(disconnected_ring_suspect)
                extra_ring_fragments_count += int(extra_ring_fragments)

                shape_samples += 1

    if n_batches == 0:
        return {"loss": None, "dice": None, "iou": None}

    if shape_diagnostics and shape_out_dir is not None and int(num_classes) == 2 and epoch is not None:
        shape_out_dir = Path(shape_out_dir).resolve()
        per_sample_path = shape_out_dir / "shape_metrics.csv"
        per_epoch_path = shape_out_dir / "shape_epoch_metrics.csv"

        _ensure_csv_header(
            per_sample_path,
            [
                "epoch",
                "sample_id",
                "gt_components",
                "pred_components",
                "gt_largest_area",
                "pred_largest_area",
                "pred_second_area",
                "pred_largest_second_ratio",
                "pred_small_components",
                "merged_suspect",
                "extra_fragments",
                "dice_leaflet",
                "iou_leaflet",
            ],
        )
        _ensure_csv_header(
            per_epoch_path,
            [
                "epoch",
                "merged_suspect_count",
                "extra_fragments_count",
                "mean_pred_components",
                "mean_gt_components",
            ],
        )

        with per_sample_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for r in shape_rows:
                w.writerow(r)

        mean_pred_components = (pred_components_sum / shape_samples) if shape_samples else None
        mean_gt_components = (gt_components_sum / shape_samples) if shape_samples else None
        with per_epoch_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    int(epoch),
                    int(merged_suspect_count),
                    int(extra_fragments_count),
                    float(mean_pred_components) if mean_pred_components is not None else None,
                    float(mean_gt_components) if mean_gt_components is not None else None,
                ]
            )
    elif shape_diagnostics and shape_out_dir is not None and int(num_classes) == 3 and epoch is not None:
        shape_out_dir = Path(shape_out_dir).resolve()
        per_sample_path = shape_out_dir / "shape_metrics.csv"
        per_epoch_path = shape_out_dir / "shape_epoch_metrics.csv"

        _ensure_csv_header(
            per_sample_path,
            [
                "epoch",
                "sample_id",
                "gt_leaflet_components",
                "pred_leaflet_components",
                "tiny_leaflet_components",
                "merged_leaflet_suspect",
                "extra_leaflet_fragments",
                "gt_ring_components",
                "pred_ring_components",
                "tiny_ring_components",
                "disconnected_ring_suspect",
                "extra_ring_fragments",
            ],
        )
        _ensure_csv_header(
            per_epoch_path,
            [
                "epoch",
                "mean_pred_leaflet_components",
                "mean_gt_leaflet_components",
                "merged_leaflet_suspect_count",
                "extra_leaflet_fragments_count",
                "mean_pred_ring_components",
                "mean_gt_ring_components",
                "disconnected_ring_suspect_count",
                "extra_ring_fragments_count",
            ],
        )

        with per_sample_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            for r in shape_rows:
                w.writerow(r)

        mean_gt_leaflet_components = (gt_leaflet_components_sum / shape_samples) if shape_samples else None
        mean_pred_leaflet_components = (pred_leaflet_components_sum / shape_samples) if shape_samples else None
        mean_gt_ring_components = (gt_ring_components_sum / shape_samples) if shape_samples else None
        mean_pred_ring_components = (pred_ring_components_sum / shape_samples) if shape_samples else None
        with per_epoch_path.open("a", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    int(epoch),
                    float(mean_pred_leaflet_components) if mean_pred_leaflet_components is not None else None,
                    float(mean_gt_leaflet_components) if mean_gt_leaflet_components is not None else None,
                    int(merged_leaflet_suspect_count),
                    int(extra_leaflet_fragments_count),
                    float(mean_pred_ring_components) if mean_pred_ring_components is not None else None,
                    float(mean_gt_ring_components) if mean_gt_ring_components is not None else None,
                    int(disconnected_ring_suspect_count),
                    int(extra_ring_fragments_count),
                ]
            )

    return {
        "loss": total_loss / n_batches,
        "dice": [v / n_batches for v in dice_sum],
        "iou": [v / n_batches for v in iou_sum],
        "shape": {
            "merged_suspect_count": int(merged_suspect_count),
            "extra_fragments_count": int(extra_fragments_count),
            "mean_pred_components": (pred_components_sum / shape_samples) if shape_samples else None,
            "mean_gt_components": (gt_components_sum / shape_samples) if shape_samples else None,
        }
        if shape_diagnostics and int(num_classes) == 2
        else {
            "mean_pred_leaflet_components": (pred_leaflet_components_sum / shape_samples) if shape_samples else None,
            "mean_gt_leaflet_components": (gt_leaflet_components_sum / shape_samples) if shape_samples else None,
            "merged_leaflet_suspect_count": int(merged_leaflet_suspect_count),
            "extra_leaflet_fragments_count": int(extra_leaflet_fragments_count),
            "mean_pred_ring_components": (pred_ring_components_sum / shape_samples) if shape_samples else None,
            "mean_gt_ring_components": (gt_ring_components_sum / shape_samples) if shape_samples else None,
            "disconnected_ring_suspect_count": int(disconnected_ring_suspect_count),
            "extra_ring_fragments_count": int(extra_ring_fragments_count),
        }
        if shape_diagnostics and int(num_classes) == 3
        else None,
    }


def format_metrics(metrics: dict) -> str:
    if metrics.get("loss") is None:
        return "val: no batches"

    parts = [f"val_loss={metrics['loss']:.6f}"]
    dice = metrics.get("dice") or []
    iou = metrics.get("iou") or []
    for i, v in enumerate(dice):
        parts.append(f"dice_c{i}={v:.4f}")
    for i, v in enumerate(iou):
        parts.append(f"iou_c{i}={v:.4f}")
    return " ".join(parts)
