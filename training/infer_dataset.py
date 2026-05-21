from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e


def _read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_mask_uint8(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _center_crop_pair(image: np.ndarray, mask: np.ndarray, crop_h: int, crop_w: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        new_w = max(w, crop_w)
        new_h = max(h, crop_h)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        h, w = image.shape[:2]
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w], mask[y0 : y0 + crop_h, x0 : x0 + crop_w]


def _simple_preprocess_uint8_rgb(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) / 255.0


def _dice_for_class(gt: np.ndarray, pred: np.ndarray, class_id: int) -> float:
    gt_c = gt == int(class_id)
    pr_c = pred == int(class_id)
    inter = int(np.logical_and(gt_c, pr_c).sum())
    gt_sum = int(gt_c.sum())
    pr_sum = int(pr_c.sum())
    denom = gt_sum + pr_sum
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def _components_areas(mask01: np.ndarray) -> list[int]:
    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return []
    num, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(int).tolist()
    areas.sort(reverse=True)
    return areas


def _shape_warnings(gt01: np.ndarray, pred01: np.ndarray) -> tuple[bool, bool, int, float | None, list[int]]:
    gt_areas = _components_areas(gt01)
    pr_areas = _components_areas(pred01)
    gt_c = len(gt_areas)
    pr_c = len(pr_areas)

    largest = int(pr_areas[0]) if pr_areas else 0
    second = int(pr_areas[1]) if len(pr_areas) > 1 else 0
    ratio = (float(largest) / float(second)) if second > 0 else None

    small_count = 0
    if pr_areas and largest > 0:
        thresh = 0.1 * float(largest)
        small_count = sum(1 for a in pr_areas[1:] if float(a) < thresh)

    merged_suspect = False
    if pr_c < gt_c:
        merged_suspect = True
    if pr_c == 1 and gt_c >= 2:
        merged_suspect = True
    if gt_c >= 2 and ratio is not None and ratio > 2.0:
        merged_suspect = True

    extra_fragments = False
    if pr_c > 3:
        extra_fragments = True
    if small_count > 0:
        extra_fragments = True

    return merged_suspect, extra_fragments, small_count, ratio, pr_areas

def _overlay_rgb(image_rgb: np.ndarray, mask: np.ndarray, alpha: float, draw_ring: bool) -> np.ndarray:
def _count_holes(mask01: np.ndarray) -> int:
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


    base = image_rgb.astype(np.float32)
    base = image_rgb.astype(np.float32)
    base = image_rgb.astype(np.float32)
    overlay = base.copy()

    green = np.array([0, 255, 0], dtype=np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)

    leaflet = mask == 1
    ring = (mask == 2) if draw_ring else None
    if leaflet.any():
        overlay[leaflet] = (1.0 - alpha) * overlay[leaflet] + alpha * green
    if ring is not None and ring.any():
        overlay[ring] = (1.0 - alpha) * overlay[ring] + alpha * red

    return np.clip(overlay, 0, 255).astype(np.uint8)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(path)


def _read_split_pairs(dataset_root: Path, split_txt: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with split_txt.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise SystemExit(f"Invalid line in {split_txt}: {line!r}")
            img_rel, mask_rel = parts
            pairs.append(((dataset_root / img_rel).resolve(), (dataset_root / mask_rel).resolve()))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch inference on val split and save overlay previews.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt (checkpoint from training/train.py)")
    parser.add_argument("--out-dir", type=Path, default=Path("training/inference_preview"))
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--shape-report", action="store_true")
    args = parser.parse_args()

    import segmentation_models_pytorch as smp

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: CUDA is not available, using CPU.")
    print(f"Device: {device}")

    ckpt = torch.load(str(args.checkpoint), map_location="cpu")
    cfg = ckpt.get("config") if isinstance(ckpt, dict) else None
    if not isinstance(cfg, dict):
        raise SystemExit(f"Checkpoint does not contain config: {args.checkpoint}")

    dataset_root = Path(cfg["dataset"]["root"]).resolve()
    val_txt = Path(cfg["dataset"]["val_txt"]).resolve()
    input_size = int(cfg["model"]["input_size"])
    encoder = cfg["model"].get("encoder") or cfg["model"].get("encoder_name")
    encoder = str(encoder) if encoder is not None else None
    num_classes = int(cfg["model"]["classes"])
    encoder_weights = cfg["model"].get("encoder_weights", None)
    target = (cfg.get("dataset") or {}).get("target", None)
    target = str(target).strip().lower() if target is not None else "multiclass"
    leaflet_only = (num_classes == 2) or (target == "leaflet_only")

    if not encoder:
        raise SystemExit("Checkpoint config missing model.encoder (or model.encoder_name)")

    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=int(cfg["model"]["in_channels"]),
        classes=num_classes,
    )
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    if encoder_weights is None:
        preprocessing_fn = _simple_preprocess_uint8_rgb
    else:
        preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder, encoder_weights)

    pairs = _read_split_pairs(dataset_root, val_txt)
    print(f"Val samples: {len(pairs)}")
    print(f"Output dir: {args.out_dir.resolve()}")

    out_dir = args.out_dir.resolve()
    orig_dir = out_dir / "original"
    gt_dir = out_dir / "gt_overlay"
    pred_dir = out_dir / "pred_overlay"
    cmp_dir = out_dir / "compare"

    for image_path, mask_path in tqdm(pairs, desc="Infer val", unit="img"):
        stem = image_path.stem
        image = _read_image_rgb(image_path)
        gt = _read_mask_uint8(mask_path)

        if leaflet_only:
            gt = (gt == 1).astype(np.uint8)

        image, gt = _center_crop_pair(image, gt, input_size, input_size)
        x_np = preprocessing_fn(image)
        x = torch.from_numpy(x_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)

        dice_leaflet = _dice_for_class(gt, pred, 1)
        if leaflet_only:
            msg = f"{image_path.name} dice_leaflet={dice_leaflet:.4f}"
            if args.shape_report:
                merged_suspect, extra_fragments, small_count, ratio, areas = _shape_warnings(gt == 1, pred == 1)
                warns = []
                if merged_suspect:
                    warns.append("merged_suspect")
                if extra_fragments:
                    warns.append("extra_fragments")
                if small_count > 0:
                    warns.append(f"small_components={small_count}")
                if ratio is not None:
                    warns.append(f"largest/second={ratio:.2f}")
                warns.append(f"pred_components={len(areas)}")
                warns.append(f"areas={areas[:10]}")
                msg = msg + " " + " ".join(warns)
            print(msg)
        else:
            dice_ring = _dice_for_class(gt, pred, 2)
            msg = f"{image_path.name} dice_leaflet={dice_leaflet:.4f} dice_fibrous_ring={dice_ring:.4f}"
            if args.shape_report:
                gt_leaf = (gt == 1).astype(np.uint8)
                pr_leaf = (pred == 1).astype(np.uint8)
                gt_ring = (gt == 2).astype(np.uint8)
                pr_ring = (pred == 2).astype(np.uint8)

                leaf_areas_gt = _components_areas(gt_leaf)
                leaf_areas_pr = _components_areas(pr_leaf)
                ring_areas_gt = _components_areas(gt_ring)
                ring_areas_pr = _components_areas(pr_ring)

                gt_leaf_c = len(leaf_areas_gt)
                pr_leaf_c = len(leaf_areas_pr)
                gt_ring_c = len(ring_areas_gt)
                pr_ring_c = len(ring_areas_pr)

                merged_leaflet = False
                if pr_leaf_c < gt_leaf_c:
                    merged_leaflet = True
                if pr_leaf_c == 1 and gt_leaf_c >= 2:
                    merged_leaflet = True
                if gt_leaf_c >= 2 and len(leaf_areas_pr) > 1:
                    largest = int(leaf_areas_pr[0]) if leaf_areas_pr else 0
                    second = int(leaf_areas_pr[1]) if len(leaf_areas_pr) > 1 else 0
                    if second > 0 and (float(largest) / float(second)) > 2.0:
                        merged_leaflet = True

                tiny_leaflet = 0
                if leaf_areas_pr:
                    largest = int(leaf_areas_pr[0])
                    if largest > 0:
                        thresh = 0.1 * float(largest)
                        tiny_leaflet = sum(1 for a in leaf_areas_pr[1:] if float(a) < thresh)

                tiny_ring = 0
                if ring_areas_pr:
                    largest = int(ring_areas_pr[0])
                    if largest > 0:
                        thresh = 0.1 * float(largest)
                        tiny_ring = sum(1 for a in ring_areas_pr[1:] if float(a) < thresh)

                ring_holes = _count_holes(pr_ring)
                warns = [
                    f"leaflet_components={pr_leaf_c}",
                    f"ring_components={pr_ring_c}",
                    f"ring_holes={ring_holes}",
                ]
                if merged_leaflet:
                    warns.append("merged_leaflet_suspect")
                if tiny_leaflet > 0:
                    warns.append(f"tiny_leaflet={tiny_leaflet}")
                if pr_ring_c > 1:
                    warns.append("disconnected_ring")
                if tiny_ring > 0:
                    warns.append(f"tiny_ring={tiny_ring}")
                msg = msg + " " + " ".join(warns)

            print(msg)

        gt_overlay = _overlay_rgb(image, gt, alpha=float(args.alpha), draw_ring=not leaflet_only)
        pred_overlay = _overlay_rgb(image, pred, alpha=float(args.alpha), draw_ring=not leaflet_only)

        cmp = np.concatenate([image, gt_overlay, pred_overlay], axis=1)

        _save_rgb(orig_dir / f"{stem}.png", image)
        _save_rgb(gt_dir / f"{stem}.png", gt_overlay)
        _save_rgb(pred_dir / f"{stem}.png", pred_overlay)
        _save_rgb(cmp_dir / f"{stem}.png", cmp)


if __name__ == "__main__":
    main()
