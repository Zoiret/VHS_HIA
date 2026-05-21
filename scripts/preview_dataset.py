import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def _read_image_rgb(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _overlay_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float,
    leaflet_color: tuple[int, int, int],
    ring_color: tuple[int, int, int],
) -> np.ndarray:
    out = image_rgb.copy()
    color = np.zeros_like(out, dtype=np.uint8)

    color[mask == 1] = np.array(leaflet_color, dtype=np.uint8)
    color[mask == 2] = np.array(ring_color, dtype=np.uint8)

    m_any = mask > 0
    if np.any(m_any):
        out[m_any] = (out[m_any].astype(np.float32) * (1.0 - alpha) + color[m_any].astype(np.float32) * alpha).astype(
            np.uint8
        )

    for class_id, c in [(1, leaflet_color), (2, ring_color)]:
        bin_m = (mask == class_id).astype(np.uint8)
        if not np.any(bin_m):
            continue
        contours, _ = cv2.findContours(bin_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, c, thickness=2)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create overlay previews for converted segmentation dataset.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/converted"),
        help="Converted dataset root (default: datasets/converted).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/previews"),
        help="Output folder for previews (default: datasets/previews).",
    )
    parser.add_argument("--max-items", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.45)
    args = parser.parse_args()

    ds_root = args.dataset_root.resolve()
    images_dir = ds_root / "images"
    masks_dir = ds_root / "masks"
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.exists() or not masks_dir.exists():
        raise SystemExit(f"Dataset not found (expected {images_dir} and {masks_dir})")

    image_paths = [p for p in images_dir.rglob("*") if p.is_file()]
    image_paths.sort(key=lambda p: str(p).lower())

    leaflet_color = (255, 0, 0)
    ring_color = (0, 255, 255)

    written = 0
    for img_path in tqdm(image_paths, desc="Preview", unit="img"):
        stem = img_path.stem
        mask_path = masks_dir / f"{stem}.png"
        if not mask_path.exists():
            continue

        img = _read_image_rgb(img_path)
        mask = _read_mask(mask_path)
        if img is None or mask is None:
            continue

        if mask.shape[0] != img.shape[0] or mask.shape[1] != img.shape[1]:
            continue

        overlay = _overlay_mask(img, mask, alpha=float(args.alpha), leaflet_color=leaflet_color, ring_color=ring_color)

        out_name = f"{stem}_overlay.png"
        out_path = out_dir / out_name
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), overlay_bgr)

        written += 1
        if written >= int(args.max_items):
            break

    print(f"Previews written: {written}")
    print(f"Output dir: {out_dir}")


if __name__ == "__main__":
    main()
