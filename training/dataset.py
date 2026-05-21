from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e


@dataclass(frozen=True)
class DatasetItem:
    image_path: Path
    mask_path: Path


def read_split_file(dataset_root: Path, split_txt: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    with split_txt.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Invalid line in {split_txt}: expected 2 tab-separated columns, got: {line!r}")
            img_rel, mask_rel = parts
            items.append(DatasetItem(image_path=(dataset_root / img_rel).resolve(), mask_path=(dataset_root / mask_rel).resolve()))
    return items


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


def _leaflet_boundary(mask_leaflet: np.ndarray, width: int) -> np.ndarray:
    w = int(width)
    if w <= 0:
        return np.zeros_like(mask_leaflet, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * w + 1, 2 * w + 1))
    dil = cv2.dilate(mask_leaflet, kernel, iterations=1)
    ero = cv2.erode(mask_leaflet, kernel, iterations=1)
    return (dil != ero).astype(np.uint8)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_txt: Path,
        num_classes: int,
        target: str | None = None,
        crop_mode: str | None = None,
        crop_padding: float = 0.0,
        boundary_cfg: dict | None = None,
        augment_fn=None,
        preprocessing_fn=None,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.split_txt = split_txt.resolve()
        self.num_classes = int(num_classes)
        self.target = (target or "multiclass").strip().lower()
        self.crop_mode = (crop_mode or "").strip().lower() or None
        self.crop_padding = float(crop_padding)
        self.boundary_cfg = boundary_cfg or {}
        self.items = read_split_file(self.dataset_root, self.split_txt)
        self.augment_fn = augment_fn
        self.preprocessing_fn = preprocessing_fn

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        image = _read_image_rgb(it.image_path)
        mask = _read_mask_uint8(it.mask_path)

        boundary = None

        if self.crop_mode == "anatomy_bbox":
            fg = mask > 0
            if fg.any():
                ys, xs = np.where(fg)
                y0 = int(ys.min())
                y1 = int(ys.max()) + 1
                x0 = int(xs.min())
                x1 = int(xs.max()) + 1

                bbox_h = y1 - y0
                bbox_w = x1 - x0
                pad = int(round(max(bbox_h, bbox_w) * self.crop_padding))
                y0 = max(0, y0 - pad)
                x0 = max(0, x0 - pad)
                y1 = min(mask.shape[0], y1 + pad)
                x1 = min(mask.shape[1], x1 + pad)

                image = image[y0:y1, x0:x1, :]
                mask = mask[y0:y1, x0:x1]

        if self.target == "leaflet_only":
            leaflet = (mask == 1).astype(np.uint8)
            if bool(self.boundary_cfg.get("enabled", False)):
                # Boundary is computed before spatial augmentations so it stays aligned with image/mask during crop/flip/rotate.
                width = int(self.boundary_cfg.get("width", 3))
                boundary = _leaflet_boundary(leaflet, width=width)
            mask = leaflet

        if self.augment_fn is not None:
            if boundary is None:
                image, mask = self.augment_fn(image, mask)
            else:
                image, mask, boundary = self.augment_fn(image, mask, boundary)

        if self.preprocessing_fn is not None:
            image = self.preprocessing_fn(image)

        if mask.ndim != 2:
            raise ValueError(f"Mask must be HxW after augmentation, got shape={mask.shape} for {it.mask_path}")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be HxWx3 RGB after preprocessing, got shape={image.shape} for {it.image_path}")

        mask_max = int(mask.max())
        if mask_max >= self.num_classes:
            raise ValueError(f"Mask has value {mask_max} >= num_classes={self.num_classes} at {it.mask_path}")

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).long()

        out = {
            "image": image_t,
            "mask": mask_t,
            "image_path": str(it.image_path),
            "mask_path": str(it.mask_path),
        }
        if boundary is not None:
            boundary_t = torch.from_numpy(boundary).float()
            out["boundary"] = boundary_t
        return out
