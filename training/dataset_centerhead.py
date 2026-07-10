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

from dataset import DatasetItem, read_split_file


@dataclass(frozen=True)
class CenterDatasetItem(DatasetItem):
    center_path: Path
    metadata_path: Path


def _read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_mask_u8(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read mask: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _read_u16(path: Path) -> np.ndarray:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"Failed to read u16 map: {path}")
    if m.ndim == 3:
        m = m[:, :, 0]
    if m.dtype != np.uint16:
        m = m.astype(np.uint16)
    return m


class SegmentationWithCenterDataset(Dataset):
    def __init__(
        self,
        dataset_root: Path,
        split_txt: Path,
        num_classes: int,
        augment_fn=None,
        preprocessing_fn=None,
        center_dir: str = "center_maps",
        metadata_dir: str = "metadata",
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.split_txt = split_txt.resolve()
        self.num_classes = int(num_classes)
        self.augment_fn = augment_fn
        self.preprocessing_fn = preprocessing_fn

        base_items = read_split_file(self.dataset_root, self.split_txt)
        out: list[CenterDatasetItem] = []
        for it in base_items:
            sid = Path(it.image_path).stem
            center_path = (self.dataset_root / center_dir / f"{sid}.png").resolve()
            meta_path = (self.dataset_root / metadata_dir / f"{sid}.json").resolve()
            out.append(CenterDatasetItem(image_path=it.image_path, mask_path=it.mask_path, center_path=center_path, metadata_path=meta_path))
        self.items = out

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        image = _read_image_rgb(it.image_path)
        mask = _read_mask_u8(it.mask_path)
        center_u16 = _read_u16(it.center_path)
        if center_u16.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Center map shape mismatch: {it.center_path} center={center_u16.shape} mask={mask.shape}")

        center = (center_u16.astype(np.float32) / 65535.0).astype(np.float32)

        if self.augment_fn is not None:
            image, mask, center = self.augment_fn(image, mask, center=center)

        if self.preprocessing_fn is not None:
            image = self.preprocessing_fn(image)

        if mask.ndim != 2:
            raise ValueError(f"Mask must be HxW after augmentation, got shape={mask.shape} for {it.mask_path}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be HxWx3 RGB after preprocessing, got shape={image.shape} for {it.image_path}")
        if center.ndim != 2:
            raise ValueError(f"Center map must be HxW after augmentation, got shape={center.shape} for {it.center_path}")

        mask_max = int(mask.max())
        if mask_max >= self.num_classes:
            raise ValueError(f"Mask has value {mask_max} >= num_classes={self.num_classes} at {it.mask_path}")

        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask).long()
        center_t = torch.from_numpy(center[None, :, :]).float()

        return {
            "image": image_t,
            "mask": mask_t,
            "center": center_t,
            "image_path": str(it.image_path),
            "mask_path": str(it.mask_path),
            "center_path": str(it.center_path),
            "metadata_path": str(it.metadata_path),
        }

