from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _connected_components(mask01: np.ndarray):
    import cv2

    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    return int(num), labels, stats


def _remove_small(mask01: np.ndarray, min_area: int) -> np.ndarray:
    import cv2

    min_area = int(min_area)
    if min_area <= 0:
        return (mask01.astype(np.uint8) > 0).astype(np.uint8)
    num, labels, stats = _connected_components(mask01)
    if num <= 1:
        return (mask01.astype(np.uint8) > 0).astype(np.uint8)
    keep = np.zeros_like(labels, dtype=np.uint8)
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            keep[labels == i] = 1
    return keep


def _keep_largest(mask01: np.ndarray, k: int) -> np.ndarray:
    import cv2

    k = int(k)
    if k <= 0:
        return np.zeros_like(mask01, dtype=np.uint8)
    num, labels, stats = _connected_components(mask01)
    if num <= 1:
        return (mask01.astype(np.uint8) > 0).astype(np.uint8)
    comps = []
    for i in range(1, num):
        comps.append((int(stats[i, cv2.CC_STAT_AREA]), i))
    comps.sort(reverse=True)
    keep_ids = {i for _, i in comps[:k]}
    out = np.zeros_like(labels, dtype=np.uint8)
    for i in keep_ids:
        out[labels == i] = 1
    return out


def _fill_holes(mask01: np.ndarray) -> np.ndarray:
    import cv2

    m = (mask01.astype(np.uint8) > 0).astype(np.uint8) * 255
    if int(m.sum()) == 0:
        return (mask01.astype(np.uint8) > 0).astype(np.uint8)
    inv = cv2.bitwise_not(m)
    h, w = inv.shape[:2]
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, ff_mask, seedPoint=(0, 0), newVal=0)
    holes = (inv > 0).astype(np.uint8)
    filled = ((m > 0).astype(np.uint8) | holes).astype(np.uint8)
    return filled


def _closing(mask01: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    r = int(radius)
    if r <= 0:
        return (mask01.astype(np.uint8) > 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    m = (mask01.astype(np.uint8) > 0).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1).astype(np.uint8)


@dataclass(frozen=True)
class ClassPostprocessCfg:
    min_area: int = 0
    keep_largest: int = 0
    fill_holes: bool = False
    closing_radius: int = 0


def _parse_class_cfg(cfg: dict | None) -> ClassPostprocessCfg:
    if not isinstance(cfg, dict):
        return ClassPostprocessCfg()
    return ClassPostprocessCfg(
        min_area=int(cfg.get("min_area", 0) or 0),
        keep_largest=int(cfg.get("keep_largest", 0) or 0),
        fill_holes=bool(cfg.get("fill_holes", False)),
        closing_radius=int(cfg.get("closing_radius", 0) or 0),
    )


def apply_postprocess(mask: np.ndarray, cfg: dict | None) -> np.ndarray:
    if not isinstance(cfg, dict):
        return mask
    if not bool(cfg.get("enabled", False)):
        return mask

    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    leaf_cfg = _parse_class_cfg(cfg.get("leaflet", None))
    ring_cfg = _parse_class_cfg(cfg.get("fibrous_ring", None)) or _parse_class_cfg(cfg.get("ring", None))

    leaflet = (mask == 1).astype(np.uint8)
    ring = (mask == 2).astype(np.uint8)

    if leaf_cfg.min_area > 0:
        leaflet = _remove_small(leaflet, min_area=leaf_cfg.min_area)
    if leaf_cfg.keep_largest > 0:
        leaflet = _keep_largest(leaflet, k=leaf_cfg.keep_largest)
    if leaf_cfg.fill_holes:
        leaflet = _fill_holes(leaflet)
    if leaf_cfg.closing_radius > 0:
        leaflet = _closing(leaflet, radius=leaf_cfg.closing_radius)

    if ring_cfg.min_area > 0:
        ring = _remove_small(ring, min_area=ring_cfg.min_area)
    if ring_cfg.keep_largest > 0:
        ring = _keep_largest(ring, k=ring_cfg.keep_largest)
    if ring_cfg.fill_holes:
        ring = _fill_holes(ring)
    if ring_cfg.closing_radius > 0:
        ring = _closing(ring, radius=ring_cfg.closing_radius)

    out = np.zeros_like(mask, dtype=np.uint8)
    out[(leaflet > 0) & (ring == 0)] = 1
    out[ring > 0] = 2
    return out


POSTPROCESS_PRESETS: dict[str, dict] = {
    "ring_erosion_r2": {"ring_erosion_radius": 2},
}


def postprocess_multiclass_mask(mask: np.ndarray, ring_erosion_radius: int = 0, *, preset: str | None = None) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask

    if mask_u8.ndim != 2:
        raise ValueError(f"mask must be HxW uint8, got shape={mask_u8.shape} dtype={mask_u8.dtype}")

    r = int(ring_erosion_radius)
    if preset is not None:
        p = POSTPROCESS_PRESETS.get(str(preset), None)
        if p is None:
            raise ValueError(f"Unknown preset: {preset!r}")
        r = int(p.get("ring_erosion_radius", r))

    if r <= 0:
        return mask_u8.copy()

    ring_src = (mask_u8 == 2)
    if not bool(np.any(ring_src)):
        return mask_u8.copy()

    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    ring_bin = (ring_src.astype(np.uint8) * 255)
    ring_eroded = cv2.erode(ring_bin, kernel, iterations=1)
    ring_dst = ring_eroded > 0

    out = mask_u8.copy()
    removed = ring_src & (~ring_dst)
    if bool(np.any(removed)):
        out[removed] = 0
    if bool(np.any(ring_dst)):
        out[ring_dst] = 2
    return out
