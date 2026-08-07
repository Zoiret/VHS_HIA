from __future__ import annotations

import random

import cv2
import numpy as np


def _flag(cfg: dict | None, key: str, default: bool) -> bool:
    if not cfg:
        return bool(default)
    v = cfg.get(key, default)
    return bool(v)


def _random_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_h: int,
    crop_w: int,
    boundary: np.ndarray | None = None,
    center: np.ndarray | None = None,
):
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        new_h = max(h, crop_h)
        new_w = max(w, crop_w)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if boundary is not None:
            boundary = cv2.resize(boundary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if center is not None:
            center = cv2.resize(center, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w

    y0 = random.randint(0, h - crop_h) if h > crop_h else 0
    x0 = random.randint(0, w - crop_w) if w > crop_w else 0
    image = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    mask = mask[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if boundary is not None:
        boundary = boundary[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if center is not None:
        center = center[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if boundary is None and center is None:
        return image, mask
    if boundary is not None and center is None:
        return image, mask, boundary
    if boundary is None and center is not None:
        return image, mask, center
    return image, mask, boundary, center


def _center_crop(
    image: np.ndarray,
    mask: np.ndarray,
    crop_h: int,
    crop_w: int,
    boundary: np.ndarray | None = None,
    center: np.ndarray | None = None,
):
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        new_h = max(h, crop_h)
        new_w = max(w, crop_w)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if boundary is not None:
            boundary = cv2.resize(boundary, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        if center is not None:
            center = cv2.resize(center, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w

    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    image = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    mask = mask[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if boundary is not None:
        boundary = boundary[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if center is not None:
        center = center[y0 : y0 + crop_h, x0 : x0 + crop_w]
    if boundary is None and center is None:
        return image, mask
    if boundary is not None and center is None:
        return image, mask, boundary
    if boundary is None and center is not None:
        return image, mask, center
    return image, mask, boundary, center


def _random_brightness_contrast(image: np.ndarray) -> np.ndarray:
    raise NotImplementedError("Use _random_brightness_contrast_with_rng")


def _maybe_float(cfg: dict | None, key: str, default: float) -> float:
    if not cfg:
        return float(default)
    value = cfg.get(key, default)
    return float(value)


def sample_train_augmentation_params(cfg: dict | None = None, *, rng=None) -> dict:
    rr = rng if rng is not None else random
    cfg = dict(cfg or {})
    rotate90 = _flag(cfg, "rotate90", True)
    hflip_enabled = _flag(cfg, "hflip", True)
    vflip_enabled = _flag(cfg, "vflip", True)
    brightness_contrast_enabled = _flag(cfg, "brightness_contrast", False)
    gamma_enabled = _flag(cfg, "gamma", False)
    contrast_limit = _maybe_float(cfg, "contrast_limit", 0.15)
    brightness_limit = _maybe_float(cfg, "brightness_limit", 20.0)
    gamma_min = _maybe_float(cfg, "gamma_min", 0.9)
    gamma_max = _maybe_float(cfg, "gamma_max", 1.1)
    return {
        "hflip": bool(hflip_enabled and rr.random() < 0.5),
        "vflip": bool(vflip_enabled and rr.random() < 0.5),
        "rot90_k": int(rr.randint(0, 3)) if rotate90 else 0,
        "brightness_delta": float(rr.uniform(-brightness_limit, brightness_limit)) if brightness_contrast_enabled else 0.0,
        "contrast_delta": float(rr.uniform(-contrast_limit, contrast_limit)) if brightness_contrast_enabled else 0.0,
        "gamma": float(rr.uniform(gamma_min, gamma_max)) if gamma_enabled else 1.0,
    }


def apply_exact_geometric_transform(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    boundary: np.ndarray | None = None,
    center: np.ndarray | None = None,
    hflip: bool = False,
    vflip: bool = False,
    rot90_k: int = 0,
):
    if bool(hflip):
        image = np.ascontiguousarray(image[:, ::-1, :])
        mask = np.ascontiguousarray(mask[:, ::-1])
        if boundary is not None:
            boundary = np.ascontiguousarray(boundary[:, ::-1])
        if center is not None:
            center = np.ascontiguousarray(center[:, ::-1])
    if bool(vflip):
        image = np.ascontiguousarray(image[::-1, :, :])
        mask = np.ascontiguousarray(mask[::-1, :])
        if boundary is not None:
            boundary = np.ascontiguousarray(boundary[::-1, :])
        if center is not None:
            center = np.ascontiguousarray(center[::-1, :])
    k = int(rot90_k) % 4
    if k:
        image = np.ascontiguousarray(np.rot90(image, k))
        mask = np.ascontiguousarray(np.rot90(mask, k))
        if boundary is not None:
            boundary = np.ascontiguousarray(np.rot90(boundary, k))
        if center is not None:
            center = np.ascontiguousarray(np.rot90(center, k))
    if boundary is None and center is None:
        return image, mask
    if boundary is not None and center is None:
        return image, mask, boundary
    if boundary is None and center is not None:
        return image, mask, center
    return image, mask, boundary, center


def transform_points_row_col_yx(
    points_yx: list[tuple[int, int]],
    shape_hw: tuple[int, int],
    *,
    hflip: bool = False,
    vflip: bool = False,
    rot90_k: int = 0,
) -> list[tuple[int, int]]:
    h, w = [int(v) for v in shape_hw]
    out: list[tuple[int, int]] = []
    for y0, x0 in points_yx:
        y = int(y0)
        x = int(x0)
        if bool(hflip):
            x = int(w - 1 - x)
        if bool(vflip):
            y = int(h - 1 - y)
        hh, ww = h, w
        for _ in range(int(rot90_k) % 4):
            y, x = int(ww - 1 - x), int(y)
            hh, ww = ww, hh
        out.append((int(y), int(x)))
    return out


def _random_brightness_contrast_with_rng(image: np.ndarray, *, contrast_delta: float, brightness_delta: float) -> np.ndarray:
    img = image.astype(np.float32)
    img = img * (1.0 + float(contrast_delta)) + float(brightness_delta)
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _random_gamma_with_value(image: np.ndarray, *, gamma: float) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    img = np.power(img, float(gamma)) * 255.0
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


class TrainAugmentations:
    def __init__(
        self,
        input_h: int,
        input_w: int,
        rotate90: bool,
        hflip: bool,
        vflip: bool,
        brightness_contrast: bool,
        gamma: bool,
        random_crop: bool | None,
        brightness_limit: float,
        contrast_limit: float,
        gamma_min: float,
        gamma_max: float,
    ) -> None:
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.rotate90 = bool(rotate90)
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        self.brightness_contrast = bool(brightness_contrast)
        self.gamma = bool(gamma)
        self.random_crop = None if random_crop is None else bool(random_crop)
        self.brightness_limit = float(brightness_limit)
        self.contrast_limit = float(contrast_limit)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)

    def __call__(self, image: np.ndarray, mask: np.ndarray, boundary: np.ndarray | None = None, center: np.ndarray | None = None):
        params = sample_train_augmentation_params(
            {
                "rotate90": self.rotate90,
                "hflip": self.hflip,
                "vflip": self.vflip,
                "brightness_contrast": self.brightness_contrast,
                "gamma": self.gamma,
                "brightness_limit": self.brightness_limit,
                "contrast_limit": self.contrast_limit,
                "gamma_min": self.gamma_min,
                "gamma_max": self.gamma_max,
            }
        )
        out = apply_exact_geometric_transform(
            image,
            mask,
            boundary=boundary,
            center=center,
            hflip=bool(params["hflip"]),
            vflip=bool(params["vflip"]),
            rot90_k=int(params["rot90_k"]),
        )
        if boundary is None and center is None:
            image, mask = out
        elif boundary is not None and center is None:
            image, mask, boundary = out
        elif boundary is None and center is not None:
            image, mask, center = out
        else:
            image, mask, boundary, center = out

        deterministic_crop = (self.random_crop is False) or ((self.random_crop is None) and (not self.rotate90) and (not self.hflip) and (not self.vflip))
        if boundary is None and center is None:
            if deterministic_crop:
                image, mask = _center_crop(image, mask, self.input_h, self.input_w)
            else:
                image, mask = _random_crop(image, mask, self.input_h, self.input_w)
        else:
            if deterministic_crop:
                out = _center_crop(image, mask, self.input_h, self.input_w, boundary=boundary, center=center)
            else:
                out = _random_crop(image, mask, self.input_h, self.input_w, boundary=boundary, center=center)
            if boundary is not None and center is None:
                image, mask, boundary = out
            elif boundary is None and center is not None:
                image, mask, center = out
            else:
                image, mask, boundary, center = out

        if self.brightness_contrast:
            image = _random_brightness_contrast_with_rng(
                image,
                contrast_delta=float(params["contrast_delta"]),
                brightness_delta=float(params["brightness_delta"]),
            )
        if self.gamma:
            image = _random_gamma_with_value(image, gamma=float(params["gamma"]))

        if boundary is None and center is None:
            return image, mask
        if boundary is not None and center is None:
            return image, mask, boundary
        if boundary is None and center is not None:
            return image, mask, center
        return image, mask, boundary, center


class ValAugmentations:
    def __init__(self, input_h: int, input_w: int) -> None:
        self.input_h = int(input_h)
        self.input_w = int(input_w)

    def __call__(self, image: np.ndarray, mask: np.ndarray, boundary: np.ndarray | None = None, center: np.ndarray | None = None):
        if boundary is None and center is None:
            return _center_crop(image, mask, self.input_h, self.input_w)
        out = _center_crop(image, mask, self.input_h, self.input_w, boundary=boundary, center=center)
        if boundary is not None and center is None:
            return out
        if boundary is None and center is not None:
            return out
        return out


def get_train_augmentations(input_h: int, input_w: int, augment_cfg: dict | None = None) -> TrainAugmentations:
    cfg = augment_cfg or {}
    return TrainAugmentations(
        input_h=input_h,
        input_w=input_w,
        rotate90=_flag(cfg, "rotate90", True),
        hflip=_flag(cfg, "hflip", True),
        vflip=_flag(cfg, "vflip", True),
        brightness_contrast=_flag(cfg, "brightness_contrast", False),
        gamma=_flag(cfg, "gamma", False),
        random_crop=cfg.get("random_crop", None),
        brightness_limit=_maybe_float(cfg, "brightness_limit", 20.0),
        contrast_limit=_maybe_float(cfg, "contrast_limit", 0.15),
        gamma_min=_maybe_float(cfg, "gamma_min", 0.9),
        gamma_max=_maybe_float(cfg, "gamma_max", 1.1),
    )


def get_val_augmentations(input_h: int, input_w: int) -> ValAugmentations:
    return ValAugmentations(input_h=input_h, input_w=input_w)
