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
    img = image.astype(np.float32)
    contrast = random.uniform(-0.15, 0.15)
    brightness = random.uniform(-20.0, 20.0)
    img = img * (1.0 + contrast) + brightness
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _random_gamma(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32) / 255.0
    gamma = random.uniform(0.9, 1.1)
    img = np.power(img, gamma) * 255.0
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
    ) -> None:
        self.input_h = int(input_h)
        self.input_w = int(input_w)
        self.rotate90 = bool(rotate90)
        self.hflip = bool(hflip)
        self.vflip = bool(vflip)
        self.brightness_contrast = bool(brightness_contrast)
        self.gamma = bool(gamma)

    def __call__(self, image: np.ndarray, mask: np.ndarray, boundary: np.ndarray | None = None, center: np.ndarray | None = None):
        if self.hflip and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
            if boundary is not None:
                boundary = np.ascontiguousarray(boundary[:, ::-1])
            if center is not None:
                center = np.ascontiguousarray(center[:, ::-1])
        if self.vflip and random.random() < 0.5:
            image = np.ascontiguousarray(image[::-1, :, :])
            mask = np.ascontiguousarray(mask[::-1, :])
            if boundary is not None:
                boundary = np.ascontiguousarray(boundary[::-1, :])
            if center is not None:
                center = np.ascontiguousarray(center[::-1, :])

        if self.rotate90:
            k = random.randint(0, 3)
            if k:
                image = np.ascontiguousarray(np.rot90(image, k))
                mask = np.ascontiguousarray(np.rot90(mask, k))
                if boundary is not None:
                    boundary = np.ascontiguousarray(np.rot90(boundary, k))
                if center is not None:
                    center = np.ascontiguousarray(np.rot90(center, k))

        deterministic_crop = (not self.rotate90) and (not self.hflip) and (not self.vflip)
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
            image = _random_brightness_contrast(image)
        if self.gamma:
            image = _random_gamma(image)

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
    )


def get_val_augmentations(input_h: int, input_w: int) -> ValAugmentations:
    return ValAugmentations(input_h=input_h, input_w=input_w)
