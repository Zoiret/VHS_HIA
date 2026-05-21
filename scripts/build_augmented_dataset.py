import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass(frozen=True)
class SplitItem:
    image_rel: str
    mask_rel: str
    image_path: Path
    mask_path: Path
    stem: str


def _read_split(dataset_root: Path, split: str) -> list[SplitItem]:
    split_path = (dataset_root / f"{split}.txt").resolve()
    if not split_path.exists():
        raise SystemExit(f"Split file not found: {split_path}")
    items: list[SplitItem] = []
    for raw in split_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            raise SystemExit(f"Invalid line in {split_path}: {line!r}")
        image_rel, mask_rel = parts
        image_path = (dataset_root / image_rel).resolve()
        mask_path = (dataset_root / mask_rel).resolve()
        stem = Path(image_rel).stem
        items.append(SplitItem(image_rel=image_rel, mask_rel=mask_rel, image_path=image_path, mask_path=mask_path, stem=stem))
    return items


def _read_image_bgr_any(path: Path) -> np.ndarray | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return img


def _read_mask_uint8(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _apply_spatial(image_bgr: np.ndarray, mask: np.ndarray, *, k90: int, hflip: bool, vflip: bool) -> tuple[np.ndarray, np.ndarray]:
    img = image_bgr
    m = mask
    k = int(k90) % 4
    if k:
        img = np.ascontiguousarray(np.rot90(img, k))
        m = np.ascontiguousarray(np.rot90(m, k))
    if bool(hflip):
        img = np.ascontiguousarray(img[:, ::-1, :])
        m = np.ascontiguousarray(m[:, ::-1])
    if bool(vflip):
        img = np.ascontiguousarray(img[::-1, :, :])
        m = np.ascontiguousarray(m[::-1, :])
    return img, m


def _apply_photometric(image_bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    img = image_bgr.astype(np.float32)
    desc_parts: list[str] = []

    contrast = float(rng.uniform(-0.12, 0.12))
    brightness = float(rng.uniform(-15.0, 15.0))
    img = img * (1.0 + contrast) + brightness
    desc_parts.append(f"bc(c={contrast:+.3f},b={brightness:+.1f})")

    gamma = float(rng.uniform(0.90, 1.10))
    img01 = np.clip(img, 0.0, 255.0) / 255.0
    img01 = np.power(img01, gamma)
    img = img01 * 255.0
    desc_parts.append(f"gamma({gamma:.3f})")

    if float(rng.random()) < 0.30:
        hsv = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hue_shift = int(rng.integers(-3, 4))
        sat_mul = float(rng.uniform(0.90, 1.10))
        val_mul = float(rng.uniform(0.95, 1.05))
        hsv[:, :, 0] = (hsv[:, :, 0] + float(hue_shift)) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_mul, 0.0, 255.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * val_mul, 0.0, 255.0)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
        desc_parts.append(f"hsv(h={hue_shift:+d},s={sat_mul:.3f},v={val_mul:.3f})")

    out = np.clip(img, 0.0, 255.0).astype(np.uint8)
    return out, ";".join(desc_parts)


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise SystemExit(f"Failed to write: {path}")


def _copy_split_as_is(dataset_root: Path, out_root: Path, split: str) -> int:
    split_path = (dataset_root / f"{split}.txt").resolve()
    out_split_path = (out_root / f"{split}.txt").resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_split_path.write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")

    items = _read_split(dataset_root, split)
    copied = 0
    for it in tqdm(items, desc=f"Copy {split}", unit="sample"):
        dst_img = (out_root / it.image_rel).resolve()
        dst_mask = (out_root / it.mask_rel).resolve()
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_mask.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(it.image_path, dst_img)
        shutil.copy2(it.mask_path, dst_mask)
        copied += 1
    return copied


def _ensure_mask_values(mask: np.ndarray) -> None:
    uniques = np.unique(mask)
    allowed = {0, 1, 2}
    bad = [int(v) for v in uniques.tolist() if int(v) not in allowed]
    if bad:
        raise SystemExit(f"Mask has unexpected values {bad}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build augmented dataset on disk (train: orig+N variants; val/test: copy as-is).")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/converted_full_multiclass"))
    parser.add_argument("--split", type=str, default="train", choices=["train"])
    parser.add_argument("--out-root", type=Path, default=Path("datasets/converted_full_multiclass_aug"))
    parser.add_argument("--num-variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    out_root = args.out_root.resolve()
    num_variants = int(args.num_variants)
    if num_variants < 0:
        raise SystemExit("--num-variants must be >= 0")

    if out_root.exists():
        try:
            has_any = any(out_root.iterdir())
        except Exception:
            has_any = True
        if has_any:
            raise SystemExit(f"out-root must be empty or not exist: {out_root}")

    rng = np.random.default_rng(int(args.seed))

    train_items = _read_split(dataset_root, "train")
    copied_val = _copy_split_as_is(dataset_root, out_root, "val")
    copied_test = _copy_split_as_is(dataset_root, out_root, "test")

    out_images_dir = (out_root / "images").resolve()
    out_masks_dir = (out_root / "masks").resolve()
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_masks_dir.mkdir(parents=True, exist_ok=True)

    meta_path = (out_root / "augment_meta.csv").resolve()
    train_out_path = (out_root / "train.txt").resolve()

    out_lines: list[str] = []
    broken_train: list[str] = []
    processed = 0

    with meta_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["original_id", "aug_id", "transform_description"])

        for it in tqdm(train_items, desc="Augment train", unit="sample"):
            img = _read_image_bgr_any(it.image_path)
            mask = _read_mask_uint8(it.mask_path)
            if img is None or mask is None:
                broken_train.append(it.stem)
                continue
            if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
                broken_train.append(it.stem)
                continue
            _ensure_mask_values(mask)

            orig_id = it.stem
            orig_aug_id = f"{orig_id}__orig"
            orig_img_rel = f"images/{orig_aug_id}.png"
            orig_mask_rel = f"masks/{orig_aug_id}.png"
            _write_png(out_root / orig_img_rel, img)
            _write_png(out_root / orig_mask_rel, mask)
            out_lines.append(f"{orig_img_rel}\t{orig_mask_rel}")
            w.writerow([orig_id, orig_aug_id, "orig"])

            used_transforms: set[tuple[int, bool, bool]] = {(0, False, False)}
            for j in range(1, num_variants + 1):
                k90 = int(rng.integers(0, 4))
                hflip = bool(rng.random() < 0.5)
                vflip = bool(rng.random() < 0.5)
                if (k90, hflip, vflip) == (0, False, False):
                    hflip = True
                if (k90, hflip, vflip) in used_transforms:
                    k90 = (k90 + 1) % 4
                used_transforms.add((k90, hflip, vflip))

                aug_img, aug_mask = _apply_spatial(img, mask, k90=k90, hflip=hflip, vflip=vflip)
                _ensure_mask_values(aug_mask)

                aug_img, photo_desc = _apply_photometric(aug_img, rng=rng)
                aug_id = f"{orig_id}__aug{j:03d}"
                img_rel = f"images/{aug_id}.png"
                mask_rel = f"masks/{aug_id}.png"
                _write_png(out_root / img_rel, aug_img)
                _write_png(out_root / mask_rel, aug_mask)
                out_lines.append(f"{img_rel}\t{mask_rel}")
                w.writerow([orig_id, aug_id, f"spatial(k90={k90 * 90},hflip={int(hflip)},vflip={int(vflip)});{photo_desc}"])

            processed += 1

    train_out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    images_count = len(list(out_images_dir.glob("*.png")))
    masks_count = len(list(out_masks_dir.glob("*.png")))
    if images_count != masks_count:
        raise SystemExit(f"Output images/masks count mismatch: images={images_count} masks={masks_count}")

    expected_train = processed * (1 + num_variants)
    if len(out_lines) != expected_train:
        raise SystemExit(f"train.txt count mismatch: got={len(out_lines)} expected={expected_train}")

    print("Augmented dataset created")
    print(f"Input root: {dataset_root}")
    print(f"Output root: {out_root}")
    print(f"Train source samples: {len(train_items)}")
    print(f"Train processed samples: {processed}")
    print(f"Train skipped (broken): {len(broken_train)}")
    if broken_train:
        print("Broken train sample ids (first 20):")
        for s in broken_train[:20]:
            print(f"- {s}")
    print(f"Train output samples (orig+aug): {len(out_lines)}")
    print(f"Val copied samples: {copied_val}")
    print(f"Test copied samples: {copied_test}")


if __name__ == "__main__":
    main()
