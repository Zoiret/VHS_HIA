import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


@dataclass(frozen=True)
class SplitEntry:
    line: str
    image_rel: str
    mask_rel: str
    image_path: Path
    mask_path: Path
    stem: str


def _read_split(split_path: Path, dataset_root: Path) -> list[SplitEntry]:
    entries: list[SplitEntry] = []
    with split_path.open("r", encoding="utf-8") as f:
        for raw in f:
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
            entries.append(
                SplitEntry(
                    line=line,
                    image_rel=image_rel,
                    mask_rel=mask_rel,
                    image_path=image_path,
                    mask_path=mask_path,
                    stem=stem,
                )
            )
    return entries


def _read_image_rgb_any(path: Path) -> np.ndarray | None:
    try:
        data = path.read_bytes()
    except Exception:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        with Image.open(path) as img:
            img.load()
            rgb = img.convert("RGB")
            return np.array(rgb)
    except Exception:
        return None


def _read_mask_uint8(path: Path) -> np.ndarray | None:
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype(np.uint8)


def _try_repair_image_from_meta(dataset_root: Path, entry: SplitEntry) -> bool:
    meta_path = (dataset_root / "meta" / f"{entry.stem}.json").resolve()
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    src = meta.get("source") if isinstance(meta, dict) else None
    if not isinstance(src, dict):
        return False
    src_image = src.get("image")
    if not isinstance(src_image, str) or not src_image:
        return False

    src_path = Path(src_image)
    if not src_path.exists():
        return False

    try:
        shutil.copy2(src_path, entry.image_path)
    except Exception:
        return False

    return _read_image_rgb_any(entry.image_path) is not None


def validate_entry(dataset_root: Path, entry: SplitEntry) -> list[str]:
    issues: list[str] = []

    if not entry.image_path.exists():
        issues.append(f"image missing: {entry.image_path}")
        return issues
    if not entry.mask_path.exists():
        issues.append(f"mask missing: {entry.mask_path}")
        return issues

    img = _read_image_rgb_any(entry.image_path)
    if img is None:
        issues.append(f"image unreadable: {entry.image_path}")
        return issues

    mask = _read_mask_uint8(entry.mask_path)
    if mask is None:
        issues.append(f"mask unreadable: {entry.mask_path}")
        return issues

    if mask.ndim != 2:
        issues.append(f"mask must be HxW, got shape={tuple(mask.shape)}: {entry.mask_path}")
        return issues

    if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
        issues.append(
            f"size mismatch: image={tuple(img.shape[:2])} mask={tuple(mask.shape[:2])} image={entry.image_path} mask={entry.mask_path}"
        )

    uniques = np.unique(mask)
    allowed = {0, 1, 2}
    bad_vals = [int(v) for v in uniques.tolist() if int(v) not in allowed]
    if bad_vals:
        issues.append(f"mask has unexpected values {bad_vals}: {entry.mask_path}")

    return issues


def _rewrite_split_without_broken(split_path: Path, broken_lines: set[str]) -> int:
    orig_lines = split_path.read_text(encoding="utf-8").splitlines()
    out_lines = []
    dropped = 0
    for l in orig_lines:
        if l.strip() in broken_lines:
            dropped += 1
            continue
        out_lines.append(l)
    split_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate converted dataset files referenced by train/val/test splits.")
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/converted"))
    parser.add_argument("--drop-broken", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    train_txt = (dataset_root / "train.txt").resolve()
    val_txt = (dataset_root / "val.txt").resolve()
    test_txt = (dataset_root / "test.txt").resolve()

    for p in [train_txt, val_txt, test_txt]:
        if not p.exists():
            raise SystemExit(f"Split file not found: {p}")

    splits = {
        "train": _read_split(train_txt, dataset_root),
        "val": _read_split(val_txt, dataset_root),
        "test": _read_split(test_txt, dataset_root),
    }

    broken: dict[str, list[tuple[SplitEntry, list[str]]]] = {"train": [], "val": [], "test": []}

    for split_name, entries in splits.items():
        for entry in tqdm(entries, desc=f"Validate {split_name}", unit="sample"):
            issues = validate_entry(dataset_root, entry)
            if issues and args.drop_broken:
                if any(x.startswith("image unreadable:") for x in issues):
                    repaired = _try_repair_image_from_meta(dataset_root, entry)
                    if repaired:
                        issues = validate_entry(dataset_root, entry)
            if issues:
                broken[split_name].append((entry, issues))

    total_broken = sum(len(v) for v in broken.values())
    print()
    print("Validation summary")
    print(f"Dataset root: {dataset_root}")
    print(f"Train samples: {len(splits['train'])}")
    print(f"Val samples: {len(splits['val'])}")
    print(f"Test samples: {len(splits['test'])}")
    print(f"Broken samples: {total_broken}")

    if total_broken:
        print()
        print("Broken list")
        for split_name in ["train", "val", "test"]:
            for entry, issues in broken[split_name]:
                print(f"[{split_name}] {entry.image_rel}\t{entry.mask_rel}")
                for issue in issues:
                    print(f"  - {issue}")

    if args.drop_broken:
        broken_lines = {entry.line for split_items in broken.values() for entry, _ in split_items}
        if not broken_lines:
            print()
            print("drop-broken: nothing to drop")
            return

        for p in [train_txt, val_txt, test_txt]:
            bak = p.with_suffix(p.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(p, bak)

        dropped_train = _rewrite_split_without_broken(train_txt, broken_lines)
        dropped_val = _rewrite_split_without_broken(val_txt, broken_lines)
        dropped_test = _rewrite_split_without_broken(test_txt, broken_lines)
        print()
        print("drop-broken: updated split files")
        print(f"Dropped from train: {dropped_train}")
        print(f"Dropped from val: {dropped_val}")
        print(f"Dropped from test: {dropped_test}")


if __name__ == "__main__":
    main()
