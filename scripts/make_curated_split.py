import argparse
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SampleRef:
    stem: str
    line: str


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
    return out


def _normalize_id(s: str) -> str:
    if "\t" in s:
        s = s.split("\t", 1)[0]
    s = s.replace("\\", "/")
    name = s.split("/")[-1]
    if name.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        return Path(name).stem
    return name


def _index_converted_splits(converted_root: Path) -> dict[str, str]:
    index: dict[str, str] = {}
    for split_name in ["train.txt", "val.txt", "test.txt"]:
        p = (converted_root / split_name).resolve()
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) != 2:
                    raise SystemExit(f"Invalid line in {p}: {line!r}")
                img_rel, _ = parts
                stem = Path(img_rel).stem
                index[stem] = line
    if not index:
        raise SystemExit(f"No samples indexed. Expected train/val/test splits in: {converted_root}")
    return index


def _split_ids(items: list[str], seed: int, val_ratio: float, test_ratio: float) -> tuple[list[str], list[str], list[str]]:
    if val_ratio < 0 or test_ratio < 0 or (val_ratio + test_ratio) >= 1.0:
        raise SystemExit("Invalid ratios: need val_ratio>=0, test_ratio>=0, and val_ratio+test_ratio<1")
    rng = random.Random(int(seed))
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_test = int(round(n * float(test_ratio)))
    n_val = int(round(n * float(val_ratio)))
    n_test = min(max(n_test, 0), n)
    n_val = min(max(n_val, 0), n - n_test)

    test = items[:n_test]
    val = items[n_test : n_test + n_val]
    train = items[n_test + n_val :]
    return train, val, test


def _write_split(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create curated train/val/test splits based on manual ratings.")
    parser.add_argument("--converted-root", type=Path, default=Path("datasets/converted"))
    parser.add_argument("--curated-dir", type=Path, default=Path("datasets/curated"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/converted_curated"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    converted_root = args.converted_root.resolve()
    curated_dir = args.curated_dir.resolve()
    out_dir = args.out_dir.resolve()

    index = _index_converted_splits(converted_root)

    clean_raw = _read_lines(curated_dir / "clean.txt")
    medium_raw = _read_lines(curated_dir / "medium.txt")
    bad_raw = _read_lines(curated_dir / "bad.txt")

    clean_ids = {_normalize_id(x) for x in clean_raw}
    medium_ids = {_normalize_id(x) for x in medium_raw}
    bad_ids = {_normalize_id(x) for x in bad_raw}

    included_ids = (clean_ids | medium_ids) - bad_ids

    missing = sorted([x for x in included_ids if x not in index])
    bad_missing = sorted([x for x in bad_ids if x not in index])

    resolved = [x for x in included_ids if x in index]
    train_ids, val_ids, test_ids = _split_ids(resolved, seed=int(args.seed), val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio))

    train_lines = [index[x] for x in train_ids]
    val_lines = [index[x] for x in val_ids]
    test_lines = [index[x] for x in test_ids]

    _write_split(out_dir / "train.txt", train_lines)
    _write_split(out_dir / "val.txt", val_lines)
    _write_split(out_dir / "test.txt", test_lines)

    print("Curated summary")
    print(f"Converted root: {converted_root}")
    print(f"Curated dir: {curated_dir}")
    print(f"Out dir: {out_dir}")
    print(f"clean: {len(clean_ids)}")
    print(f"medium: {len(medium_ids)}")
    print(f"bad: {len(bad_ids)}")
    print(f"included (clean+medium-bad): {len(included_ids)}")
    print(f"resolved (found in converted): {len(resolved)}")
    print(f"train: {len(train_lines)}")
    print(f"val: {len(val_lines)}")
    print(f"test: {len(test_lines)}")
    if missing:
        print()
        print("WARNING: included ids not found in datasets/converted splits:")
        for x in missing[:50]:
            print(f"- {x}")
        if len(missing) > 50:
            print(f"... ({len(missing) - 50} more)")
    if bad_missing:
        print()
        print("NOTE: bad ids not found in datasets/converted (ok):")
        for x in bad_missing[:50]:
            print(f"- {x}")
        if len(bad_missing) > 50:
            print(f"... ({len(bad_missing) - 50} more)")


if __name__ == "__main__":
    main()

