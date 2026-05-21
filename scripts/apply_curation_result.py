import argparse
import json
from pathlib import Path


def _normalize_id(s: str) -> str:
    s = str(s).strip()
    if not s:
        return ""
    s = s.replace("\\", "/").split("/")[-1]
    return Path(s).stem


def _write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply curation_result.json to datasets/curated/*.txt lists.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, default=Path("datasets/converted"))
    parser.add_argument("--curated-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/curated"))
    args = parser.parse_args()

    in_path = args.input.resolve()
    out_dir = (args.curated_dir or args.out_dir).resolve()

    obj = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise SystemExit("curation_result.json must be a JSON object")

    clean_raw = obj.get("clean", [])
    medium_raw = obj.get("medium", [])
    bad_raw = obj.get("bad", [])
    if not isinstance(clean_raw, list) or not isinstance(medium_raw, list) or not isinstance(bad_raw, list):
        raise SystemExit('curation_result.json must contain lists: "clean", "medium", "bad"')

    clean = sorted({x for x in (_normalize_id(v) for v in clean_raw) if x})
    medium = sorted({x for x in (_normalize_id(v) for v in medium_raw) if x})
    bad = sorted({x for x in (_normalize_id(v) for v in bad_raw) if x})

    overlap = (set(clean) & set(medium)) | (set(clean) & set(bad)) | (set(medium) & set(bad))
    if overlap:
        items = ", ".join(sorted(list(overlap))[:20])
        raise SystemExit(f"Some sample_ids appear in multiple categories: {items}")

    _write_list(out_dir / "clean.txt", clean)
    _write_list(out_dir / "medium.txt", medium)
    _write_list(out_dir / "bad.txt", bad)

    print("Applied curation_result.json")
    print(f"Input: {in_path}")
    print(f"Output dir: {out_dir}")
    print(f"clean: {len(clean)}")
    print(f"medium: {len(medium)}")
    print(f"bad: {len(bad)}")


if __name__ == "__main__":
    main()
