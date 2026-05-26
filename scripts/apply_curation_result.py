import argparse
import json
import re
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


def _extract_all_ids_from_gallery_html(path: Path) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    ids = re.findall(r'<div class="card [^"]*" data-id="([^"]+)"', text)
    out = []
    for s in ids:
        nid = _normalize_id(s)
        if nid:
            out.append(nid)
    return sorted(set(out))


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
    has_unset_key = "unset" in obj
    unset_raw = obj.get("unset", [])
    if not isinstance(clean_raw, list) or not isinstance(medium_raw, list) or not isinstance(bad_raw, list) or not isinstance(unset_raw, list):
        raise SystemExit('curation_result.json must contain lists: "clean", "medium", "bad", optional "unset"')

    clean = sorted({x for x in (_normalize_id(v) for v in clean_raw) if x})
    medium = sorted({x for x in (_normalize_id(v) for v in medium_raw) if x})
    bad = sorted({x for x in (_normalize_id(v) for v in bad_raw) if x})
    unset = sorted({x for x in (_normalize_id(v) for v in unset_raw) if x})
    if not has_unset_key:
        gallery_ids = _extract_all_ids_from_gallery_html(out_dir / "curation_gallery.html")
        if gallery_ids:
            used = set(clean) | set(medium) | set(bad)
            unset = sorted([x for x in gallery_ids if x not in used])
        else:
            print("WARNING: curation_result.json has no 'unset' list and curation_gallery.html not found; unset will be treated as empty.")

    overlap = (
        (set(clean) & set(medium))
        | (set(clean) & set(bad))
        | (set(medium) & set(bad))
        | (set(clean) & set(unset))
        | (set(medium) & set(unset))
        | (set(bad) & set(unset))
    )
    if overlap:
        items = ", ".join(sorted(list(overlap))[:20])
        raise SystemExit(f"Some sample_ids appear in multiple categories: {items}")

    _write_list(out_dir / "clean.txt", clean)
    _write_list(out_dir / "medium.txt", medium)
    _write_list(out_dir / "bad.txt", bad)
    _write_list(out_dir / "unset.txt", unset)

    print("Applied curation_result.json")
    print(f"Input: {in_path}")
    print(f"Output dir: {out_dir}")
    print(f"clean: {len(clean)}")
    print(f"medium: {len(medium)}")
    print(f"bad: {len(bad)}")
    print(f"unset: {len(unset)}")


if __name__ == "__main__":
    main()
