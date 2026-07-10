from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np


def _read_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _palette_rgb(n: int) -> list[tuple[int, int, int]]:
    base = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
    ]
    if n <= len(base):
        return base[:n]
    out = []
    for i in range(n):
        c = base[i % len(base)]
        k = 1 + (i // len(base))
        out.append((max(0, c[0] - 15 * k), max(0, c[1] - 10 * k), max(0, c[2] - 5 * k)))
    return out


def _colorize_semantic(mask_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    out[mask_u8 == 1] = (0, 255, 0)
    out[mask_u8 == 2] = (255, 0, 0)
    return out


def _colorize_instances(mask_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    colors = _palette_rgb(3)
    for i in [1, 2, 3]:
        out[mask_u8 == i] = colors[i - 1]
    return out


def _find_contours(mask01_u8: np.ndarray):
    res = cv2.findContours(mask01_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(res) == 2:
        contours, hierarchy = res
        return contours, hierarchy
    _, contours, hierarchy = res
    return contours, hierarchy


def _draw_contours_rgb(image_rgb_u8: np.ndarray, mask01: np.ndarray, *, color_rgb, thickness: int) -> None:
    m = (mask01.astype(np.uint8) * 255)
    if not np.any(m):
        return
    contours, _ = _find_contours(m)
    if not contours:
        return
    cv2.drawContours(image_rgb_u8, contours, -1, tuple(int(x) for x in color_rgb), int(thickness))


def _overlay_instances(image_rgb: np.ndarray, inst_u8: np.ndarray, sem_u8: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    colors = _palette_rgb(3)
    for i in [1, 2, 3]:
        _draw_contours_rgb(out, inst_u8 == i, color_rgb=colors[i - 1], thickness=2)
    _draw_contours_rgb(out, sem_u8 == 2, color_rgb=(255, 0, 0), thickness=2)
    return out


def _text(img: np.ndarray, x: int, y: int, s: str, *, scale: float = 0.55, color=(255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), tuple(int(c) for c in color), int(thickness), cv2.LINE_AA)


def _make_compare(*, original: np.ndarray, sem: np.ndarray, inst: np.ndarray, overlay: np.ndarray, meta_lines: list[str], legend_lines: list[str]) -> np.ndarray:
    h, w = original.shape[:2]
    grid = np.concatenate([original, _colorize_semantic(sem), _colorize_instances(inst), overlay], axis=1)
    header_h = 160
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)
    y = 26
    for line in meta_lines:
        _text(header, 12, y, line, scale=0.65, thickness=2)
        y += 24
    y = 26
    x0 = int(grid.shape[1] * 0.60)
    for line in legend_lines:
        _text(header, x0, y, line, scale=0.55, thickness=1)
        y += 20
    out = np.concatenate([header, grid], axis=0)
    cap_y = header_h + 28
    _text(out, 12, cap_y, "ORIGINAL", scale=0.8, thickness=2)
    _text(out, 12 + w, cap_y, "SEMANTIC (leaflet=green, annulus=red)", scale=0.8, thickness=2)
    _text(out, 12 + 2 * w, cap_y, "INSTANCES (A/B/C technical colors)", scale=0.8, thickness=2)
    _text(out, 12 + 3 * w, cap_y, "OVERLAY (instance contours + annulus contour)", scale=0.8, thickness=2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/leaflet_instance_dataset_gallery"))
    ap.add_argument("--n-per-quality", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    ds = args.dataset_root.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    cur = json.loads(args.curation_json.resolve().read_text(encoding="utf-8"))
    clean = [str(x) for x in cur.get("clean", [])]
    medium = [str(x) for x in cur.get("medium", [])]
    bad = [str(x) for x in cur.get("bad", [])]

    manifest_path = ds / "instance_dataset_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    rows = manifest_path.read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",")
    idx = {k: i for i, k in enumerate(header)}
    by_sample = {}
    for line in rows[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        s = parts[idx["sample"]]
        by_sample[s] = {
            "split": parts[idx["split"]],
            "source_leaf_objects": int(parts[idx["source_leaf_objects"]]),
            "selected_instances": int(parts[idx["selected_instances"]]),
            "overlap_pixels": int(parts[idx["overlap_pixels"]]),
            "warning_count": int(parts[idx["warning_count"]]),
            "conversion_ok": int(parts[idx["conversion_ok"]]),
        }

    def pick(ids: list[str], n: int) -> list[str]:
        r = random.Random(int(args.seed))
        present = [x for x in ids if x in by_sample]
        r.shuffle(present)
        return present[: int(n)]

    picked = {
        "clean": pick(clean, args.n_per_quality),
        "medium": pick(medium, args.n_per_quality),
        "bad": pick(bad, args.n_per_quality),
    }

    specials = set()
    for s, info in by_sample.items():
        if info["source_leaf_objects"] > 3:
            specials.add(s)
        if info["overlap_pixels"] > 0:
            specials.add(s)
        if info["warning_count"] > 0:
            specials.add(s)
        if info["conversion_ok"] == 0:
            specials.add(s)

    legend_lines = [
        "Legend:",
        "Semantic: leaflet union=green, annulus=red, bg=black",
        "Instances: A/B/C are technical IDs per-image (not anatomical)",
        "Instance colors: A=red, B=green, C=yellow (see panel)",
        "Contours: instance contours colored; annulus contour red",
    ]

    def render_one(sample_id: str, subdir: Path) -> None:
        img_p = ds / "images" / f"{sample_id}.png"
        sem_p = ds / "semantic_masks" / f"{sample_id}.png"
        inst_p = ds / "instance_masks" / f"{sample_id}.png"
        meta_p = ds / "metadata" / f"{sample_id}.json"
        if not (img_p.exists() and sem_p.exists() and inst_p.exists() and meta_p.exists()):
            return

        img = _read_rgb_u8(img_p)
        sem = _read_u8(sem_p)
        inst = _read_u8(inst_p)
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        info = by_sample.get(sample_id, {})

        overlay = _overlay_instances(img, inst, sem)
        meta_lines = [
            f"sample={sample_id}  split={info.get('split','')}  quality={meta.get('quality')}",
            f"source Leaf objects={meta.get('leaflet_source_objects_total')}  selected instances={meta.get('leaflet_selected_instances')}",
            f"overlap_pixels={meta.get('overlap_pixels')}  warnings={len(meta.get('warnings') or [])}",
            "A/B/C are technical instance IDs per-image (not Leaflet 1/2/3)",
        ]
        compare = _make_compare(original=img, sem=sem, inst=inst, overlay=overlay, meta_lines=meta_lines, legend_lines=legend_lines)

        subdir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(subdir / "compare.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

    for q, ids in picked.items():
        for sid in ids:
            render_one(sid, out_root / q / sid)

    for sid in sorted(list(specials)):
        render_one(sid, out_root / "special_cases" / sid)

    out_root.joinpath("gallery_manifest.json").write_text(
        json.dumps(
            {
                "dataset_root": str(ds),
                "out_dir": str(out_root),
                "picked_counts": {k: len(v) for k, v in picked.items()},
                "special_cases_count": int(len(specials)),
                "picked": picked,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Output: {out_root}")
    print(f"Picked: clean={len(picked['clean'])} medium={len(picked['medium'])} bad={len(picked['bad'])}")
    print(f"Special cases: {len(specials)}")


if __name__ == "__main__":
    main()

