from __future__ import annotations

import argparse
import csv
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


def _read_u16(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(str(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return arr


def _center_crop_to_shape(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == target_h and w == target_w:
        return arr
    if h < target_h or w < target_w:
        raise ValueError(f"Cannot crop {target_h}x{target_w} from {h}x{w}")
    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    if arr.ndim == 2:
        return arr[y0 : y0 + target_h, x0 : x0 + target_w]
    return arr[y0 : y0 + target_h, x0 : x0 + target_w, :]


def _palette_instances() -> dict[int, tuple[int, int, int]]:
    return {1: (230, 25, 75), 2: (60, 180, 75), 3: (255, 225, 25)}


def _colorize_semantic(sem_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((sem_u8.shape[0], sem_u8.shape[1], 3), dtype=np.uint8)
    out[sem_u8 == 1] = (0, 255, 0)
    out[sem_u8 == 2] = (255, 0, 0)
    return out


def _colorize_instances(inst_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((inst_u8.shape[0], inst_u8.shape[1], 3), dtype=np.uint8)
    pal = _palette_instances()
    for i in [1, 2, 3]:
        out[inst_u8 == i] = pal[i]
    return out


def _heatmap_u16(u16: np.ndarray) -> np.ndarray:
    x = (u16.astype(np.float32) / 65535.0) * 255.0
    x8 = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(x8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)


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


def _overlay_centers(image_rgb: np.ndarray, center_u16: np.ndarray, inst_u8: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    cm = center_u16.astype(np.float32) / 65535.0
    pal = _palette_instances()
    for iid in [1, 2, 3]:
        m = inst_u8 == iid
        if int(np.sum(m)) == 0:
            continue
        yy, xx = np.where(m)
        vals = cm[yy, xx]
        if vals.size == 0:
            continue
        k = int(np.argmax(vals))
        y = int(yy[k])
        x = int(xx[k])
        cv2.circle(out, (x, y), 6, tuple(int(c) for c in pal[iid]), thickness=2)
    return out


def _text(img: np.ndarray, x: int, y: int, s: str, *, scale: float = 0.55, color=(255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), tuple(int(c) for c in color), int(thickness), cv2.LINE_AA)


def _make_compare(*, panels: list[np.ndarray], header_lines: list[str], legend_lines: list[str]) -> np.ndarray:
    grid = np.concatenate(panels, axis=1)
    header_h = 190
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)
    y = 26
    for line in header_lines:
        _text(header, 12, y, line, scale=0.65, thickness=2)
        y += 24
    y = 26
    x0 = int(grid.shape[1] * 0.60)
    for line in legend_lines:
        _text(header, x0, y, line, scale=0.55, thickness=1)
        y += 20
    return np.concatenate([header, grid], axis=0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distance-root", type=Path, default=Path("datasets/converted_leaflet_distance"))
    ap.add_argument("--source-instances-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/leaflet_distance_dataset_gallery"))
    ap.add_argument("--n-per-quality", type=int, default=20)
    ap.add_argument("--random-3instance", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    d_root = args.distance_root.resolve()
    s_root = args.source_instances_root.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = d_root / "distance_dataset_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    by_sample = {r["sample"]: r for r in rows}

    cur = json.loads(args.curation_json.resolve().read_text(encoding="utf-8"))
    clean = [str(x) for x in cur.get("clean", [])]
    medium = [str(x) for x in cur.get("medium", [])]
    bad = [str(x) for x in cur.get("bad", [])]

    rng = random.Random(int(args.seed))

    def pick(ids: list[str], n: int) -> list[str]:
        present = [x for x in ids if x in by_sample]
        rng.shuffle(present)
        return present[: int(n)]

    picked = {"clean": pick(clean, args.n_per_quality), "medium": pick(medium, args.n_per_quality), "bad": pick(bad, args.n_per_quality)}

    one_inst = sorted([s for s, r in by_sample.items() if int(r["instance_count"]) == 1])
    two_inst = sorted([s for s, r in by_sample.items() if int(r["instance_count"]) == 2])
    three_inst = [s for s, r in by_sample.items() if int(r["instance_count"]) == 3]
    rng.shuffle(three_inst)
    three_inst = three_inst[: int(args.random_3instance)]

    special = sorted(set(one_inst) | set(two_inst) | set(three_inst))

    legend_lines = [
        "Legend:",
        "Semantic: leaflet=green, ring=red, bg=black",
        "Instances: A/B/C are technical IDs per-image (not anatomical)",
        "Instance colors: A=red, B=green, C=yellow",
        "Distance map: per-instance normalized DT, max-composed (turbo colormap)",
        "Centers: Gaussian peaks at argmax DT per instance (turbo colormap + markers)",
    ]

    def render(sid: str, dst_dir: Path) -> None:
        img_p = d_root / "images" / f"{sid}.png"
        sem_p = d_root / "semantic_masks" / f"{sid}.png"
        dist_p = d_root / "distance_maps" / f"{sid}.png"
        center_p = d_root / "center_maps" / f"{sid}.png"
        inst_p = s_root / "instance_masks" / f"{sid}.png"
        if not img_p.exists() or not sem_p.exists() or not dist_p.exists() or not inst_p.exists():
            return
        img = _read_rgb_u8(img_p)
        sem = _read_u8(sem_p)
        dist_u16 = _read_u16(dist_p)
        inst_src = _read_u8(inst_p)
        center_u16 = _read_u16(center_p) if center_p.exists() else np.zeros_like(dist_u16)
        inst = _center_crop_to_shape(inst_src, sem.shape[0], sem.shape[1])

        heat_dist = _heatmap_u16(dist_u16)
        heat_center = _heatmap_u16(center_u16)
        overlay_centers = _overlay_centers(img, center_u16, inst)

        row = by_sample.get(sid, {})
        header_lines = [
            f"sample={sid} split={row.get('split','')} quality={row.get('quality','')}",
            f"instances={row.get('instance_count','')}  dist_format=png_u16  center_sigma=4px",
            "A/B/C are technical instance IDs per-image (not Leaflet 1/2/3)",
        ]

        panels = [img, _colorize_semantic(sem), _colorize_instances(inst), heat_dist, heat_center, overlay_centers]
        compare = _make_compare(panels=panels, header_lines=header_lines, legend_lines=legend_lines)
        dst_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_dir / "compare.png"), cv2.cvtColor(compare, cv2.COLOR_RGB2BGR))

    for q, ids in picked.items():
        for sid in ids:
            render(sid, out_root / q / sid)

    for sid in special:
        render(sid, out_root / "special_cases" / sid)

    out_root.joinpath("gallery_manifest.json").write_text(
        json.dumps(
            {
                "distance_root": str(d_root),
                "source_instances_root": str(s_root),
                "out_dir": str(out_root),
                "picked_counts": {k: len(v) for k, v in picked.items()},
                "special_counts": {"1_instance": len(one_inst), "2_instance": len(two_inst), "random_3_instance": len(three_inst)},
                "picked": picked,
                "special_cases": special,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Output: {out_root}")
    print(f"Picked: clean={len(picked['clean'])} medium={len(picked['medium'])} bad={len(picked['bad'])}")
    print(f"Special: one={len(one_inst)} two={len(two_inst)} random3={len(three_inst)} total_special={len(special)}")


if __name__ == "__main__":
    main()
