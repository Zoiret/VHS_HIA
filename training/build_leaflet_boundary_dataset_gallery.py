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


def _palette_rgb() -> dict[int, tuple[int, int, int]]:
    return {
        1: (230, 25, 75),
        2: (60, 180, 75),
        3: (255, 225, 25),
    }


def _colorize_instances(inst_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((inst_u8.shape[0], inst_u8.shape[1], 3), dtype=np.uint8)
    pal = _palette_rgb()
    for i in [1, 2, 3]:
        out[inst_u8 == i] = pal[i]
    return out


def _colorize_target(mask_u8: np.ndarray) -> np.ndarray:
    out = np.zeros((mask_u8.shape[0], mask_u8.shape[1], 3), dtype=np.uint8)
    out[mask_u8 == 1] = (0, 255, 0)
    out[mask_u8 == 2] = (255, 0, 0)
    out[mask_u8 == 3] = (255, 255, 0)
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


def _overlay_instances(image_rgb: np.ndarray, inst_u8: np.ndarray, target_u8: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    pal = _palette_rgb()
    for i in [1, 2, 3]:
        _draw_contours_rgb(out, inst_u8 == i, color_rgb=pal[i], thickness=2)
    _draw_contours_rgb(out, target_u8 == 2, color_rgb=(255, 0, 0), thickness=2)
    return out


def _overlay_boundary(image_rgb: np.ndarray, target_u8: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    _draw_contours_rgb(out, target_u8 == 3, color_rgb=(255, 255, 0), thickness=2)
    _draw_contours_rgb(out, target_u8 == 2, color_rgb=(255, 0, 0), thickness=2)
    return out


def _text(img: np.ndarray, x: int, y: int, s: str, *, scale: float = 0.6, color=(255, 255, 255), thickness: int = 1) -> None:
    cv2.putText(img, s, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, float(scale), tuple(int(c) for c in color), int(thickness), cv2.LINE_AA)


def _make_compare(
    *,
    original: np.ndarray,
    inst: np.ndarray,
    target: np.ndarray,
    overlay_inst: np.ndarray,
    overlay_bnd: np.ndarray,
    header_lines: list[str],
    legend_lines: list[str],
) -> np.ndarray:
    panels = [
        original,
        _colorize_instances(inst),
        _colorize_target(target),
        overlay_inst,
        overlay_bnd,
    ]
    grid = np.concatenate(panels, axis=1)
    header_h = 170
    header = np.zeros((header_h, grid.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 20, 20)

    y = 26
    for line in header_lines:
        _text(header, 12, y, line, scale=0.65, thickness=2)
        y += 24

    y = 26
    x0 = int(grid.shape[1] * 0.62)
    for line in legend_lines:
        _text(header, x0, y, line, scale=0.55, thickness=1)
        y += 20

    out = np.concatenate([header, grid], axis=0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary-root", type=Path, default=Path("datasets/converted_leaflet_boundary"))
    ap.add_argument("--source-instances-root", type=Path, default=Path("datasets/converted_leaflet_instances"))
    ap.add_argument("--curation-json", type=Path, default=Path("server_assets/curation/curation_result.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("training/analysis/leaflet_boundary_dataset_gallery"))
    ap.add_argument("--n-per-quality", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    b_root = args.boundary_root.resolve()
    s_root = args.source_instances_root.resolve()
    out_root = args.out_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    manifest_path = b_root / "boundary_dataset_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    manifest_rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8")))
    by_sample = {r["sample"]: r for r in manifest_rows}

    frac_vals = [float(r["boundary_fraction"]) for r in manifest_rows]
    p95 = float(np.percentile(np.asarray(frac_vals, dtype=np.float64), 95)) if frac_vals else 0.0

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

    one_leaflet = sorted([r["sample"] for r in manifest_rows if int(r["instance_count"]) == 1])
    suspicious = sorted([r["sample"] for r in manifest_rows if float(r["boundary_fraction"]) > p95])
    special = sorted(
        {
            *one_leaflet,
            *suspicious,
            *[r["sample"] for r in manifest_rows if int(r.get("close_pairs_no_boundary", "0")) > 0],
        }
    )

    legend_lines = [
        "Legend:",
        "Target labels: 0 bg, 1 leaflet interior (green), 2 ring (red), 3 boundary (yellow)",
        "Instances: A/B/C are technical IDs per-image (not anatomical)",
        "Instance colors: A=red, B=green, C=yellow",
        "Overlays: instance contours colored; boundary contour yellow; ring contour red",
    ]

    def render(sample_id: str, dst_dir: Path) -> None:
        b_img = b_root / "images" / f"{sample_id}.png"
        b_mask = b_root / "masks" / f"{sample_id}.png"
        s_inst = s_root / "instance_masks" / f"{sample_id}.png"
        if not b_img.exists() or not b_mask.exists() or not s_inst.exists():
            return
        img = _read_rgb_u8(b_img)
        target = _read_u8(b_mask)
        inst = _read_u8(s_inst)
        if inst.shape != target.shape:
            return

        overlay_inst = _overlay_instances(img, inst, target)
        overlay_bnd = _overlay_boundary(img, target)

        row = by_sample.get(sample_id, {})
        header_lines = [
            f"sample={sample_id} split={row.get('split','')} quality={row.get('quality','')}",
            f"instances={row.get('instance_count','')}  boundary_pixels={row.get('boundary_pixels','')}  boundary_frac={float(row.get('boundary_fraction',0.0)):.4f}",
            "A/B/C are technical instance IDs per-image (not Leaflet 1/2/3)",
        ]

        compare = _make_compare(
            original=img,
            inst=inst,
            target=target,
            overlay_inst=overlay_inst,
            overlay_bnd=overlay_bnd,
            header_lines=header_lines,
            legend_lines=legend_lines,
        )

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
                "boundary_root": str(b_root),
                "source_instances_root": str(s_root),
                "out_dir": str(out_root),
                "picked_counts": {k: len(v) for k, v in picked.items()},
                "special_cases_count": int(len(special)),
                "p95_boundary_fraction": float(p95),
                "picked": picked,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Output: {out_root}")
    print(f"Picked: clean={len(picked['clean'])} medium={len(picked['medium'])} bad={len(picked['bad'])}")
    print(f"Special cases: {len(special)} (includes all instance_count=1 and p95 boundary fraction cases)")


if __name__ == "__main__":
    main()

