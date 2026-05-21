import argparse
import html
import json
import os
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Curation Gallery</title>
    <style>
      :root {{
        --bg: #0f1216;
        --panel: #161b22;
        --text: #e6edf3;
        --muted: #9da7b3;
        --border: #2b313a;
        --clean: #2ea043;
        --medium: #d29922;
        --bad: #f85149;
        --unset: #6e7681;
      }}

      body {{
        margin: 0;
        font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        background: var(--bg);
        color: var(--text);
      }}

      header {{
        position: sticky;
        top: 0;
        z-index: 10;
        background: rgba(15, 18, 22, 0.92);
        backdrop-filter: blur(6px);
        border-bottom: 1px solid var(--border);
        padding: 12px 16px;
      }}

      .row {{
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        align-items: center;
      }}

      .pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 10px;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--panel);
        font-size: 13px;
        color: var(--muted);
      }}

      .pill b {{ color: var(--text); font-weight: 600; }}

      input[type="text"] {{
        min-width: 280px;
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        outline: none;
      }}

      select {{
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        outline: none;
      }}

      button {{
        padding: 8px 10px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        cursor: pointer;
        font-weight: 600;
      }}

      button:hover {{
        border-color: #3b424c;
      }}

      .toggle {{
        display: inline-flex;
        border: 1px solid var(--border);
        border-radius: 10px;
        overflow: hidden;
        background: var(--panel);
      }}
      .toggle button {{
        border: 0;
        border-radius: 0;
      }}
      .toggle button.active {{
        background: #202734;
      }}

      main {{
        padding: 16px;
      }}

      .grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(420px, 1fr));
        gap: 12px;
      }}

      .card {{
        border: 2px solid var(--unset);
        border-radius: 12px;
        overflow: hidden;
        background: var(--panel);
      }}

      .card.clean {{ border-color: var(--clean); }}
      .card.medium {{ border-color: var(--medium); }}
      .card.bad {{ border-color: var(--bad); }}
      .card.unset {{ border-color: var(--unset); }}

      .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 10px;
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
      }}

      .card-header-right {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }}

      .id {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
        color: var(--text);
        word-break: break-all;
      }}

      .src {{
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        color: var(--muted);
      }}

      .status-badge {{
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        color: var(--muted);
      }}

      .status-badge.clean {{ color: var(--clean); }}
      .status-badge.medium {{ color: var(--medium); }}
      .status-badge.bad {{ color: var(--bad); }}
      .status-badge.unset {{ color: var(--unset); }}

      img {{
        width: 100%;
        height: auto;
        display: block;
        background: #0b0e12;
      }}

      body.view-original img.img-preview {{ display: none; }}
      body.view-overlay img.img-original {{ display: none; }}

      .actions {{
        display: flex;
        gap: 8px;
        padding: 10px 12px 12px;
        border-top: 1px solid var(--border);
      }}

      .btn-clean {{ border-color: rgba(46,160,67,0.7); }}
      .btn-medium {{ border-color: rgba(210,153,34,0.7); }}
      .btn-bad {{ border-color: rgba(248,81,73,0.7); }}

      .export {{
        margin-top: 14px;
        display: grid;
        gap: 8px;
      }}

      textarea {{
        width: 100%;
        min-height: 140px;
        padding: 10px;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        font-size: 12px;
      }}
    </style>
  </head>
  <body>
    <header>
      <div class="row">
        <div class="pill"><b>Samples</b> <span id="countTotal">0</span></div>
        <div class="pill"><b>Clean</b> <span id="countClean">0</span></div>
        <div class="pill"><b>Medium</b> <span id="countMedium">0</span></div>
        <div class="pill"><b>Bad</b> <span id="countBad">0</span></div>
        <div class="pill"><b>Unset</b> <span id="countUnset">0</span></div>
        <div class="toggle" role="group" aria-label="View mode">
          <button id="btnViewOverlay" class="active">Show GT overlay</button>
          <button id="btnViewOriginal">Show original</button>
        </div>
        <select id="statusFilter">
          <option value="all">all</option>
          <option value="unset">unset</option>
          <option value="clean">clean</option>
          <option value="medium">medium</option>
          <option value="bad">bad</option>
        </select>
        <button id="btnOnlyUnset">Show only unset</button>
        <input id="search" type="text" placeholder="Search by sample id..." />
        <button id="btnExport">Export</button>
        <button id="btnDownload">Download JSON</button>
        <button id="btnCopy">Copy to clipboard</button>
      </div>
      <div class="export">
        <textarea id="exportText" placeholder="Export JSON will appear here..."></textarea>
      </div>
    </header>
    <main>
      <div class="grid" id="grid">
        {cards_html}
      </div>
    </main>
    <script>
      const allowed = new Set(["clean", "medium", "bad", "unset"]);

      function setStatus(id, status) {{
        if (!allowed.has(status)) return;
        const card = document.querySelector(`.card[data-id="${{CSS.escape(id)}}"]`);
        if (!card) return;
        card.dataset.status = status;
        card.classList.remove("clean", "medium", "bad", "unset");
        card.classList.add(status);
        const badge = card.querySelector(".status-badge");
        badge.textContent = status;
        badge.classList.remove("clean", "medium", "bad", "unset");
        badge.classList.add(status);
        updateCounts();
      }}

      function updateCounts() {{
        const cards = Array.from(document.querySelectorAll(".card"));
        let clean = 0, medium = 0, bad = 0, unset = 0;
        for (const c of cards) {{
          const s = c.dataset.status || "unset";
          if (s === "clean") clean++;
          else if (s === "medium") medium++;
          else if (s === "bad") bad++;
          else unset++;
        }}
        document.getElementById("countTotal").textContent = String(cards.length);
        document.getElementById("countClean").textContent = String(clean);
        document.getElementById("countMedium").textContent = String(medium);
        document.getElementById("countBad").textContent = String(bad);
        document.getElementById("countUnset").textContent = String(unset);
      }}

      function buildExport() {{
        const out = {{ clean: [], medium: [], bad: [] }};
        const cards = Array.from(document.querySelectorAll(".card"));
        for (const c of cards) {{
          const id = c.dataset.id;
          const s = c.dataset.status || "unset";
          if (s === "clean") out.clean.push(id);
          else if (s === "medium") out.medium.push(id);
          else if (s === "bad") out.bad.push(id);
        }}
        return out;
      }}

      function renderExport() {{
        const obj = buildExport();
        const text = JSON.stringify(obj, null, 2);
        const ta = document.getElementById("exportText");
        ta.value = text;
        return text;
      }}

      function downloadExport() {{
        const text = renderExport();
        const blob = new Blob([text], {{ type: "application/json" }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "curation_result.json";
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      }}

      async function copyExport() {{
        const text = renderExport();
        const ta = document.getElementById("exportText");
        ta.focus();
        ta.select();
        try {{
          if (navigator.clipboard && window.isSecureContext) {{
            await navigator.clipboard.writeText(text);
            return;
          }}
        }} catch (_) {{}}
        document.execCommand("copy");
      }}

      function applyFilters() {{
        const q = (document.getElementById("search").value || "").toLowerCase().trim();
        const status = document.getElementById("statusFilter").value || "all";
        const cards = Array.from(document.querySelectorAll(".card"));
        for (const c of cards) {{
          const id = (c.dataset.id || "").toLowerCase();
          const s = c.dataset.status || "unset";
          const okText = (q === "" || id.includes(q));
          const okStatus = (status === "all" || s === status);
          c.style.display = (okText && okStatus) ? "" : "none";
        }}
      }}

      document.getElementById("search").addEventListener("input", applyFilters);
      document.getElementById("statusFilter").addEventListener("change", applyFilters);
      document.getElementById("btnExport").addEventListener("click", renderExport);
      document.getElementById("btnDownload").addEventListener("click", downloadExport);
      document.getElementById("btnCopy").addEventListener("click", copyExport);
      document.getElementById("btnOnlyUnset").addEventListener("click", () => {{
        document.getElementById("statusFilter").value = "unset";
        applyFilters();
      }});

      function setView(mode) {{
        document.body.classList.remove("view-overlay", "view-original");
        document.body.classList.add(mode === "original" ? "view-original" : "view-overlay");
        document.getElementById("btnViewOverlay").classList.toggle("active", mode !== "original");
        document.getElementById("btnViewOriginal").classList.toggle("active", mode === "original");
      }}

      document.getElementById("btnViewOverlay").addEventListener("click", () => setView("overlay"));
      document.getElementById("btnViewOriginal").addEventListener("click", () => setView("original"));

      document.getElementById("grid").addEventListener("click", (e) => {{
        const btn = e.target.closest("button[data-status]");
        if (!btn) return;
        const card = btn.closest(".card");
        if (!card) return;
        const id = card.dataset.id;
        const status = btn.dataset.status;
        setStatus(id, status);
      }});

      updateCounts();
      applyFilters();
      setView("overlay");
    </script>
  </body>
</html>
"""


def _make_card(sample_id: str, rel_preview_src: str, rel_original_src: str, preview_label: str) -> str:
    safe_id = sample_id.replace("\\", "/").split("/")[-1]
    safe_id = Path(safe_id).stem
    sid_attr = html.escape(safe_id, quote=True)
    sid_text = html.escape(safe_id, quote=False)
    src_preview = html.escape(rel_preview_src, quote=True)
    src_original = html.escape(rel_original_src, quote=True)
    src_label = html.escape(preview_label, quote=False)
    return f"""
<div class="card unset" data-id="{sid_attr}" data-status="unset">
  <div class="card-header">
    <div class="id">{sid_text}</div>
    <div class="card-header-right">
      <div class="src">source: {src_label}</div>
      <div class="status-badge unset">unset</div>
    </div>
  </div>
  <img class="img-preview" src="{src_preview}" loading="lazy" />
  <img class="img-original" src="{src_original}" loading="lazy" />
  <div class="actions">
    <button class="btn-clean" data-status="clean">clean</button>
    <button class="btn-medium" data-status="medium">medium</button>
    <button class="btn-bad" data-status="bad">bad</button>
    <button data-status="unset">unset</button>
  </div>
</div>
""".strip()


def _rel_path(from_dir: Path, target: Path) -> str:
    return Path(os.path.relpath(target, from_dir)).as_posix()


def _collect_source_inference(compare_dir: Path) -> list[tuple[str, Path]]:
    images = sorted([p for p in compare_dir.glob("*.png") if p.is_file()])
    return [(p.stem, p) for p in images]


def _collect_source_dataset(images_dir: Path) -> list[tuple[str, Path]]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    paths = sorted([p for p in images_dir.glob("*") if p.is_file() and p.suffix.lower() in exts])
    return [(p.stem, p) for p in paths]


def _read_image_rgb_any(path: Path):
    import cv2
    import numpy as np
    from PIL import Image

    try:
        data = path.read_bytes()
    except Exception:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        with Image.open(path) as im:
            im.load()
            return np.array(im.convert("RGB"))
    except Exception:
        return None


def _read_mask_uint8(path: Path):
    import cv2

    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return None
    if m.ndim == 3:
        m = m[:, :, 0]
    return m.astype("uint8")


def _write_gt_overlay_preview(image_path: Path, mask_path: Path, out_path: Path, alpha: float) -> bool:
    import numpy as np
    from PIL import Image

    img = _read_image_rgb_any(image_path)
    if img is None:
        return False
    mask = _read_mask_uint8(mask_path)
    if mask is None:
        return False
    if mask.ndim != 2:
        return False
    if img.shape[0] != mask.shape[0] or img.shape[1] != mask.shape[1]:
        return False

    base = img.astype(np.float32)
    overlay = base.copy()
    green = np.array([0, 255, 0], dtype=np.float32)
    red = np.array([255, 0, 0], dtype=np.float32)

    leaflet = mask == 1
    ring = mask == 2
    if leaflet.any():
        overlay[leaflet] = (1.0 - alpha) * overlay[leaflet] + alpha * green
    if ring.any():
        overlay[ring] = (1.0 - alpha) * overlay[ring] + alpha * red

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(out_path)
    return True


def _select_preview_path(sample_id: str, dataset_image_path: Path, dataset_mask_path: Path, generated_dir: Path) -> tuple[str, Path]:
    p1 = Path("training/inference_preview/compare") / f"{sample_id}.png"
    if p1.exists():
        return "inference", p1.resolve()
    p2 = Path("datasets/previews") / f"{sample_id}.png"
    if p2.exists():
        return "gt_overlay", p2.resolve()
    p3 = (generated_dir / f"{sample_id}.png").resolve()
    if p3.exists():
        return "gt_overlay", p3

    if dataset_mask_path.exists():
        ok = _write_gt_overlay_preview(
            image_path=dataset_image_path,
            mask_path=dataset_mask_path,
            out_path=p3,
            alpha=0.45,
        )
        if ok and p3.exists():
            return "gt_overlay", p3

    return "original", dataset_image_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local HTML gallery for manual curation.")
    parser.add_argument("--source", type=str, choices=["dataset", "inference"], default="dataset")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--compare-dir", type=Path, default=Path("training/inference_preview/compare"))
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets/converted"))
    parser.add_argument("--dataset-images-dir", type=Path, default=None)
    parser.add_argument("--dataset-masks-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("datasets/curated/curation_gallery.html"))
    parser.add_argument("--preview-dir", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--generated-previews-dir", type=Path, default=None)
    args = parser.parse_args()

    out_html = (args.out_html or args.out).resolve()
    rel_base = out_html.parent.resolve()

    items: list[tuple[str, Path]] = []
    if args.source == "inference":
        compare_dir = args.compare_dir.resolve()
        if not compare_dir.exists():
            raise SystemExit(f"Compare dir not found: {compare_dir}")
        items = _collect_source_inference(compare_dir)
    else:
        dataset_root = args.dataset_root.resolve()
        images_dir = (args.dataset_images_dir or (dataset_root / "images")).resolve()
        if not images_dir.exists():
            raise SystemExit(f"Dataset images dir not found: {images_dir}")
        items = _collect_source_dataset(images_dir)

    if not items:
        raise SystemExit("No samples found for gallery")

    out_html.parent.mkdir(parents=True, exist_ok=True)

    cards = []
    if args.limit is not None:
        items = items[: int(args.limit)]

    dataset_root = args.dataset_root.resolve()
    dataset_images_dir = (args.dataset_images_dir or (dataset_root / "images")).resolve()
    dataset_masks_dir = (args.dataset_masks_dir or (dataset_root / "masks")).resolve()

    default_out = Path("datasets/curated/curation_gallery.html").resolve()
    if args.generated_previews_dir is not None:
        generated_dir = args.generated_previews_dir.resolve()
    elif args.preview_dir is not None:
        generated_dir = args.preview_dir.resolve()
    else:
        if out_html == default_out:
            generated_dir = Path("datasets/curated/generated_previews").resolve()
        else:
            generated_dir = (out_html.parent / "generated_previews").resolve()

    for sample_id, src_path in items:
        original_path = None
        if args.source == "dataset":
            original_path = src_path.resolve()
        else:
            candidate = (dataset_images_dir / f"{sample_id}{src_path.suffix}").resolve()
            if not candidate.exists():
                candidate = (dataset_images_dir / f"{sample_id}.png").resolve()
            original_path = candidate if candidate.exists() else src_path.resolve()

        mask_path = (dataset_masks_dir / f"{sample_id}.png").resolve()

        preview_label = "inference" if args.source == "inference" else "original"
        preview_path = src_path.resolve()
        if args.source == "dataset":
            preview_label, preview_path = _select_preview_path(
                sample_id=sample_id,
                dataset_image_path=original_path,
                dataset_mask_path=mask_path,
                generated_dir=generated_dir,
            )
        else:
            if (Path("training/inference_preview/compare") / f"{sample_id}.png").exists():
                preview_label = "inference"
            elif (Path("datasets/previews") / f"{sample_id}.png").exists() or (generated_dir / f"{sample_id}.png").exists():
                preview_label = "gt_overlay"
            else:
                preview_label = "original"

        rel_preview = _rel_path(rel_base, preview_path.resolve())
        rel_original = _rel_path(rel_base, original_path.resolve())
        cards.append(_make_card(sample_id=sample_id, rel_preview_src=rel_preview, rel_original_src=rel_original, preview_label=preview_label))

    html_text = HTML_TEMPLATE.format(cards_html="\n".join(cards))
    out_html.write_text(html_text, encoding="utf-8")
    print(f"Wrote: {out_html}")
    print(f"Source: {args.source}")
    print(f"Samples: {len(items)}")


if __name__ == "__main__":
    main()
