from __future__ import annotations

import contextlib
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e

from augmentations import get_train_augmentations, get_val_augmentations
from dataset import read_split_file
from losses import CombinedCrossEntropyDiceLoss
from metrics import compute_per_class_metrics_from_logits


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_CONFIG = REPO_ROOT / "training" / "configs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep.yaml"
DEFAULT_SEMANTIC_CHECKPOINT = (
    REPO_ROOT / "training" / "runs" / "unetpp_effb3_a100_multiclass_curated_finetune_stage2_lr1e5_100ep" / "best_mean_fg.pth"
)
DEFAULT_SEMANTIC_DATASET_ROOT = REPO_ROOT / "datasets" / "converted_full_multiclass"
DEFAULT_SEMANTIC_TRAIN_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "train.txt"
DEFAULT_SEMANTIC_VAL_SPLIT = REPO_ROOT / "datasets" / "converted_full_multiclass_curated" / "val.txt"
DEFAULT_INSTANCE_ROOT = REPO_ROOT / "datasets" / "converted_leaflet_instances"
DEFAULT_RESEARCH_MANIFEST = REPO_ROOT / "training" / "manifests" / "center_full_val_manifest.jsonl"
DEFAULT_PREP_OUTPUT_DIR = REPO_ROOT / "training" / "analysis" / "semantic_topology_aux_finetune_prep"


@dataclass(frozen=True)
class TopologyTargetContract:
    boundary_width_px: int
    separation_width_px: int
    narrow_width_threshold_px: int
    source_split_txt: str
    source_instance_root: str
    selection_rule: str
    train_only: bool = True


def _resolve_repo_path(path_like: str | Path | None, default: Path) -> Path:
    if path_like is None:
        return default.resolve()
    path = Path(path_like)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as e:
        raise SystemExit(
            "PyYAML is not installed. Install training deps with:\n"
            "  py -m pip install -r requirements-train.txt"
        ) from e
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"Expected YAML dict at {path}")
    return data


def _simple_preprocess_uint8_rgb(image_rgb_u8: np.ndarray) -> np.ndarray:
    return image_rgb_u8.astype(np.float32) / 255.0


def _autocast_ctx(device: torch.device, enabled: bool):
    if device.type == "cuda" and bool(enabled):
        return torch.amp.autocast("cuda", enabled=True)
    return contextlib.nullcontext()


def _amp_enabled(cfg: dict[str, Any], device: torch.device) -> bool:
    train_cfg = cfg.get("train") or {}
    if not isinstance(train_cfg, dict):
        return device.type == "cuda"
    v = train_cfg.get("amp", None)
    if v is None:
        return device.type == "cuda"
    return bool(v) and device.type == "cuda"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_image_rgb(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _read_u8(path: Path) -> np.ndarray:
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def _center_crop_like_validation(image: np.ndarray, crop_h: int, crop_w: int, *, is_mask: bool) -> np.ndarray:
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        new_h = max(h, crop_h)
        new_w = max(w, crop_w)
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        image = cv2.resize(image, (new_w, new_h), interpolation=interp)
        h, w = image.shape[:2]
    y0 = (h - crop_h) // 2 if h > crop_h else 0
    x0 = (w - crop_w) // 2 if w > crop_w else 0
    if image.ndim == 2:
        return image[y0 : y0 + crop_h, x0 : x0 + crop_w]
    return image[y0 : y0 + crop_h, x0 : x0 + crop_w, :]


def _leaflet_union_from_instance_mask(instance_mask_u8: np.ndarray) -> np.ndarray:
    return (instance_mask_u8 > 0).astype(np.uint8)


def _positive_instance_ids(instance_mask_u8: np.ndarray) -> list[int]:
    return [int(v) for v in np.unique(instance_mask_u8) if int(v) > 0]


def _ellipse_kernel(radius_px: int) -> np.ndarray:
    r = max(int(radius_px), 0)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _morph_band(mask01: np.ndarray, radius_px: int) -> np.ndarray:
    if int(radius_px) <= 0:
        return np.zeros_like(mask01, dtype=np.uint8)
    kernel = _ellipse_kernel(int(radius_px))
    dil = cv2.dilate(mask01.astype(np.uint8), kernel, iterations=1)
    ero = cv2.erode(mask01.astype(np.uint8), kernel, iterations=1)
    return (dil != ero).astype(np.uint8)


def _boundary_mask(mask01: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask01.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
    return ((mask01.astype(np.uint8) > 0) & (eroded == 0)).astype(np.uint8)


def _pairwise_gap_distances(instance_mask_u8: np.ndarray) -> list[float]:
    ids = _positive_instance_ids(instance_mask_u8)
    out: list[float] = []
    boundaries = {inst_id: _boundary_mask((instance_mask_u8 == inst_id).astype(np.uint8)) for inst_id in ids}
    for i, left_id in enumerate(ids):
        for right_id in ids[i + 1 :]:
            right_boundary = boundaries[int(right_id)]
            inv = (1 - right_boundary.astype(np.uint8)).astype(np.uint8)
            dt = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
            left_pts = boundaries[int(left_id)].astype(bool)
            if not np.any(left_pts):
                continue
            out.append(float(np.min(dt[left_pts])))
    return out


def _local_widths(instance_mask_u8: np.ndarray) -> np.ndarray:
    widths: list[np.ndarray] = []
    for inst_id in _positive_instance_ids(instance_mask_u8):
        inst01 = (instance_mask_u8 == int(inst_id)).astype(np.uint8)
        dt = cv2.distanceTransform(inst01, cv2.DIST_L2, 5).astype(np.float32)
        vals = (2.0 * dt[inst01 > 0]).astype(np.float32)
        if vals.size > 0:
            widths.append(vals)
    if not widths:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(widths, axis=0).astype(np.float32)


def _instance_component_counts(instance_mask_u8: np.ndarray) -> dict[int, int]:
    out: dict[int, int] = {}
    for inst_id in _positive_instance_ids(instance_mask_u8):
        inst01 = (instance_mask_u8 == int(inst_id)).astype(np.uint8)
        n, _labels = cv2.connectedComponents(inst01, connectivity=8)
        out[int(inst_id)] = max(int(n) - 1, 0)
    return out


def _quantiles(values: np.ndarray | list[float], qs: list[float]) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return {f"p{int(q)}": None for q in qs}
    return {f"p{int(q)}": float(np.percentile(arr, q)) for q in qs}


def choose_topology_target_contract(
    *,
    dataset_root: Path,
    train_split_txt: Path,
    instance_root: Path,
) -> tuple[TopologyTargetContract, dict[str, Any]]:
    items = read_split_file(dataset_root.resolve(), train_split_txt.resolve())
    gap_values: list[float] = []
    width_values: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    disconnected_examples = 0

    for item in items:
        sample_id = Path(item.image_path).stem
        instance_path = instance_root.resolve() / "instance_masks" / f"{sample_id}.png"
        inst = _read_u8(instance_path)
        ids = _positive_instance_ids(inst)
        gaps = _pairwise_gap_distances(inst)
        widths = _local_widths(inst)
        comp_counts = _instance_component_counts(inst)
        if any(int(v) > 1 for v in comp_counts.values()):
            disconnected_examples += 1
        gap_values.extend(gaps)
        width_values.extend(widths.tolist())
        sample_rows.append(
            {
                "sample_id": sample_id,
                "gt_count": int(len(ids)),
                "pair_gap_min_px": float(min(gaps)) if gaps else None,
                "pair_gap_mean_px": float(np.mean(gaps)) if gaps else None,
                "local_width_min_px": float(np.min(widths)) if widths.size else None,
                "local_width_p10_px": float(np.percentile(widths, 10)) if widths.size else None,
                "local_width_median_px": float(np.median(widths)) if widths.size else None,
                "has_disconnected_instance": int(any(int(v) > 1 for v in comp_counts.values())),
            }
        )

    gap_arr = np.asarray(gap_values, dtype=np.float32)
    width_arr = np.asarray(width_values, dtype=np.float32)
    gap_p10 = float(np.percentile(gap_arr, 10)) if gap_arr.size else 8.0
    width_p25 = float(np.percentile(width_arr, 25)) if width_arr.size else 8.0
    width_p20 = float(np.percentile(width_arr, 20)) if width_arr.size else 6.0

    boundary_width_px = int(np.clip(round(min(gap_p10 / 3.0, width_p25 / 4.0)), 2, 4))
    separation_width_px = int(boundary_width_px)
    narrow_width_threshold_px = int(np.clip(round(max(width_p20, float(2 * boundary_width_px + 1))), 5, 12))

    contract = TopologyTargetContract(
        boundary_width_px=int(boundary_width_px),
        separation_width_px=int(separation_width_px),
        narrow_width_threshold_px=int(narrow_width_threshold_px),
        source_split_txt=str(train_split_txt.resolve()),
        source_instance_root=str(instance_root.resolve()),
        selection_rule=(
            "boundary_width_px = clamp(round(min(p10_inter_instance_gap/3, p25_local_width/4)), 2, 4); "
            "separation_width_px = boundary_width_px; "
            "narrow_width_threshold_px = clamp(round(max(p20_local_width, 2*boundary_width_px+1)), 5, 12)"
        ),
        train_only=True,
    )
    audit = {
        "sample_count": int(len(sample_rows)),
        "pair_gap_distance_px": {
            "count": int(gap_arr.size),
            "mean": float(gap_arr.mean()) if gap_arr.size else None,
            "median": float(np.median(gap_arr)) if gap_arr.size else None,
            **_quantiles(gap_arr, [5, 10, 25, 75, 90, 95]),
        },
        "local_leaflet_width_px": {
            "count": int(width_arr.size),
            "mean": float(width_arr.mean()) if width_arr.size else None,
            "median": float(np.median(width_arr)) if width_arr.size else None,
            **_quantiles(width_arr, [5, 10, 20, 25, 50, 75, 90, 95]),
        },
        "disconnected_same_leaflet_samples": int(disconnected_examples),
        "chosen_contract": asdict(contract),
        "sample_rows": sample_rows,
    }
    return contract, audit


def generate_topology_target(
    instance_mask_u8: np.ndarray,
    contract: TopologyTargetContract,
    *,
    return_parts: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
    instance_mask_u8 = instance_mask_u8.astype(np.uint8)
    ids = _positive_instance_ids(instance_mask_u8)
    h, w = instance_mask_u8.shape[:2]
    union_fg = (instance_mask_u8 > 0).astype(np.uint8)

    boundary = np.zeros((h, w), dtype=np.uint8)
    separation = np.zeros((h, w), dtype=np.uint8)
    narrow = np.zeros((h, w), dtype=np.uint8)
    sep_kernel = _ellipse_kernel(int(contract.separation_width_px))

    inst_masks: dict[int, np.ndarray] = {}
    for inst_id in ids:
        inst01 = (instance_mask_u8 == int(inst_id)).astype(np.uint8)
        inst_masks[int(inst_id)] = inst01
        boundary = np.maximum(boundary, _morph_band(inst01, int(contract.boundary_width_px)))
        dt = cv2.distanceTransform(inst01, cv2.DIST_L2, 5).astype(np.float32)
        width_map = 2.0 * dt
        narrow = np.maximum(
            narrow,
            ((inst01 > 0) & (width_map > 0.0) & (width_map <= float(contract.narrow_width_threshold_px))).astype(np.uint8),
        )

    for idx, left_id in enumerate(ids):
        left_dil = cv2.dilate(inst_masks[int(left_id)], sep_kernel, iterations=1)
        for right_id in ids[idx + 1 :]:
            right_dil = cv2.dilate(inst_masks[int(right_id)], sep_kernel, iterations=1)
            overlap = ((left_dil > 0) & (right_dil > 0) & (union_fg == 0)).astype(np.uint8)
            separation = np.maximum(separation, overlap)

    target = ((boundary > 0) | (separation > 0) | (narrow > 0)).astype(np.uint8)
    if not return_parts:
        return target
    return target, {
        "boundary": boundary.astype(np.uint8),
        "separation": separation.astype(np.uint8),
        "narrow": narrow.astype(np.uint8),
    }


class SemanticTopologyAuxDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_root: Path,
        split_txt: Path,
        instance_root: Path,
        contract: TopologyTargetContract,
        num_classes: int,
        input_size: int,
        augment_cfg: dict[str, Any] | None,
        training: bool,
    ) -> None:
        self.dataset_root = dataset_root.resolve()
        self.split_txt = split_txt.resolve()
        self.instance_root = instance_root.resolve()
        self.contract = contract
        self.num_classes = int(num_classes)
        self.input_size = int(input_size)
        self.items = read_split_file(self.dataset_root, self.split_txt)
        self.augment_fn = get_train_augmentations(self.input_size, self.input_size, augment_cfg) if training else get_val_augmentations(self.input_size, self.input_size)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx]
        sample_id = Path(item.image_path).stem
        image = _read_image_rgb(item.image_path)
        mask = _read_u8(item.mask_path)
        instance_mask = _read_u8(self.instance_root / "instance_masks" / f"{sample_id}.png")
        topology_target = generate_topology_target(instance_mask, self.contract)
        image, mask, topology_target = self.augment_fn(image, mask, topology_target)
        image = _simple_preprocess_uint8_rgb(image)
        image_t = torch.from_numpy(image.transpose(2, 0, 1)).float()
        mask_t = torch.from_numpy(mask.astype(np.uint8)).long()
        topology_t = torch.from_numpy(topology_target.astype(np.float32))
        return {
            "image": image_t,
            "mask": mask_t,
            "topology_target": topology_t,
            "sample_id": sample_id,
            "image_path": str(item.image_path),
            "mask_path": str(item.mask_path),
            "instance_path": str((self.instance_root / "instance_masks" / f"{sample_id}.png").resolve()),
        }


class BinaryBCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1e-7) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float()
        if target.ndim == 3:
            target = target.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, target)
        prob = torch.sigmoid(logits)
        inter = torch.sum(prob * target, dim=(1, 2, 3))
        denom = torch.sum(prob, dim=(1, 2, 3)) + torch.sum(target, dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * inter + self.eps) / (denom + self.eps))
        return self.bce_weight * bce + self.dice_weight * dice_loss.mean()


class SemanticTopologyAuxLoss(nn.Module):
    def __init__(
        self,
        *,
        semantic_loss: nn.Module,
        lambda_topology: float,
        topology_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.semantic_loss = semantic_loss
        self.lambda_topology = float(lambda_topology)
        self.topology_loss = topology_loss if topology_loss is not None else BinaryBCEDiceLoss()

    def forward(
        self,
        *,
        semantic_logits: torch.Tensor,
        semantic_target: torch.Tensor,
        topology_logits: torch.Tensor,
        topology_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        semantic = self.semantic_loss(semantic_logits, semantic_target)
        topology = self.topology_loss(topology_logits, topology_target)
        total = semantic + float(self.lambda_topology) * topology
        return {
            "semantic_loss": semantic,
            "topology_loss": topology,
            "combined_loss": total,
        }


class TopologyHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 16) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(int(in_channels), int(hidden_channels), kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(hidden_channels), 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UnetPlusPlusSemanticTopologyAux(nn.Module):
    def __init__(
        self,
        *,
        encoder_name: str,
        encoder_weights: str | None,
        in_channels: int,
        classes: int,
        topology_hidden_channels: int = 16,
    ) -> None:
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as e:
            raise SystemExit(
                "segmentation-models-pytorch is not installed. Install training deps with:\n"
                "  py -m pip install -r requirements-train.txt"
            ) from e

        self.base = smp.UnetPlusPlus(
            encoder_name=str(encoder_name),
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=int(classes),
        )
        decoder_out_channels = list(getattr(self.base.decoder, "out_channels", []))
        if not decoder_out_channels:
            raise RuntimeError("Failed to read Unet++ decoder out_channels")
        self.decoder_feature_channels = int(decoder_out_channels[-1])
        self.topology_head = TopologyHead(self.decoder_feature_channels, hidden_channels=int(topology_hidden_channels))
        self.attachment_module_path = "base.decoder.blocks.x_0_4"

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def segmentation_head(self):
        return self.base.segmentation_head

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.base.encoder(x)
        decoder_feature = self.base.decoder(features)
        semantic_logits = self.base.segmentation_head(decoder_feature)
        topology_logits = self.topology_head(decoder_feature)
        return {
            "semantic_logits": semantic_logits,
            "topology_logits": topology_logits,
            "decoder_feature": decoder_feature,
        }


def build_model_from_cfg(cfg: dict[str, Any]) -> UnetPlusPlusSemanticTopologyAux:
    model_cfg = cfg.get("model") or {}
    topology_cfg = cfg.get("topology_aux") or {}
    encoder_name = model_cfg.get("encoder") or model_cfg.get("encoder_name")
    if not encoder_name:
        raise SystemExit("Config: model.encoder_name is required")
    return UnetPlusPlusSemanticTopologyAux(
        encoder_name=str(encoder_name),
        encoder_weights=model_cfg.get("encoder_weights", None),
        in_channels=int(model_cfg["in_channels"]),
        classes=int(model_cfg["classes"]),
        topology_hidden_channels=int(topology_cfg.get("hidden_channels", 16)),
    )


def load_semantic_checkpoint(model: UnetPlusPlusSemanticTopologyAux, checkpoint_path: Path) -> dict[str, Any]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")
    incompat = model.base.load_state_dict(state, strict=True)
    missing = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    if missing or unexpected:
        raise RuntimeError(f"Unexpected checkpoint incompatibility: missing={missing[:5]} unexpected={unexpected[:5]}")
    return {
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint_path.resolve()),
    }


def build_semantic_loss_from_cfg(cfg: dict[str, Any], device: torch.device) -> CombinedCrossEntropyDiceLoss:
    num_classes = int(cfg["model"]["classes"])
    loss_cfg = cfg.get("loss") or {}
    ce_coef = float(loss_cfg.get("ce_coef", 0.3))
    dice_coef = float(loss_cfg.get("dice_coef", 0.7))
    class_weights = None
    ce_class_weights_cfg = loss_cfg.get("ce_class_weights", None)
    if int(num_classes) == 3 and isinstance(ce_class_weights_cfg, list) and len(ce_class_weights_cfg) == 3:
        class_weights = torch.tensor([float(x) for x in ce_class_weights_cfg], dtype=torch.float32, device=device)
    return CombinedCrossEntropyDiceLoss(
        num_classes=num_classes,
        ce_coef=ce_coef,
        dice_coef=dice_coef,
        class_weights=class_weights,
        boundary_enabled=False,
        boundary_coef=0.0,
        boundary_mode="weight_map",
    ).to(device)


def build_combined_loss_from_cfg(cfg: dict[str, Any], device: torch.device) -> SemanticTopologyAuxLoss:
    topology_cfg = cfg.get("topology_aux") or {}
    semantic_loss = build_semantic_loss_from_cfg(cfg, device)
    return SemanticTopologyAuxLoss(
        semantic_loss=semantic_loss,
        lambda_topology=float(topology_cfg.get("lambda_topology", 0.2)),
        topology_loss=BinaryBCEDiceLoss(
            bce_weight=float(topology_cfg.get("topology_bce_weight", 1.0)),
            dice_weight=float(topology_cfg.get("topology_dice_weight", 1.0)),
        ),
    ).to(device)


def _module_param_names(model: nn.Module, module_path: str) -> list[str]:
    module = dict(model.named_modules()).get(module_path, None)
    if module is None:
        raise KeyError(f"Module path not found: {module_path}")
    names: list[str] = []
    prefix = module_path + "."
    for name, _param in model.named_parameters():
        if name.startswith(prefix):
            names.append(name)
    return names


def apply_training_policy(model: UnetPlusPlusSemanticTopologyAux, cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = cfg.get("train") or {}
    trainable_decoder_modules = list(train_cfg.get("trainable_decoder_modules", ["base.decoder.blocks.x_0_4"]))
    for param in model.parameters():
        param.requires_grad = False

    trainable_prefixes = ["base.segmentation_head.", "topology_head."]
    selected_decoder_param_names: list[str] = []
    for module_path in trainable_decoder_modules:
        names = _module_param_names(model, str(module_path))
        if not names:
            raise SystemExit(f"Selected decoder module has no parameters: {module_path}")
        selected_decoder_param_names.extend(names)
        for name, param in model.named_parameters():
            if name in names:
                param.requires_grad = True

    for name, param in model.named_parameters():
        if any(name.startswith(prefix) for prefix in trainable_prefixes):
            param.requires_grad = True

    total_params = int(sum(int(p.numel()) for p in model.parameters()))
    trainable_names = [name for name, param in model.named_parameters() if bool(param.requires_grad)]
    trainable_params = int(sum(int(p.numel()) for _name, p in model.named_parameters() if bool(p.requires_grad)))
    encoder_frozen_names = [name for name, param in model.named_parameters() if name.startswith("base.encoder.") and not bool(param.requires_grad)]
    segmentation_head_names = [name for name, _p in model.named_parameters() if name.startswith("base.segmentation_head.")]
    topology_head_names = [name for name, _p in model.named_parameters() if name.startswith("topology_head.")]

    return {
        "trainable_decoder_modules": trainable_decoder_modules,
        "selected_decoder_param_names": sorted(set(selected_decoder_param_names)),
        "segmentation_head_param_names": segmentation_head_names,
        "topology_head_param_names": topology_head_names,
        "trainable_names": trainable_names,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "decoder_trainable_params": int(sum(int(p.numel()) for name, p in model.named_parameters() if name in set(selected_decoder_param_names))),
        "segmentation_head_trainable_params": int(sum(int(p.numel()) for name, p in model.named_parameters() if name.startswith("base.segmentation_head."))),
        "topology_head_trainable_params": int(sum(int(p.numel()) for name, p in model.named_parameters() if name.startswith("topology_head."))),
        "encoder_frozen_param_names": encoder_frozen_names,
        "encoder_frozen_params": int(sum(int(p.numel()) for name, p in model.named_parameters() if name.startswith("base.encoder.") and not bool(p.requires_grad))),
        "all_frozen_param_names": [name for name, p in model.named_parameters() if not bool(p.requires_grad)],
    }


def set_train_modes(model: UnetPlusPlusSemanticTopologyAux, freeze_info: dict[str, Any]) -> None:
    model.train()
    model.base.eval()
    model.base.encoder.eval()
    model.base.decoder.eval()
    model.base.segmentation_head.train()
    model.topology_head.train()
    for module_path in freeze_info["trainable_decoder_modules"]:
        module = dict(model.named_modules()).get(str(module_path), None)
        if module is None:
            raise KeyError(f"Module path not found: {module_path}")
        module.train()


def build_optimizer_groups(
    model: UnetPlusPlusSemanticTopologyAux,
    cfg: dict[str, Any],
    freeze_info: dict[str, Any],
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    train_cfg = cfg.get("train") or {}
    decoder_lr = float(train_cfg.get("lr_decoder", train_cfg.get("lr", 1.0e-5)))
    seg_lr = float(train_cfg.get("lr_segmentation_head", train_cfg.get("lr", 1.0e-5)))
    topology_lr = float(train_cfg.get("lr_topology_head", 1.0e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1.0e-5))

    decoder_named = [(name, p) for name, p in model.named_parameters() if name in set(freeze_info["selected_decoder_param_names"]) and p.requires_grad]
    seg_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.segmentation_head.") and p.requires_grad]
    topo_named = [(name, p) for name, p in model.named_parameters() if name.startswith("topology_head.") and p.requires_grad]
    group_specs = [
        {"name": "decoder", "named_params": decoder_named, "lr": decoder_lr},
        {"name": "segmentation_head", "named_params": seg_named, "lr": seg_lr},
        {"name": "topology_head", "named_params": topo_named, "lr": topology_lr},
    ]
    seen: set[str] = set()
    for group in group_specs:
        for name, param in group["named_params"]:
            if name in seen:
                raise SystemExit(f"Optimizer overlap detected for parameter: {name}")
            seen.add(name)
            if not bool(param.requires_grad):
                raise SystemExit(f"Frozen parameter included in optimizer: {name}")
    optimizer = torch.optim.AdamW(
        [{"params": [p for _n, p in group["named_params"]], "lr": float(group["lr"])} for group in group_specs if group["named_params"]],
        weight_decay=weight_decay,
    )
    meta: list[dict[str, Any]] = []
    for group in group_specs:
        meta.append(
            {
                "name": group["name"],
                "lr": float(group["lr"]),
                "param_count": int(sum(int(p.numel()) for _n, p in group["named_params"])),
                "parameter_names": [name for name, _p in group["named_params"]],
            }
        )
    return optimizer, meta


def _collect_batchnorm_stats(model: nn.Module) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    out: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, module in model.named_modules():
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is None or running_var is None:
            continue
        if not torch.is_tensor(running_mean) or not torch.is_tensor(running_var):
            continue
        out.append((name, running_mean.detach().clone(), running_var.detach().clone()))
    return out


def _max_bn_delta_filtered(
    model: nn.Module,
    ref: list[tuple[str, torch.Tensor, torch.Tensor]],
    *,
    include_prefixes: list[str] | None = None,
    exclude_prefixes: list[str] | None = None,
) -> float | None:
    modules = dict(model.named_modules())
    max_delta = None
    for name, running_mean_ref, running_var_ref in ref:
        if include_prefixes is not None and not any(str(name).startswith(prefix) for prefix in include_prefixes):
            continue
        if exclude_prefixes is not None and any(str(name).startswith(prefix) for prefix in exclude_prefixes):
            continue
        module = modules.get(name, None)
        if module is None:
            continue
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is None or running_var is None:
            continue
        d1 = float((running_mean.detach() - running_mean_ref).abs().max().item()) if running_mean.numel() else 0.0
        d2 = float((running_var.detach() - running_var_ref).abs().max().item()) if running_var.numel() else 0.0
        cur = max(d1, d2)
        max_delta = cur if max_delta is None else max(float(max_delta), float(cur))
    return float(max_delta) if max_delta is not None else None


def _snapshot_named_parameters(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor]:
    return {str(name): param.detach().clone() for name, param in named_params}


def _max_parameter_delta_from_snapshot(named_params: list[tuple[str, torch.nn.Parameter]], snap: dict[str, torch.Tensor]) -> float:
    max_delta = 0.0
    for name, param in named_params:
        ref = snap.get(str(name), None)
        if ref is None:
            continue
        delta = float((param.detach() - ref).abs().max().item()) if param.numel() else 0.0
        max_delta = max(max_delta, delta)
    return float(max_delta)


def _named_grad_l2_norm(named_params: list[tuple[str, torch.nn.Parameter]]) -> float:
    s = 0.0
    for _name, param in named_params:
        if param.grad is None:
            continue
        s += float(torch.sum(param.grad.detach().float() ** 2).item())
    return float(np.sqrt(max(s, 0.0)))


def _count_present_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> int:
    return int(sum(1 for _name, param in named_params if param.grad is not None))


def _all_grads_finite(named_params: list[tuple[str, torch.nn.Parameter]]) -> bool:
    for _name, param in named_params:
        if param.grad is None:
            continue
        if not bool(torch.isfinite(param.grad.detach()).all().item()):
            return False
    return True


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, cfg: dict[str, Any], extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(epoch),
            "config": cfg,
            "extra": extra,
        },
        str(path),
    )


def _predict_semantic_mask_from_model(
    model: UnetPlusPlusSemanticTopologyAux,
    rgb_u8: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    image = _simple_preprocess_uint8_rgb(rgb_u8)
    image_t = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        with _autocast_ctx(device, enabled=use_amp):
            outputs = model(image_t)
            logits = outputs["semantic_logits"]
        pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
    return pred


def compute_semantic_metrics(model: UnetPlusPlusSemanticTopologyAux, loader, device: torch.device, loss_fn: SemanticTopologyAuxLoss, use_amp: bool) -> dict[str, Any]:
    model.eval()
    total_semantic_loss = 0.0
    total_topology_loss = 0.0
    total_combined_loss = 0.0
    n_batches = 0
    num_classes = None
    dice_sum: list[float] | None = None
    iou_sum: list[float] | None = None
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            topology_target = batch["topology_target"].to(device, non_blocking=True)
            with _autocast_ctx(device, enabled=use_amp):
                outputs = model(images)
                loss_dict = loss_fn(
                    semantic_logits=outputs["semantic_logits"],
                    semantic_target=masks,
                    topology_logits=outputs["topology_logits"],
                    topology_target=topology_target,
                )
            metrics = compute_per_class_metrics_from_logits(outputs["semantic_logits"], masks, num_classes=int(outputs["semantic_logits"].shape[1]))
            if dice_sum is None:
                num_classes = int(outputs["semantic_logits"].shape[1])
                dice_sum = [0.0 for _ in range(int(num_classes))]
                iou_sum = [0.0 for _ in range(int(num_classes))]
            for i in range(int(num_classes)):
                dice_sum[i] += float(metrics.dice[i])
                iou_sum[i] += float(metrics.iou[i])
            total_semantic_loss += float(loss_dict["semantic_loss"].item())
            total_topology_loss += float(loss_dict["topology_loss"].item())
            total_combined_loss += float(loss_dict["combined_loss"].item())
            n_batches += 1
    if not n_batches or dice_sum is None or iou_sum is None or num_classes is None:
        raise RuntimeError("No validation batches available")
    mean_dice_fg = float(sum(dice_sum[1:]) / max(len(dice_sum[1:]), 1) / n_batches)
    mean_iou_fg = float(sum(iou_sum[1:]) / max(len(iou_sum[1:]), 1) / n_batches)
    return {
        "semantic_loss": float(total_semantic_loss / n_batches),
        "topology_loss": float(total_topology_loss / n_batches),
        "combined_loss": float(total_combined_loss / n_batches),
        "dice": [float(v / n_batches) for v in dice_sum],
        "iou": [float(v / n_batches) for v in iou_sum],
        "leaflet_dice": float(dice_sum[1] / n_batches) if num_classes > 1 else None,
        "leaflet_iou": float(iou_sum[1] / n_batches) if num_classes > 1 else None,
        "ring_dice": float(dice_sum[2] / n_batches) if num_classes > 2 else None,
        "ring_iou": float(iou_sum[2] / n_batches) if num_classes > 2 else None,
        "mean_dice_fg": mean_dice_fg,
        "mean_iou_fg": mean_iou_fg,
    }


def evaluate_oracle_k_reconstruction(
    model: UnetPlusPlusSemanticTopologyAux,
    *,
    manifest_path: Path,
    image_root: Path,
    device: torch.device,
    use_amp: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    import leaflet_oracle_count_geometric_split_audit as base_audit
    import leaflet_oracle_count_geometric_split_forensic as forensic
    import leaflet_oracle_k_constrained_normalization_audit as k_audit

    rows = _read_jsonl(manifest_path.resolve())
    if limit is not None:
        rows = rows[: int(limit)]
    if any(bool(row.get("present_in_authoritative_106_holdout", False)) for row in rows):
        raise SystemExit("Authoritative holdout samples are not allowed in oracle-K validation")

    per_sample: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample"])
        rgb = _read_image_rgb((image_root / str(row["image_rel"])).resolve())
        gt_inst_full = _read_u8((image_root / str(row["instance_mask_rel"])).resolve())
        target_h = int(row["image_height"])
        target_w = int(row["image_width"])
        rgb = _center_crop_like_validation(rgb, target_h, target_w, is_mask=False)
        gt_inst = _center_crop_like_validation(gt_inst_full, target_h, target_w, is_mask=True)
        gt_k = int(row["gt_instance_count"])
        pred_sem = _predict_semantic_mask_from_model(model, rgb, device=device, use_amp=use_amp)
        pred_union = (pred_sem == 1).astype(np.uint8)
        normalized = k_audit.normalize_mask_exact_k(pred_union, gt_k, "centroid_distance_k_normalizer")
        pred_inst = normalized["labels"].astype(np.uint8)
        metrics = base_audit.compute_detailed_instance_metrics(gt_inst, pred_inst, gt_k=gt_k, pred_k=int(normalized["final_group_count"]))
        topo = forensic.classify_semantic_topology(gt_inst, pred_union)
        if topo["bridge"] and topo["missing"]:
            topo_class = "both"
        elif topo["bridge"]:
            topo_class = "false_bridges"
        elif topo["missing"]:
            topo_class = "missing_pixels"
        else:
            topo_class = "other"
        per_sample.append(
            {
                "sample_id": sample_id,
                "patient_id": str(row["patient_id"]),
                "gt_count": gt_k,
                "exact_k": float(metrics["instance_exact_count_acc"]),
                "mean_matched_iou": float(metrics["instance_mean_matched_iou"]),
                "all_iou_ge_0.50": float(metrics["all_iou_ge_0.50"]),
                "all_iou_ge_0.70": float(metrics["all_iou_ge_0.70"]),
                "topology_class": topo_class,
            }
        )

    def _mean(key: str, subset: list[dict[str, Any]]) -> float:
        return float(sum(float(row[key]) for row in subset) / len(subset)) if subset else 0.0

    gt2 = [row for row in per_sample if int(row["gt_count"]) == 2]
    gt3 = [row for row in per_sample if int(row["gt_count"]) == 3]
    topo_counts = {
        "missing_pixels": int(sum(1 for row in per_sample if row["topology_class"] == "missing_pixels")),
        "false_bridges": int(sum(1 for row in per_sample if row["topology_class"] == "false_bridges")),
        "both": int(sum(1 for row in per_sample if row["topology_class"] == "both")),
        "other": int(sum(1 for row in per_sample if row["topology_class"] == "other")),
    }
    return {
        "samples": int(len(per_sample)),
        "method_key": "centroid_distance_k_normalizer",
        "exact_k": _mean("exact_k", per_sample),
        "mean_matched_iou": _mean("mean_matched_iou", per_sample),
        "all_iou_ge_0.50": _mean("all_iou_ge_0.50", per_sample),
        "all_iou_ge_0.70": _mean("all_iou_ge_0.70", per_sample),
        "gt1_success": _mean("all_iou_ge_0.50", [row for row in per_sample if int(row["gt_count"]) == 1]),
        "gt2_success": _mean("all_iou_ge_0.50", gt2),
        "gt3_success": _mean("all_iou_ge_0.50", gt3),
        "topology_failure_classes": topo_counts,
        "per_sample": per_sample,
        "analysis_only": True,
        "oracle_k_source": "manifest.gt_instance_count",
        "holdout_used": False,
    }


def topology_reconstruction_better(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None:
        return True
    keys = [
        ("all_iou_ge_0.50", True),
        ("mean_matched_iou", True),
        ("gt2_success", True),
        ("semantic_mean_fg", True),
        ("epoch", False),
    ]
    for key, descending in keys:
        left = candidate.get(key, None)
        right = best.get(key, None)
        if left == right:
            continue
        if descending:
            return float(left) > float(right)
        return int(left) < int(right)
    return False


def build_validation_contract(research_manifest: Path) -> dict[str, Any]:
    return {
        "semantic_metrics": [
            "leaflet_dice",
            "leaflet_iou",
            "ring_dice",
            "ring_iou",
            "mean_dice_fg",
            "mean_iou_fg",
        ],
        "topology_reconstruction_metrics": [
            "exact_k",
            "mean_matched_iou",
            "all_iou_ge_0.50",
            "all_iou_ge_0.70",
            "gt1_success",
            "gt2_success",
            "gt3_success",
            "topology_failure_classes",
        ],
        "research_manifest": str(research_manifest.resolve()),
        "normalizer_method": "centroid_distance_k_normalizer",
        "oracle_k_analysis_only": True,
        "holdout_used": False,
        "checkpoint_rules": {
            "best_mean_fg.pth": "highest semantic mean_dice_fg",
            "best_topology_reconstruction.pth": [
                "highest all_iou_ge_0.50",
                "higher mean_matched_iou",
                "higher gt2_success",
                "higher semantic_mean_fg",
                "earlier epoch",
            ],
        },
    }


def build_visual_target_audit(
    *,
    dataset_root: Path,
    train_split_txt: Path,
    instance_root: Path,
    contract: TopologyTargetContract,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    items = read_split_file(dataset_root.resolve(), train_split_txt.resolve())
    rows: list[dict[str, Any]] = []

    def _overlay(rgb: np.ndarray, target01: np.ndarray) -> np.ndarray:
        out = rgb.copy().astype(np.float32)
        mask = target01.astype(bool)
        out[mask] = out[mask] * 0.45 + np.asarray([255.0, 0.0, 0.0], dtype=np.float32) * 0.55
        return out.astype(np.uint8)

    candidates: dict[str, tuple[float, str]] = {
        "gt1": (float("-inf"), ""),
        "gt2": (float("-inf"), ""),
        "gt3": (float("-inf"), ""),
        "narrow_leaflet": (float("-inf"), ""),
        "close_neighbors": (float("-inf"), ""),
        "disconnected_same_leaflet": (float("-inf"), ""),
    }

    for item in items:
        sample_id = Path(item.image_path).stem
        image = _read_image_rgb(item.image_path)
        semantic_mask = _read_u8(item.mask_path)
        instance_mask = _read_u8(instance_root.resolve() / "instance_masks" / f"{sample_id}.png")
        target, parts = generate_topology_target(instance_mask, contract, return_parts=True)
        gt_count = len(_positive_instance_ids(instance_mask))
        gaps = _pairwise_gap_distances(instance_mask)
        widths = _local_widths(instance_mask)
        comp_counts = _instance_component_counts(instance_mask)
        disconnected = int(any(int(v) > 1 for v in comp_counts.values()))
        narrow_score = -float(np.percentile(widths, 10)) if widths.size else 0.0
        close_score = -float(min(gaps)) if gaps else 0.0
        if gt_count == 1 and narrow_score > candidates["gt1"][0]:
            candidates["gt1"] = (narrow_score, sample_id)
        if gt_count == 2 and close_score > candidates["gt2"][0]:
            candidates["gt2"] = (close_score, sample_id)
        if gt_count == 3 and close_score > candidates["gt3"][0]:
            candidates["gt3"] = (close_score, sample_id)
        if narrow_score > candidates["narrow_leaflet"][0]:
            candidates["narrow_leaflet"] = (narrow_score, sample_id)
        if close_score > candidates["close_neighbors"][0]:
            candidates["close_neighbors"] = (close_score, sample_id)
        if disconnected > 0 and float(disconnected) > candidates["disconnected_same_leaflet"][0]:
            candidates["disconnected_same_leaflet"] = (float(disconnected), sample_id)
        rows.append(
            {
                "sample_id": sample_id,
                "gt_count": int(gt_count),
                "pair_gap_min_px": float(min(gaps)) if gaps else None,
                "local_width_p10_px": float(np.percentile(widths, 10)) if widths.size else None,
                "disconnected_same_leaflet": int(disconnected),
                "target_fraction": float(np.mean(target > 0)),
            }
        )

    sample_lookup = {Path(item.image_path).stem: item for item in items}
    saved_files: list[str] = []
    for category, (_score, sample_id) in candidates.items():
        if not sample_id:
            continue
        item = sample_lookup[sample_id]
        image = _center_crop_like_validation(_read_image_rgb(item.image_path), 768, 768, is_mask=False)
        semantic_mask = _center_crop_like_validation(_read_u8(item.mask_path), 768, 768, is_mask=True)
        instance_mask = _center_crop_like_validation(_read_u8(instance_root.resolve() / "instance_masks" / f"{sample_id}.png"), 768, 768, is_mask=True)
        target, parts = generate_topology_target(instance_mask, contract, return_parts=True)
        gt_leaflet = (semantic_mask == 1).astype(np.uint8)
        overlay = _overlay(image, target)
        semantic_rgb = np.zeros((gt_leaflet.shape[0], gt_leaflet.shape[1], 3), dtype=np.uint8)
        semantic_rgb[gt_leaflet > 0] = np.asarray([0, 255, 0], dtype=np.uint8)
        target_rgb = np.zeros_like(semantic_rgb)
        target_rgb[parts["boundary"] > 0] = np.asarray([255, 255, 0], dtype=np.uint8)
        target_rgb[parts["separation"] > 0] = np.asarray([255, 0, 255], dtype=np.uint8)
        target_rgb[parts["narrow"] > 0] = np.asarray([255, 0, 0], dtype=np.uint8)
        inst_vis = cv2.applyColorMap(((instance_mask.astype(np.float32) / max(float(instance_mask.max()), 1.0)) * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_TURBO)
        panel_top = np.concatenate([cv2.cvtColor(image, cv2.COLOR_RGB2BGR), cv2.cvtColor(semantic_rgb, cv2.COLOR_RGB2BGR)], axis=1)
        panel_bottom = np.concatenate([inst_vis, cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)], axis=1)
        if panel_bottom.shape[1] != panel_top.shape[1]:
            pad_w = panel_bottom.shape[1] - panel_top.shape[1]
            if pad_w > 0:
                pad = np.zeros((panel_top.shape[0], pad_w, 3), dtype=np.uint8)
                panel_top = np.concatenate([panel_top, pad], axis=1)
        grid = np.concatenate([panel_top, panel_bottom], axis=0)
        out_path = output_dir / f"{category}_{sample_id}.png"
        cv2.imwrite(str(out_path), grid)
        saved_files.append(str(out_path.resolve()))
    return {
        "saved_files": saved_files,
        "candidate_rows": rows,
        "selected_categories": {category: sample_id for category, (_score, sample_id) in candidates.items() if sample_id},
    }


def run_smoke_test(
    *,
    cfg: dict[str, Any],
    contract: TopologyTargetContract,
    device: torch.device,
) -> dict[str, Any]:
    dataset_root = _resolve_repo_path((cfg.get("dataset") or {}).get("root", DEFAULT_SEMANTIC_DATASET_ROOT), DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split_txt = _resolve_repo_path((cfg.get("dataset") or {}).get("train_txt", DEFAULT_SEMANTIC_TRAIN_SPLIT), DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = _resolve_repo_path((cfg.get("dataset") or {}).get("instance_root", DEFAULT_INSTANCE_ROOT), DEFAULT_INSTANCE_ROOT)
    model = build_model_from_cfg(cfg).to(device)
    checkpoint_info = load_semantic_checkpoint(
        model,
        _resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", DEFAULT_SEMANTIC_CHECKPOINT), DEFAULT_SEMANTIC_CHECKPOINT),
    )
    freeze_info = apply_training_policy(model, cfg)
    set_train_modes(model, freeze_info)
    optimizer, optimizer_meta = build_optimizer_groups(model, cfg, freeze_info)
    loss_fn = build_combined_loss_from_cfg(cfg, device)

    train_ds = SemanticTopologyAuxDataset(
        dataset_root=dataset_root,
        split_txt=train_split_txt,
        instance_root=instance_root,
        contract=contract,
        num_classes=int(cfg["model"]["classes"]),
        input_size=int(cfg["model"]["input_size"]),
        augment_cfg=cfg.get("augment", None),
        training=True,
    )
    batch = next(iter(torch.utils.data.DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=0)))
    images = batch["image"].to(device)
    semantic_target = batch["mask"].to(device)
    topology_target = batch["topology_target"].to(device)
    use_amp = _amp_enabled(cfg, device)

    trainable_decoder_named = [(name, p) for name, p in model.named_parameters() if name in set(freeze_info["selected_decoder_param_names"])]
    segmentation_head_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.segmentation_head.")]
    topology_head_named = [(name, p) for name, p in model.named_parameters() if name.startswith("topology_head.")]
    frozen_named = [(name, p) for name, p in model.named_parameters() if not bool(p.requires_grad)]
    encoder_frozen_named = [(name, p) for name, p in model.named_parameters() if name.startswith("base.encoder.") and not bool(p.requires_grad)]

    trainable_snap = {
        "decoder": _snapshot_named_parameters(trainable_decoder_named),
        "segmentation_head": _snapshot_named_parameters(segmentation_head_named),
        "topology_head": _snapshot_named_parameters(topology_head_named),
    }
    frozen_snap = _snapshot_named_parameters(frozen_named)
    bn_ref = _collect_batchnorm_stats(model.base)

    optimizer.zero_grad(set_to_none=True)
    with _autocast_ctx(device, enabled=use_amp):
        outputs = model(images)
        loss_dict = loss_fn(
            semantic_logits=outputs["semantic_logits"],
            semantic_target=semantic_target,
            topology_logits=outputs["topology_logits"],
            topology_target=topology_target,
        )
    loss_dict["combined_loss"].backward()
    decoder_grad_norm = _named_grad_l2_norm(trainable_decoder_named)
    segmentation_head_grad_norm = _named_grad_l2_norm(segmentation_head_named)
    topology_head_grad_norm = _named_grad_l2_norm(topology_head_named)
    optimizer.step()

    selected_prefixes = [str(path).replace("base.", "", 1) for path in freeze_info["trainable_decoder_modules"]]
    frozen_bn_delta = _max_bn_delta_filtered(model.base, bn_ref, exclude_prefixes=selected_prefixes)
    selected_bn_delta = _max_bn_delta_filtered(model.base, bn_ref, include_prefixes=selected_prefixes)
    frozen_parameter_delta = _max_parameter_delta_from_snapshot(frozen_named, frozen_snap)

    summary = {
        "device": str(device),
        "cpu_only": bool(device.type != "cuda"),
        "checkpoint": checkpoint_info,
        "semantic_forward": bool(outputs["semantic_logits"].shape[1] == int(cfg["model"]["classes"])),
        "semantic_logits_shape": list(outputs["semantic_logits"].shape),
        "topology_logits_shape": list(outputs["topology_logits"].shape),
        "decoder_feature_shape": list(outputs["decoder_feature"].shape),
        "topology_target_shape": list(topology_target.shape),
        "semantic_loss_finite": bool(torch.isfinite(loss_dict["semantic_loss"]).all().item()),
        "topology_loss_finite": bool(torch.isfinite(loss_dict["topology_loss"]).all().item()),
        "combined_loss_finite": bool(torch.isfinite(loss_dict["combined_loss"]).all().item()),
        "semantic_loss": float(loss_dict["semantic_loss"].detach().cpu().item()),
        "topology_loss": float(loss_dict["topology_loss"].detach().cpu().item()),
        "combined_loss": float(loss_dict["combined_loss"].detach().cpu().item()),
        "backward_finite": bool(
            _all_grads_finite(trainable_decoder_named) and _all_grads_finite(segmentation_head_named) and _all_grads_finite(topology_head_named)
        ),
        "topology_head_grad_norm": float(topology_head_grad_norm),
        "segmentation_head_grad_norm": float(segmentation_head_grad_norm),
        "selected_decoder_grad_norm": float(decoder_grad_norm),
        "topology_head_grad_present": int(_count_present_grads(topology_head_named)),
        "segmentation_head_grad_present": int(_count_present_grads(segmentation_head_named)),
        "selected_decoder_grad_present": int(_count_present_grads(trainable_decoder_named)),
        "frozen_encoder_grad_count": int(_count_present_grads(encoder_frozen_named)),
        "frozen_parameter_max_delta": float(frozen_parameter_delta),
        "frozen_bn_max_delta": float(frozen_bn_delta) if frozen_bn_delta is not None else None,
        "selected_bn_max_delta": float(selected_bn_delta) if selected_bn_delta is not None else None,
        "selected_decoder_parameter_delta": float(_max_parameter_delta_from_snapshot(trainable_decoder_named, trainable_snap["decoder"])),
        "segmentation_head_parameter_delta": float(_max_parameter_delta_from_snapshot(segmentation_head_named, trainable_snap["segmentation_head"])),
        "topology_head_parameter_delta": float(_max_parameter_delta_from_snapshot(topology_head_named, trainable_snap["topology_head"])),
        "selected_decoder_train_mode": bool(all(dict(model.named_modules())[module_path].training for module_path in freeze_info["trainable_decoder_modules"])),
        "segmentation_head_train_mode": bool(model.base.segmentation_head.training),
        "topology_head_train_mode": bool(model.topology_head.training),
        "encoder_eval_mode": bool(not model.base.encoder.training),
        "base_eval_mode": bool(not model.base.training),
        "optimizer_groups": optimizer_meta,
        "trainable_parameter_names": freeze_info["trainable_names"],
    }
    return summary


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def prepare_experiment(
    *,
    cfg: dict[str, Any],
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = cfg.get("dataset") or {}
    dataset_root = _resolve_repo_path(dataset_cfg.get("root", DEFAULT_SEMANTIC_DATASET_ROOT), DEFAULT_SEMANTIC_DATASET_ROOT)
    train_split_txt = _resolve_repo_path(dataset_cfg.get("train_txt", DEFAULT_SEMANTIC_TRAIN_SPLIT), DEFAULT_SEMANTIC_TRAIN_SPLIT)
    instance_root = _resolve_repo_path(dataset_cfg.get("instance_root", DEFAULT_INSTANCE_ROOT), DEFAULT_INSTANCE_ROOT)
    research_manifest = _resolve_repo_path(dataset_cfg.get("research_val_manifest", DEFAULT_RESEARCH_MANIFEST), DEFAULT_RESEARCH_MANIFEST)

    contract, target_audit = choose_topology_target_contract(
        dataset_root=dataset_root,
        train_split_txt=train_split_txt,
        instance_root=instance_root,
    )
    target_rows = []
    sample_rows = target_audit.pop("sample_rows")
    target_fractions: list[float] = []
    for row in sample_rows:
        sample_id = str(row["sample_id"])
        target = generate_topology_target(_read_u8(instance_root / "instance_masks" / f"{sample_id}.png"), contract)
        frac = float(np.mean(target > 0))
        target_rows.append({**row, "topology_target_fraction": frac})
        target_fractions.append(frac)
    target_audit["topology_target_fraction"] = {
        "mean": float(np.mean(target_fractions)) if target_fractions else None,
        "median": float(np.median(target_fractions)) if target_fractions else None,
        **_quantiles(np.asarray(target_fractions, dtype=np.float32), [5, 10, 25, 75, 90, 95]),
    }

    model = build_model_from_cfg(cfg)
    checkpoint_info = load_semantic_checkpoint(
        model,
        _resolve_repo_path((cfg.get("train") or {}).get("init_checkpoint", DEFAULT_SEMANTIC_CHECKPOINT), DEFAULT_SEMANTIC_CHECKPOINT),
    )
    freeze_info = apply_training_policy(model, cfg)
    optimizer, optimizer_meta = build_optimizer_groups(model, cfg, freeze_info)
    del optimizer
    topology_head_params = int(sum(int(p.numel()) for name, p in model.named_parameters() if name.startswith("topology_head.")))
    total_additional_params = int(topology_head_params)
    visual_summary = build_visual_target_audit(
        dataset_root=dataset_root,
        train_split_txt=train_split_txt,
        instance_root=instance_root,
        contract=contract,
        output_dir=output_dir / "visual_target_audit",
    )
    smoke_summary = run_smoke_test(cfg=cfg, contract=contract, device=device)

    semantic_path = {
        "decoder_feature_module_path": "base.decoder.blocks.x_0_4",
        "decoder_feature_channels": int(model.decoder_feature_channels),
        "decoder_feature_shape_at_input_768": [1, int(model.decoder_feature_channels), 768, 768],
        "segmentation_head_class": type(model.base.segmentation_head).__name__,
        "segmentation_head_repr": repr(model.base.segmentation_head),
        "segmentation_head_param_names": freeze_info["segmentation_head_param_names"],
        "current_semantic_checkpoint": checkpoint_info,
        "baseline_trainable_modules": "full model trainable in training/train.py semantic fine-tuning path",
    }
    validation_contract = build_validation_contract(research_manifest)
    readiness = {
        "status": "ready_for_training"
        if (
            smoke_summary["semantic_forward"]
            and smoke_summary["semantic_loss_finite"]
            and smoke_summary["topology_loss_finite"]
            and smoke_summary["combined_loss_finite"]
            and smoke_summary["backward_finite"]
            and float(smoke_summary["topology_head_grad_norm"]) > 0.0
            and float(smoke_summary["segmentation_head_grad_norm"]) > 0.0
            and float(smoke_summary["selected_decoder_grad_norm"]) > 0.0
            and int(smoke_summary["frozen_encoder_grad_count"]) == 0
            and float(smoke_summary["frozen_parameter_max_delta"]) == 0.0
            and (smoke_summary["frozen_bn_max_delta"] is None or float(smoke_summary["frozen_bn_max_delta"]) == 0.0)
            and float(smoke_summary["selected_decoder_parameter_delta"]) > 0.0
            and float(smoke_summary["segmentation_head_parameter_delta"]) > 0.0
            and float(smoke_summary["topology_head_parameter_delta"]) > 0.0
        )
        else "blocked",
        "training_launched": False,
        "holdout_used": False,
        "production_changed": False,
    }

    _write_json(output_dir / "semantic_path_summary.json", semantic_path)
    _write_json(output_dir / "topology_target_audit.json", target_audit)
    _write_csv(output_dir / "topology_target_audit_rows.csv", target_rows)
    _write_json(output_dir / "trainable_contract.json", {**freeze_info, "optimizer_groups": optimizer_meta})
    _write_json(output_dir / "smoke_test_summary.json", smoke_summary)
    _write_json(output_dir / "validation_contract.json", validation_contract)
    _write_json(output_dir / "readiness_summary.json", readiness)

    return {
        "semantic_path": semantic_path,
        "target_contract": asdict(contract),
        "target_audit": target_audit,
        "trainable_contract": {**freeze_info, "optimizer_groups": optimizer_meta},
        "smoke": smoke_summary,
        "validation_contract": validation_contract,
        "visual_target_audit": visual_summary,
        "readiness": readiness,
        "topology_head_parameter_count": int(topology_head_params),
        "total_additional_parameters": int(total_additional_params),
    }
