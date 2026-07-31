from __future__ import annotations

import types

import torch
import torch.nn.functional as F


class SpatialResidualBlock(torch.nn.Module):
    def __init__(self, channels: int, *, dilation: int):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(int(channels), int(channels), kernel_size=3, padding=int(dilation), dilation=int(dilation), bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=int(channels)),
            torch.nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SpatialDilatedCenterHead(torch.nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(int(in_channels), 64, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=64),
            torch.nn.SiLU(inplace=True),
        )
        self.blocks = torch.nn.ModuleList(
            [
                SpatialResidualBlock(64, dilation=1),
                SpatialResidualBlock(64, dilation=2),
                SpatialResidualBlock(64, dilation=4),
                SpatialResidualBlock(64, dilation=8),
            ]
        )
        self.refine = torch.nn.Sequential(
            torch.nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            torch.nn.GroupNorm(num_groups=8, num_channels=32),
            torch.nn.SiLU(inplace=True),
        )
        self.out_conv = torch.nn.Conv2d(32, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.refine(x)
        return self.out_conv(x)


class UnetPlusPlusSemanticCenterHead(torch.nn.Module):
    def __init__(self, encoder_name: str, encoder_weights, in_channels: int, classes: int, center_head_type: str = "linear_1x1", center_feature: dict | None = None):
        super().__init__()
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch.base import SegmentationHead

        self.base = smp.UnetPlusPlus(
            encoder_name=str(encoder_name),
            encoder_weights=encoder_weights,
            in_channels=int(in_channels),
            classes=int(classes),
        )
        decoder = getattr(self.base, "decoder", None)
        out_channels = list(getattr(decoder, "out_channels", [])) if decoder is not None else []
        if not out_channels:
            raise RuntimeError("Unet++ decoder out_channels not found")

        center_feature_cfg = dict(center_feature or {})
        center_feature_path = str(center_feature_cfg.get("module_path", "") or "").strip() or None
        center_feature_expected_channels = center_feature_cfg.get("expected_channels", None)
        center_feature_adapter_out_channels = center_feature_cfg.get("adapter_out_channels", None)
        center_feature_native_stride = center_feature_cfg.get("native_stride", None)
        center_feature_upsample_logits_to_target = bool(center_feature_cfg.get("upsample_logits_to_target", False))
        self.center_feature_cfg = {
            "module_path": center_feature_path,
            "expected_channels": int(center_feature_expected_channels) if center_feature_expected_channels is not None else None,
            "adapter_out_channels": int(center_feature_adapter_out_channels) if center_feature_adapter_out_channels is not None else None,
            "native_stride": int(center_feature_native_stride) if center_feature_native_stride is not None else 1,
            "upsample_logits_to_target": bool(center_feature_upsample_logits_to_target),
        }
        self.center_feature_module_path = self.center_feature_cfg["module_path"]
        self.center_feature_expected_channels = self.center_feature_cfg["expected_channels"]
        self.center_feature_adapter_out_channels = self.center_feature_cfg["adapter_out_channels"]
        self.center_feature_native_stride = self.center_feature_cfg["native_stride"]
        self.center_feature_upsample_logits_to_target = self.center_feature_cfg["upsample_logits_to_target"]
        self._captured_center_feature: torch.Tensor | None = None
        self._captured_center_feature_call_count = 0
        self._captured_center_feature_actual_path: str | None = None
        self._center_feature_hook_handle = None

        if self.center_feature_module_path is not None and self.center_feature_expected_channels is None:
            raise ValueError("center_feature.expected_channels is required when center_feature.module_path is set")

        center_head_in_channels = int(out_channels[-1])
        if self.center_feature_module_path is not None:
            center_head_in_channels = int(self.center_feature_expected_channels)
        if self.center_feature_adapter_out_channels is not None:
            self.center_adapter = torch.nn.Conv2d(
                int(center_head_in_channels),
                int(self.center_feature_adapter_out_channels),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            torch.nn.init.kaiming_normal_(self.center_adapter.weight, mode="fan_out", nonlinearity="relu")
            center_head_in_channels = int(self.center_feature_adapter_out_channels)
        else:
            self.center_adapter = None

        self.center_head_type = str(center_head_type).strip().lower() or "linear_1x1"
        if self.center_head_type == "linear_1x1":
            self.center_head = SegmentationHead(in_channels=int(center_head_in_channels), out_channels=1, activation=None, kernel_size=3)
        elif self.center_head_type == "spatial_dilated":
            self.center_head = SpatialDilatedCenterHead(in_channels=int(center_head_in_channels))
        else:
            raise ValueError(f"Unsupported center_head_type: {center_head_type}")

        if self.center_feature_module_path is not None:
            self._register_center_feature_hook(self.center_feature_module_path)
        self.freeze_base = False

    def _register_center_feature_hook(self, module_path: str) -> None:
        self.close_center_feature_hook()
        mod = dict(self.named_modules()).get(str(module_path), None)
        if mod is None:
            raise ValueError(f"Unknown center_feature.module_path: {module_path}")
        self._captured_center_feature_actual_path = str(module_path)
        self._center_feature_hook_handle = mod.register_forward_hook(self._center_feature_hook)

    def _center_feature_hook(self, _module, _inputs, output) -> None:
        self._captured_center_feature_call_count += 1
        if torch.is_tensor(output):
            self._captured_center_feature = output
        elif isinstance(output, (list, tuple)) and output and torch.is_tensor(output[0]):
            self._captured_center_feature = output[0]
        else:
            self._captured_center_feature = None

    def clear_center_feature_capture(self) -> None:
        self._captured_center_feature = None
        self._captured_center_feature_call_count = 0

    def close_center_feature_hook(self) -> None:
        handle = getattr(self, "_center_feature_hook_handle", None)
        if handle is not None:
            handle.remove()
            self._center_feature_hook_handle = None

    def center_feature_capture_info(self) -> dict:
        feat = self._captured_center_feature
        return {
            "configured_module_path": self.center_feature_module_path,
            "actual_module_path": self._captured_center_feature_actual_path,
            "hook_call_count": int(self._captured_center_feature_call_count),
            "captured_shape": list(feat.shape) if torch.is_tensor(feat) else None,
            "captured_dtype": str(feat.dtype).replace("torch.", "") if torch.is_tensor(feat) else None,
            "expected_channels": self.center_feature_expected_channels,
            "actual_channels": int(feat.shape[1]) if torch.is_tensor(feat) and feat.ndim >= 2 else None,
            "native_stride": int(self.center_feature_native_stride or 1),
            "upsample_logits_to_target": bool(self.center_feature_upsample_logits_to_target),
            "adapter_out_channels": self.center_feature_adapter_out_channels,
        }

    def resolve_center_features(self, decoder_output: torch.Tensor) -> torch.Tensor:
        if self.center_feature_module_path is None:
            return decoder_output
        if self._captured_center_feature_call_count != 1:
            raise RuntimeError(
                f"Configured center feature hook must be called exactly once for {self.center_feature_module_path}; "
                f"got {self._captured_center_feature_call_count}"
            )
        feat = self._captured_center_feature
        if not torch.is_tensor(feat):
            raise RuntimeError(f"Configured center feature tensor missing for {self.center_feature_module_path}")
        if self.center_feature_expected_channels is not None and int(feat.shape[1]) != int(self.center_feature_expected_channels):
            raise RuntimeError(
                f"Captured center feature channels mismatch for {self.center_feature_module_path}: "
                f"expected {int(self.center_feature_expected_channels)}, got {int(feat.shape[1])}"
            )
        return feat

    def upsample_center_logits(self, center_logits: torch.Tensor) -> torch.Tensor:
        if not bool(self.center_feature_upsample_logits_to_target):
            return center_logits
        stride = max(int(self.center_feature_native_stride or 1), 1)
        target_h = int(center_logits.shape[-2]) * stride
        target_w = int(center_logits.shape[-1]) * stride
        return F.interpolate(center_logits, size=(target_h, target_w), mode="bilinear", align_corners=False)

    def forward_center_from_features(self, center_features: torch.Tensor) -> torch.Tensor:
        x = center_features
        if self.center_adapter is not None:
            x = self.center_adapter(x)
        center_logits = self.center_head(x)
        return self.upsample_center_logits(center_logits)

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def segmentation_head(self):
        return self.base.segmentation_head

    def center_head_output_layer(self) -> torch.nn.Module:
        if self.center_head_type == "linear_1x1":
            try:
                layer0 = self.center_head[0]
            except Exception as e:
                raise RuntimeError("center_head[0] not found for linear_1x1") from e
            return layer0
        if self.center_head_type == "spatial_dilated":
            return self.center_head.out_conv
        raise RuntimeError(f"Unsupported center_head_type: {self.center_head_type}")

    def forward_base(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.clear_center_feature_capture()
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        semantic = self.segmentation_head(decoder_output)
        return semantic, decoder_output

    def forward_center(self, decoder_features: torch.Tensor) -> torch.Tensor:
        center_features = self.resolve_center_features(decoder_features)
        return self.forward_center_from_features(center_features)

    def forward(self, x: torch.Tensor) -> dict:
        if bool(getattr(self, "freeze_base", False)):
            with torch.no_grad():
                semantic, decoder_output = self.forward_base(x)
            center = self.forward_center(decoder_output.detach())
        else:
            semantic, decoder_output = self.forward_base(x)
            center = self.forward_center(decoder_output)
        return {"semantic": semantic, "center": center}

    def __del__(self):
        try:
            self.close_center_feature_hook()
        except Exception:
            pass


def load_semantic_checkpoint_non_strict(model: torch.nn.Module, checkpoint_path: str) -> tuple[list[str], list[str]]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state = ckpt.get("model") if isinstance(ckpt, dict) else None
    if state is None:
        state = ckpt
    if not isinstance(state, dict):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")

    model_state = model.state_dict()
    remapped = {}
    for k, v in state.items():
        if k in model_state:
            remapped[k] = v
            continue
        bk = f"base.{k}"
        if bk in model_state:
            remapped[bk] = v
            continue

    incompat = model.load_state_dict(remapped, strict=False)
    missing_keys = list(getattr(incompat, "missing_keys", [])) if incompat is not None else []
    unexpected_keys = list(getattr(incompat, "unexpected_keys", [])) if incompat is not None else []
    return missing_keys, unexpected_keys
