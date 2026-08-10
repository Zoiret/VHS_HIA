from __future__ import annotations

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


def _groupnorm_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if int(channels) % int(groups) == 0:
            return int(groups)
    return 1


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
        context_feature_module_path = str(center_feature_cfg.get("context_module_path", "") or "").strip() or None
        context_feature_expected_channels = center_feature_cfg.get("context_expected_channels", None)
        context_feature_native_stride = center_feature_cfg.get("context_native_stride", None)
        primary_projection_out_channels = center_feature_cfg.get("primary_projection_out_channels", None)
        context_projection_out_channels = center_feature_cfg.get("context_projection_out_channels", None)
        fusion_out_channels = center_feature_cfg.get("fusion_out_channels", None)
        fusion_method = str(center_feature_cfg.get("fusion_method", "concat") or "concat").strip().lower()
        self.center_feature_cfg = {
            "module_path": center_feature_path,
            "expected_channels": int(center_feature_expected_channels) if center_feature_expected_channels is not None else None,
            "adapter_out_channels": int(center_feature_adapter_out_channels) if center_feature_adapter_out_channels is not None else None,
            "native_stride": int(center_feature_native_stride) if center_feature_native_stride is not None else 1,
            "upsample_logits_to_target": bool(center_feature_upsample_logits_to_target),
            "context_module_path": context_feature_module_path,
            "context_expected_channels": int(context_feature_expected_channels) if context_feature_expected_channels is not None else None,
            "context_native_stride": int(context_feature_native_stride) if context_feature_native_stride is not None else None,
            "primary_projection_out_channels": int(primary_projection_out_channels) if primary_projection_out_channels is not None else None,
            "context_projection_out_channels": int(context_projection_out_channels) if context_projection_out_channels is not None else None,
            "fusion_out_channels": int(fusion_out_channels) if fusion_out_channels is not None else None,
            "fusion_method": fusion_method,
        }
        self.center_feature_module_path = self.center_feature_cfg["module_path"]
        self.center_feature_expected_channels = self.center_feature_cfg["expected_channels"]
        self.center_feature_adapter_out_channels = self.center_feature_cfg["adapter_out_channels"]
        self.center_feature_native_stride = self.center_feature_cfg["native_stride"]
        self.center_feature_upsample_logits_to_target = self.center_feature_cfg["upsample_logits_to_target"]
        self.context_feature_module_path = self.center_feature_cfg["context_module_path"]
        self.context_feature_expected_channels = self.center_feature_cfg["context_expected_channels"]
        self.context_feature_native_stride = self.center_feature_cfg["context_native_stride"]
        self.primary_projection_out_channels = self.center_feature_cfg["primary_projection_out_channels"]
        self.context_projection_out_channels = self.center_feature_cfg["context_projection_out_channels"]
        self.fusion_out_channels = self.center_feature_cfg["fusion_out_channels"]
        self.fusion_method = self.center_feature_cfg["fusion_method"]
        self.multiscale_enabled = bool(self.context_feature_module_path is not None)
        self._captured_center_features: dict[str, torch.Tensor | None] = {}
        self._captured_center_feature_call_counts: dict[str, int] = {}
        self._decoder_capture_paths: dict[str, str] = {}

        if self.center_feature_module_path is not None and self.center_feature_expected_channels is None:
            raise ValueError("center_feature.expected_channels is required when center_feature.module_path is set")
        if self.multiscale_enabled and self.context_feature_expected_channels is None:
            raise ValueError("center_feature.context_expected_channels is required when center_feature.context_module_path is set")
        if self.multiscale_enabled and self.context_feature_native_stride is None:
            raise ValueError("center_feature.context_native_stride is required when center_feature.context_module_path is set")
        if self.multiscale_enabled and self.primary_projection_out_channels is None:
            raise ValueError("center_feature.primary_projection_out_channels is required for multiscale center path")
        if self.multiscale_enabled and self.context_projection_out_channels is None:
            raise ValueError("center_feature.context_projection_out_channels is required for multiscale center path")
        if self.multiscale_enabled and self.fusion_out_channels is None:
            raise ValueError("center_feature.fusion_out_channels is required for multiscale center path")
        if self.multiscale_enabled and self.fusion_method != "concat":
            raise ValueError(f"Unsupported center_feature.fusion_method: {self.fusion_method}")

        center_head_in_channels = int(out_channels[-1])
        if self.center_feature_module_path is not None:
            center_head_in_channels = int(self.center_feature_expected_channels)
        self.center_primary_projection = None
        self.center_context_projection = None
        self.center_fusion_adapter = None
        if self.multiscale_enabled:
            self.center_primary_projection = torch.nn.Conv2d(
                int(self.center_feature_expected_channels),
                int(self.primary_projection_out_channels),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.center_context_projection = torch.nn.Conv2d(
                int(self.context_feature_expected_channels),
                int(self.context_projection_out_channels),
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            fused_in_channels = int(self.primary_projection_out_channels) + int(self.context_projection_out_channels)
            self.center_fusion_adapter = torch.nn.Sequential(
                torch.nn.Conv2d(int(fused_in_channels), int(self.fusion_out_channels), kernel_size=3, stride=1, padding=1, bias=False),
                torch.nn.GroupNorm(num_groups=_groupnorm_groups(int(self.fusion_out_channels)), num_channels=int(self.fusion_out_channels)),
                torch.nn.SiLU(inplace=True),
            )
            torch.nn.init.kaiming_normal_(self.center_primary_projection.weight, mode="fan_out", nonlinearity="relu")
            torch.nn.init.kaiming_normal_(self.center_context_projection.weight, mode="fan_out", nonlinearity="relu")
            torch.nn.init.kaiming_normal_(self.center_fusion_adapter[0].weight, mode="fan_out", nonlinearity="relu")
            center_head_in_channels = int(self.fusion_out_channels)
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
            self._decoder_capture_paths[str(self.center_feature_module_path)] = self._decoder_block_name_from_module_path(self.center_feature_module_path)
        if self.context_feature_module_path is not None:
            self._decoder_capture_paths[str(self.context_feature_module_path)] = self._decoder_block_name_from_module_path(self.context_feature_module_path)
        self.freeze_base = False

    @staticmethod
    def _decoder_block_module_prefix() -> str:
        return "base.decoder.blocks."

    def _decoder_block_name_from_module_path(self, module_path: str) -> str:
        prefix = self._decoder_block_module_prefix()
        path = str(module_path or "").strip()
        if not path.startswith(prefix):
            raise ValueError(
                f"center_feature.module_path must target a decoder block materialized in the same SMP decoder forward pass; got {module_path}"
            )
        block_name = path[len(prefix):]
        if not block_name:
            raise ValueError(f"Decoder block name missing in module path: {module_path}")
        if block_name not in getattr(self.base.decoder, "blocks", {}):
            raise ValueError(f"Unknown decoder block module path: {module_path}")
        return str(block_name)

    def _record_captured_decoder_feature(self, block_name: str, output: torch.Tensor) -> None:
        for module_path, candidate_block_name in self._decoder_capture_paths.items():
            if str(candidate_block_name) != str(block_name):
                continue
            self._captured_center_feature_call_counts[str(module_path)] = int(self._captured_center_feature_call_counts.get(str(module_path), 0)) + 1
            self._captured_center_features[str(module_path)] = output

    def _forward_decoder_with_captures(self, encoder_features) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        decoder = self.base.decoder
        if not hasattr(decoder, "blocks") or not hasattr(decoder, "depth") or not hasattr(decoder, "in_channels"):
            raise RuntimeError("Installed SMP UnetPlusPlusDecoder does not expose the expected block topology")

        features = encoder_features[1:]
        features = features[::-1]

        dense_x: dict[str, torch.Tensor] = {}
        for layer_idx in range(len(decoder.in_channels) - 1):
            for depth_idx in range(int(decoder.depth) - int(layer_idx)):
                block_name = f"x_{depth_idx}_{depth_idx}" if layer_idx == 0 else f"x_{depth_idx}_{depth_idx + layer_idx}"
                block = decoder.blocks[block_name]
                if layer_idx == 0:
                    output = block(features[depth_idx], features[depth_idx + 1])
                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f"x_{idx}_{dense_l_i}"] for idx in range(depth_idx + 1, dense_l_i + 1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i + 1]], dim=1)
                    output = block(dense_x[f"x_{depth_idx}_{dense_l_i - 1}"], cat_features)
                dense_x[block_name] = output
                self._record_captured_decoder_feature(block_name, output)

        final_block_name = f"x_{0}_{decoder.depth}"
        final_output = decoder.blocks[final_block_name](dense_x[f"x_{0}_{decoder.depth - 1}"])
        dense_x[final_block_name] = final_output
        self._record_captured_decoder_feature(final_block_name, final_output)
        return final_output, dense_x

    def clear_center_feature_capture(self) -> None:
        for path in list(self._decoder_capture_paths):
            self._captured_center_features[str(path)] = None
            self._captured_center_feature_call_counts[str(path)] = 0

    def close_center_feature_hooks(self) -> None:
        return None

    def close_center_feature_hook(self) -> None:
        self.close_center_feature_hooks()

    def center_branch_parameter_prefixes(self) -> list[str]:
        prefixes = []
        for attr in (
            "center_primary_projection",
            "center_context_projection",
            "center_fusion_adapter",
            "center_adapter",
            "center_head",
        ):
            if getattr(self, attr, None) is not None:
                prefixes.append(f"{attr}.")
        return prefixes

    def center_branch_modules(self) -> list[torch.nn.Module]:
        out = []
        for attr in (
            "center_primary_projection",
            "center_context_projection",
            "center_fusion_adapter",
            "center_adapter",
            "center_head",
        ):
            module = getattr(self, attr, None)
            if module is not None:
                out.append(module)
        return out

    def center_feature_capture_info(self) -> dict:
        feat = self._captured_center_features.get(str(self.center_feature_module_path))
        return {
            "configured_module_path": self.center_feature_module_path,
            "actual_module_path": self.center_feature_module_path,
            "hook_call_count": int(self._captured_center_feature_call_counts.get(str(self.center_feature_module_path), 0)),
            "captured_shape": list(feat.shape) if torch.is_tensor(feat) else None,
            "captured_dtype": str(feat.dtype).replace("torch.", "") if torch.is_tensor(feat) else None,
            "expected_channels": self.center_feature_expected_channels,
            "actual_channels": int(feat.shape[1]) if torch.is_tensor(feat) and feat.ndim >= 2 else None,
            "native_stride": int(self.center_feature_native_stride or 1),
            "upsample_logits_to_target": bool(self.center_feature_upsample_logits_to_target),
            "adapter_out_channels": self.center_feature_adapter_out_channels,
            "multiscale_enabled": bool(self.multiscale_enabled),
            "context": {
                "configured_module_path": self.context_feature_module_path,
                "hook_call_count": int(self._captured_center_feature_call_counts.get(str(self.context_feature_module_path), 0)) if self.context_feature_module_path else None,
                "captured_shape": list(self._captured_center_features[str(self.context_feature_module_path)].shape) if self.context_feature_module_path and torch.is_tensor(self._captured_center_features.get(str(self.context_feature_module_path))) else None,
                "captured_dtype": str(self._captured_center_features[str(self.context_feature_module_path)].dtype).replace("torch.", "") if self.context_feature_module_path and torch.is_tensor(self._captured_center_features.get(str(self.context_feature_module_path))) else None,
                "expected_channels": self.context_feature_expected_channels,
                "native_stride": self.context_feature_native_stride,
                "projection_out_channels": self.context_projection_out_channels,
            },
            "access_requires_hooks": False,
            "decoder_capture_paths": dict(self._decoder_capture_paths),
            "primary_projection_out_channels": self.primary_projection_out_channels,
            "fusion_method": self.fusion_method,
            "fusion_out_channels": self.fusion_out_channels,
        }

    def resolve_center_features(self, decoder_output: torch.Tensor) -> torch.Tensor:
        if self.center_feature_module_path is None:
            return decoder_output
        if int(self._captured_center_feature_call_counts.get(str(self.center_feature_module_path), 0)) != 1:
            raise RuntimeError(
                f"Configured center feature capture must occur exactly once for {self.center_feature_module_path}; "
                f"got {int(self._captured_center_feature_call_counts.get(str(self.center_feature_module_path), 0))}"
            )
        feat = self._captured_center_features.get(str(self.center_feature_module_path))
        if not torch.is_tensor(feat):
            raise RuntimeError(f"Configured center feature tensor missing for {self.center_feature_module_path}")
        if self.center_feature_expected_channels is not None and int(feat.shape[1]) != int(self.center_feature_expected_channels):
            raise RuntimeError(
                f"Captured center feature channels mismatch for {self.center_feature_module_path}: "
                f"expected {int(self.center_feature_expected_channels)}, got {int(feat.shape[1])}"
            )
        if not self.multiscale_enabled:
            return feat
        if int(self._captured_center_feature_call_counts.get(str(self.context_feature_module_path), 0)) != 1:
            raise RuntimeError(
                f"Configured context feature capture must occur exactly once for {self.context_feature_module_path}; "
                f"got {int(self._captured_center_feature_call_counts.get(str(self.context_feature_module_path), 0))}"
            )
        context_feat = self._captured_center_features.get(str(self.context_feature_module_path))
        if not torch.is_tensor(context_feat):
            raise RuntimeError(f"Configured context feature tensor missing for {self.context_feature_module_path}")
        if self.context_feature_expected_channels is not None and int(context_feat.shape[1]) != int(self.context_feature_expected_channels):
            raise RuntimeError(
                f"Captured context feature channels mismatch for {self.context_feature_module_path}: "
                f"expected {int(self.context_feature_expected_channels)}, got {int(context_feat.shape[1])}"
            )
        primary_proj = self.center_primary_projection(feat)
        context_proj = self.center_context_projection(context_feat)
        if list(context_proj.shape[-2:]) != list(primary_proj.shape[-2:]):
            context_proj = F.interpolate(context_proj, size=primary_proj.shape[-2:], mode="bilinear", align_corners=False)
        if self.fusion_method != "concat":
            raise RuntimeError(f"Unsupported fusion method at runtime: {self.fusion_method}")
        fused = torch.cat([primary_proj, context_proj], dim=1)
        return self.center_fusion_adapter(fused)

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
        decoder_output, _dense_x = self._forward_decoder_with_captures(features)
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
