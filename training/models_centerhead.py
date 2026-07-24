from __future__ import annotations

import types

import torch


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
    def __init__(self, encoder_name: str, encoder_weights, in_channels: int, classes: int, center_head_type: str = "linear_1x1"):
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

        self.center_head_type = str(center_head_type).strip().lower() or "linear_1x1"
        if self.center_head_type == "linear_1x1":
            self.center_head = SegmentationHead(in_channels=int(out_channels[-1]), out_channels=1, activation=None, kernel_size=3)
        elif self.center_head_type == "spatial_dilated":
            self.center_head = SpatialDilatedCenterHead(in_channels=int(out_channels[-1]))
        else:
            raise ValueError(f"Unsupported center_head_type: {center_head_type}")
        self.freeze_base = False

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

    def forward(self, x: torch.Tensor) -> dict:
        if bool(getattr(self, "freeze_base", False)):
            with torch.no_grad():
                features = self.encoder(x)
                decoder_output = self.decoder(features)
                semantic = self.segmentation_head(decoder_output)
            center = self.center_head(decoder_output.detach())
        else:
            features = self.encoder(x)
            decoder_output = self.decoder(features)
            semantic = self.segmentation_head(decoder_output)
            center = self.center_head(decoder_output)
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
