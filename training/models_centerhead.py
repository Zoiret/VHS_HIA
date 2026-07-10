from __future__ import annotations

import types

import torch


class UnetPlusPlusSemanticCenterHead(torch.nn.Module):
    def __init__(self, encoder_name: str, encoder_weights, in_channels: int, classes: int):
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

        self.center_head = SegmentationHead(in_channels=int(out_channels[-1]), out_channels=1, activation=None, kernel_size=3)

    @property
    def encoder(self):
        return self.base.encoder

    @property
    def decoder(self):
        return self.base.decoder

    @property
    def segmentation_head(self):
        return self.base.segmentation_head

    def forward(self, x: torch.Tensor) -> dict:
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
