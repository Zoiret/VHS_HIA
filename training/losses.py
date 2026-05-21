from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e


class CombinedCrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ce_coef: float = 1.0,
        dice_coef: float = 1.0,
        class_weights: torch.Tensor | None = None,
        boundary_enabled: bool = False,
        boundary_coef: float = 0.0,
        boundary_mode: str = "weight_map",
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ce_coef = float(ce_coef)
        self.dice_coef = float(dice_coef)
        self.class_weights = class_weights
        self.boundary_enabled = bool(boundary_enabled)
        self.boundary_coef = float(boundary_coef)
        self.boundary_mode = str(boundary_mode).strip().lower()

        self.ce = nn.CrossEntropyLoss(weight=class_weights)

        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as e:
            raise SystemExit(
                "segmentation-models-pytorch is not installed. Install training deps with:\n"
                "  py -m pip install -r requirements-train.txt"
            ) from e

        self.dice = smp.losses.DiceLoss(mode="multiclass", from_logits=True)

    def forward(self, logits: torch.Tensor, target: torch.Tensor, boundary_target: torch.Tensor | None = None) -> torch.Tensor:
        if (
            self.boundary_enabled
            and boundary_target is not None
            and int(self.num_classes) == 2
            and float(self.boundary_coef) > 0.0
            and self.boundary_mode == "weight_map"
        ):
            ce_per_pixel = F.cross_entropy(logits, target, weight=self.class_weights, reduction="none")
            weight_map = 1.0 + float(self.boundary_coef) * boundary_target.float()
            loss_ce = (ce_per_pixel * weight_map).mean()
        else:
            loss_ce = self.ce(logits, target)

        loss_dice = self.dice(logits, target)
        loss = self.ce_coef * loss_ce + self.dice_coef * loss_dice
        return loss
