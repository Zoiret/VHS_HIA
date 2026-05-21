from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError as e:
    raise SystemExit(
        "PyTorch is not installed. Install training deps with:\n"
        "  py -m pip install -r requirements-train.txt"
    ) from e


@dataclass(frozen=True)
class PerClassMetrics:
    dice: list[float]
    iou: list[float]


def _confusion_stats(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = pred.view(-1)
    target = target.view(-1)
    tp = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    fp = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    fn = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)

    for c in range(num_classes):
        p = pred == c
        t = target == c
        tp[c] = torch.sum(p & t).float()
        fp[c] = torch.sum(p & ~t).float()
        fn[c] = torch.sum(~p & t).float()
    return tp, fp, fn


@torch.no_grad()
def compute_per_class_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-7) -> PerClassMetrics:
    pred = torch.argmax(logits, dim=1)
    tp, fp, fn = _confusion_stats(pred, target, int(num_classes))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)

    return PerClassMetrics(dice=dice.detach().cpu().tolist(), iou=iou.detach().cpu().tolist())
