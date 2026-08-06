from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import torch


_THIS_DIR = Path(__file__).resolve().parent


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base_block = torch.nn.Linear(4, 4, bias=False)
        self.center_adapter = torch.nn.Linear(4, 4, bias=False)
        self.center_head = torch.nn.Linear(4, 1, bias=True)


class TestTrainCenterheadFullBaseline(unittest.TestCase):
    @staticmethod
    def _mod():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import train_centerhead as mod

        return mod

    def test_best_threshold_selection_uses_required_tie_break(self):
        mod = self._mod()
        rows = [
            {
                "threshold": 0.05,
                "center_f1_mean_samples": 0.50,
                "strict_marker_contract_pass_rate": 0.60,
                "exact_center_count_accuracy": 0.70,
                "localization_error_px": 3.0,
            },
            {
                "threshold": 0.03,
                "center_f1_mean_samples": 0.50,
                "strict_marker_contract_pass_rate": 0.80,
                "exact_center_count_accuracy": 0.60,
                "localization_error_px": 4.0,
            },
            {
                "threshold": 0.02,
                "center_f1_mean_samples": 0.50,
                "strict_marker_contract_pass_rate": 0.80,
                "exact_center_count_accuracy": 0.80,
                "localization_error_px": 2.0,
            },
        ]
        best = mod._select_best_threshold_row(rows, primary_metric="center_f1_mean_samples")
        self.assertEqual(float(best["threshold"]), 0.02)

    def test_epoch_candidate_prefers_earlier_epoch_on_full_tie(self):
        mod = self._mod()
        incumbent = {
            "center_f1_mean_samples": 0.8,
            "strict_marker_contract_pass_rate": 0.7,
            "exact_center_count_accuracy": 0.6,
            "localization_error_px": 1.0,
        }
        candidate = dict(incumbent)
        self.assertFalse(mod._is_better_epoch_candidate(candidate, incumbent, epoch=6, incumbent_epoch=5, primary_metric="center_f1_mean_samples"))
        self.assertTrue(mod._is_better_epoch_candidate(candidate, incumbent, epoch=4, incumbent_epoch=5, primary_metric="center_f1_mean_samples"))

    def test_frozen_optimizer_contains_only_center_parameters(self):
        mod = self._mod()
        model = _TinyModel()
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("center_head.") or name.startswith("center_adapter.")
        cfg = {"train": {"lr": 1e-3, "lr_center_head": 1e-3, "weight_decay": 0.0}}
        optimizer, meta = mod._build_optimizer_groups(model, cfg, freeze_info=None, freeze_base=True)
        self.assertEqual(len(optimizer.param_groups), 1)
        names = meta[0]["parameter_names"]
        self.assertTrue(all(name.startswith("center_head.") or name.startswith("center_adapter.") for name in names))
        self.assertFalse(any(name.startswith("base_block.") for name in names))

    def test_validation_reports_write_per_patient_and_threshold_tables(self):
        mod = self._mod()
        val_metrics = {
            "per_patient_center_metrics": {
                "m01_p01": {
                    "sample_count": 3,
                    "center_precision": 0.7,
                    "center_recall": 0.8,
                    "center_f1": 0.75,
                    "center_precision_mean_samples": 0.71,
                    "center_recall_mean_samples": 0.81,
                    "center_f1_mean_samples": 0.76,
                    "exact_center_count_accuracy": 0.67,
                    "strict_marker_contract_pass_count": 2,
                    "strict_marker_contract_pass_rate": 0.67,
                    "localization_error_px": 2.5,
                }
            },
            "per_gt_count_center_metrics": {
                "1": {
                    "sample_count": 2,
                    "center_precision": 1.0,
                    "center_recall": 1.0,
                    "center_f1": 1.0,
                    "center_precision_mean_samples": 1.0,
                    "center_recall_mean_samples": 1.0,
                    "center_f1_mean_samples": 1.0,
                    "exact_center_count_accuracy": 1.0,
                    "strict_marker_contract_pass_count": 2,
                    "strict_marker_contract_pass_rate": 1.0,
                    "localization_error_px": 1.0,
                }
            },
        }
        sweep = {
            "rows": [
                {
                    "threshold": 0.03,
                    "center_precision": 0.8,
                    "center_recall": 0.7,
                    "center_f1": 0.75,
                    "center_precision_mean_samples": 0.82,
                    "center_recall_mean_samples": 0.72,
                    "center_f1_mean_samples": 0.77,
                    "predicted_center_count_mean": 2.0,
                    "predicted_center_count_median": 2.0,
                    "exact_center_count_accuracy": 0.7,
                    "strict_marker_contract_pass_count": 5,
                    "strict_marker_contract_pass_rate": 0.71,
                    "missing_gt_instances": 1,
                    "gt_instances_with_multiple_markers": 0,
                    "markers_outside_all_gt_instances": 0,
                    "localization_error_px": 2.0,
                    "sample_count_gt1": 2,
                    "sample_count_gt2": 3,
                    "sample_count_gt3": 2,
                    "pass_count_gt1": 2,
                    "pass_count_gt2": 2,
                    "pass_count_gt3": 1,
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            mod._write_validation_reports(out_dir, epoch=1, val_metrics=val_metrics, sweep_res=sweep, locked_threshold=0.03)
            self.assertTrue((out_dir / "validation_per_patient_metrics.csv").exists())
            self.assertTrue((out_dir / "validation_gt_count_metrics.csv").exists())
            self.assertTrue((out_dir / "validation_threshold_summary.csv").exists())
            with (out_dir / "validation_threshold_summary.csv").open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["is_locked_reference_threshold"], "True")


if __name__ == "__main__":
    unittest.main()
