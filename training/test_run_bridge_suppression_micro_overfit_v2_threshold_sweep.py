from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


_THIS_DIR = Path(__file__).resolve().parent


def _fake_recon(
    *,
    pixel_f1: float,
    full_mean_iou: float,
    full_success50: int,
    positive_mean_iou: float,
    positive_success50: int,
    negative_regresses: int,
    negative_topology_changes: int,
    negative_removed_fraction: float,
    negative_bridge_pixels: int,
    negative_zero_removal: int = 0,
    full_gt2_success: str = "1/3",
    full_gt3_success: str = "2/3",
    positive_gt2_success: str = "1/3",
    positive_gt3_success: str = "2/3",
) -> dict:
    pixel = {
        "precision": pixel_f1,
        "recall": pixel_f1,
        "f1": pixel_f1,
        "dice": pixel_f1,
        "tp": 10,
        "fp": 2,
        "fn": 3,
    }
    return {
        "pixel": pixel,
        "positive_subset": {
            "pixel": pixel,
            "reconstruction": {
                "p50_minus_predicted_bridge": {
                    "n": 6,
                    "mean_matched_iou": positive_mean_iou,
                    "all_iou_ge_0.50_count": positive_success50,
                    "all_iou_ge_0.50_rate": positive_success50 / 6.0,
                    "gt2_success": positive_gt2_success,
                    "gt3_success": positive_gt3_success,
                }
            },
        },
        "negative_subset": {
            "predicted_bridge_pixels": negative_bridge_pixels,
            "fraction_of_candidate_pixels_removed": negative_removed_fraction,
            "samples_with_zero_predicted_removal": negative_zero_removal,
            "starting_mean_matched_iou": 0.60,
            "refined_mean_matched_iou": 0.55,
            "num_improves": 0,
            "num_unchanged": 0 if negative_regresses > 0 else 4,
            "num_regresses": negative_regresses,
            "num_component_topology_changes": negative_topology_changes,
        },
        "removal_calibration": {
            "all_removed_over_candidate": 0.20,
            "positive_removed_over_candidate": 0.25,
            "negative_removed_over_candidate": negative_removed_fraction,
            "positive_gt_bridge_over_candidate": 0.16,
        },
        "reconstruction": {
            "p50_minus_predicted_bridge": {
                "n": 10,
                "mean_matched_iou": full_mean_iou,
                "all_iou_ge_0.50_count": full_success50,
                "all_iou_ge_0.50_rate": full_success50 / 10.0,
                "gt2_success": full_gt2_success,
                "gt3_success": full_gt3_success,
            }
        },
        "per_sample": [],
    }


class TestBridgeThresholdSweep(unittest.TestCase):
    @staticmethod
    def _mods():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import bridge_suppression_head as bridge
        import run_bridge_suppression_micro_overfit_v2_threshold_sweep as sweep

        return bridge, sweep

    def test_flatten_threshold_metrics_preserves_negative_metrics(self):
        _bridge, sweep = self._mods()
        row = sweep.flatten_threshold_metrics(
            0.65,
            _fake_recon(
                pixel_f1=0.58,
                full_mean_iou=0.56,
                full_success50=6,
                positive_mean_iou=0.57,
                positive_success50=3,
                negative_regresses=0,
                negative_topology_changes=0,
                negative_removed_fraction=0.04,
                negative_bridge_pixels=123,
            ),
        )
        self.assertEqual(row["threshold"], 0.65)
        self.assertEqual(row["negative_predicted_bridge_pixels"], 123)
        self.assertEqual(row["negative_num_regresses"], 0)
        self.assertEqual(row["negative_num_component_topology_changes"], 0)
        self.assertEqual(row["positive_success50_count"], 3)
        self.assertEqual(row["full_success50_count"], 6)

    def test_lexicographic_ranking_prioritizes_negative_preservation(self):
        _bridge, sweep = self._mods()
        rows = [
            {
                "threshold": 0.50,
                "negative_num_component_topology_changes": 2,
                "negative_num_regresses": 2,
                "positive_success50_count": 5,
                "positive_mean_iou": 0.80,
                "full_mean_iou": 0.80,
                "pixel_f1": 0.90,
            },
            {
                "threshold": 0.80,
                "negative_num_component_topology_changes": 0,
                "negative_num_regresses": 0,
                "positive_success50_count": 3,
                "positive_mean_iou": 0.60,
                "full_mean_iou": 0.65,
                "pixel_f1": 0.50,
            },
        ]
        best = sweep.select_best_threshold(rows)
        self.assertEqual(best["threshold"], 0.80)

    def test_safe_threshold_requires_positive_success_and_zero_negative_damage(self):
        _bridge, sweep = self._mods()
        rows = [
            {
                "threshold": 0.60,
                "negative_num_component_topology_changes": 0,
                "negative_num_regresses": 0,
                "positive_success50_count": 2,
                "positive_mean_iou": 0.50,
                "full_mean_iou": 0.55,
                "pixel_f1": 0.60,
            },
            {
                "threshold": 0.75,
                "negative_num_component_topology_changes": 0,
                "negative_num_regresses": 0,
                "positive_success50_count": 3,
                "positive_mean_iou": 0.58,
                "full_mean_iou": 0.59,
                "pixel_f1": 0.57,
            },
        ]
        safe = sweep.find_safe_threshold(rows)
        self.assertIsNotNone(safe)
        self.assertEqual(safe["threshold"], 0.75)

    def test_run_threshold_sweep_uses_thresholds_and_no_backward(self):
        bridge, sweep = self._mods()
        model = mock.Mock()
        cached_records = [{"sample_id": "s1", "candidate_mask": torch.ones((1, 2, 2)), "bridge_target": torch.zeros((1, 2, 2)), "p_leaf": torch.ones((1, 2, 2))}]
        thresholds = [0.50, 0.75]
        seen_thresholds: list[float] = []

        def fake_eval(_model, _records, _device, *, threshold):
            seen_thresholds.append(float(threshold))
            return _fake_recon(
                pixel_f1=0.5 + float(threshold) * 0.1,
                full_mean_iou=0.5,
                full_success50=5,
                positive_mean_iou=0.5,
                positive_success50=3,
                negative_regresses=0,
                negative_topology_changes=0,
                negative_removed_fraction=0.1,
                negative_bridge_pixels=10,
            )

        with mock.patch.object(bridge, "evaluate_reconstruction_levels_on_cached", side_effect=fake_eval), \
             mock.patch.object(torch.Tensor, "backward", side_effect=AssertionError("backward should not run during threshold sweep")):
            out = sweep.run_threshold_sweep(model=model, cached_records=cached_records, device=torch.device("cpu"), thresholds=thresholds)
        self.assertEqual(seen_thresholds, thresholds)
        self.assertEqual([row["threshold"] for row in out["rows"]], thresholds)

    def test_threshold_sweep_does_not_mutate_cached_candidate_or_targets(self):
        bridge, sweep = self._mods()
        model = mock.Mock()
        cached_records = [
            {
                "sample_id": "s1",
                "candidate_mask": torch.ones((1, 2, 2), dtype=torch.float32),
                "bridge_target": torch.zeros((1, 2, 2), dtype=torch.float32),
                "p_leaf": torch.full((1, 2, 2), 0.5, dtype=torch.float32),
            }
        ]
        before_candidate = cached_records[0]["candidate_mask"].clone()
        before_target = cached_records[0]["bridge_target"].clone()
        before_p_leaf = cached_records[0]["p_leaf"].clone()
        with mock.patch.object(bridge, "evaluate_reconstruction_levels_on_cached", return_value=_fake_recon(
            pixel_f1=0.5,
            full_mean_iou=0.5,
            full_success50=5,
            positive_mean_iou=0.5,
            positive_success50=3,
            negative_regresses=0,
            negative_topology_changes=0,
            negative_removed_fraction=0.1,
            negative_bridge_pixels=10,
        )):
            sweep.run_threshold_sweep(model=model, cached_records=cached_records, device=torch.device("cpu"), thresholds=[0.5, 0.9])
        self.assertTrue(torch.equal(cached_records[0]["candidate_mask"], before_candidate))
        self.assertTrue(torch.equal(cached_records[0]["bridge_target"], before_target))
        self.assertTrue(torch.equal(cached_records[0]["p_leaf"], before_p_leaf))

    def test_resolve_checkpoint_reports_identical_weights(self):
        _bridge, sweep = self._mods()
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            payload = {
                "model": {"x": torch.tensor([1.0, 2.0])},
                "optimizer": {},
                "step": 300,
                "extra": {},
            }
            for name in ("best_reconstruction.pth", "best_pixel_f1.pth", "last.pth"):
                torch.save(payload, run_dir / name)
            info = sweep.resolve_checkpoint_to_evaluate(run_dir=run_dir)
            self.assertTrue(info["best_pixel_best_reconstruction_last_identical"])
            self.assertEqual(Path(info["evaluated"]["path"]).name, "best_reconstruction.pth")

    def test_inspect_checkpoint_uses_shared_canonical_model_hash(self):
        bridge, sweep = self._mods()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "best_reconstruction.pth"
            torch.save({"model": {"x": torch.tensor([1.0, 2.0])}, "step": 300}, path)
            with mock.patch.object(bridge, "canonical_model_state_sha256", return_value="historical-sha") as hash_mock:
                info = sweep.inspect_checkpoint(path)
            hash_mock.assert_called_once()
            self.assertEqual(info["model_state_sha256"], "historical-sha")

    def test_locked_v2_manifest_counts_and_ids_unchanged(self):
        _bridge, _sweep = self._mods()
        manifest_path = Path("training/manifests/bridge_suppression_micro_overfit_v2_manifest.json").resolve()
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["bridge_positive_count"], 6)
        self.assertEqual(payload["bridge_negative_count"], 4)
        self.assertEqual(len(payload["sample_ids"]), 10)
        self.assertEqual(payload["sample_ids"], [
            "m11_p01_s06",
            "m10_p01_s11",
            "m09_p01_s27",
            "m11_p02_s03",
            "m03_p01_s10",
            "m12_p02_s08",
            "m13_p03_s05",
            "m12_p01_s00",
            "m13_p02_s08",
            "m15_p01_s08",
        ])


if __name__ == "__main__":
    unittest.main()
