from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


_THIS_DIR = Path(__file__).resolve().parent


class TestTrainBridgeSuppressionHead(unittest.TestCase):
    @staticmethod
    def _mods():
        if str(_THIS_DIR) not in sys.path:
            sys.path.insert(0, str(_THIS_DIR))
        import bridge_suppression_head as bridge
        import train_bridge_suppression_head as runner

        return bridge, runner

    def test_smoke_only_does_not_require_full_train_val_audit_outputs(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg["train"]["save_dir"] = str(Path(td) / "micro")
            cfg["reserved_full_run"]["save_dir"] = str(Path(td) / "future_full")
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml").resolve())
            fake_record = {
                "sample_id": "m00_p00_s00",
                "image_path": str((bridge.REPO_ROOT / "datasets" / "converted_full_multiclass" / "dummy.png").resolve()),
                "gt_semantic": __import__("numpy").zeros((16, 16), dtype="uint8"),
                "bridge_target": __import__("numpy").zeros((16, 16), dtype="uint8"),
            }
            with mock.patch.object(bridge, "_build_split_items", return_value=[{"sample_id": "m00_p00_s00"}]), \
                 mock.patch.object(bridge, "build_model_from_cfg"), \
                 mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_sha256": bridge.CHECKPOINT_SHA256 if hasattr(bridge, "CHECKPOINT_SHA256") else ""}), \
                 mock.patch.object(bridge, "build_optimizer", return_value=(mock.Mock(), {"total_trainable_params": 1})), \
                 mock.patch.object(bridge, "mine_bridge_records_for_split", return_value=[fake_record]), \
                 mock.patch.object(runner, "_smoke_step", return_value={"status": "pass"}):
                summary = runner.run_pipeline(cfg, smoke_only=True)
            self.assertEqual(summary["smoke"]["status"], "pass")
            self.assertIn("a100_smoke_command", summary)

    def test_reserved_full_run_dir_blocker(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml")
        with tempfile.TemporaryDirectory() as td:
            micro = Path(td) / "micro"
            future = Path(td) / "future"
            future.mkdir(parents=True, exist_ok=True)
            cfg["train"]["save_dir"] = str(micro)
            cfg["reserved_full_run"]["save_dir"] = str(future)
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit.yaml").resolve())
            with self.assertRaises(SystemExit):
                runner.run_pipeline(cfg, smoke_only=True)

    def test_v2_config_locks_manifest_and_disables_augmentation(self):
        bridge, _runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        self.assertIn("bridge_suppression_micro_overfit_v2_manifest.json", str(cfg["micro_overfit"]["manifest_path"]))
        self.assertEqual(int(cfg["micro_overfit"]["max_steps"]), 300)
        self.assertEqual(float(cfg["bridge_head"]["candidate_threshold"]), 0.50)
        self.assertEqual(float(cfg["bridge_head"]["remove_threshold"]), 0.50)
        self.assertTrue(all(bool(v) is False for v in (cfg.get("augment") or {}).values()))

    def test_run_pipeline_uses_locked_manifest_and_never_regenerates(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg["train"]["save_dir"] = str(Path(td) / "micro_v2")
            cfg["reserved_full_run"]["save_dir"] = str(Path(td) / "future_full")
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml").resolve())
            manifest_payload = {
                "source_split": "datasets/converted_full_multiclass_curated/train.txt",
                "source_split_canonical_sha256": "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82",
                "sample_ids": ["s1"] * 10,
                "rows": [{"sample_id": f"s{i}", "patient_id": f"p{i}", "gt_count": 2 if i < 5 else 3, "bridge_positive": 1 if i < 6 else 0, "bridge_pixels": 1 if i < 6 else 0, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 1 if i < 6 else 0, "reason_selected": "x"} for i in range(10)],
            }
            manifest_payload["sample_ids"] = [row["sample_id"] for row in manifest_payload["rows"]]
            fake_records = [
                {"sample_id": row["sample_id"], "patient_id": row["patient_id"], "gt_count": row["gt_count"], "bridge_positive": row["bridge_positive"], "bridge_pixels": row["bridge_pixels"], "candidate_pixels": row["candidate_pixels"], "topology_changes_if_oracle_removed": row["topology_changes_if_oracle_removed"]}
                for row in manifest_payload["rows"]
            ]
            with mock.patch.object(bridge, "build_model_from_cfg", return_value=mock.Mock()), \
                 mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_sha256": "x"}), \
                 mock.patch.object(bridge, "read_locked_micro_manifest", return_value=manifest_payload), \
                 mock.patch.object(bridge, "validate_locked_manifest_source_split", return_value={"status": "pass", "resolved_source_split": str(bridge.DEFAULT_TRAIN_SPLIT.resolve()), "actual_source_split_canonical_sha256": manifest_payload["source_split_canonical_sha256"]}), \
                 mock.patch.object(bridge, "mine_bridge_records_for_split", side_effect=[fake_records, fake_records, fake_records]), \
                 mock.patch.object(bridge, "summarize_bridge_records", return_value={"sample_count": 1}), \
                 mock.patch.object(bridge, "build_validation_audit", return_value={"verdict": "valid_for_bridge_head_development"}), \
                 mock.patch.object(bridge, "validate_locked_micro_records", return_value={"status": "pass", "sample_ids": manifest_payload["sample_ids"]}), \
                 mock.patch.object(bridge, "build_optimizer", return_value=(mock.Mock(), {"total_trainable_params": 1})), \
                 mock.patch.object(bridge, "cache_microset_features", return_value=fake_records), \
                 mock.patch.object(runner, "_smoke_step", return_value={"status": "pass"}), \
                 mock.patch.object(runner, "_run_micro_overfit", return_value={"final": {"step": 1}}), \
                 mock.patch.object(bridge, "save_train_target_visual_audit", return_value={}), \
                 mock.patch.object(bridge, "write_micro_manifest", side_effect=AssertionError("must not regenerate manifest")):
                runner.run_pipeline(cfg, smoke_only=False)

    def test_manifest_block_happens_before_optimizer_creation(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg["train"]["save_dir"] = str(Path(td) / "micro_v2")
            cfg["reserved_full_run"]["save_dir"] = str(Path(td) / "future_full")
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml").resolve())
            manifest_payload = {
                "source_split": "datasets/converted_full_multiclass_curated/train.txt",
                "source_split_canonical_sha256": "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82",
                "sample_ids": ["s1"],
                "rows": [{"sample_id": "s1", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "bridge_pixels": 1, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 1, "reason_selected": "x"}],
            }
            fake_record = {"sample_id": "s1", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "bridge_pixels": 1, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 1}
            build_optimizer = mock.Mock(return_value=(mock.Mock(), {"total_trainable_params": 1}))
            with mock.patch.object(bridge, "build_model_from_cfg", return_value=mock.Mock()), \
                 mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_sha256": "x"}), \
                 mock.patch.object(bridge, "read_locked_micro_manifest", return_value=manifest_payload), \
                 mock.patch.object(bridge, "validate_locked_manifest_source_split", return_value={"status": "pass", "resolved_source_split": str(bridge.DEFAULT_TRAIN_SPLIT.resolve()), "actual_source_split_canonical_sha256": manifest_payload["source_split_canonical_sha256"]}), \
                 mock.patch.object(bridge, "mine_bridge_records_for_split", side_effect=[[fake_record], [fake_record], [fake_record]]), \
                 mock.patch.object(bridge, "summarize_bridge_records", return_value={"sample_count": 1}), \
                 mock.patch.object(bridge, "build_validation_audit", return_value={"verdict": "valid_for_bridge_head_development"}), \
                 mock.patch.object(bridge, "validate_locked_micro_records", return_value={"status": "blocked", "sample_ids": ["s1"], "actual_sample_ids": []}), \
                 mock.patch.object(bridge, "build_optimizer", build_optimizer), \
                 mock.patch.object(bridge, "save_train_target_visual_audit", return_value={}):
                with self.assertRaises(SystemExit):
                    runner.run_pipeline(cfg, smoke_only=False)
            build_optimizer.assert_not_called()

    def test_split_or_sha_block_happens_before_optimizer_creation(self):
        bridge, runner = self._mods()
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg["train"]["save_dir"] = str(Path(td) / "micro_v2")
            cfg["reserved_full_run"]["save_dir"] = str(Path(td) / "future_full")
            cfg["_config_path"] = str((bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_suppression_frozen_semantic_micro_overfit_v2.yaml").resolve())
            manifest_payload = {
                "source_split": r"E:\3d_visual\ml\datasets\converted_full_multiclass_curated\train.txt",
                "source_split_canonical_sha256": "f5e920ffaf54c0a0034c457cf3c951f71e186a9f35e3fe67a5eee95737b2ee82",
                "sample_ids": ["s1"],
                "rows": [{"sample_id": "s1", "patient_id": "p1", "gt_count": 2, "bridge_positive": 1, "bridge_pixels": 1, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 1, "reason_selected": "x"}],
            }
            fake_train_record = {"sample_id": "train", "patient_id": "p0", "gt_count": 2, "bridge_positive": 0, "bridge_pixels": 0, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 0}
            fake_val_record = {"sample_id": "val", "patient_id": "p9", "gt_count": 3, "bridge_positive": 0, "bridge_pixels": 0, "candidate_pixels": 10, "topology_changes_if_oracle_removed": 0}
            build_optimizer = mock.Mock(return_value=(mock.Mock(), {"total_trainable_params": 1}))
            with mock.patch.object(bridge, "build_model_from_cfg", return_value=mock.Mock()), \
                 mock.patch.object(bridge, "load_semantic_checkpoint", return_value={"checkpoint_sha256": "x"}), \
                 mock.patch.object(bridge, "read_locked_micro_manifest", return_value=manifest_payload), \
                 mock.patch.object(bridge, "mine_bridge_records_for_split", side_effect=[[fake_train_record], [fake_val_record], AssertionError("must not resolve microset after split validation failure")]), \
                 mock.patch.object(bridge, "summarize_bridge_records", return_value={"sample_count": 1}), \
                 mock.patch.object(bridge, "build_validation_audit", return_value={"verdict": "valid_for_bridge_head_development"}), \
                 mock.patch.object(bridge, "validate_locked_manifest_source_split", return_value={"status": "blocked", "error": "bad split"}), \
                 mock.patch.object(bridge, "build_optimizer", build_optimizer), \
                 mock.patch.object(bridge, "save_train_target_visual_audit", return_value={}):
                with self.assertRaises(SystemExit):
                    runner.run_pipeline(cfg, smoke_only=False)
            build_optimizer.assert_not_called()

    def test_run_micro_overfit_saves_separate_pixel_and_reconstruction_checkpoints(self):
        bridge, runner = self._mods()

        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Linear(1, 1)
                for p in self.base.parameters():
                    p.requires_grad = False
                self.context_projection = torch.nn.Conv2d(1, 1, 1)
                self.bridge_head = torch.nn.Conv2d(1, 1, 1)

            def bridge_forward_from_cached(self, *, x_0_4, x_2_2, p_leaf):
                logits = self.bridge_head(x_0_4[:, :1])
                return {"bridge_logits": logits, "candidate_mask": (p_leaf >= 0).float()}

        model = FakeModel()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
        cached_records = [
            {
                "sample_id": "s1",
                "x_0_4": torch.ones((1, 2, 2), dtype=torch.float32),
                "x_2_2": torch.ones((1, 1, 1), dtype=torch.float32),
                "p_leaf": torch.ones((1, 2, 2), dtype=torch.float32),
                "candidate_mask": torch.ones((1, 2, 2), dtype=torch.float32),
                "bridge_target": torch.zeros((1, 2, 2), dtype=torch.float32),
                "gt_instances": __import__("numpy").ones((2, 2), dtype="uint8"),
                "candidate_mask_np": __import__("numpy").ones((2, 2), dtype="uint8"),
                "oracle_removed_mask": __import__("numpy").ones((2, 2), dtype="uint8"),
                "image_path": "x",
                "patient_id": "p1",
                "gt_count": 2,
                "bridge_positive": 1,
                "candidate_pixels": 4,
                "bridge_pixels": 1,
            }
        ]
        loss_fn = bridge.CandidateBalancedBCEDiceLoss()
        recon_seq = [
            {
                "positive_subset": {"pixel": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "dice": 0.5, "tp": 1, "fp": 1, "fn": 1}, "reconstruction": {"p50_start": {"mean_matched_iou": 0.1, "all_iou_ge_0.50_count": 0}, "p50_minus_predicted_bridge": {"mean_matched_iou": 0.4, "all_iou_ge_0.50_count": 1}, "p50_minus_gt_oracle_bridge": {"mean_matched_iou": 0.9, "all_iou_ge_0.50_count": 1}}},
                "negative_subset": {"predicted_bridge_pixels": 0, "fraction_of_candidate_pixels_removed": 0.0, "samples_with_zero_predicted_removal": 0, "starting_mean_matched_iou": 0.0, "refined_mean_matched_iou": 0.0, "num_improves": 0, "num_unchanged": 0, "num_regresses": 0, "num_component_topology_changes": 0},
                "removal_calibration": {"all_removed_over_candidate": 0.1, "positive_removed_over_candidate": 0.1, "negative_removed_over_candidate": 0.0, "positive_gt_bridge_over_candidate": 0.2},
                "reconstruction": {"p50_start": {"mean_matched_iou": 0.1, "all_iou_ge_0.50_count": 0}, "p50_minus_predicted_bridge": {"mean_matched_iou": 0.4, "all_iou_ge_0.50_count": 1}, "p50_minus_gt_oracle_bridge": {"mean_matched_iou": 0.9, "all_iou_ge_0.50_count": 1}},
            },
            {
                "positive_subset": {"pixel": {"precision": 0.2, "recall": 0.2, "f1": 0.2, "dice": 0.2, "tp": 1, "fp": 4, "fn": 4}, "reconstruction": {"p50_start": {"mean_matched_iou": 0.1, "all_iou_ge_0.50_count": 0}, "p50_minus_predicted_bridge": {"mean_matched_iou": 0.3, "all_iou_ge_0.50_count": 2}, "p50_minus_gt_oracle_bridge": {"mean_matched_iou": 0.9, "all_iou_ge_0.50_count": 1}}},
                "negative_subset": {"predicted_bridge_pixels": 0, "fraction_of_candidate_pixels_removed": 0.0, "samples_with_zero_predicted_removal": 0, "starting_mean_matched_iou": 0.0, "refined_mean_matched_iou": 0.0, "num_improves": 0, "num_unchanged": 0, "num_regresses": 0, "num_component_topology_changes": 0},
                "removal_calibration": {"all_removed_over_candidate": 0.2, "positive_removed_over_candidate": 0.2, "negative_removed_over_candidate": 0.0, "positive_gt_bridge_over_candidate": 0.2},
                "reconstruction": {"p50_start": {"mean_matched_iou": 0.1, "all_iou_ge_0.50_count": 0}, "p50_minus_predicted_bridge": {"mean_matched_iou": 0.3, "all_iou_ge_0.50_count": 2}, "p50_minus_gt_oracle_bridge": {"mean_matched_iou": 0.9, "all_iou_ge_0.50_count": 1}},
            },
        ]
        pixel_seq = [
            {"precision": 0.6, "recall": 0.6, "f1": 0.6, "dice": 0.6, "tp": 1, "fp": 1, "fn": 1},
            {"precision": 0.3, "recall": 0.3, "f1": 0.3, "dice": 0.3, "tp": 1, "fp": 2, "fn": 2},
        ]
        saved = []

        with tempfile.TemporaryDirectory() as td, \
             mock.patch.object(bridge, "evaluate_reconstruction_levels_on_cached", side_effect=recon_seq), \
             mock.patch.object(bridge, "compute_binary_metrics_from_domain", side_effect=pixel_seq), \
             mock.patch.object(bridge, "save_checkpoint", side_effect=lambda path, *args, **kwargs: saved.append(Path(path).name)):
            summary = runner._run_micro_overfit(
                model=model,
                cached_records=cached_records,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device("cpu"),
                max_steps=2,
                log_every=1,
                save_dir=Path(td),
                cfg={"seed": 1337},
            )
        self.assertEqual(summary["best_pixel"]["step"], 1)
        self.assertEqual(summary["best_reconstruction"]["step"], 2)
        self.assertIn("best_pixel_f1.pth", saved)
        self.assertIn("best_reconstruction.pth", saved)
        self.assertIn("last.pth", saved)


if __name__ == "__main__":
    unittest.main()
