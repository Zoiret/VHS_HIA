from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_suppression_head as bridge_head
import bridge_suppression_head as bridge
import train_bridge_presence_gate_v4_patient_disjoint_dev as runner


class TestTrainBridgePresenceGateV4PatientDisjointDev(unittest.TestCase):
    def _fake_prepared(self):
        gate_model = torch.nn.Sequential(torch.nn.Linear(105, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
        train_cache = [
            {
                "sample_id": "t1",
                "gate_target": 1,
                "bridge_positive": 1,
                "original_v2_remove_pixels": 2,
                "closed": {"candidate_pixels": 10, "predicted_removed_pixels": 0, "predicted_mean_iou": 0.40, "start_mean_iou": 0.40, "predicted_success50": 0, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.0},
                "open": {"candidate_pixels": 10, "predicted_removed_pixels": 2, "predicted_mean_iou": 0.60, "start_mean_iou": 0.40, "predicted_success50": 1, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.1},
            }
        ]
        val_cache = [
            {
                "sample_id": "v1",
                "gate_target": 1,
                "bridge_positive": 1,
                "original_v2_remove_pixels": 2,
                "closed": {"candidate_pixels": 10, "predicted_removed_pixels": 0, "predicted_mean_iou": 0.4234538944043546, "start_mean_iou": 0.4234538944043546, "predicted_success50": 1, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.0},
                "open": {"candidate_pixels": 10, "predicted_removed_pixels": 2, "predicted_mean_iou": 0.5061690111512952, "start_mean_iou": 0.4234538944043546, "predicted_success50": 1, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.1},
            },
            {
                "sample_id": "v2",
                "gate_target": 1,
                "bridge_positive": 1,
                "original_v2_remove_pixels": 2,
                "closed": {"candidate_pixels": 10, "predicted_removed_pixels": 0, "predicted_mean_iou": 0.4234538944043546, "start_mean_iou": 0.4234538944043546, "predicted_success50": 0, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.0},
                "open": {"candidate_pixels": 10, "predicted_removed_pixels": 2, "predicted_mean_iou": 0.5061690111512952, "start_mean_iou": 0.4234538944043546, "predicted_success50": 1, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.1},
            },
            {
                "sample_id": "v3",
                "gate_target": 1,
                "bridge_positive": 1,
                "original_v2_remove_pixels": 2,
                "closed": {"candidate_pixels": 10, "predicted_removed_pixels": 0, "predicted_mean_iou": 0.4234538944043546, "start_mean_iou": 0.4234538944043546, "predicted_success50": 0, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.0},
                "open": {"candidate_pixels": 10, "predicted_removed_pixels": 2, "predicted_mean_iou": 0.5233930737601301, "start_mean_iou": 0.4234538944043546, "predicted_success50": 1, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.1},
            },
        ] + [
            {
                "sample_id": f"n{i}",
                "gate_target": 0,
                "bridge_positive": 0,
                "original_v2_remove_pixels": 2,
                "closed": {"candidate_pixels": 10, "predicted_removed_pixels": 0, "predicted_mean_iou": 0.8, "start_mean_iou": 0.8, "predicted_success50": 1, "component_topology_changed": 0, "predicted_reconstruction_runtime_seconds": 0.0},
                "open": {"candidate_pixels": 10, "predicted_removed_pixels": 2, "predicted_mean_iou": 0.7 if i <= 20 else 0.8, "start_mean_iou": 0.8, "predicted_success50": 1, "component_topology_changed": 1 if i <= 24 else 0, "predicted_reconstruction_runtime_seconds": 0.1},
            }
            for i in range(1, 25)
        ]
        return {
            "manifest_stage": {
                "manifest": {
                    "contract": {
                        "train_summary": {"sample_count": 121, "patient_count": 17},
                        "val_summary": {"sample_count": 36, "patient_count": 5},
                    }
                }
            },
            "contract": {"train_summary": {"sample_count": 121, "patient_count": 17}},
            "device": torch.device("cpu"),
            "frozen_model": mock.Mock(base=torch.nn.Linear(2, 2), bridge_head=torch.nn.Linear(2, 2)),
            "frozen_v2_checkpoint": {"checkpoint_file_sha256": "abc", "checkpoint_model_state_sha256": "def"},
            "train_prepared": {
                "cached_records": [{"sample_id": "t1", "x_0_4": torch.ones((1, 2, 2)), "x_2_2": torch.ones((1, 1, 1)), "p_leaf": torch.ones((1, 2, 2))}],
                "frozen_logits": torch.ones((1, 1, 2, 2)),
                "frozen_logit_diagnostics": {"input_devices": {"x_0_4": "cpu", "x_2_2": "cpu", "p_leaf": "cpu"}},
                "gate_model": gate_model,
                "features_t": torch.ones((1, 105)),
                "targets_t": torch.ones((1, 1)),
                "feature_rows": [{"sample_id": "t1", "candidate_fraction": 0.1541646271944046}],
                "selector_input_rows": [{"sample_id": "t1", "patient_id": "t1", "gt_count": 2, "bridge_target": 1, "candidate_pixels": 90930, "candidate_fraction_denominator": 589824, "candidate_fraction": 0.1541646271944046}],
                "selector_audit": {
                    "selector_input_rows": [{"sample_id": "t1", "patient_id": "t1", "gt_count": 2, "bridge_target": 1, "candidate_pixels": 90930, "candidate_fraction_denominator": 589824, "candidate_fraction": 0.1541646271944046}],
                    "selector_input_summary": {
                        "row_count": 1,
                        "scientific_selector_input_sha256": "sha-train",
                        "audit_row_sha256": "audit-train",
                    },
                    "selected_rule": {"scalar": "candidate_fraction", "direction": "ge", "threshold": 0.1541646271944046, "tp": 1, "tn": 0, "fp": 0, "fn": 0, "balanced_accuracy": 0.7364329268292682},
                },
                "frozen_rule_validation": {"matches_frozen_rule": True, "scientific_selector_input_sha256": "sha-train"},
                "hard_gate_state_cache": train_cache,
            },
            "val_prepared": {
                "features_t": torch.ones((len(val_cache), 105)),
                "feature_rows": [{"sample_id": row["sample_id"], "candidate_fraction": 0.1541646271944046} for row in val_cache],
                "hard_gate_state_cache": val_cache,
                "state_summary": {
                    "always_closed": {"positive_success50": 1, "positive_mean_matched_iou": 0.4234538944043546},
                    "always_open": {"positive_success50": 2, "positive_mean_matched_iou": 0.5061690111512952, "negative_regressions": 20, "negative_topology_changes": 24},
                    "two_state_positive_success50_union_upper_bound": 3,
                },
            },
            "success_criteria_v2": {
                "utility": {"positive_success50_min": 2, "positive_mean_matched_iou_min": 0.47342348408224233},
                "safety": {"negative_regressions": 0, "negative_topology_changes": 0},
            },
        }

    def test_fixed_300_step_training_contract_and_train_only_checkpoint_selection(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
        with tempfile.TemporaryDirectory() as td:
            cfg = dict(cfg)
            cfg["train"] = dict(cfg["train"])
            cfg["train"]["save_dir"] = str(Path(td) / "run")
            cfg["future_training"] = dict(cfg["future_training"])
            cfg["future_training"]["max_steps"] = 6
            cfg["future_training"]["progress_epochs"] = 2
            gate = torch.nn.Linear(105, 1)
            features = torch.ones((2, 105))
            targets = torch.tensor([[1.0], [0.0]])
            out = runner._train_only_run(
                cfg=cfg,
                save_dir=Path(cfg["train"]["save_dir"]),
                device=torch.device("cpu"),
                gate_model=gate,
                features_t=features,
                targets_t=targets,
            )
            self.assertEqual(out["max_steps"], 6)
            self.assertTrue((Path(cfg["train"]["save_dir"]) / "best_train_loss.pth").exists())
            self.assertTrue((Path(cfg["train"]["save_dir"]) / "last.pth").exists())

    def test_validation_is_not_used_during_optimization_or_threshold_selection(self):
        prepared = self._fake_prepared()
        with mock.patch.object(runner, "_prepare_training_inputs_core", return_value=prepared), \
             mock.patch.object(dev, "assert_locked_val_references"), \
             mock.patch.object(dev, "assert_locked_active_success_criterion_v2"), \
             mock.patch.object(dev, "snapshot_frozen_backbone_state", return_value={"named": [], "params": {}, "bn": {}}), \
             mock.patch.object(dev, "frozen_backbone_invariant_deltas", return_value={"semantic_parameter_max_delta": 0.0, "semantic_bn_state_max_delta": 0.0, "v2_pixel_head_parameter_max_delta": 0.0}), \
             mock.patch.object(runner, "_train_only_run", return_value={"best_train_loss": 0.1, "best_train_loss_step": 1, "history": [], "optimizer_name": "AdamW", "max_steps": 300}), \
             mock.patch.object(runner, "_load_gate_checkpoint", return_value={"step": 1}), \
             mock.patch.object(runner, "_gate_probabilities", side_effect=[np.array([1.0]), np.ones((27,), dtype=np.float64)]), \
             mock.patch.object(dev, "evaluate_fixed_scalar_rule", wraps=dev.evaluate_fixed_scalar_rule) as scalar_mock:
            with tempfile.TemporaryDirectory() as td:
                cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
                cfg = dict(cfg)
                cfg["train"] = dict(cfg["train"])
                cfg["train"]["save_dir"] = str(Path(td) / "run")
                summary = runner.run_pipeline(cfg)
        self.assertFalse(summary["validation_isolation"]["used_during_optimization"])
        self.assertFalse(summary["validation_isolation"]["used_for_checkpoint_selection"])
        self.assertFalse(summary["validation_isolation"]["used_for_threshold_selection"])
        self.assertEqual(summary["training_contract"]["threshold"], 0.50)
        self.assertEqual(scalar_mock.call_args.kwargs["scalar_rule"], dev.FROZEN_SIMPLE_SCALAR_RULE)

    def test_identical_selector_input_hashes_between_preflight_and_training_paths(self):
        prepared = self._fake_prepared()
        with tempfile.TemporaryDirectory() as td:
            analysis_dir = Path(td) / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            bridge._write_json(analysis_dir / "gate_train_selector_audit.json", prepared["train_prepared"]["selector_audit"])
            with mock.patch.object(runner, "_prepare_training_inputs_core", return_value=prepared), \
                 mock.patch.object(dev, "assert_locked_val_references"), \
                 mock.patch.object(dev, "assert_locked_active_success_criterion_v2"), \
                 mock.patch.object(dev, "snapshot_frozen_backbone_state", return_value={"named": [], "params": {}, "bn": {}}), \
                 mock.patch.object(dev, "frozen_backbone_invariant_deltas", return_value={"semantic_parameter_max_delta": 0.0, "semantic_bn_state_max_delta": 0.0, "v2_pixel_head_parameter_max_delta": 0.0}), \
                 mock.patch.object(runner, "_train_only_run", return_value={"best_train_loss": 0.1, "best_train_loss_step": 1, "history": [], "optimizer_name": "AdamW", "max_steps": 300}), \
                 mock.patch.object(runner, "_load_gate_checkpoint", return_value={"step": 1}), \
                 mock.patch.object(runner, "_gate_probabilities", side_effect=[np.array([1.0]), np.ones((27,), dtype=np.float64)]):
                cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
                cfg = dict(cfg)
                cfg["train"] = dict(cfg["train"])
                cfg["train"]["save_dir"] = str(Path(td) / "run")
                cfg["analysis"] = {"feature_audit_dir": str(analysis_dir)}
                summary = runner.run_pipeline(cfg)
        self.assertTrue(summary["train_selector_input_comparison"]["identical"])

    def test_training_preparation_enforces_train_only_not_val(self):
        manifest_stage = {
            "manifest": {"contract": {"train_sample_ids": ["t1"], "val_sample_ids": ["v1"]}},
            "device": torch.device("cpu"),
        }
        fake_prepared = {
            "simple_scalar_rule": {"train_simple_gate_threshold_exists": True},
            "state_summary": {"always_closed": {}, "two_state_positive_success50_union_upper_bound": 0},
            "hard_gate_state_cache": [],
        }
        with mock.patch.object(runner.preflight_runner, "_prepare_manifest", return_value=manifest_stage), \
             mock.patch.object(runner.gate_v4, "load_frozen_v2_pixel_model_from_cfg", return_value=(mock.Mock(), {})), \
             mock.patch.object(dev, "_prepare_split_preflight_core", side_effect=[fake_prepared, fake_prepared]) as prepare_mock, \
             mock.patch.object(dev, "build_predeclared_success_criteria_v2", return_value={}):
            runner._prepare_training_inputs_core({}, enforce_frozen_scalar_rule=True)
        self.assertTrue(prepare_mock.call_args_list[0].kwargs["enforce_frozen_scalar_rule"])
        self.assertFalse(prepare_mock.call_args_list[1].kwargs["enforce_frozen_scalar_rule"])

    def test_frozen_baseline_mismatch_still_fails_closed(self):
        feature_rows = [
            {"sample_id": "a", "bridge_positive_target": 1, "candidate_fraction": 0.10},
            {"sample_id": "b", "bridge_positive_target": 0, "candidate_fraction": 0.05},
        ]
        with self.assertRaises(SystemExit):
            dev.select_train_only_scalar_rule(feature_rows)

    def test_v2_success_criterion_exact_and_no_validation_threshold_sweep(self):
        prepared = self._fake_prepared()
        trained_eval = {
            "gated_reconstruction": {
                "positive_success50": 2,
                "positive_mean_matched_iou": 0.48,
                "negative_regressions": 0,
                "negative_topology_changes": 0,
            }
        }
        decision = dev.evaluate_success_against_locked_v2_criterion(
            trained_payload=trained_eval,
            success_criteria_v2=prepared["success_criteria_v2"],
            always_closed_payload={"gated_reconstruction": {"positive_success50": 1, "positive_mean_matched_iou": 0.4234538944043546}},
        )
        self.assertTrue(decision["pass"])
        self.assertEqual(decision["status_text"], "YES")

    def test_simple_scalar_baseline_is_frozen(self):
        self.assertEqual(dev.FROZEN_SIMPLE_SCALAR_RULE["scalar"], "candidate_fraction")
        self.assertEqual(dev.FROZEN_SIMPLE_SCALAR_RULE["direction"], "ge")
        self.assertAlmostEqual(dev.FROZEN_SIMPLE_SCALAR_RULE["threshold"], 0.1541646271944046)

    def test_locked_val_references_and_success_criterion_fail_closed_on_drift(self):
        with self.assertRaises(SystemExit):
            dev.assert_locked_val_references(
                always_closed={"positive_success50": 0, "positive_mean_matched_iou": 0.1},
                always_open={"positive_success50": 0, "positive_mean_matched_iou": 0.1, "negative_regressions": 0, "negative_topology_changes": 0},
                safe_two_state_oracle={"positive_success50": 0, "positive_mean_matched_iou": 0.1, "negative_regressions": 0, "negative_topology_changes": 0, "positive_open_count": 0, "positive_closed_count": 0},
                union_upper_bound=0,
            )
        with self.assertRaises(SystemExit):
            dev.assert_locked_active_success_criterion_v2(
                {"utility": {"positive_success50_min": 3, "positive_mean_matched_iou_min": 0.4}, "safety": {"negative_regressions": 0, "negative_topology_changes": 0}}
            )

    def test_snapshot_named_parameters_are_detached_cpu_clones(self):
        layer = torch.nn.Linear(3, 2)
        snap = bridge_head._snapshot_named_parameters(list(layer.named_parameters()))
        for name, param in layer.named_parameters():
            self.assertEqual(snap[name].device.type, "cpu")
            self.assertFalse(snap[name].requires_grad)
            self.assertNotEqual(snap[name].data_ptr(), param.detach().cpu().data_ptr())

    def test_collect_batchnorm_stats_are_detached_cpu_clones(self):
        bn = torch.nn.BatchNorm2d(2)
        stats = bridge_head._collect_batchnorm_stats(bn)
        self.assertEqual(len(stats), 1)
        _name, running_mean, running_var = stats[0]
        self.assertEqual(running_mean.device.type, "cpu")
        self.assertEqual(running_var.device.type, "cpu")
        self.assertFalse(running_mean.requires_grad)
        self.assertFalse(running_var.requires_grad)

    def test_max_parameter_delta_from_cpu_snapshot_is_zero_for_unchanged_cpu_model(self):
        layer = torch.nn.Linear(3, 2)
        named = list(layer.named_parameters())
        snap = bridge_head._snapshot_named_parameters(named)
        self.assertEqual(bridge_head._max_parameter_delta_from_snapshot(named, snap), 0.0)

    def test_invariant_delta_zero_for_unchanged_model(self):
        model = mock.Mock()
        model.base = torch.nn.Sequential(torch.nn.BatchNorm2d(2))
        model.bridge_head = torch.nn.Conv2d(2, 1, kernel_size=1)
        model.named_parameters = lambda: list(model.base.named_parameters()) + [(f"bridge_head.{k}", v) for k, v in model.bridge_head.named_parameters()]
        snapshot = dev.snapshot_frozen_backbone_state(model)
        deltas = dev.frozen_backbone_invariant_deltas(model, snapshot)
        self.assertEqual(deltas["semantic_parameter_max_delta"], 0.0)
        self.assertEqual(deltas["semantic_bn_state_max_delta"], 0.0)
        self.assertEqual(deltas["v2_pixel_head_parameter_max_delta"], 0.0)

    def test_invariant_delta_detects_parameter_mutation(self):
        model = mock.Mock()
        model.base = torch.nn.Sequential(torch.nn.BatchNorm2d(2))
        model.bridge_head = torch.nn.Conv2d(2, 1, kernel_size=1)
        model.named_parameters = lambda: list(model.base.named_parameters()) + [(f"bridge_head.{k}", v) for k, v in model.bridge_head.named_parameters()]
        snapshot = dev.snapshot_frozen_backbone_state(model)
        with torch.no_grad():
            model.base[0].weight.add_(1.0)
        deltas = dev.frozen_backbone_invariant_deltas(model, snapshot)
        self.assertGreater(deltas["semantic_parameter_max_delta"], 0.0)

    def test_invariant_delta_detects_bn_buffer_mutation(self):
        model = mock.Mock()
        model.base = torch.nn.Sequential(torch.nn.BatchNorm2d(2))
        model.bridge_head = torch.nn.Conv2d(2, 1, kernel_size=1)
        model.named_parameters = lambda: list(model.base.named_parameters()) + [(f"bridge_head.{k}", v) for k, v in model.bridge_head.named_parameters()]
        snapshot = dev.snapshot_frozen_backbone_state(model)
        with torch.no_grad():
            model.base[0].running_mean.add_(1.0)
        deltas = dev.frozen_backbone_invariant_deltas(model, snapshot)
        self.assertGreater(deltas["semantic_bn_state_max_delta"], 0.0)

    def test_validation_is_not_invoked_before_invariant_verification(self):
        prepared = self._fake_prepared()
        with mock.patch.object(runner, "_prepare_training_inputs_core", return_value=prepared), \
             mock.patch.object(dev, "assert_locked_val_references"), \
             mock.patch.object(dev, "assert_locked_active_success_criterion_v2"), \
             mock.patch.object(dev, "snapshot_frozen_backbone_state", return_value={"named": [], "params": {}, "bn": {}}), \
             mock.patch.object(dev, "frozen_backbone_invariant_deltas", side_effect=SystemExit("invariant failure")), \
             mock.patch.object(runner, "_train_only_run", return_value={"best_train_loss": 0.1, "best_train_loss_step": 1, "history": [], "optimizer_name": "AdamW", "max_steps": 300}), \
             mock.patch.object(runner, "_load_gate_checkpoint", return_value={"step": 300}), \
             mock.patch.object(runner, "_evaluate_split") as eval_mock:
            with tempfile.TemporaryDirectory() as td:
                cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
                cfg = dict(cfg)
                cfg["train"] = dict(cfg["train"])
                cfg["train"]["save_dir"] = str(Path(td) / "run")
                with self.assertRaises(SystemExit):
                    runner.run_pipeline(cfg)
        eval_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
