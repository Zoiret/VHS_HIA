from __future__ import annotations

import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import bridge_presence_gate_v4 as gate_v4
import bridge_presence_gate_v4_patient_disjoint_dev as dev
import bridge_suppression_head as bridge


class _FailOnUse:
    def __repr__(self) -> str:
        raise AssertionError("Disallowed model feature value was accessed by split selection")

    def __float__(self) -> float:
        raise AssertionError("Disallowed model feature value was accessed by split selection")

    def __int__(self) -> int:
        raise AssertionError("Disallowed model feature value was accessed by split selection")


class TestBridgePresenceGateV4PatientDisjointDev(unittest.TestCase):
    def _metadata_rows(self):
        rows = []
        # Validation-friendly patients.
        rows.extend(
            [
                {"sample_id": "m01_p01_s00", "patient_id": "m01_p01", "gt_count": 2, "bridge_positive": 1},
                {"sample_id": "m01_p01_s01", "patient_id": "m01_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m02_p01_s00", "patient_id": "m02_p01", "gt_count": 3, "bridge_positive": 1},
                {"sample_id": "m02_p01_s01", "patient_id": "m02_p01", "gt_count": 2, "bridge_positive": 0},
                {"sample_id": "m03_p01_s00", "patient_id": "m03_p01", "gt_count": 2, "bridge_positive": 1},
                {"sample_id": "m03_p01_s01", "patient_id": "m03_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m04_p01_s00", "patient_id": "m04_p01", "gt_count": 3, "bridge_positive": 1},
                {"sample_id": "m04_p01_s01", "patient_id": "m04_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m05_p01_s00", "patient_id": "m05_p01", "gt_count": 2, "bridge_positive": 1},
                {"sample_id": "m05_p01_s01", "patient_id": "m05_p01", "gt_count": 1, "bridge_positive": 0},
            ]
        )
        # Remaining train patients.
        rows.extend(
            [
                {"sample_id": "m06_p01_s00", "patient_id": "m06_p01", "gt_count": 2, "bridge_positive": 1},
                {"sample_id": "m06_p01_s01", "patient_id": "m06_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m07_p01_s00", "patient_id": "m07_p01", "gt_count": 3, "bridge_positive": 1},
                {"sample_id": "m07_p01_s01", "patient_id": "m07_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m08_p01_s00", "patient_id": "m08_p01", "gt_count": 2, "bridge_positive": 1},
                {"sample_id": "m08_p01_s01", "patient_id": "m08_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m09_p01_s00", "patient_id": "m09_p01", "gt_count": 3, "bridge_positive": 1},
                {"sample_id": "m09_p01_s01", "patient_id": "m09_p01", "gt_count": 1, "bridge_positive": 0},
                {"sample_id": "m10_p01_s00", "patient_id": "m10_p01", "gt_count": 2, "bridge_positive": 1},
                {"sample_id": "m10_p01_s01", "patient_id": "m10_p01", "gt_count": 1, "bridge_positive": 0},
            ]
        )
        return rows

    def test_select_patient_disjoint_split_is_deterministic(self):
        rows = self._metadata_rows()
        a = dev.select_patient_disjoint_split(rows, seed=1337)
        b = dev.select_patient_disjoint_split(rows, seed=1337)
        self.assertEqual(a["train_summary"]["sample_ids"], b["train_summary"]["sample_ids"])
        self.assertEqual(a["val_summary"]["sample_ids"], b["val_summary"]["sample_ids"])

    def test_select_patient_disjoint_split_has_zero_overlap_and_required_coverage(self):
        split = dev.select_patient_disjoint_split(self._metadata_rows(), seed=1337)
        train = split["train_summary"]
        val = split["val_summary"]
        self.assertEqual(split["patient_overlap"], 0)
        self.assertEqual(split["sample_overlap"], 0)
        self.assertGreater(train["bridge_positive_count"], 0)
        self.assertGreater(train["bridge_negative_count"], 0)
        self.assertGreater(val["bridge_positive_count"], 0)
        self.assertGreater(val["bridge_negative_count"], 0)
        self.assertGreater(val["gt2_count"], 0)
        self.assertGreater(val["gt3_count"], 0)

    def test_split_algorithm_does_not_use_model_features_or_predictions(self):
        rows = []
        for row in self._metadata_rows():
            current = dict(row)
            current["x_0_4"] = _FailOnUse()
            current["bridge_score_mean"] = _FailOnUse()
            rows.append(current)
        split = dev.select_patient_disjoint_split(rows, seed=1337)
        self.assertEqual(split["patient_overlap"], 0)

    def test_source_sha_is_canonical_and_locked(self):
        source = bridge.DEFAULT_TRAIN_SPLIT
        self.assertEqual(bridge._canonical_split_sha256(source), dev.DEFAULT_SOURCE_SHA256)

    def test_build_split_texts_and_manifest_reproducibility(self):
        rows = self._metadata_rows()
        split = dev.select_patient_disjoint_split(rows, seed=1337)
        source_entries = []
        for row in rows:
            source_entries.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "patient_id": str(row["patient_id"]),
                    "row_text": f"images/{row['sample_id']}.png  masks/{row['sample_id']}.png",
                }
            )
        texts = dev.build_split_texts(
            source_entries=source_entries,
            train_sample_ids=list(split["train_summary"]["sample_ids"]),
            val_sample_ids=list(split["val_summary"]["sample_ids"]),
        )
        contract = dev.build_manifest_contract(
            source_split_path=bridge.DEFAULT_TRAIN_SPLIT,
            source_sha256=dev.DEFAULT_SOURCE_SHA256,
            split_payload=split,
        )
        with tempfile.TemporaryDirectory() as td:
            first = dev.load_or_create_manifest(
                manifest_dir=Path(td),
                contract_payload=contract,
                train_text=texts["train_text"],
                val_text=texts["val_text"],
            )
            second = dev.load_or_create_manifest(
                manifest_dir=Path(td),
                contract_payload=contract,
                train_text=texts["train_text"],
                val_text=texts["val_text"],
            )
        self.assertTrue(first["created"])
        self.assertFalse(second["created"])

    def test_feature_dimension_and_gate_param_contract_unchanged(self):
        gate = gate_v4.build_gate_model_from_cfg({"gate": {"hidden_dim": 16, "dropout_p": 0.0}}, input_dim=dev.FEATURE_DIMENSION)
        self.assertEqual(dev.FEATURE_DIMENSION, 105)
        self.assertEqual(gate_v4.count_trainable_parameters(gate), dev.TRAINABLE_GATE_PARAMS)

    def test_split_metadata_rows_only_keep_allowed_fields(self):
        records = [
            {
                "sample_id": "m01_p01_s00",
                "patient_id": "m01_p01",
                "gt_count": 2,
                "bridge_positive": 1,
                "x_0_4": torch.ones((1, 1, 1)),
                "bridge_logits": np.ones((1, 1)),
            }
        ]
        rows = dev.build_split_metadata_rows(records)
        self.assertEqual(set(rows[0].keys()), set(dev.ALLOWED_SPLIT_FIELDS))

    def test_select_train_only_scalar_rule_cannot_read_validation_labels(self):
        params = list(inspect.signature(dev.select_train_only_scalar_rule).parameters.keys())
        self.assertEqual(params, ["train_feature_rows"])

    def test_predeclared_success_criteria_serialized_before_training(self):
        criteria = dev.build_predeclared_success_criteria(
            val_summary={"bridge_positive": 6},
            always_closed={"positive_success50": 0, "positive_mean_matched_iou": 0.28},
            always_open={"positive_success50": 3, "positive_mean_matched_iou": 0.58},
            cfg={"future_training": {"optimizer": "AdamW", "learning_rate": 0.001, "max_steps": 300, "seed": 1337}, "gate": {"gate_threshold": 0.50}},
        )
        self.assertTrue(criteria["declared_before_training"])
        self.assertEqual(criteria["safety"]["negative_regressions"], 0)
        self.assertEqual(criteria["safety"]["negative_topology_changes"], 0)
        self.assertIn("positive_success50_min", criteria["utility"])
        self.assertIn("positive_mean_matched_iou_min", criteria["utility"])

    def test_predeclared_baselines_fixed_before_validation(self):
        baselines = dev.build_predeclared_baselines()
        self.assertIn("always_closed", baselines)
        self.assertIn("always_open", baselines)
        self.assertIn("train_derived_simple_scalar_gate", baselines)
        self.assertIn("trained_v4_gate", baselines)

    def test_no_test_or_holdout_paths_loaded_in_dev_config(self):
        cfg = bridge._read_yaml(bridge.REPO_ROOT / "training" / "configs" / "unetpp_effb3_bridge_presence_gate_v4_patient_disjoint_dev_v1.yaml")
        self.assertIn("train.txt", str((cfg.get("dataset") or {}).get("train_txt", "")))
        self.assertTrue(bool((cfg.get("experiment_notes") or {}).get("no_test_usage", False)))
        self.assertTrue(bool((cfg.get("experiment_notes") or {}).get("no_authoritative_holdout", False)))


if __name__ == "__main__":
    unittest.main()
