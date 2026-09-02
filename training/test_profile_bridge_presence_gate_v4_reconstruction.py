from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import profile_bridge_presence_gate_v4_reconstruction as mod


class TestProfileBridgePresenceGateV4Reconstruction(unittest.TestCase):
    def test_build_profile_writes_summary_and_projects_runtime(self):
        fake_profile = {
            "rows": [
                {
                    "reference_centroid_distance_computation_seconds": 8.0,
                    "reference_call_counts": {"group_pair_evaluations": 12},
                    "reference_total_reconstruction_seconds": 10.0,
                    "optimized_total_reconstruction_seconds": 2.0,
                },
                {
                    "reference_centroid_distance_computation_seconds": 4.0,
                    "reference_call_counts": {"group_pair_evaluations": 6},
                    "reference_total_reconstruction_seconds": 5.0,
                    "optimized_total_reconstruction_seconds": 1.0,
                },
            ],
            "reference_total_reconstruction_seconds": 15.0,
            "optimized_total_reconstruction_seconds": 3.0,
            "reference_mean_per_state_seconds": 7.5,
            "optimized_mean_per_state_seconds": 1.5,
            "speedup_factor": 5.0,
            "slowest_reference_state": {"sample_id": "a", "state": "OPEN"},
            "slowest_optimized_state": {"sample_id": "b", "state": "CLOSED"},
        }
        fake_prepared = {"cached_micro": ["x"], "pixel_remove_masks": ["y"]}
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            with mock.patch.object(mod.runner, "_prepare_v4_inputs", return_value=fake_prepared), \
                 mock.patch.object(mod.gate_v4, "profile_hard_gate_reconstruction_states", return_value=fake_profile), \
                 mock.patch.object(mod, "runtime_threading_snapshot", return_value={"cpu_affinity": [0, 1]}):
                summary = mod.build_profile({"seed": 1}, out_dir)
            self.assertEqual(summary["hotspot"]["reference_group_pair_evaluations"], 18)
            self.assertAlmostEqual(summary["hotspot"]["reference_centroid_distance_fraction"], 12.0 / 15.0)
            self.assertAlmostEqual(summary["projected_157_sample_two_state_cache_seconds"], 1.5 * 314.0)
            saved = json.loads((out_dir / "reconstruction_profile_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["threading_snapshot"]["cpu_affinity"], [0, 1])


if __name__ == "__main__":
    unittest.main()
