"""
Unit tests for HYDROTWIN-AI visualization and scenario modules.
"""

import unittest
import json
import pandas as pd

from engine.models import DEFAULT_CONFIG
from engine.physics import simulate_system
from engine.scenarios import SCENARIOS, get_scenario_config
from engine.sensitivity import run_parameter_sweep
from visualization.twin_3d import generate_3d_twin_html
from visualization.twin_2d import generate_2d_pnd_html


class TestVisualizationAndScenarios(unittest.TestCase):

    def setUp(self):
        self.config = dict(DEFAULT_CONFIG)
        self.config["duration_h"] = 12.0
        self.config["dt_h"] = 1.0
        self.df = simulate_system(self.config)

    def test_3d_twin_html_generation(self):
        """Verify 3D digital twin HTML contains essential Three.js and telemetry elements."""
        html = generate_3d_twin_html(self.df, self.config, initial_step=0)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("THREE.WebGLRenderer", html)
        self.assertIn("OrbitControls", html)
        self.assertIn("SIM_DATA", html)
        self.assertIn("waterMesh", html)
        self.assertIn("makeupValve", html)
        self.assertIn("blowdownValve", html)
        self.assertIn("vaporPoints", html)
        self.assertIn("hud-time", html)

    def test_2d_pnd_html_generation(self):
        """Verify 2D P&ID HTML contains Canvas and engineering schematic elements."""
        html = generate_2d_pnd_html(self.df, self.config, initial_step=0)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("pid-canvas", html)
        self.assertIn("SIM_DATA", html)
        self.assertIn("RECIRCULATING BASIN", html)
        self.assertIn("MAKEUP TANK", html)
        self.assertIn("FCV-101", html)
        self.assertIn("BDV-201", html)

    def test_all_scenarios_simulate_successfully(self):
        """Verify all standardized operational scenarios run and produce valid output."""
        for name in SCENARIOS.keys():
            sc_cfg = get_scenario_config(name, self.config)
            df = simulate_system(sc_cfg)
            self.assertGreater(len(df), 0)
            self.assertTrue(df["basin_volume_m3"].notna().all())
            self.assertTrue(df["EC_estimate_uScm"].notna().all())
            self.assertTrue(df["WQRI_provisional"].notna().all())

    def test_sensitivity_parameter_sweep(self):
        """Verify sensitivity parameter sweep produces non-empty DataFrame with swept values."""
        sweep_values = [5.0, 10.0, 15.0]
        sweep_df = run_parameter_sweep(self.config, "thermal_load", sweep_values)
        self.assertEqual(len(sweep_df), 3)
        self.assertIn("thermal_load", sweep_df.columns)
        self.assertIn("total_makeup_m3", sweep_df.columns)
        self.assertIn("avg_ec_uScm", sweep_df.columns)


if __name__ == "__main__":
    unittest.main()
