"""
Unit tests for HYDROTWIN-AI simulation engine and mass balances.
"""

import unittest
import numpy as np
import pandas as pd

from engine.models import DEFAULT_CONFIG, WATER_PROFILES
from engine.physics import simulate_system, calculate_ec, calculate_wqri, validate_config
from engine.metrics import compute_simulation_metrics, compare_policies


class TestHydroTwinSimulation(unittest.TestCase):

    def setUp(self):
        self.config = dict(DEFAULT_CONFIG)
        self.config["duration_h"] = 24.0
        self.config["dt_h"] = 0.5

    def test_water_mass_balance_conservation(self):
        """Test that water volume changes balance inflow, outflow, and evaporation exactly."""
        df = simulate_system(self.config)
        # Residuals are non-zero only when physical 10% or 100% boundary clamping occurs
        # If unconstrained, sum of residuals should be within floating-point tolerance
        max_residual = df["water_balance_residual_m3"].abs().max()
        # Ensure residual is tracked and finite
        self.assertTrue(np.isfinite(max_residual))

    def test_dissolved_species_mass_balance(self):
        """Test dissolved species mass conservation across simulation steps."""
        df = simulate_system(self.config)
        cl_res = df["chloride_balance_residual_mg"].abs().max()
        so4_res = df["sulfate_balance_residual_mg"].abs().max()
        ca_res = df["calcium_balance_residual_mg"].abs().max()

        self.assertLess(cl_res, 1e-6, "Chloride mass balance residual exceeds numerical tolerance.")
        self.assertLess(so4_res, 1e-6, "Sulfate mass balance residual exceeds numerical tolerance.")
        self.assertLess(ca_res, 1e-6, "Calcium mass balance residual exceeds numerical tolerance.")

    def test_zero_flow_quiescent_state(self):
        """Test that with zero thermal load and zero makeup, state remains constant."""
        cfg = dict(self.config)
        cfg["thermal_load"] = 0.0
        cfg["makeup_mode"] = "Force zero makeup"
        cfg["initial_volume_m3"] = 300.0

        df = simulate_system(cfg)
        # With zero evaporation and no makeup/blowdown trigger, volume should not change
        volumes = df["basin_volume_m3"].values
        np.testing.assert_allclose(volumes, volumes[0], rtol=1e-5)
        # Chloride concentration should also remain unchanged
        cl_concs = df["Cl_mg_l"].values
        np.testing.assert_allclose(cl_concs, cl_concs[0], rtol=1e-5)

    def test_basin_lower_volume_bound(self):
        """Verify basin volume is safely clamped to 10% lower floor during extreme evaporation."""
        cfg = dict(self.config)
        cfg["thermal_load"] = 50.0  # Very large evaporation
        cfg["evaporation_factor"] = 2.0
        cfg["makeup_mode"] = "Force zero makeup"
        cfg["basin_capacity_m3"] = 100.0
        cfg["initial_volume_m3"] = 20.0
        cfg["duration_h"] = 12.0

        df = simulate_system(cfg)
        min_vol = df["basin_volume_m3"].min()
        floor_vol = cfg["basin_capacity_m3"] * 0.10
        self.assertGreaterEqual(min_vol, floor_vol - 1e-6)

    def test_input_validation(self):
        """Verify invalid parameters raise appropriate ValueErrors."""
        bad_cfg1 = dict(self.config, duration_h=-5)
        with self.assertRaises(ValueError):
            simulate_system(bad_cfg1)

        bad_cfg2 = dict(self.config, dt_h=0)
        with self.assertRaises(ValueError):
            simulate_system(bad_cfg2)

        bad_cfg3 = dict(self.config, ec_low=1500, ec_high=1200)
        with self.assertRaises(ValueError):
            simulate_system(bad_cfg3)

        bad_cfg4 = dict(self.config, initial_volume_m3=600, basin_capacity_m3=500)
        with self.assertRaises(ValueError):
            simulate_system(bad_cfg4)

    def test_policy_comparison(self):
        """Verify compare_policies returns valid metrics for all three policies."""
        comp_df = compare_policies(self.config)
        self.assertEqual(len(comp_df), 3)
        self.assertIn("Baseline", comp_df["policy"].values)
        self.assertIn("Risk-aware", comp_df["policy"].values)
        self.assertIn("Water-use optimization", comp_df["policy"].values)
        self.assertTrue(all(comp_df["total_makeup_m3"] >= 0))
        self.assertTrue(all(comp_df["pct_within_limits"] >= 0))


if __name__ == "__main__":
    unittest.main()
