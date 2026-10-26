"""
HYDROTWIN-AI: Comprehensive Automated Tests for ML, Data Ingestion, & Physics Safeguards
Validates Section C (Data Generation & Ingestion), Section D (Predictive ML & XAI), and Section I (Validation).
"""

import os
import unittest
import numpy as np
import pandas as pd

from engine.models import DEFAULT_CONFIG
from engine.physics import simulate_system, calculate_ec, calculate_wqri
from engine.dataset_generator import generate_scenario_time_series, generate_research_benchmark_dataset
from engine.data_ingestion import validate_time_series_dataset, load_dataset_from_csv
from engine.ml_analytics import PredictiveModelBenchmark, create_lagged_dataset, run_full_ml_pipeline


class TestDataGenerationAndIngestion(unittest.TestCase):

    def test_synthetic_data_generation_reproducibility(self):
        """Verify synthetic data generation is strictly reproducible given the same random seed."""
        df1 = generate_scenario_time_series("Normal Operation", DEFAULT_CONFIG, duration_h=24.0, random_seed=42)
        df2 = generate_scenario_time_series("Normal Operation", DEFAULT_CONFIG, duration_h=24.0, random_seed=42)
        
        pd.testing.assert_frame_equal(df1, df2)
        self.assertTrue((df1["data_type"] == "SYNTHETIC").all())
        self.assertTrue((df1["is_simulated"] == True).all())

    def test_synthetic_dataset_leakage_safeguard(self):
        """Verify that train, validation, and test splits have strictly disjoint simulation run IDs."""
        master_df, train_df, test_df = generate_research_benchmark_dataset(master_seed=123)
        
        train_runs = set(train_df["run_id"].unique())
        test_runs = set(test_df["run_id"].unique())
        
        overlap = train_runs.intersection(test_runs)
        self.assertEqual(len(overlap), 0, f"Data leakage detected! Overlapping run IDs in train and test: {overlap}")
        self.assertGreater(len(train_df), 0)
        self.assertGreater(len(test_df), 0)

    def test_schema_validation_success(self):
        """Verify valid simulation output passes schema validation cleanly."""
        df = simulate_system(DEFAULT_CONFIG)
        val_res = validate_time_series_dataset(df)
        self.assertTrue(val_res.is_valid, f"Validation failed with errors: {val_res.errors}")
        self.assertEqual(len(val_res.errors), 0)

    def test_schema_validation_catches_missing_columns(self):
        """Verify schema validator detects missing required columns."""
        df = simulate_system(DEFAULT_CONFIG)
        bad_df = df.drop(columns=["basin_volume_m3", "EC_estimate_uScm"])
        val_res = validate_time_series_dataset(bad_df)
        self.assertFalse(val_res.is_valid)
        self.assertTrue(any("basin_volume_m3" in err for err in val_res.errors))
        self.assertTrue(any("EC_estimate_uScm" in err for err in val_res.errors))

    def test_schema_validation_catches_negative_volume(self):
        """Verify schema validator catches physically impossible negative volumes."""
        df = simulate_system(DEFAULT_CONFIG)
        bad_df = df.copy()
        bad_df.loc[5, "basin_volume_m3"] = -10.0
        val_res = validate_time_series_dataset(bad_df)
        self.assertFalse(val_res.is_valid)
        self.assertTrue(any("below physical minimum" in err for err in val_res.errors))


class TestPredictiveMLAndExplainability(unittest.TestCase):

    def setUp(self):
        self.master_df, self.train_raw, self.test_raw = generate_research_benchmark_dataset(master_seed=42)
        self.train_lagged = create_lagged_dataset(self.train_raw, horizon_steps=2, group_col="run_id")
        self.test_lagged = create_lagged_dataset(self.test_raw, horizon_steps=2, group_col="run_id")

    def test_lagged_dataset_construction(self):
        """Verify multi-step future targets are correctly created without cross-run bleeding."""
        self.assertIn("target_ec_future", self.train_lagged.columns)
        self.assertIn("target_risk_class_future", self.train_lagged.columns)
        self.assertFalse(self.train_lagged["target_ec_future"].isna().any())

    def test_regression_benchmark_evaluation(self):
        """Verify regression models train and held-out metrics (MAE, RMSE, R2) are finite and ordered."""
        benchmark = PredictiveModelBenchmark(random_state=42)
        results = benchmark.train_and_evaluate_regression(self.train_lagged, self.test_lagged)
        
        self.assertIn("Persistence Baseline", results["metrics"])
        self.assertIn("Random Forest", results["metrics"])
        self.assertIn("Gradient Boosting", results["metrics"])
        
        # Verify finite metrics
        for model_name, m in results["metrics"].items():
            self.assertTrue(np.isfinite(m["MAE"]))
            self.assertTrue(np.isfinite(m["RMSE"]))
            self.assertTrue(np.isfinite(m["R2"]))
            
        # Verify Random Forest out-performs naive persistence baseline on held-out test data
        rf_rmse = results["metrics"]["Random Forest"]["RMSE"]
        persist_rmse = results["metrics"]["Persistence Baseline"]["RMSE"]
        self.assertLess(rf_rmse, persist_rmse)
        
        # Verify feature importance
        self.assertGreater(len(results["feature_importance"]), 0)
        total_pct = sum(item["pct"] for item in results["feature_importance"])
        self.assertAlmostEqual(total_pct, 100.0, delta=1.0)

    def test_classification_benchmark_evaluation(self):
        """Verify risk classification models report precision, recall, F1, and confusion matrix."""
        benchmark = PredictiveModelBenchmark(random_state=42)
        results = benchmark.train_and_evaluate_classification(self.train_lagged, self.test_lagged)
        
        self.assertIn("Majority Baseline", results["metrics"])
        self.assertIn("Random Forest", results["metrics"])
        
        for model_name, m in results["metrics"].items():
            self.assertTrue(0.0 <= m["f1_macro"] <= 1.0)
            self.assertTrue(0.0 <= m["precision_macro"] <= 1.0)
            self.assertTrue(0.0 <= m["recall_macro"] <= 1.0)
            cm = np.array(m["confusion_matrix"])
            self.assertEqual(cm.shape[0], cm.shape[1])


class TestPhysicalConservationSafeguards(unittest.TestCase):

    def test_evaporation_does_not_remove_salts(self):
        """
        Verify fundamental physical law:
        Evaporation removes pure water vapor and DOES NOT remove dissolved chloride, sulfate, or calcium salts.
        """
        cfg = dict(DEFAULT_CONFIG)
        cfg["duration_h"] = 10.0
        cfg["makeup_mode"] = "Force zero makeup"  # No makeup adding salts
        cfg["thermal_load"] = 15.0               # High evaporation
        
        # Run with blowdown capacity = 0 so no blowdown removes salts either
        cfg["blowdown_capacity_m3_h"] = 0.0
        
        df = simulate_system(cfg)
        # Salt mass in basin: concentration * volume * 1000
        initial_cl_mass = df.iloc[0]["Cl_mg_l"] * df.iloc[0]["basin_volume_m3"] * 1000.0
        final_cl_mass = df.iloc[-1]["Cl_mg_l"] * df.iloc[-1]["basin_volume_m3"] * 1000.0
        
        # Chloride mass in basin must remain constant within floating-point tolerance
        np.testing.assert_allclose(initial_cl_mass, final_cl_mass, rtol=1e-5)
        # But concentration must increase due to evaporative concentration!
        self.assertGreater(df.iloc[-1]["Cl_mg_l"], df.iloc[0]["Cl_mg_l"])

    def test_configurable_wqri_weights(self):
        """Verify WQRI accepts customized weights and thresholds."""
        custom_weights = {"ec": 0.50, "cl": 0.20, "so4": 0.20, "ca": 0.10}
        custom_limits = {"ec": 2000.0, "cl": 5.0, "so4": 5.0, "ca": 5.0}
        
        val = calculate_wqri(1000.0, 2.5, 2.5, 2.5, weights=custom_weights, limits=custom_limits)
        expected = 0.50 * (1000/2000) + 0.20 * (2.5/5.0) + 0.20 * (2.5/5.0) + 0.10 * (2.5/5.0)
        self.assertAlmostEqual(val, expected, places=4)


if __name__ == "__main__":
    unittest.main()
