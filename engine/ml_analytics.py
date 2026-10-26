"""
HYDROTWIN-AI: Predictive Machine Learning & Explainable AI (XAI)
Provides supervised multi-step time-series forecasting for all approved water quality parameters
(EC, pH, TDS, Hardness), risk classification, model benchmarking, and feature attribution.

DISCLAIMER:
All models trained on synthetic datasets measure mathematical agreement with the
recirculating mass-balance simulator. They DO NOT certify accuracy against physical equipment
until fine-tuned or validated with calibrated physical plant sensor records.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


FEATURE_COLUMNS = [
    "basin_volume_m3",
    "EC_estimate_uScm",
    "pH",
    "TDS_mg_l",
    "hardness_mg_l",
    "temperature_C",
    "WQRI_provisional",
    "Cl_mg_l",
    "SO4_mg_l",
    "Ca_mg_l",
    "makeup_m3_h",
    "blowdown_m3_h",
    "evaporation_m3_h",
    "thermal_load_MW",
    "cycles_of_concentration",
]

FEATURE_LABELS = {
    "basin_volume_m3": "Basin Water Inventory (m³)",
    "EC_estimate_uScm": "Electrical Conductivity (µS/cm)",
    "pH": "Water pH (acidity / alkalinity)",
    "TDS_mg_l": "Total Dissolved Solids (mg/L)",
    "hardness_mg_l": "Total Hardness as CaCO3 (mg/L)",
    "temperature_C": "Basin Water Temperature (°C)",
    "WQRI_provisional": "Water Quality Risk Index [0-1]",
    "Cl_mg_l": "Chloride Concentration (mg/L)",
    "SO4_mg_l": "Sulfate Concentration (mg/L)",
    "Ca_mg_l": "Calcium Concentration (mg/L)",
    "makeup_m3_h": "Makeup Water Inflow (m³/h)",
    "blowdown_m3_h": "Blowdown Bleed Outflow (m³/h)",
    "evaporation_m3_h": "Evaporation Rate (m³/h)",
    "thermal_load_MW": "IT Thermal Load (MW)",
    "cycles_of_concentration": "Cycles of Concentration (CoC)",
}

TARGET_MAP = {
    "target_ec_future": "EC_estimate_uScm",
    "target_ph_future": "pH",
    "target_tds_future": "TDS_mg_l",
    "target_hardness_future": "hardness_mg_l",
    "target_wqri_future": "WQRI_provisional",
}


def create_lagged_dataset(
    df: pd.DataFrame,
    horizon_steps: int = 4,
    group_col: str = "run_id",
) -> pd.DataFrame:
    """
    Constructs multi-step forward prediction targets grouped by run_id
    to prevent cross-run target contamination.
    
    Target variables created:
    - target_ec_future: EC at t + horizon_steps
    - target_ph_future: pH at t + horizon_steps
    - target_tds_future: TDS at t + horizon_steps
    - target_hardness_future: Hardness at t + horizon_steps
    - target_wqri_future: WQRI at t + horizon_steps
    - target_risk_class_future: Risk category (Normal / Elevated / Critical) at t + horizon_steps
    """
    dfs = []
    groups = df.groupby(group_col) if group_col in df.columns else [(None, df)]
    
    for _, group in groups:
        g = group.copy().sort_values("time_h").reset_index(drop=True)
        
        # Multi-parameter forecasting targets
        g["target_ec_future"] = g["EC_estimate_uScm"].shift(-horizon_steps)
        if "pH" in g.columns:
            g["target_ph_future"] = g["pH"].shift(-horizon_steps)
        else:
            g["target_ph_future"] = 7.6
            
        if "TDS_mg_l" in g.columns:
            g["target_tds_future"] = g["TDS_mg_l"].shift(-horizon_steps)
        else:
            g["target_tds_future"] = g["target_ec_future"] * 0.65
            
        if "hardness_mg_l" in g.columns:
            g["target_hardness_future"] = g["hardness_mg_l"].shift(-horizon_steps)
        else:
            g["target_hardness_future"] = 100.0
            
        g["target_wqri_future"] = g["WQRI_provisional"].shift(-horizon_steps)
        
        # Risk classification target
        def classify_risk(val: float) -> str:
            if pd.isna(val):
                return np.nan
            if val >= 0.60:
                return "Critical"
            elif val >= 0.40:
                return "Elevated"
            else:
                return "Normal"
                
        g["target_risk_class_future"] = g["target_wqri_future"].apply(classify_risk)
        # Drop rows where future shift exceeds run horizon
        g = g.dropna(subset=["target_ec_future", "target_wqri_future"]).reset_index(drop=True)
        dfs.append(g)
        
    return pd.concat(dfs, ignore_index=True)


class PredictiveModelBenchmark:
    """
    Trains, benchmarks, and explains regression and classification models
    on held-out simulation data partitioned strictly by run_id.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.regression_results: Dict[str, Any] = {}
        self.classification_results: Dict[str, Any] = {}
        self.multi_parameter_results: Dict[str, Any] = {}
        self.best_regressor = None
        self.best_classifier = None
        self.trained_features: List[str] = FEATURE_COLUMNS
        
    def train_and_evaluate_regression(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "target_ec_future",
    ) -> Dict[str, Any]:
        """
        Trains and compares 4 regression models on held-out test data:
        1. Persistence Baseline (current value predicts future value)
        2. Ridge Regression (linear baseline)
        3. Random Forest Regressor
        4. Gradient Boosting Regressor
        """
        avail_features = [col for col in self.trained_features if col in train_df.columns and col in test_df.columns]
        
        X_train = train_df[avail_features].values
        y_train = train_df[target_col].values
        X_test = test_df[avail_features].values
        y_test = test_df[target_col].values
        
        # 1. Persistence Baseline
        current_col = TARGET_MAP.get(target_col, "EC_estimate_uScm")
        ec_idx = avail_features.index(current_col) if current_col in avail_features else 0
        y_pred_persist = X_test[:, ec_idx]
        
        models = {
            "Persistence Baseline": None,
            "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=self.random_state))]),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=self.random_state),
        }
        
        results = {}
        fitted_models = {}
        
        for name, model in models.items():
            if name == "Persistence Baseline":
                y_pred = y_pred_persist
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                fitted_models[name] = model
                
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(root_mean_squared_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))
            
            results[name] = {
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "predictions": y_pred[:100].tolist(),  # First 100 for plotting
            }
            
        # Select best model based on held-out RMSE
        comp_models = [m for m in results.keys() if m != "Persistence Baseline"]
        best_name = min(comp_models, key=lambda k: results[k]["RMSE"])
        self.best_regressor = fitted_models[best_name]
        
        # Feature importances for best tree model
        feature_importance = []
        if hasattr(self.best_regressor, "feature_importances_"):
            importances = self.best_regressor.feature_importances_
            for feat, imp in sorted(zip(avail_features, importances), key=lambda x: x[1], reverse=True):
                feature_importance.append({
                    "feature": feat,
                    "label": FEATURE_LABELS.get(feat, feat),
                    "importance": float(imp),
                    "pct": float(imp * 100.0),
                })
        elif hasattr(self.best_regressor, "named_steps") and hasattr(self.best_regressor.named_steps["ridge"], "coef_"):
            coefs = np.abs(self.best_regressor.named_steps["ridge"].coef_)
            total = coefs.sum() if coefs.sum() > 0 else 1.0
            for feat, imp in sorted(zip(avail_features, coefs / total), key=lambda x: x[1], reverse=True):
                feature_importance.append({
                    "feature": feat,
                    "label": FEATURE_LABELS.get(feat, feat),
                    "importance": float(imp),
                    "pct": float(imp * 100.0),
                })
                
        out = {
            "metrics": results,
            "best_model": best_name,
            "target": target_col,
            "test_actuals": y_test[:100].tolist(),
            "feature_importance": feature_importance,
            "feature_list": avail_features,
            "test_samples": len(y_test),
        }
        if target_col == "target_ec_future":
            self.regression_results = out
        return out

    def train_multi_parameter_forecast(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Trains multi-step forecasts for all approved water-quality parameters:
        EC, pH, TDS, and Hardness.
        """
        targets = {
            "Conductivity (EC)": "target_ec_future",
            "Water pH": "target_ph_future",
            "Total Dissolved Solids (TDS)": "target_tds_future",
            "Total Hardness (CaCO3)": "target_hardness_future",
        }
        summary = {}
        for label, target_col in targets.items():
            if target_col in train_df.columns and target_col in test_df.columns:
                res = self.train_and_evaluate_regression(train_df, test_df, target_col=target_col)
                best_m = res["best_model"]
                summary[label] = {
                    "target_column": target_col,
                    "best_model": best_m,
                    "MAE": res["metrics"][best_m]["MAE"],
                    "RMSE": res["metrics"][best_m]["RMSE"],
                    "R2": res["metrics"][best_m]["R2"],
                    "persistence_RMSE": res["metrics"]["Persistence Baseline"]["RMSE"],
                }
        self.multi_parameter_results = summary
        return summary

    def train_and_evaluate_classification(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str = "target_risk_class_future",
    ) -> Dict[str, Any]:
        """
        Trains and compares 3 classification models for water quality risk state:
        1. Majority Class Dummy Baseline
        2. Logistic Regression Baseline
        3. Random Forest Classifier
        """
        avail_features = [col for col in self.trained_features if col in train_df.columns and col in test_df.columns]
        
        tr = train_df.dropna(subset=[target_col]).copy()
        te = test_df.dropna(subset=[target_col]).copy()
        
        X_train = tr[avail_features].values
        y_train = tr[target_col].values
        X_test = te[avail_features].values
        y_test = te[target_col].values
        
        classes = sorted(list(set(y_train)))
        
        models = {
            "Majority Baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, random_state=self.random_state))]),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=self.random_state, n_jobs=-1),
        }
        
        results = {}
        fitted_models = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            fitted_models[name] = model
            
            p_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
            r_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
            f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
            f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            
            cm = confusion_matrix(y_test, y_pred, labels=classes).tolist()
            
            results[name] = {
                "precision_macro": p_macro,
                "recall_macro": r_macro,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "confusion_matrix": cm,
                "classes": classes,
            }
            
        best_name = max(results.keys(), key=lambda k: results[k]["f1_macro"])
        self.best_classifier = fitted_models[best_name]
        
        self.classification_results = {
            "metrics": results,
            "best_model": best_name,
            "classes": classes,
            "target": target_col,
            "test_samples": len(y_test),
        }
        return self.classification_results

    def save_models(self, directory: str) -> None:
        """Saves trained models and benchmark metadata to disk."""
        os.makedirs(directory, exist_ok=True)
        if self.best_regressor is not None:
            joblib.dump(self.best_regressor, os.path.join(directory, "best_ec_regressor.joblib"))
        if self.best_classifier is not None:
            joblib.dump(self.best_classifier, os.path.join(directory, "best_risk_classifier.joblib"))
        meta = {
            "regression_results": {k: v for k, v in self.regression_results.items() if k not in ["predictions", "test_actuals"]},
            "classification_results": self.classification_results,
            "multi_parameter_results": self.multi_parameter_results,
        }
        joblib.dump(meta, os.path.join(directory, "benchmark_summary.joblib"))


def run_full_ml_pipeline(
    random_seed: int = 42,
    horizon_steps: int = 4,
    save_artifacts: bool = True,
) -> Tuple[PredictiveModelBenchmark, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end execution:
    1. Generates 18-run benchmark dataset partitioned by run_id
    2. Builds lagged forecasting targets (t + horizon_steps) for EC, pH, TDS, Hardness
    3. Trains and benchmarks regression models across parameters
    4. Trains and benchmarks risk classifiers
    5. Saves serialized models and benchmark summaries
    """
    from engine.dataset_generator import generate_research_benchmark_dataset
    
    master_df, train_raw, test_raw = generate_research_benchmark_dataset(master_seed=random_seed)
    
    # Construct multi-step forecasting targets
    train_lagged = create_lagged_dataset(train_raw, horizon_steps=horizon_steps, group_col="run_id")
    test_lagged = create_lagged_dataset(test_raw, horizon_steps=horizon_steps, group_col="run_id")
    
    benchmark = PredictiveModelBenchmark(random_state=random_seed)
    benchmark.train_and_evaluate_regression(train_lagged, test_lagged, target_col="target_ec_future")
    benchmark.train_multi_parameter_forecast(train_lagged, test_lagged)
    benchmark.train_and_evaluate_classification(train_lagged, test_lagged, target_col="target_risk_class_future")
    
    if save_artifacts:
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "trained_models")
        benchmark.save_models(assets_dir)
        
    return benchmark, train_lagged, test_lagged
