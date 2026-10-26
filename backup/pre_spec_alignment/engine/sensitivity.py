"""
HYDROTWIN-AI: Sensitivity Analysis Engine
Executes parametric sweeps across thermal, hydraulic, and water-quality setpoints.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd
from engine.physics import simulate_system
from engine.metrics import compute_simulation_metrics


def run_parameter_sweep(
    base_config: Dict[str, Any],
    param_name: str,
    param_values: List[float],
) -> pd.DataFrame:
    """
    Executes a 1D sensitivity sweep over a selected parameter.
    Returns a DataFrame containing parameter values and resultant KPIs.
    """
    records = []
    for val in param_values:
        cfg = dict(base_config)
        cfg[param_name] = float(val)
        try:
            df = simulate_system(cfg)
            metrics = compute_simulation_metrics(df, cfg)
            metrics[param_name] = float(val)
            records.append(metrics)
        except Exception as e:
            continue

    return pd.DataFrame(records)
