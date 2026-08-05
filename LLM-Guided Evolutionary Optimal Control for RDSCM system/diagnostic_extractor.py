import re
from typing import Dict, Any, List

import numpy as np


def extract_metrics(version: str,
                    best_params: Dict[str, float],
                    best_cost: float,
                    final_state: np.ndarray,
                    control_logs: Dict[str, float],
                    param_bounds: Dict[str, tuple]) -> Dict[str, Any]:
    Iud, Iad = 1.2, 1.2
    LSud, LIud, LRud = 0.5, 0.2, 0.8

    final_means = {
        'Iu': np.mean(final_state[:, 1]),
        'Ia': np.mean(final_state[:, 7]),
        'LSu': np.mean(final_state[:, 3]),
        'LIu': np.mean(final_state[:, 4]),
        'LRu': np.mean(final_state[:, 5]),
    }
    deviations = {
        'Iu': final_means['Iu'] - Iud,
        'Ia': final_means['Ia'] - Iad,
        'LSu': final_means['LSu'] - LSud,
        'LIu': final_means['LIu'] - LIud,
        'LRu': final_means['LRu'] - LRud,
    }

    anomalies = []


    for name, dev in deviations.items():
        if abs(dev) > 0.3:
            anomalies.append(f"{name} deviation {dev:+.3f} exceeds 0.3")

    return {
        'version': version,
        'total_cost': best_cost,
        'params': best_params,
        'deviations': deviations,
        'control_activity': control_logs,
        'anomalies': anomalies,
        'param_bounds': param_bounds,
    }

def format_metrics_for_llm(metrics: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"Version: {metrics['version']}")
    lines.append(f"Best Total Cost: {metrics['total_cost']:.6f}")
    lines.append("Parameter Values:")
    for k, v in metrics['params'].items():
        lines.append(f"  {k} = {v:.4f}")
    lines.append("Final State Deviations:")
    for k, v in metrics['deviations'].items():
        lines.append(f"  {k} = {v:+.4f}")
    lines.append("Control Activity:")
    for k, v in metrics['control_activity'].items():
        lines.append(f"  {k} = {v:.4f}")
    if metrics['anomalies']:
        lines.append("Anomalies:")
        for a in metrics['anomalies']:
            lines.append(f"  ⚠️ {a}")
    return "\n".join(lines)


def detect_structural_anomalies(code_str: str, metrics: dict, param_bounds: dict) -> List[str]:
    anomalies = []

    # 1. Check return shape
    if 'return' in code_str:
        return_section = code_str.split('return')[-1].split('\n')[0]
        if 'zeros' not in return_section and 'ones' not in return_section and 'stack' not in return_section:
            if 'column_stack' not in return_section and 'hstack' not in return_section:
                anomalies.append("[Format Warning] Return value may not be explicitly constructed. Ensure shape is (M+1, 6).")


    # 2. Vectorization violations
    if re.search(r'float\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)', code_str):
        anomalies.append("[Vectorization Warning] Found float(variable) usage. If variable is an array, this will crash. Replace with .astype(float).")
    if re.search(r'int\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)', code_str):
        anomalies.append("[Vectorization Warning] Found int(variable) usage. If variable is an array, this will crash. Remove it.")
    if re.search(r'bool\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\)', code_str):
        anomalies.append("[Vectorization Warning] Found bool(variable) usage. If variable is an array, this will crash. Remove it.")

    # 3. Spatial loops
    if re.search(r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(\s*x\s*\)\s*\)', code_str):
        anomalies.append("[Vectorization Warning] Detected for loop over spatial points. This violates vectorization requirement.")

    # 4. np.where usage
    if 'np.where' not in code_str:
        anomalies.append("[Vectorization Suggestion] np.where not used for generating control values. Ensure final control values are strictly 0/1 or 0/0.5.")

    return anomalies
