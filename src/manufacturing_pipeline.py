"""Manufacturing analytics pipeline for plating equipment logs.

This module provides notebook-friendly functions to:
1) load raw Excel data
2) select relevant process variables
3) clean data
4) engineer features
5) detect anomalies
6) generate AI-style insights
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_COLUMN_GROUPS: Dict[str, List[str]] = {
    "context": ["Time", "L1 Lot ID", "L1 Product Name"],
    "process": ["Running Speed", "Target Speed", "Pump Speed", "Pressure", "Flow"],
    "chemical": ["Ag Deplating pH", "Ag Plating pH", "Cu Strike pH", "Conductivity"],
}


def _to_snake_case(name: str) -> str:
    """Normalize a column name to snake_case."""
    cleaned = str(name).strip().lower()
    out = []
    prev_is_underscore = False
    for char in cleaned:
        if char.isalnum():
            out.append(char)
            prev_is_underscore = False
        else:
            if not prev_is_underscore:
                out.append("_")
                prev_is_underscore = True
    normalized = "".join(out).strip("_")
    return normalized or "col"


def _build_column_alias_map(columns: Iterable[str]) -> Dict[str, str]:
    """Map normalized names to original columns for resilient matching."""
    alias_map: Dict[str, str] = {}
    for col in columns:
        alias_map[_to_snake_case(col)] = col
    return alias_map


def _resolve_requested_columns(df: pd.DataFrame, requested_columns: List[str]) -> Tuple[List[str], List[str]]:
    """Resolve requested column names using exact + snake_case aliases."""
    available = set(df.columns)
    alias_map = _build_column_alias_map(df.columns)

    selected: List[str] = []
    missing: List[str] = []

    for name in requested_columns:
        if name in available:
            selected.append(name)
            continue

        alias = _to_snake_case(name)
        resolved = alias_map.get(alias)
        if resolved is not None:
            selected.append(resolved)
        else:
            missing.append(name)

    # De-duplicate while preserving order.
    selected = list(dict.fromkeys(selected))
    return selected, missing


def load_data(file_path: str, time_col: str = "Time") -> pd.DataFrame:
    """Load raw Excel, parse datetime, and sort by time defensively."""
    df = pd.read_excel(file_path)

    resolved_time_col = time_col
    if time_col not in df.columns:
        alias_map = _build_column_alias_map(df.columns)
        resolved_time_col = alias_map.get(_to_snake_case(time_col), "")

    if not resolved_time_col:
        raise ValueError(f"Time column '{time_col}' was not found in the input file.")

    df[resolved_time_col] = pd.to_datetime(df[resolved_time_col], errors="coerce")
    df = df.sort_values(resolved_time_col).reset_index(drop=True)

    if resolved_time_col != time_col:
        df = df.rename(columns={resolved_time_col: time_col})

    return df


def select_columns(
    df: pd.DataFrame,
    column_groups: Optional[Dict[str, List[str]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Select relevant feature-layer columns and report missing fields."""
    groups = column_groups or DEFAULT_COLUMN_GROUPS
    requested_columns: List[str] = [col for cols in groups.values() for col in cols]

    selected_columns, missing_columns = _resolve_requested_columns(df, requested_columns)
    if not selected_columns:
        raise ValueError("No requested columns were found. Check source headers.")

    result = df.loc[:, selected_columns].copy()
    metadata = {
        "selected_columns": selected_columns,
        "missing_columns": missing_columns,
        "requested_count": len(requested_columns),
        "selected_count": len(selected_columns),
    }
    return result, metadata


def clean_data(
    df: pd.DataFrame,
    row_nan_ratio_threshold: float = 0.5,
    ffill_limit: int = 3,
) -> pd.DataFrame:
    """Clean and normalize data.

    - Drop rows with too many NaN values
    - Forward-fill small gaps
    - Cast numeric-like columns
    - Normalize column names to snake_case
    """
    cleaned = df.copy()

    min_non_na = max(int(np.ceil(cleaned.shape[1] * (1 - row_nan_ratio_threshold))), 1)
    cleaned = cleaned.dropna(axis=0, thresh=min_non_na)

    # Preserve product/lot IDs as string-like and parse numeric for others.
    protected_cols = {"time", "l1_lot_id", "l1_product_name"}
    renamed_columns = {col: _to_snake_case(col) for col in cleaned.columns}
    cleaned = cleaned.rename(columns=renamed_columns)

    for col in cleaned.columns:
        if col in protected_cols:
            continue
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        cleaned[numeric_cols] = cleaned[numeric_cols].ffill(limit=ffill_limit)

    if "time" in cleaned.columns:
        cleaned = cleaned.sort_values("time").reset_index(drop=True)

    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features used for diagnostics and anomaly detection."""
    out = df.copy()

    if {"running_speed", "target_speed"}.issubset(out.columns):
        out["speed_diff"] = out["running_speed"] - out["target_speed"]
    else:
        out["speed_diff"] = np.nan

    ph_cols = [c for c in ["ag_deplating_ph", "ag_plating_ph", "cu_strike_ph"] if c in out.columns]
    if ph_cols:
        out["ph_range"] = out[ph_cols].max(axis=1) - out[ph_cols].min(axis=1)
    else:
        out["ph_range"] = np.nan

    if "pressure" in out.columns:
        out["pressure_variation"] = out["pressure"].rolling(window=10, min_periods=3).std()
    else:
        out["pressure_variation"] = np.nan

    return out


def detect_anomaly(
    df: pd.DataFrame,
    threshold: Optional[float] = None,
    quantile: float = 0.98,
) -> pd.DataFrame:
    """Compute z-score based anomaly score and flag anomalies."""
    analyzed = df.copy()
    numeric_cols = analyzed.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        analyzed["anomaly_score"] = 0.0
        analyzed["anomaly_flag"] = False
        analyzed["segment"] = "NORMAL"
        return analyzed

    numeric = analyzed[numeric_cols]
    means = numeric.mean(skipna=True)
    stds = numeric.std(skipna=True).replace(0, np.nan)
    zscores = (numeric - means) / stds
    zscores = zscores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    analyzed["anomaly_score"] = zscores.abs().sum(axis=1)

    if threshold is None:
        threshold = float(analyzed["anomaly_score"].quantile(quantile))

    analyzed["anomaly_flag"] = analyzed["anomaly_score"] > threshold
    analyzed["segment"] = np.where(analyzed["anomaly_flag"], "ABNORMAL", "NORMAL")

    z_cols = {f"z_{col}": zscores[col] for col in numeric_cols}
    analyzed = analyzed.assign(**z_cols)
    analyzed.attrs["anomaly_threshold"] = threshold
    analyzed.attrs["anomaly_numeric_cols"] = numeric_cols

    return analyzed


def generate_insight(df_analyzed: pd.DataFrame, top_k: int = 5) -> Dict[str, List[str]]:
    """Generate structured AI-style insights from anomaly results."""
    z_cols = [c for c in df_analyzed.columns if c.startswith("z_")]
    if not z_cols:
        return {
            "top_anomaly_features": [],
            "key_observations": ["No numeric signals available for anomaly analysis."],
            "possible_impacts": ["Insufficient telemetry to infer process risks."],
        }

    abnormal_mask = df_analyzed.get("anomaly_flag", pd.Series(False, index=df_analyzed.index))
    abnormal_df = df_analyzed.loc[abnormal_mask]
    if abnormal_df.empty:
        abnormal_df = df_analyzed

    top_scores = abnormal_df[z_cols].abs().mean().sort_values(ascending=False)
    top_features = [name.replace("z_", "") for name in top_scores.head(top_k).index]

    observations: List[str] = []
    impacts: List[str] = []

    if "ph_range" in top_features:
        observations.append("pH instability correlates with speed fluctuation in abnormal windows.")
        impacts.append("Bath chemistry drift may increase defect and rework risk.")

    if "pressure_variation" in top_features or "pressure" in top_features:
        observations.append("Pressure variation indicates process inconsistency and mechanical transients.")
        impacts.append("Inconsistent deposition can reduce yield and throughput stability.")

    if "speed_diff" in top_features:
        observations.append("Running speed diverges from target speed during high anomaly periods.")
        impacts.append("Line speed mismatch can impact plating uniformity and cycle-time predictability.")

    if not observations:
        observations.append("Multiple process variables deviate simultaneously in anomaly segments.")
    if not impacts:
        impacts.append("Combined signal drift suggests elevated yield-impact probability.")

    return {
        "top_anomaly_features": top_features,
        "key_observations": observations,
        "possible_impacts": impacts,
    }


def print_insights(summary_dict: Dict[str, List[str]]) -> None:
    """Pretty-print generated insights for notebook demos."""
    print("\n=== Anomaly Insight Summary ===")
    print("Top anomaly features:", ", ".join(summary_dict.get("top_anomaly_features", [])) or "N/A")

    print("\nKey observations:")
    for item in summary_dict.get("key_observations", []):
        print(f"- {item}")

    print("\nPossible impacts:")
    for item in summary_dict.get("possible_impacts", []):
        print(f"- {item}")


def generate_llm_insight(
    summary_dict: Dict[str, List[str]],
    model: str = "qwen2.5:3b",
    base_url: str = "http://localhost:11434",
    timeout: int = 30,
) -> str:
    """Optional hook: ask local Ollama API for a natural-language explanation."""
    import json
    import urllib.error
    import urllib.request

    prompt = (
        "You are a manufacturing analytics assistant. "
        "Given this anomaly summary JSON, provide a concise explanation in plain English with likely root causes and actions.\n\n"
        f"{json.dumps(summary_dict, ensure_ascii=False)}"
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
        body = json.loads(raw)
        return str(body.get("response", "")).strip() or "LLM returned an empty response."
    except urllib.error.URLError as exc:
        return f"LLM call failed (URL error): {exc}"
    except Exception as exc:  # Defensive on purpose for notebook scenarios.
        return f"LLM call failed: {exc}"


def _extract_anomaly_periods(df_analyzed: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Group contiguous anomaly rows into time periods and return the most severe windows."""
    if "anomaly_flag" not in df_analyzed.columns or not bool(df_analyzed["anomaly_flag"].any()):
        return pd.DataFrame(columns=["start_time", "end_time", "rows", "max_anomaly_score"])

    work = df_analyzed.copy()
    work["block_id"] = (work["anomaly_flag"] != work["anomaly_flag"].shift(fill_value=False)).cumsum()
    abnormal = work[work["anomaly_flag"]].copy()
    if abnormal.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "rows", "max_anomaly_score"])

    if "time" in abnormal.columns:
        period_df = (
            abnormal.groupby("block_id", as_index=False)
            .agg(
                start_time=("time", "min"),
                end_time=("time", "max"),
                rows=("anomaly_flag", "size"),
                max_anomaly_score=("anomaly_score", "max"),
            )
            .sort_values(["max_anomaly_score", "rows"], ascending=[False, False])
            .head(top_n)
            .reset_index(drop=True)
        )
    else:
        period_df = (
            abnormal.groupby("block_id", as_index=False)
            .agg(
                start_time=("block_id", "min"),
                end_time=("block_id", "max"),
                rows=("anomaly_flag", "size"),
                max_anomaly_score=("anomaly_score", "max"),
            )
            .sort_values(["max_anomaly_score", "rows"], ascending=[False, False])
            .head(top_n)
            .reset_index(drop=True)
        )

    return period_df


def run_pipeline(file_path: str) -> Dict[str, Any]:
    """Execute full pipeline and return reusable outputs with final insights."""
    df_raw = load_data(file_path=file_path)
    df_selected, column_report = select_columns(df_raw)
    df_clean = clean_data(df_selected)
    df_features = engineer_features(df_clean)
    df_analyzed = detect_anomaly(df_features)
    summary = generate_insight(df_analyzed)
    anomaly_periods = _extract_anomaly_periods(df_analyzed, top_n=5)

    print("\n=== Key anomaly periods ===")
    if anomaly_periods.empty:
        print("No anomaly periods identified.")
    else:
        print(anomaly_periods.to_string(index=False))

    demo_explanation = (
        "This pipeline converts raw plating logs into anomaly-ranked process signals, helping teams spot likely yield-impact drivers "
        "and prioritize corrective actions even when MES traceability is unavailable."
    )

    return {
        "df_clean": df_clean,
        "df_analyzed": df_analyzed,
        "summary": summary,
        "anomaly_periods": anomaly_periods,
        "column_report": column_report,
        "demo_explanation": demo_explanation,
    }


if __name__ == "__main__":
    # Example usage for local runs:
    # python src/manufacturing_pipeline.py
    FILE = "Y2026M04D20data V1.0.xlsx"
    try:
        outputs = run_pipeline(FILE)
        print_insights(outputs["summary"])
        print("\nDemo explanation:")
        print(outputs["demo_explanation"])
        print(
            f"\nRows in clean data: {len(outputs['df_clean'])}, "
            f"rows analyzed: {len(outputs['df_analyzed'])}"
        )
    except FileNotFoundError:
        print(f"Input file not found: {FILE}")
