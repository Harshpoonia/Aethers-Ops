"""Shared preprocessing (used identically in training, inference and CSV ingestion)."""
import numpy as np, pandas as pd
from simulator.sim import CFG, METRICS, SERVICES
LO = np.array([CFG["bounds"][m][0] for m in METRICS], np.float32); HI = np.array([CFG["bounds"][m][1] for m in METRICS], np.float32)

def sanitize(X):
    """Numeric conversion, missing-value handling (non-finite -> column lower bound... forward-fill done in validate_frame), outlier clipping to physical bounds."""
    X = np.asarray(X, np.float32); X = np.where(np.isfinite(X), X, LO); return np.clip(X, LO, HI)

def validate_frame(df):
    """Long-format telemetry frame: columns timestamp, service, + METRICS. Returns (clean_df, report)."""
    rep = {}; need = ["timestamp", "service"] + METRICS; miss = [c for c in need if c not in df.columns]
    if miss: raise ValueError(f"schema validation failed, missing columns: {miss}")
    df = df.copy(); df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    rep["bad_timestamps"] = int(df["timestamp"].isna().sum()); df = df.dropna(subset=["timestamp"])
    rep["unknown_services"] = int((~df["service"].isin(SERVICES)).sum()); df = df[df["service"].isin(SERVICES)]
    for m in METRICS: df[m] = pd.to_numeric(df[m], errors="coerce")
    rep["duplicates"] = int(df.duplicated(["timestamp", "service"]).sum()); df = df.drop_duplicates(["timestamp", "service"], keep="last")
    df = df.sort_values(["service", "timestamp"]); rep["missing_values"] = int(df[METRICS].isna().sum().sum())
    df[METRICS] = df.groupby("service")[METRICS].transform(lambda s: s.ffill().bfill())
    arr = df[METRICS].to_numpy(np.float32); rep["outliers_clipped"] = int(((arr < LO) | (arr > HI)).sum()); df[METRICS] = sanitize(arr)
    return df, rep
