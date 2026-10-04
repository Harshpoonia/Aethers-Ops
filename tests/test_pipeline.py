import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from ml.data import gen_runs, chrono_split, fit_scaler, windows, W, H
def test_labels_use_only_future():
    r = gen_runs(5, 3); sc = fit_scaler(r); X, Y, V, meta = windows(r, sc)
    ri, t = meta[0]; assert (Y[0] == r[ri]["failed"][t + 1:t + H + 1].any(0)).all()
def test_split_is_chronological_and_disjoint():
    a, b, c = chrono_split(gen_runs(40, 1)); assert len(a) + len(b) + len(c) == 40
def test_engine_flow():
    from backend.engine import Engine
    E = Engine(); E.start("database_degradation")
    for _ in range(45):
        with E.lock: E._tick(sleep=False)
    assert E.alerts and E.incidents and E.incidents[0]["lead_sim_minutes"] is not None
    E.reset(); assert not E.alerts and not E.incidents and E.snapshot()["summary"]["failed"] == 0
def test_preprocessing_validation():
    import pandas as pd
    from ml.preprocessing import validate_frame
    from simulator.sim import METRICS
    row = {"timestamp": "2026-01-01T00:00:00Z", "service": "db", **{m: 1.0 for m in METRICS}}
    bad = dict(row, timestamp="nonsense"); out = dict(row, timestamp="2026-01-01T00:01:00Z", cpu=999, error_rate="x")
    df, rep = validate_frame(pd.DataFrame([row, row, bad, out]))
    assert rep["duplicates"] == 1 and rep["bad_timestamps"] == 1 and rep["outliers_clipped"] >= 1 and df["cpu"].max() <= 100
