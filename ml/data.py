import numpy as np, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from simulator.sim import Sim, SERVICES, N, F, CFG
from ml.preprocessing import sanitize
W = CFG["window_minutes"]; H = CFG["horizon_minutes"]

def make_plan(rng):
    u = rng.random()
    if u < 0.22: return {"kind": "normal"}, 70
    tgt = SERVICES[rng.integers(N)]; start = int(rng.integers(15, 30))
    if u < 0.37: return {"kind": "blip", "target": tgt, "start": start, "ramp": int(rng.integers(8, 20)), "level": float(rng.uniform(.45, .75))}, 70
    if u < 0.50: return {"kind": "sudden_failure", "target": tgt, "start": start, "ramp": 0, "level": 1.0}, start + 15
    ramp = int(rng.integers(8, 30)); kind = str(rng.choice(["database_degradation", "payment_degradation", "cpu_saturation", "network_latency"]))
    if kind == "database_degradation": tgt = "db"
    if kind == "payment_degradation": tgt = "payment"
    return {"kind": kind, "target": tgt, "start": start, "ramp": ramp, "level": float(rng.uniform(.95, 1.1))}, start + ramp + 12

def gen_runs(n, seed):
    rng = np.random.default_rng(seed); runs = []
    for k in range(n):  # generation order == chronological order of the dataset
        plan, T = make_plan(rng); s = Sim(int(rng.integers(1 << 30)), plan)
        X, Fl = [], []
        for _ in range(T):
            m, f, _t = s.step(); X.append(m); Fl.append(f)
        runs.append({"X": np.array(X, np.float32), "failed": np.array(Fl), "plan": plan, "fail_t": list(s.fail_t)})
    return runs

def chrono_split(runs, cfg=CFG["split"]):
    a = int(len(runs) * cfg["train"]); b = a + int(len(runs) * cfg["val"])
    return runs[:a], runs[a:b], runs[b:]

def fit_scaler(runs):
    X = np.concatenate([r["X"].reshape(-1, F) for r in runs])
    X = X[np.concatenate([~r["failed"].reshape(-1) for r in runs])]  # train runs only, healthy samples
    return {"mean": X.mean(0).tolist(), "std": (X.std(0) + 1e-6).tolist()}

def scale(X, sc): return ((X - np.array(sc["mean"], np.float32)) / np.array(sc["std"], np.float32)).astype(np.float32)

def windows(runs, sc):
    """Xw[B,W,N,F], y[B,N], valid[B,N], meta. Label uses ONLY ticks (t, t+H]; input only ticks <= t."""
    Xw, Y, V, meta = [], [], [], []
    for ri, r in enumerate(runs):
        X = scale(sanitize(r["X"]), sc); T = len(X); fl = r["failed"]
        for t in range(W - 1, T - H):
            Xw.append(X[t - W + 1:t + 1]); Y.append(fl[t + 1:t + H + 1].any(0)); V.append(~fl[t]); meta.append((ri, t))
    return np.array(Xw), np.array(Y, np.float32), np.array(V), meta

def flat(Xw):
    """Per-node features [B*N, W*F + 2F + N]: own window, deltas, node one-hot."""
    B = Xw.shape[0]; own = Xw.transpose(0, 2, 1, 3)
    f = own.reshape(B, N, -1); d = own[:, :, -1] - own[:, :, 0]; sl = own[:, :, -1] - own[:, :, -3]
    oh = np.broadcast_to(np.eye(N, dtype=np.float32), (B, N, N))
    return np.concatenate([f, d, sl, oh], -1).reshape(B * N, -1)
