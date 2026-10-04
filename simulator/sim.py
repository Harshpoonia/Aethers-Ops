"""Simulated Distributed-Service Environment. 1 tick = 1 simulated minute."""
import json, pathlib
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = json.load(open(ROOT / "config.json"))
SERVICES = CFG["services"]; N = len(SERVICES)
IDX = {s: i for i, s in enumerate(SERVICES)}
CALLS = CFG["topology"]; METRICS = CFG["metrics"]; F = len(METRICS)
FAIL_ERR = CFG["failure"]["error_rate_pct"]
PROP = 0.95  # fraction of a dependency's degradation inherited by its caller (1 tick lag)

def adjacency():
    """Symmetric-normalised adjacency with self loops (for the GCN)."""
    A = np.eye(N, dtype=np.float32)
    for c, cs in CALLS.items():
        for x in cs:
            A[IDX[c], IDX[x]] = A[IDX[x], IDX[c]] = 1
    d = A.sum(1) ** -0.5
    return (A * d[:, None] * d[None, :]).astype(np.float32)

SCENARIOS = {  # kind -> (default target, ramp ticks, level)
    "normal": (None, 0, 0),
    "database_degradation": ("db", 16, 1.0),
    "payment_degradation": ("payment", 14, 1.0),
    "cpu_saturation": ("order", 12, 1.0),
    "network_latency": ("product", 14, 1.0),
    "sudden_failure": ("payment", 0, 1.0),
}

class Sim:
    def __init__(self, seed=0, plan=None):
        self.rng = np.random.default_rng(seed); self.t = 0
        self.plan = plan or {"kind": "normal"}
        self.eff = np.zeros(N); self.failed = np.zeros(N, bool)
        self.fail_t = [None] * N

    def _own(self):
        p, own = self.plan, np.zeros(N); k = p["kind"]
        if k == "normal": return own
        i, dt = IDX[p["target"]], self.t - p["start"]
        if dt < 0: return own
        if k == "blip": own[i] = p["level"] * np.sin(np.pi * min(dt / p["ramp"], 1))
        elif k == "sudden_failure": own[i] = 1.0
        else: own[i] = p["level"] * min(dt / max(p["ramp"], 1), 1)
        return own

    def step(self):
        eff = self._own()
        for caller, callees in CALLS.items():
            c = IDX[caller]
            eff[c] = max(eff[c], PROP * max(self.eff[IDX[x]] for x in callees))
        eff = np.clip(eff, 0, 1); e = eff
        r = self.rng; n = lambda s=0.03: r.normal(1, s, N)
        traffic = 1 + 0.08 * np.sin(self.t / 6.0)
        m = np.zeros((N, F))
        m[:, 0] = np.clip((38 + 50 * e ** 0.8) * n(), 0, 100)
        m[:, 1] = np.clip((50 + 25 * e) * n(0.015), 0, 100)
        m[:, 2] = 120 * traffic * (1 - 0.5 * e) * n()
        m[:, 3] = (100 * (1 + 0.3 * e) + 1900 * e ** 3) * n(0.04)
        m[:, 4] = np.clip((0.2 + 30 * e ** 3) * n(0.05), 0, 100)
        m[:, 5] = 30 * traffic * (1 + 0.8 * e) * n()
        m[:, 6] = (60 + 140 * e) * n()
        newly = (~self.failed) & (m[:, 4] >= FAIL_ERR)
        for i in np.where(newly)[0]: self.fail_t[i] = self.t
        self.failed |= newly
        for i in np.where(self.failed)[0]: m[i] = [5, 40, 0, 5000, 100, 0, 0]
        self.eff = eff; out = (m.copy(), self.failed.copy(), self.t); self.t += 1
        return out
