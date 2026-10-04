import threading, time, collections, datetime as dt
import numpy as np
from simulator.sim import Sim, SERVICES, SCENARIOS, METRICS, CFG, N, IDX, CALLS
from ml.inference import Predictor
from backend.db import DB
W, H = CFG["window_minutes"], CFG["horizon_minutes"]
now = lambda: dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

class Engine:
    def __init__(self):
        self.pred = Predictor(); self.db = DB(); self.lock = threading.RLock(); self.mode = "demo"; self.version = 0
        self.threshold = float(self.pred.threshold); self.reset(); self.th = threading.Thread(target=self._loop, daemon=True); self.th.start()

    # ---- control ----
    def reset(self):
        with self.lock:
            self.sim = Sim(int(time.time()) % 100000); self.hist = collections.deque(maxlen=120); self.ph = collections.deque(maxlen=120)
            self.prob = np.zeros(N); self.alerts = []; self.incidents = []; self.events = []; self.scenario = {"kind": "normal"}
            self.alert_of = {}; self.last_failed = np.zeros(N, bool)
            for _ in range(W + 2): self._tick(sleep=False)   # warm-up history (normal operation)
            self.events.append({"ts": now(), "type": "reset", "msg": "System reset; all services healthy"}); self.version += 1

    def start(self, kind, target=None):
        if kind not in SCENARIOS: raise ValueError("unknown scenario")
        with self.lock:
            d_t, ramp, level = SCENARIOS[kind]; plan = {"kind": kind}
            if kind != "normal":
                plan.update(target=target or d_t, start=self.sim.t + 2, ramp=ramp, level=level)
            self.sim.plan = plan; self.scenario = dict(plan, started_wall=now(), started_tick=self.sim.t)
            self.events.append({"ts": now(), "type": "scenario", "msg": f"Scenario started: {kind}"}); self.db.scenario(kind, plan.get("target")); self.version += 1

    def set_mode(self, mode):
        if mode not in ("demo", "real"): raise ValueError
        self.mode = mode; self.version += 1

    # ---- core ----
    def _tick(self, sleep=True):
        m, failed, t = self.sim.step(); self.hist.append(m)
        if len(self.hist) >= W: self.prob = self.pred.predict(np.array(list(self.hist)[-W:]))
        wall = now()
        self.ph.append(self.prob.copy())
        for i, s in enumerate(SERVICES):
            if failed[i] and not self.last_failed[i]:               # actual failure transition
                a = self.alert_of.get(s); lead = (t - a["tick"]) if a else None
                inc = {"service": s, "failure_tick": t, "failure_wall": wall, "alert_tick": a["tick"] if a else None, "alert_wall": a["wall"] if a else None,
                       "lead_sim_minutes": lead, "correct": a is not None, "within_horizon": a is not None and lead <= H, "probability_at_alert": a["p"] if a else None, "scenario": self.scenario.get("kind")}
                self.incidents.append(inc); self.db.incident(inc); self.events.append({"ts": wall, "type": "failure", "msg": f"{s} FAILED" + (f" (warned {lead} sim-min earlier)" if a else " (NO prior warning)")})
            elif (not failed[i]) and self.prob[i] >= self.threshold and s not in self.alert_of:   # model-driven alert
                a = {"service": s, "tick": t, "wall": wall, "p": float(self.prob[i]), "horizon": H, "model": self.pred.version}
                self.alert_of[s] = a; self.alerts.append(a); self.db.alert(a); self.events.append({"ts": wall, "type": "alert", "msg": f"EARLY WARNING {s}: p={self.prob[i]:.2f}"})
        sc = self.scenario   # persist telemetry only while a scenario is running (max 120 ticks) so an idle server never fills the database
        if sleep and sc.get("kind") != "normal" and sc.get("started_tick") is not None and t - sc["started_tick"] < 120:
            self.db.tick(t, m, failed, self.prob, self.pred.version, H)
        self.last_failed = failed.copy(); self.version += 1

    def _loop(self):
        while True:
            try:
                with self.lock: self._tick()
            except Exception as e: print("tick error", e)
            time.sleep(CFG["demo"]["tick_seconds_demo" if self.mode == "demo" else "tick_seconds_real"])

    def state_of(self, i):
        f, p, m = self.last_failed[i], self.prob[i], self.hist[-1][i]
        return "failed" if f else "at_risk" if p >= self.threshold else "degraded" if (m[4] > 1.0 or m[3] > 250) else "healthy"

    def signals(self, i):
        w = np.array(list(self.hist)[-W:])[:, i]; base = np.array(list(self.hist)[:W])[:, i].mean(0); out = []
        for k, lab in ((3, "Increasing latency"), (4, "Increasing error rate"), (0, "Increasing CPU")):
            if w[-1, k] > 1.25 * base[k]: out.append(lab)
        if any(self.state_of(IDX[c]) != "healthy" for c in CALLS.get(SERVICES[i], [])): out.append("Dependency degradation")
        return out

    def snapshot(self):
        with self.lock:
            svc = {}
            for i, s in enumerate(SERVICES):
                st = self.state_of(i)
                svc[s] = dict(id=s, label=CFG["labels"][s], state=st, probability=float(self.prob[i]), metrics=dict(zip(METRICS, map(float, self.hist[-1][i]))),
                              series={k: [float(h[i][j]) for h in list(self.hist)[-60:]] for j, k in enumerate(METRICS)},
                              prob_series=[float(p[i]) for p in list(self.ph)[-60:]], signals=self.signals(i) if st == "at_risk" else [])
            c = lambda x: sum(v["state"] == x for v in svc.values())
            return dict(version=self.version, sim_tick=self.sim.t, mode=self.mode, scenario=self.scenario, services=svc, edges=CALLS,
                        summary=dict(total=N, healthy=c("healthy"), degraded=c("degraded"), at_risk=c("at_risk"), failed=c("failed"), horizon_minutes=H, window_minutes=W,
                                     model=self.pred.name, model_version=self.pred.version, threshold=self.threshold),
                        alerts=self.alerts[-20:], incidents=self.incidents[-20:], events=self.events[-30:])

    def call(self, path):
        for s in path:
            if self.last_failed[IDX[s]]: return False, s
        return True, None

    def checkout(self):
        path = ["gateway", "order", "payment", "db"]
        for s in path:
            if self.last_failed[IDX[s]]: return False, s
        return True, None
