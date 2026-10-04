"""Optional PostgreSQL persistence. Enabled only when DATABASE_URL is set; writes are queued to a worker thread so
the simulation/API never block on the database. Without DATABASE_URL (or psycopg) this is a no-op."""
import os, queue, threading, pathlib
from simulator.sim import CFG, SERVICES
class DB:
    def __init__(self):
        self.q = queue.Queue(maxsize=5000); self.ok = False; self.err = None
        url = os.getenv("DATABASE_URL")
        if not url: self.err = "DATABASE_URL not set (persistence disabled)"; return
        try:
            import psycopg; self.conn = psycopg.connect(url, autocommit=True)
            self.conn.execute((pathlib.Path(__file__).parent / "schema.sql").read_text())
            for s in SERVICES: self.conn.execute("INSERT INTO services VALUES (%s,%s) ON CONFLICT DO NOTHING", (s, CFG["labels"][s]))
            self.ok = True; threading.Thread(target=self._run, daemon=True).start()
        except Exception as e: self.err = f"{type(e).__name__}: {e}"
    def put(self, sql, args):
        if self.ok:
            try: self.q.put_nowait((sql, args))
            except queue.Full: pass
    def _run(self):
        while True:
            sql, args = self.q.get()
            try: self.conn.execute(sql, args)
            except Exception as e: self.err = str(e)
    def tick(self, t, M, failed, prob, ver, H):
        for i, s in enumerate(SERVICES):
            self.put("INSERT INTO telemetry(sim_tick,service_id,cpu,memory,request_rate,latency_ms,error_rate,network_mbps,connections,failed) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", (t, s, *map(float, M[i]), bool(failed[i])))
            self.put("INSERT INTO predictions(sim_tick,service_id,probability,horizon_minutes,model_version) VALUES(%s,%s,%s,%s,%s)", (t, s, float(prob[i]), H, ver))
    def alert(self, a): self.put("INSERT INTO alerts(sim_tick,service_id,probability,model_version) VALUES(%s,%s,%s,%s)", (a["tick"], a["service"], a["p"], a["model"]))
    def incident(self, i): self.put("INSERT INTO incidents(service_id,failure_tick,alert_tick,lead_sim_minutes,correct,within_horizon,scenario) VALUES(%s,%s,%s,%s,%s,%s,%s)", (i["service"], i["failure_tick"], i["alert_tick"], i["lead_sim_minutes"], i["correct"], i["within_horizon"], i["scenario"]))
    def event(self, e): self.put("INSERT INTO events(type,msg) VALUES(%s,%s)", (e["type"], e["msg"]))
    def scenario(self, k, tgt): self.put("INSERT INTO scenarios(kind,target) VALUES(%s,%s)", (k, tgt))
