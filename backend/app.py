import asyncio, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from backend.engine import Engine
from simulator.sim import SCENARIOS, CFG

ROOT = pathlib.Path(__file__).resolve().parents[1]
app = FastAPI(title="AetherOps"); E = Engine()

@app.get("/api/health")
def health(): return {"status": "ok", "model": E.pred.version, "tick": E.sim.t, "database": "connected" if E.db.ok else E.db.err}
@app.get("/api/services")
def services(): return list(E.snapshot()["services"].values())
@app.get("/api/services/{sid}")
def service(sid: str):
    s = E.snapshot()["services"].get(sid)
    if not s: raise HTTPException(404, "unknown service")
    return s
@app.get("/api/telemetry")
def telemetry(): return {k: v["series"] for k, v in E.snapshot()["services"].items()}
@app.get("/api/predictions")
def predictions(): return {k: {"timestamp": E.snapshot()["sim_tick"], "service_id": k, "failure_probability": v["probability"],
        "predicted_state": "FAILURE_LIKELY_WITHIN_HORIZON" if v["state"] in ("at_risk",) else v["state"].upper(), "horizon_minutes": E.pred.meta["horizon"], "model_version": E.pred.version}
        for k, v in E.snapshot()["services"].items()}
@app.get("/api/alerts")
def alerts(): return E.snapshot()["alerts"]
@app.get("/api/topology")
def topology(): return {"services": CFG["services"], "labels": CFG["labels"], "edges": CFG["topology"]}
@app.get("/api/scenarios")
def scenarios(): return [{"id": k, "target": v[0]} for k, v in SCENARIOS.items()]
@app.post("/api/scenarios/start")
def start(body: dict):
    try: E.start(body.get("scenario", "normal"), body.get("target"))
    except ValueError as e: raise HTTPException(400, str(e))
    return {"ok": True}
@app.post("/api/scenarios/reset")
def reset(): E.reset(); return {"ok": True}
@app.post("/api/mode")
def mode(body: dict):
    try: E.set_mode(body.get("mode"))
    except ValueError: raise HTTPException(400, "mode must be demo|real")
    return {"ok": True, "mode": E.mode}
@app.get("/api/evaluation")
def evaluation(): return json.load(open(ROOT / "ml/artifacts/evaluation.json"))
@app.get("/api/multiseed")
def multiseed():
    f = ROOT / "ml/artifacts/multiseed.json"
    if not f.exists(): raise HTTPException(404, "multiseed.json not found; run python -m ml.multiseed")
    return json.load(open(f))
@app.post("/api/checkout")
def checkout():
    ok, bad = E.checkout()
    if ok: return {"ok": True, "order": "ORD-%d" % E.sim.t}
    return JSONResponse({"ok": False, "failed_service": bad, "message": f"{E.snapshot()['services'][bad]['label']} unavailable. Order cannot be completed."}, status_code=503)

def _guard(path, ok_body):
    ok, bad = E.call(path)
    if ok: return ok_body
    return JSONResponse({"ok": False, "failed_service": bad, "message": f"{E.snapshot()['services'][bad]['label']} unavailable."}, status_code=503)
@app.get("/api/product/{pid}")
def product(pid: int): return _guard(["gateway", "product"], {"id": pid, "description": "In stock. Served by Gateway → Product Service."})
@app.post("/api/login")
def login(body: dict): return _guard(["gateway", "auth"], {"ok": True, "user": body.get("user") or "guest"})

@app.websocket("/ws/telemetry")
async def ws(sock: WebSocket):
    await sock.accept(); last = -1
    try:
        while True:
            if E.version != last: last = E.version; await sock.send_text(json.dumps(E.snapshot()))
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError): pass

app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")
@app.get("/")
def index(): return FileResponse(ROOT / "frontend/index.html")
