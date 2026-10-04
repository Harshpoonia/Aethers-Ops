# AetherOps — predictive failure detection (Simulated Distributed-Service Environment)

```
Demo application → Services → Telemetry → Preprocessing → Graph + temporal window
→ model (GCN+LSTM if trained; see metadata) → Failure probability → Early warning → Actual failure → Lead-time validation
```
## Run locally
`pip install -r requirements.txt && python -m ml.train && uvicorn backend.app:app --port 8000` → http://localhost:8000 (Shop · Login · `#/monitor` · `#/evaluation`).
Docker: `docker compose up --build` (app + Postgres). Set `DATABASE_URL` (see `.env.example`) to persist telemetry/predictions/alerts/incidents; unset = in-memory only.
## Method
- `simulator/sim.py`: 6 services, explicit topology (`config.json`), degradation propagates to callers with 1-tick lag. 1 tick = 1 simulated minute.
- Failure: observed error rate >= 20% (health check failing) or explicit sudden-failure injection; sticky until reset. Same definition for training labels and live runs.
- Window 10 min, horizon 5 min, alert threshold — all in `config.json`. Label(t) = failure in (t, t+5]; input uses ticks <= t only.
- Chronological split by run order 70/15/15; scaler fit on train only; alert threshold picked on validation; active model chosen on validation PR-AUC.
- `ml/preprocessing.py`: schema/timestamp validation, duplicates, missing values, numeric conversion, outlier clipping (shared by training and inference).
- Contributing signals in the UI are simple rule-based indicators (latency/error/CPU vs. baseline, unhealthy dependencies), not causal proof.
## API
GET `/api/health /services /services/{id} /telemetry /predictions /alerts /topology /scenarios /evaluation`; POST `/api/scenarios/start {"scenario":...}`, `/api/scenarios/reset`, `/api/mode {"mode":"demo|real"}`, `/api/checkout`, `/api/login`; WS `/ws/telemetry`.
## Deploy
Backend+frontend: Render (`render.yaml`, Docker). Database: Neon — set `DATABASE_URL` in Render env. Commit trained `ml/artifacts/` so deploys never retrain.
## Status
Verified here: simulator, preprocessing, baseline training/evaluation, engine alert→failure→lead-time→reset flow, unit tests. Not verified: web server/WebSocket/UI in a browser, Postgres writes, Docker build, PyTorch models, deployment.
