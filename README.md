# AetherOps

**Predictive failure detection for distributed systems, using a Graph Convolutional Network and an LSTM.**

AetherOps predicts, every minute, the probability that each service in a distributed application will fail within the next **5 minutes**, so an early warning can be raised before users notice. It includes a simulator of a 10-service e-commerce application, an ML pipeline, a FastAPI backend and a live web dashboard.

**Live demo:** https://aethers-ops.onrender.com (free tier, the first load can take about a minute to wake up)

> Capstone project, Thapar Institute of Engineering and Technology, Patiala.

---

## Why

In a microservice system, a failure spreads to every service that calls the failing one. A slow database raises errors in payment, then order, then the gateway, where customers finally see it. Threshold alerts fire after this damage is done. AetherOps tries to warn earlier by reading both the **trend over time** (LSTM) and the **dependency structure** (GCN).

## Features

- **Simulator** of 10 services with 7 metrics each (CPU, memory, request rate, latency, error rate, network, connections). Stress propagates to calling services with a one-minute delay.
- **9 scenarios:** normal, sudden failure, and seven gradual degradations (database, payment, inventory database, search, cache, CPU saturation, network latency).
- **5 models compared:** logistic regression, random forest, LSTM, GCN, GCN+LSTM.
- **Live dashboard:** dependency graph with per-service risk, alerts, user journeys (checkout, browse, search, login), timeline, and a check of every prediction against the real failure.
- **Evaluation page** with all models side by side, plus repeated runs over several seeds.
- **Optional PostgreSQL (Neon)** storage for telemetry, predictions, alerts and incidents.

## Architecture

```
Simulator -> telemetry (7 metrics x 10 services, per minute)
          -> preprocessing -> 10-minute window + dependency graph
          -> model (GCN + LSTM) -> failure probability per service
          -> alert above threshold -> dashboard
          -> actual failure -> lead-time check
```

**Dependency graph**

```
gateway -> auth -> cache
        -> product -> inventory -> inventory_db
                   -> search
        -> order -> payment -> db
```

**Model (GCN+LSTM).** A learned node embedding (8) is concatenated with the metrics and projected to 32 units. One graph convolution step with the normalised adjacency `D^-1/2 (A+I) D^-1/2` and a skip connection mixes information between connected services. An LSTM(32) reads the 10 time steps for every service, and a small head outputs the failure probability. Training uses a positively weighted, masked binary cross-entropy (Adam, learning rate 0.002, batch size 128, early stopping).

## Method

| Item | Choice |
|---|---|
| Window / horizon | 10 minutes / 5 minutes |
| Label | 1 if the service fails in (t, t+5]; already-failed samples are masked |
| Failure | error rate >= 20%, or an injected sudden failure (sticky until reset) |
| Data | 360 simulated runs, chronological split 251 train / 54 validation / 55 test |
| Scaler | fitted on healthy training samples only |
| Threshold | chosen on validation data (maximum F1) |
| Reproducibility | fixed seeds, deterministic training |

## Results

Held-out test set: 55 runs, 104 incidents.

| Model | Precision | Recall | F1 | PR-AUC | Detected | Mean lead (min) |
|---|---|---|---|---|---|---|
| Logistic regression | .650 | .802 | .718 | .716 | 87/104 | 4.77 |
| Random forest | .796 | .737 | .766 | .794 | 86/104 | 4.47 |
| LSTM | .649 | .840 | .732 | .782 | 88/104 | 4.98 |
| GCN | .669 | .778 | .719 | .754 | 96/104 | 4.23 |
| **GCN+LSTM** | .636 | **.854** | .729 | .779 | **98/104** | 4.54 |

Median warning time is 5 minutes for all models. Averaged over five seeds, GCN+LSTM detects **93.4%** of incidents (the highest), with PR-AUC 0.784 +/- 0.083.

**Honest reading.** GCN+LSTM catches the most incidents early, but random forest has slightly better precision, F1 and PR-AUC, and the differences between the better models are within noise (standard deviation about 0.08). We do not claim GCN+LSTM is the most accurate model overall. All 6 misses of GCN+LSTM are sudden failures, which have no warning signs for the failing service. Precision is about 0.64, so roughly one alert in three is false.

## Getting started (Windows, PowerShell)

```powershell
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# train all models (writes ml/artifacts/)
python -m ml.train --epochs 15

# run the app
uvicorn backend.app:app --port 8000
```

Open http://localhost:8000 (Shop, Login, `#/monitor`, `#/evaluation`).

Optional:

```powershell
python -m ml.multiseed        # repeat the experiment over five seeds
python -m pytest tests -q     # run the tests
$env:DATABASE_URL = "postgresql://user:password@host/db"   # enable PostgreSQL storage
```

Without `DATABASE_URL` the app keeps everything in memory.

## Using the dashboard

1. Open **Monitor** and wait for the services to turn green.
2. Start a scenario, for example *Database degradation*.
3. Watch the risk climb on the degrading service while it is still healthy. An early warning appears.
4. See the failure spread up the dependency chain, and the prediction check compare each alert with the real failure.
5. Reset and try another scenario. *Sudden failure* shows the known limitation.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health`, `/api/services`, `/api/services/{id}` | Health and per-service state |
| GET | `/api/telemetry`, `/api/predictions`, `/api/alerts` | Live data |
| GET | `/api/topology`, `/api/journeys`, `/api/scenarios`, `/api/evaluation` | Structure and evaluation |
| POST | `/api/scenarios/start`, `/api/scenarios/reset` | Control scenarios |
| POST | `/api/mode`, `/api/checkout`, `/api/login`, `/api/search` | Demo actions |
| WS | `/ws/telemetry` | Live updates |

## Project structure

```
config.json            window, horizon, thresholds, services, journeys
simulator/sim.py       simulator and scenarios
ml/                    data, preprocessing, training, multi-seed, artifacts/
backend/               FastAPI app and live engine
frontend/index.html    dashboard
tests/                 pipeline tests
topology/              dependency graph
docker/, render.yaml   deployment
```

## Deployment

The backend and frontend deploy to Render (`render.yaml`, Docker). The database is Neon: set `DATABASE_URL` in the Render environment. Commit the trained `ml/artifacts/` so deployments never retrain.

## Limitations and future work

- Data is **simulated**. Validation on real telemetry or public microservice traces is the main next step.
- Sudden failures cannot be predicted for the failing service itself, only for the services above it.
- Failures propagate upward only. Effects on called services are not modelled.
- About one alert in three is a false alarm.
- Future work: real data, downstream effects, fewer false alarms, automatic remediation.

## Documentation

- Capstone report and IEEE-format paper (see `docs/`).
- Presentation slides.

## Team

[Harsh Poonia] ([2025010039]), [Ishika Sidana] ([2025010047]), [Kartik Baghel] ([2025010055]), [Mansha Setia] ([2025010066])
Mentor: [Dr. Chinu], Computer Science and Engineering Department, TIET Patiala.

## License

This project is released under the [MIT License](LICENSE).