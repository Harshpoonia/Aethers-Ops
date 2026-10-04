CREATE TABLE IF NOT EXISTS services(id TEXT PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS scenarios(id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT now(), kind TEXT, target TEXT);
CREATE TABLE IF NOT EXISTS telemetry(ts TIMESTAMPTZ DEFAULT now(), sim_tick INT, service_id TEXT REFERENCES services(id), cpu REAL, memory REAL, request_rate REAL, latency_ms REAL, error_rate REAL, network_mbps REAL, connections REAL, failed BOOLEAN);
CREATE TABLE IF NOT EXISTS predictions(ts TIMESTAMPTZ DEFAULT now(), sim_tick INT, service_id TEXT REFERENCES services(id), probability REAL, horizon_minutes INT, model_version TEXT);
CREATE TABLE IF NOT EXISTS alerts(ts TIMESTAMPTZ DEFAULT now(), sim_tick INT, service_id TEXT REFERENCES services(id), probability REAL, model_version TEXT);
CREATE TABLE IF NOT EXISTS incidents(ts TIMESTAMPTZ DEFAULT now(), service_id TEXT REFERENCES services(id), failure_tick INT, alert_tick INT, lead_sim_minutes INT, correct BOOLEAN, within_horizon BOOLEAN, scenario TEXT);
CREATE TABLE IF NOT EXISTS events(ts TIMESTAMPTZ DEFAULT now(), type TEXT, msg TEXT);
CREATE INDEX IF NOT EXISTS telemetry_idx ON telemetry(service_id, ts);
