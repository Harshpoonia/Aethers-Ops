"""Train + evaluate baselines; save artifacts. Run: python -m ml.train"""
import json, pathlib, sys, time, joblib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, roc_auc_score
from ml.data import *
ART = pathlib.Path(__file__).parent / "artifacts"; ART.mkdir(exist_ok=True)

def metrics(y, p, thr):
    yh = p >= thr; tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum()); fn = int((~yh & (y == 1)).sum()); tn = int((~yh & (y == 0)).sum())
    pr, rc, f1, _ = precision_recall_fscore_support(y, yh, average="binary", zero_division=0)
    return dict(precision=pr, recall=rc, f1=f1, pr_auc=average_precision_score(y, p), roc_auc=roc_auc_score(y, p),
                fpr=fp / max(fp + tn, 1), fnr=fn / max(fn + tp, 1), tp=tp, fp=fp, fn=fn, tn=tn, threshold=thr)

def pick_threshold(y, p):
    best = max(np.linspace(.1, .9, 33), key=lambda t: precision_recall_fscore_support(y, p >= t, average="binary", zero_division=0)[2])
    return float(best)

def early_warning(runs, meta, P, Y, V, thr):
    """Per incident (run,node first failure): detected if first alert in [fail-H, fail-1]; lead=fail-alert."""
    P = P.reshape(len(meta), N); alerts = {}
    for k, (ri, t) in enumerate(meta):
        for n in range(N):
            if V[k, n] and P[k, n] >= thr: alerts.setdefault((ri, n), []).append(t)
    inc, det, leads, by_kind, false_eps = 0, 0, [], {}, 0
    for ri, r in enumerate(runs):
        for n, ft in enumerate(r["fail_t"]):
            a = alerts.get((ri, n), [])
            if ft is None:
                false_eps += bool(a); continue
            inc += 1; ok = [x for x in a if ft - H <= x < ft]; k = r["plan"]["kind"]; d = by_kind.setdefault(k, [0, 0]); d[1] += 1
            if ok: det += 1; leads.append(ft - ok[0]); d[0] += 1
    return dict(incidents=inc, detected=det, missed=inc - det, pct_detected=det / max(inc, 1), mean_lead_min=float(np.mean(leads)) if leads else None,
                median_lead_min=float(np.median(leads)) if leads else None, false_alert_episodes=false_eps,
                by_scenario={k: {"detected": v[0], "incidents": v[1]} for k, v in by_kind.items()})

def main():
    cfg = CFG["training"]; runs = gen_runs(cfg["n_runs"], cfg["seed"]); tr, va, te = chrono_split(runs)
    sc = fit_scaler(tr); json.dump(sc, open(ART / "scaler.json", "w"))
    D = {k: windows(r, sc) for k, r in (("tr", tr), ("va", va), ("te", te))}
    (Xtr, Ytr, Vtr, _), (Xva, Yva, Vva, _), (Xte, Yte, Vte, mte) = D["tr"], D["va"], D["te"]
    ftr, fva, fte = flat(Xtr), flat(Xva), flat(Xte)
    mtr, mva, mte_ = Vtr.reshape(-1), Vva.reshape(-1), Vte.reshape(-1)
    ytr, yva, yte = Ytr.reshape(-1), Yva.reshape(-1), Yte.reshape(-1)
    res, preds = {}, {}
    sk = {"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
          "random_forest": RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=0)}
    for name, m in sk.items():
        m.fit(ftr[mtr], ytr[mtr]); joblib.dump(m, ART / f"{name}.joblib")
        preds[name] = (m.predict_proba(fva)[:, 1], m.predict_proba(fte)[:, 1])
    try:
        from ml.torch_models import train_torch, predict_torch; import torch
        for name in ("lstm", "gcn", "gcn_lstm"):
            m = train_torch(name, Xtr, Ytr, Vtr, Xva, Yva, Vva); torch.save(m.state_dict(), ART / f"{name}.pt")
            preds[name] = (predict_torch(m, Xva).reshape(-1), predict_torch(m, Xte).reshape(-1))
    except ImportError:
        print("PyTorch not installed: LSTM/GCN/GCN+LSTM NOT trained in this run.")
    for name, (pv, pt) in preds.items():
        thr = pick_threshold(yva[mva], pv[mva])      # threshold chosen on validation only
        r = metrics(yte[mte_], pt[mte_], thr); r["early_warning"] = early_warning(te, mte, pt, Yte, Vte, thr)
        r["val_pr_auc"] = float(average_precision_score(yva[mva], pv[mva])); res[name] = r
        print(f"{name:20s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} PR-AUC={r['pr_auc']:.3f} ROC={r['roc_auc']:.3f} "
              f"detected={r['early_warning']['detected']}/{r['early_warning']['incidents']} lead={r['early_warning']['mean_lead_min']}")
    active = max(res, key=lambda k: res[k]["val_pr_auc"])   # selected on VALIDATION, not test
    meta = dict(model_version=f"{active}-{time.strftime('%Y%m%d')}", active_model=active, window=W, horizon=H, threshold=res[active]["threshold"],
                trained_models=list(res), split={"train_runs": len(tr), "val_runs": len(va), "test_runs": len(te), "note": "chronological by run order; scaler fit on train only"},
                data="synthetic output of simulator/sim.py", note="GCN+LSTM only listed if PyTorch was available at training time")
    json.dump(meta, open(ART / "metadata.json", "w"), indent=1); json.dump({"models": res, "meta": meta}, open(ART / "evaluation.json", "w"), indent=1, default=float)
    print("ACTIVE:", active)

if __name__ == "__main__": main()
