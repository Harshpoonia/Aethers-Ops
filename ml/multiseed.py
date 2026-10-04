"""Repeat the full experiment over several seeds (new simulated dataset + new model init each time) and report mean/std.
Does NOT touch the saved model artifacts. Run: python -m ml.multiseed --seeds 5 --epochs 15"""
import argparse, json, pathlib, sys, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from ml.data import *
from ml.train import metrics, pick_threshold, early_warning, ART
KEYS = ["precision", "recall", "f1", "pr_auc", "roc_auc", "fpr", "fnr"]

def one_seed(seed, n_runs, epochs, torch_models):
    runs = gen_runs(n_runs, seed); tr, va, te = chrono_split(runs); sc = fit_scaler(tr)
    (Xtr, Ytr, Vtr, _), (Xva, Yva, Vva, _), (Xte, Yte, Vte, mte) = (windows(r, sc) for r in (tr, va, te))
    ftr, fva, fte = flat(Xtr), flat(Xva), flat(Xte); mtr, mva, mt = Vtr.reshape(-1), Vva.reshape(-1), Vte.reshape(-1)
    ytr, yva, yte = Ytr.reshape(-1), Yva.reshape(-1), Yte.reshape(-1); preds = {}
    for name, m in {"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                    "random_forest": RandomForestClassifier(200, min_samples_leaf=3, class_weight="balanced", n_jobs=-1, random_state=seed)}.items():
        m.fit(ftr[mtr], ytr[mtr]); preds[name] = (m.predict_proba(fva)[:, 1], m.predict_proba(fte)[:, 1])
    if torch_models:
        from ml.torch_models import train_torch, predict_torch
        for name in torch_models:
            m = train_torch(name, Xtr, Ytr, Vtr, Xva, Yva, Vva, epochs=epochs, seed=seed)
            preds[name] = (predict_torch(m, Xva).reshape(-1), predict_torch(m, Xte).reshape(-1))
    out = {}
    for name, (pv, pt) in preds.items():
        thr = pick_threshold(yva[mva], pv[mva]); r = metrics(yte[mt], pt[mt], thr); ew = early_warning(te, mte, pt, Yte, Vte, thr)
        r["detected_pct"] = ew["pct_detected"]; r["mean_lead_min"] = ew["mean_lead_min"]; r["false_alert_episodes"] = ew["false_alert_episodes"]; out[name] = r
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--runs", type=int, default=CFG["training"]["n_runs"])
    ap.add_argument("--epochs", type=int, default=15); ap.add_argument("--models", default="lstm,gcn,gcn_lstm"); ap.add_argument("--first-seed", type=int, default=100)
    a = ap.parse_args(); tm = [x for x in a.models.split(",") if x and x != "none"]
    try: import torch
    except ImportError: tm = []; print("PyTorch not installed: only sklearn baselines")
    per = {}
    for k in range(a.seeds):
        seed = a.first_seed + k; print(f"=== seed {seed} ({k+1}/{a.seeds}) ===", flush=True); per[seed] = one_seed(seed, a.runs, a.epochs, tm)
        for n, r in per[seed].items(): print(f"  {n:20s} PR-AUC={r['pr_auc']:.3f} F1={r['f1']:.3f} recall={r['recall']:.3f} detected={r['detected_pct']:.2f}", flush=True)
    names = list(next(iter(per.values()))); summ = {}
    for n in names:
        summ[n] = {}
        for k in KEYS + ["detected_pct", "mean_lead_min"]:
            v = np.array([per[s][n][k] for s in per if per[s][n][k] is not None], float); summ[n][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0}
    cmp = None
    if "gcn_lstm" in names:
        base = [n for n in names if n != "gcn_lstm"]; wins = {}
        for b in base:
            d = [per[s]["gcn_lstm"]["pr_auc"] - per[s][b]["pr_auc"] for s in per]
            wins[b] = {"seeds_won": int(sum(x > 0 for x in d)), "seeds": len(d), "mean_pr_auc_diff": float(np.mean(d))}
        cmp = {"metric": "PR-AUC, gcn_lstm minus baseline, per seed", "vs": wins}
    res = {"seeds": list(per), "runs_per_seed": a.runs, "epochs": a.epochs, "summary": summ, "gcn_lstm_vs_baselines": cmp, "per_seed": per}
    json.dump(res, open(ART / "multiseed.json", "w"), indent=1, default=float)
    print("\nMEAN ± STD over", len(per), "seeds")
    for n in names: print(f"{n:20s} " + "  ".join(f"{k}={summ[n][k]['mean']:.3f}±{summ[n][k]['std']:.3f}" for k in ("pr_auc", "f1", "recall", "detected_pct")))
    if cmp:
        for b, w in cmp["vs"].items(): print(f"gcn_lstm vs {b}: won PR-AUC on {w['seeds_won']}/{w['seeds']} seeds, mean diff {w['mean_pr_auc_diff']:+.3f}")
if __name__ == "__main__": main()
