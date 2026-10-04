import json, pathlib, joblib, numpy as np
from ml.data import scale, flat
from ml.preprocessing import sanitize
ART = pathlib.Path(__file__).parent / "artifacts"

class Predictor:
    """Loads a pre-trained artifact (never retrains). Input: raw window [W,N,F]; output: probs [N]."""
    def __init__(self):
        self.meta = json.load(open(ART / "metadata.json")); self.sc = json.load(open(ART / "scaler.json"))
        self.name = self.meta["active_model"]; self.version = self.meta["model_version"]; self.threshold = self.meta["threshold"]
        if self.name in ("lstm", "gcn", "gcn_lstm"):
            import torch; from ml.torch_models import Net, VARIANTS
            self.m = Net(**VARIANTS[self.name]); self.m.load_state_dict(torch.load(ART / f"{self.name}.pt")); self.m.eval(); self.torch = torch
        else: self.m = joblib.load(ART / f"{self.name}.joblib")

    def predict(self, win):
        x = scale(sanitize(win), self.sc)[None]
        if self.name in ("lstm", "gcn", "gcn_lstm"):
            with self.torch.no_grad(): return self.torch.sigmoid(self.m(self.torch.tensor(x)))[0].numpy()
        return self.m.predict_proba(flat(x))[:, 1]
