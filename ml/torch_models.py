"""GCN / LSTM / GCN+LSTM in plain PyTorch (dense normalised adjacency; mathematically a standard
Kipf-Welling GCN layer, so PyTorch Geometric is not required for a 6-node graph)."""
import torch, torch.nn as nn
from simulator.sim import adjacency, N, F

class Net(nn.Module):
    def __init__(self, use_graph=True, use_lstm=True, hid=32):
        super().__init__()
        self.g, self.l = use_graph, use_lstm
        self.register_buffer("A", torch.tensor(adjacency()))
        self.emb = nn.Embedding(N, 8)
        self.inp = nn.Linear(F + 8, hid); self.gcn = nn.Linear(hid, hid)
        self.lstm = nn.LSTM(hid, hid, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hid * 2, hid), nn.ReLU(), nn.Linear(hid, 1))

    def forward(self, x):                      # x [B,W,N,F]
        B, W = x.shape[:2]
        e = self.emb.weight[None, None].expand(B, W, N, 8)
        h = torch.relu(self.inp(torch.cat([x, e], -1)))          # [B,W,N,H]
        z = h
        if self.g: z = torch.relu(torch.einsum("ij,bwjh->bwih", self.A, self.gcn(h))) + h  # GCN + skip
        if self.l:
            s = z.permute(0, 2, 1, 3).reshape(B * N, W, -1); o, _ = self.lstm(s); t = o[:, -1].reshape(B, N, -1)
        else: t = z[:, -1]
        return self.out(torch.cat([t, h[:, -1]], -1)).squeeze(-1)  # logits [B,N]

VARIANTS = {"lstm": dict(use_graph=False, use_lstm=True), "gcn": dict(use_graph=True, use_lstm=False),
            "gcn_lstm": dict(use_graph=True, use_lstm=True)}

def train_torch(name, Xtr, Ytr, Vtr, Xva, Yva, Vva, epochs=30, seed=0):
    from sklearn.metrics import average_precision_score
    torch.manual_seed(seed); m = Net(**VARIANTS[name]); opt = torch.optim.Adam(m.parameters(), 2e-3)
    pw = torch.tensor(float((1 - Ytr[Vtr]).sum() / max(Ytr[Vtr].sum(), 1)))
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    Xt, Yt, Vt = map(torch.tensor, (Xtr, Ytr, Vtr.astype("float32"))); best, bs = -1, None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 128):
            b = perm[i:i + 128]; l = (lossf(m(Xt[b]), Yt[b]) * Vt[b]).sum() / Vt[b].sum()
            opt.zero_grad(); l.backward(); opt.step()
        p = predict_torch(m, Xva); sc = average_precision_score(Yva[Vva], p[Vva])
        if sc > best: best, bs = sc, {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(bs); return m

@torch.no_grad()
def predict_torch(m, X):
    m.eval(); return torch.cat([torch.sigmoid(m(torch.tensor(X[i:i + 512]))) for i in range(0, len(X), 512)]).numpy()
