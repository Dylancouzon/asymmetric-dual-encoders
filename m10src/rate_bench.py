"""Throughput microbenchmark: is the M10 build overhead-bound at batch 32?

Random token ids only. No corpus, no protected path, no teacher. Measures steps/s for the
M10 recipe shape (bge-small + 3-layer pooled 1152->1024 head) at the mandate's batch 32
under the 75/25 example mix, and prices two cheap trainer changes.
"""
import os, time, sys, json
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Nano(nn.Module):
    def __init__(self, layers=(12, 8, 4)):
        super().__init__()
        self.bb = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        self.layers = layers
        self.head = nn.Linear(384 * len(layers), 1024)
    def forward(self, ii, am):
        hs = self.bb(input_ids=ii, attention_mask=am, output_hidden_states=True).hidden_states
        feat = torch.cat([hs[i] for i in self.layers], dim=-1)      # per token
        v = self.head(feat)                                          # per token, exportable
        m = am.unsqueeze(-1).to(v.dtype)
        v = (v * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(v.float(), dim=-1, eps=1e-12)

def bench(model, opt, chunks, steps=60, warm=10, label=""):
    tgt = {n: F.normalize(torch.randn(n, 1024, device=DEV), dim=-1) for n, _ in chunks}
    nex = sum(n for n, _ in chunks)
    for it in range(warm + steps):
        if it == warm:
            torch.cuda.synchronize(); t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for n, L in chunks:
            ii = torch.randint(1000, 20000, (n, L), device=DEV)
            am = torch.ones(n, L, dtype=torch.long, device=DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                v = model(ii, am)
            loss = ((v - tgt[n]) ** 2).sum(-1).mean() * (n / nex)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = steps / dt
    print(f"{label:38s} {sps:7.2f} steps/s  {sps*nex:9.0f} ex/s  "
          f"200M ex -> {200e6/(sps*nex)/3600:6.1f} GPU-h", flush=True)
    return sps * nex

model = Nano().to(DEV)
print(f"params {sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)
res = {}
# mandate shape: batch 32 examples, 75/25 mix -> 24 queries + 8 documents, chunked as M9 chunks
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
res["bs32_2chunk_plainadamw"] = bench(model, opt, [(24, 64), (8, 256)], label="bs32 75/25, 2 chunks, AdamW")
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01, fused=True)
res["bs32_2chunk_fused"] = bench(model, opt, [(24, 64), (8, 256)], label="bs32 75/25, 2 chunks, fused AdamW")
res["bs32_1chunk_fused"] = bench(model, opt, [(32, 128)], label="bs32 single padded chunk(128), fused")
res["bs32_q64_fused"] = bench(model, opt, [(32, 64)], label="bs32 all-query len64, fused")
res["bs128_fused"] = bench(model, opt, [(96, 64), (32, 256)], label="bs128 75/25, fused AdamW")
res["bs512_fused"] = bench(model, opt, [(384, 64), (128, 256)], label="bs512 75/25, fused AdamW")
print(json.dumps(res, indent=1))
