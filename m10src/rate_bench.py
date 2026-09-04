import os, time, json
os.environ.setdefault("HF_HUB_OFFLINE","1")
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
DEV="cuda"; torch.backends.cuda.matmul.allow_tf32=True
class Nano(nn.Module):
    def __init__(s, layers=(12,8,4)):
        super().__init__(); s.bb=AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
        s.layers=layers; s.head=nn.Linear(384*len(layers),1024)
    def forward(s, ii, am):
        hs=s.bb(input_ids=ii,attention_mask=am,output_hidden_states=True).hidden_states
        v=s.head(torch.cat([hs[i] for i in s.layers],-1))
        m=am.unsqueeze(-1).to(v.dtype); v=(v*m).sum(1)/m.sum(1).clamp(min=1)
        return F.normalize(v.float(),dim=-1,eps=1e-12)
def bench(model,opt,chunks,steps=300,warm=20,label=""):
    torch.cuda.reset_peak_memory_stats(); r0=torch.cuda.memory_stats().get("num_alloc_retries",0)
    nex=sum(n for n,_ in chunks); ntok=sum(n*L for n,L in chunks)
    tgt={n:F.normalize(torch.randn(n,1024,device=DEV),dim=-1) for n,_ in chunks}
    for it in range(warm+steps):
        if it==warm: torch.cuda.synchronize(); t0=time.time()
        opt.zero_grad(set_to_none=True)
        for n,L in chunks:
            ii=torch.randint(1000,20000,(n,L),device=DEV); am=torch.ones(n,L,dtype=torch.long,device=DEV)
            with torch.autocast("cuda",dtype=torch.bfloat16): v=model(ii,am)
            (((v-tgt[n])**2).sum(-1).mean()*(n/nex)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    torch.cuda.synchronize(); dt=time.time()-t0; sps=steps/dt
    retries=torch.cuda.memory_stats().get("num_alloc_retries",0)-r0
    peak=torch.cuda.max_memory_allocated()/2**30
    print(f"{label:34s} {sps:7.2f} st/s {sps*nex:8.0f} ex/s {sps*ntok:9.0f} tok/s "
          f"peak {peak:5.2f}GB retries {retries:3d}  200M->{200e6/(sps*nex)/3600:6.1f}h",flush=True)
    return dict(steps_per_s=round(sps,2),examples_per_s=round(sps*nex,1),padded_tok_per_s=round(sps*ntok),
                peak_gb=round(peak,2),alloc_retries=retries,gpu_hours_200m=round(200e6/(sps*nex)/3600,1))
m=Nano().to(DEV); opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=0.01,fused=True)
res={}
res["bs32_query_bucket_64"]      = bench(m,opt,[(32,64)],  label="bs32 query bucket (32x64)")
res["bs32_doc_bucket_256"]       = bench(m,opt,[(32,256)], label="bs32 DOC bucket (32x256)  <-- new")
res["bs32_mixed_pad_128"]        = bench(m,opt,[(32,128)], label="bs32 mixed padded (32x128)")
res["bs64_query_bucket_64"]      = bench(m,opt,[(64,64)],  label="bs64 query bucket (64x64)  <-- new")
res["bs128_query_bucket_64"]     = bench(m,opt,[(128,64)], label="bs128 query bucket (128x64)")
res["bs128_doc_bucket_256"]      = bench(m,opt,[(128,256)],label="bs128 DOC bucket (128x256)")
res["bs512_query_bucket_64"]     = bench(m,opt,[(512,64)], label="bs512 query bucket (512x64)")
print(json.dumps(res,indent=1))
