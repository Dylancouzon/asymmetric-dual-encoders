#!/usr/bin/env python3
"""Reconcile the reviewed 144-hour build cap against live balance; never rent or edit config."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone

REPO=Path('/home/dylan/asymetric-dual-encoders/work/m13cloud')
KEYS=Path('/home/dylan/.config/runpod')
OUTPUT=REPO/'results/m13_build_allocation.json'
REQUIRED=('m13/LOTTE_GATE.json','m13/LOTTE_GATE_MANIFEST.json','results/m8_lotte_pin.json',
 'm13/LOTTE_GATE_REGISTRATION.json','m13/build_config.json','m10/screen_registry.json',
 'results/m10_screen_verdicts.json','results/m13_cloud_gate_chain_upload.json',
 'm13/build_transfer_manifest.json','results/m10_arm_E-bs32.json',
 'results/m13_encode_benchmark_fp16.json','results/m13_gate_chain_allocation.json')

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 if OUTPUT.exists(): raise RuntimeError('Existing allocation must not be overwritten')
 subprocess.run(['git','diff','--quiet','HEAD'],cwd=REPO,check=True)
 subprocess.run(['git','ls-files','--error-unmatch',*REQUIRED],cwd=REPO,check=True,stdout=subprocess.DEVNULL)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 pushed=subprocess.check_output(['git','ls-remote','--exit-code','origin','refs/heads/m13-stage1-execution-prep'],cwd=REPO,text=True,timeout=30).split()[0]
 if head!=pushed: raise RuntimeError('Require committed and pushed gate prerequisites')
 bound={p:sha(REPO/p) for p in REQUIRED}
 load=lambda p:json.loads((REPO/p).read_text())
 gate=load('results/m13_cloud_gate_chain_upload.json')
 if gate.get('status')!='PASSED' or gate.get('pod_final_status')!='EXITED' or not gate.get('backup_verified'): raise RuntimeError('Require completed verified gate chain')
 pod_id=gate.get('pod_id')
 if pod_id!='wnzk8eeqrrkw4m': raise RuntimeError('Unreviewed build Pod')
 if load('results/m10_screen_verdicts.json')['selected']['batch']!='bs32': raise RuntimeError('Unreviewed selected branch')
 measured=load('results/m10_arm_E-bs32.json')['throughput_ex_per_s']
 if measured!=883.8: raise RuntimeError('Measured rate changed')
 benchmark=load('results/m13_encode_benchmark_fp16.json')
 if benchmark.get('status')!='PASSED' or benchmark.get('encode_dtype')!='fp16' or sha(REPO/'results/m13_encode_benchmark_fp16.json')!=gate.get('benchmark_sha256'): raise RuntimeError('Require verified admitted fp16 timing')
 headers={'Authorization':'Bearer '+(KEYS/'api_key').read_text().strip(),'Content-Type':'application/json','User-Agent':'m13-build-allocation/1.0'}
 def get(url,body=None):
  req=urllib.request.Request(url,headers=headers,data=None if body is None else json.dumps(body).encode())
  with urllib.request.urlopen(req,timeout=30) as f:return json.load(f)
 pod=get('https://rest.runpod.io/v1/pods/'+pod_id)
 if pod.get('desiredStatus')!='EXITED' or not 0<float(pod['costPerHr'])<=1.59: raise RuntimeError('Require stopped approved-price Pod')
 balance=float(get('https://api.runpod.io/graphql',{'query':'query { myself { clientBalance } }'})['data']['myself']['clientBalance'])
 prior=load('results/m13_gate_chain_allocation.json')
 baseline=prior['account_balance_usd']+prior['spent_to_date_usd']
 if not math.isclose(baseline,505,abs_tol=1e-6) or not math.isfinite(balance) or balance<0 or balance>baseline: raise RuntimeError('Account funding changed; ledger reconciliation required')
 spent=baseline-balance
 if spent+1e-6<prior['spent_to_date_usd']: raise RuntimeError('Account funding increased; ledger reconciliation required')
 cfg=load('m13/build_config.json')
 if cfg['budget']['fixed_usd']['persistent_disk_and_egress']!=309: raise RuntimeError('Unreviewed storage reserve')
 sys.path.insert(0,str(REPO/'m13src'))
 import build_lock as BL
 price=1.59+530*.10/720; rate=measured/2; hours=144
 model=BL.allocation(cfg,rate,price)
 extra=max(0,hours-model['mandatory_hours']['build'])
 projected=model['committed_usd']+spent+(2+extra)*price
 cap=math.ceil(hours*price*100)/100
 if projected>1000 or cap>balance or cap>1000-spent: raise RuntimeError('Build and all mandatory reserves exceed available funds')
 if any(sha(REPO/p)!=h for p,h in bound.items()): raise RuntimeError('Allocation inputs changed during reconciliation')
 result={'status':'PASSED','measured_utc':datetime.now(timezone.utc).isoformat(),
  'budget_ceiling_usd':1000,'pod_id':pod_id,'ssh_config':str(KEYS/'m13_gate_chain_ssh_config'),'ssh_alias':'m13-gate-chain',
  'max_hours':hours,'max_cost_usd':cap,'rate_ex_per_s':rate,'selected_training_rate_ex_s':measured,
  'total_price_usd_per_hour':price,'remaining_budget_usd':1000-spent,'account_balance_usd':balance,
  'spent_to_date_usd':spent,'funding_baseline_usd':baseline,'model_allocation':model,
  'prior_setup_supplement_hours':2,'build_supplement_hours':extra,
  'projected_total_usd':math.ceil(projected*100)/100,'remaining_headroom_usd':math.floor((1000-projected)*100)/100,
  'limitations':'Conservative ceiling, not expected duration or spend. Existing E and LoTTE model lines remain even though completed stages enter paid spend. Three-disk 30-day reserve and hourly running storage both retained. No extensions or new Pod creation.',
  'artifact_sha256':bound,'prerequisite_commit':head,'allocation_script_sha256':sha(__file__)}
 with OUTPUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:result[k] for k in ('status','max_hours','max_cost_usd','account_balance_usd','projected_total_usd','remaining_headroom_usd')}))

if __name__=='__main__':main()
