#!/usr/bin/env python3
"""Additional transfer stress: frozen winner, 52 features, larger N, ±2pp.

Does not select or change the release configuration. Extra columns are noise,
so this deliberately tests the 7-to-52-dimensional transfer cost.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from celeus import Settings
from validate_synthetic import make_pool, simulate, summarize, write_json, core_hash, SCENARIOS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",type=Path,default=Path(__file__).parent/"validation/release/selected_celeus.json")
    p.add_argument("--out",type=Path,default=Path("validation/transfer_stress"))
    p.add_argument("--seeds",type=int,default=20)
    p.add_argument("--n",type=int,default=1600)
    p.add_argument("--epsilon",type=float,default=.02)
    args=p.parse_args()
    if args.seeds<1 or args.n<50 or not 0<args.epsilon<=.5:p.error("Invalid settings")
    cfg=json.loads(args.selection.read_text())
    if cfg["core_sha256"]!=core_hash() or cfg["audit_pass"] is not True:raise ValueError("Stale/failed selected config")
    if args.out.exists():raise FileExistsError("Use a new output directory")
    args.out.mkdir(parents=True)
    methods={"evalue":None,"celeus_uniform":Settings(),cfg["name"]:Settings(**cfg["settings"])}
    rows=[]
    with threadpool_limits(limits=1):
        for j,scenario in enumerate(SCENARIOS):
            for k in range(args.seeds):
                seed=410000+j*1000+k
                X,y=make_pool(scenario,args.n,seed)
                extra=np.random.default_rng(seed+99000).normal(size=(args.n,45))
                X=np.column_stack([X,extra])
                for name,settings in methods.items():
                    result=simulate(X,y,settings,seed+700000,.0125,args.epsilon,10)
                    rows.append({"phase":"transfer_stress","method":name,"scenario":scenario,"seed":seed,**result})
                if (k+1)%5==0:print(scenario,k+1,flush=True)
    summary=summarize(rows,.0125)
    write_json(args.out/"protocol.json",{"n":args.n,"seeds_per_scenario":args.seeds,"dimension":52,
        "epsilon_half_width":args.epsilon,"alpha_per_benchmark":.0125,"selection_frozen":cfg["name"],
        "seed_start":410000,"selected_configuration_unchanged":True,"core_sha256":core_hash(),"api_calls":0})
    write_json(args.out/"summary.json",summary);write_json(args.out/"trials.json",rows)
    print("Stress results saved; release selection was not changed",flush=True)

if __name__=="__main__":main()
