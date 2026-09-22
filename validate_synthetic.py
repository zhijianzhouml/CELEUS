#!/usr/bin/env python3
"""Independent offline validation and selection. NEVER imports API code.

Development seeds select the fastest eligible CELEUS candidate. Disjoint audit
seeds report pathwise coverage and cost, never re-rank. Small exact path checks
live in test_celeus.py. Synthetic performance is NOT evidence of savings on a real model.
"""
from __future__ import annotations
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np
import scipy
from scipy.special import expit
from scipy.stats import beta as beta_dist, binom
from celeus import CELEUS, Settings
from evalue import BinaryWORCS, EmptyConfidenceSet

SCENARIOS=("linear","heterogeneous","nonlinear","uninformative","rare_errors","clustered")
CANDIDATES={
    "celeus_uniform":Settings(acquisition="uniform"),
    "celeus_mix04":Settings(acquisition="mixture",beta=.4),
    "celeus_water04":Settings(acquisition="waterfill",beta=.4),
    "celeus_water08":Settings(acquisition="waterfill",beta=.8),
    "celeus_guarded_ig":Settings(acquisition="guarded_ig",beta=.4),
}
CONTROLS={"evalue":None,"corrected_no_surrogate":Settings(surrogate=False)}
ROOT=Path(__file__).resolve().parent

def core_hash():
    h=hashlib.sha256()
    for name in ("celeus.py","evalue.py","features.py","validate_synthetic.py"):
        h.update(name.encode());h.update((ROOT/name).read_bytes())
    return h.hexdigest()

def make_pool(scenario,N,seed):
    rng=np.random.default_rng(seed)
    x=rng.normal(size=(N,6))
    if scenario=="linear":
        p=expit(3*x[:,0]-2*x[:,1]-.6)
    elif scenario=="heterogeneous":
        group=rng.random(N)<.75
        x[:,0]=np.where(group,-2.,2.)
        p=np.where(group,.015,.5)
    elif scenario=="nonlinear":
        p=np.where(x[:,0]*x[:,1]>0,.92,.08)
    elif scenario=="uninformative":
        p=np.full(N,.30)
    elif scenario=="rare_errors":
        p=expit(-4.7+1.2*x[:,0])
    elif scenario=="clustered":
        group=rng.integers(0,12,N)
        effects=rng.normal(0,2,12)
        x[:,0]=group/6-1
        p=expit(effects[group]+x[:,1])
    else:
        raise ValueError(scenario)
    # Generate fixed binary outcomes ONCE. The policy receives X only.
    y=rng.binomial(1,p)
    return np.column_stack([np.ones(N),x]),y

def oracle_for(y):
    paid=set()
    def query(index):
        if index in paid: raise AssertionError("Duplicate WOR query")
        paid.add(index)
        return int(y[index])
    return query

def simulate(X,y,settings,seed,alpha,epsilon,check_every):
    N=len(y)
    truth=float(y.mean())
    oracle=oracle_for(y)
    covered=True; empty=False; stop=None; widths=[]; mix=[]
    start=time.perf_counter()
    if settings is None:
        cs=BinaryWORCS(N,str(alpha))
        order=np.random.default_rng(seed).permutation(N)
        for t,i in enumerate(order,1):
            try: cs.update(oracle(int(i)),invert=t%check_every==0)
            except EmptyConfidenceSet: empty=True;covered=False;break
            lo,hi=cs.interval
            covered &= lo-1e-9 <= truth <= hi+1e-9
            widths.append(hi-lo)
            if stop is None and hi-lo<=2*epsilon: stop=t
        count=cs.t
    else:
        engine=CELEUS(X,settings,alpha=alpha,epsilon=epsilon,seed=seed,check_every=check_every)
        for t in range(1,N+1):
            plan=engine.plan()
            mix.append(plan.mix_weight)
            try: engine.observe(plan,oracle(plan.index))
            except EmptyConfidenceSet: empty=True;covered=False;break
            lo,hi=engine.cs.interval
            covered &= lo-1e-9 <= truth <= hi+1e-9
            widths.append(hi-lo)
            if stop is None and hi-lo<=2*epsilon: stop=t
        count=engine.cs.t
    # Continue through census after the stopping time solely for stronger
    # pathwise auditing. Report first-stop cost, not these offline audit calls.
    return {"first_stop":stop if stop is not None else N,"reached_precision":stop is not None,
            "path_covered":bool(covered),"empty":empty,"truth":truth,
            "stop_fraction":(stop if stop is not None else N)/N,
            "mean_width":float(np.mean(widths)) if widths else 1.,
            "adaptive_fraction":float(np.mean(np.array(mix)>0)) if mix else 0.,
            "audit_steps":count,"cpu_seconds_full_path":time.perf_counter()-start}

def cp_interval(failures,runs):
    low=0. if failures==0 else float(beta_dist.ppf(.025,failures,runs-failures+1))
    high=1. if failures==runs else float(beta_dist.ppf(.975,failures+1,runs-failures))
    return [low,high]

def summarize(rows,alpha):
    out=[]
    cells=sorted({(r["phase"],r["method"],r["scenario"]) for r in rows})
    for phase,method,scenario in cells:
        rr=[r for r in rows if (r["phase"],r["method"],r["scenario"])==(phase,method,scenario)]
        fail=sum(not r["path_covered"] for r in rr)
        out.append({"phase":phase,"method":method,"scenario":scenario,"runs":len(rr),
                    "failures":fail,"path_coverage":1-fail/len(rr),"failure_95pct_CP":cp_interval(fail,len(rr)),
                    "violation_pvalue":float(binom.sf(fail-1,len(rr),alpha)),
                    "mean_queries":float(np.mean([r["first_stop"] for r in rr])),
                    "median_queries":float(np.median([r["first_stop"] for r in rr])),
                    "p90_queries":float(np.quantile([r["first_stop"] for r in rr],.9)),
                    "precision_rate":float(np.mean([r["reached_precision"] for r in rr])),
                    "empty_count":sum(r["empty"] for r in rr),
                    "mean_cpu_seconds_full_path":float(np.mean([r["cpu_seconds_full_path"] for r in rr]))})
    return out

def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)

def run(args):
    out=args.out
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Validation output must be new; keep the prior selection frozen")
    out.mkdir(parents=True,exist_ok=True)
    protocol={"N":args.n,"alpha_per_benchmark":args.alpha,"epsilon_half_width":args.epsilon,
              "check_every":args.check_every,"dev_seeds":args.dev_seeds,"audit_seeds":args.audit_seeds,
              "development_seed_start":11000,"audit_seed_start":81000,"scenarios":SCENARIOS,
              "candidate_settings":{k:asdict(v) for k,v in CANDIDATES.items()},
              "selection_metric":"macro mean first-stop fraction; CELEUS candidates only; coverage screening first",
              "coverage_screen":"one-sided exact binomial p < .01/(methods*scenarios) flags an implementation concern; passing does NOT prove coverage",
              "core_sha256":core_hash(),"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
              "synthetic_only":True,"extra_llm_calls":0,"model_calls":0}
    write_json(out/"protocol.json",protocol)
    rows=[]
    selected=None
    for phase,seeds,offset in [("development",args.dev_seeds,11000),("audit",args.audit_seeds,81000)]:
        methods={**CONTROLS,**CANDIDATES} if phase=="development" else {**CONTROLS,selected:CANDIDATES[selected],"celeus_uniform":CANDIDATES["celeus_uniform"]}
        for sc_index,scenario in enumerate(SCENARIOS):
            for k in range(seeds):
                seed=offset+1000*sc_index+k
                X,y=make_pool(scenario,args.n,seed)
                for name,settings in methods.items():
                    result=simulate(X,y,settings,seed+700000,args.alpha,args.epsilon,args.check_every)
                    rows.append({"phase":phase,"method":name,"scenario":scenario,"seed":seed,**result})
                if (k+1)%5==0 or k+1==seeds:
                    print(f"{phase} {scenario}: {k+1}/{seeds}",flush=True)
            write_json(out/"progress.json",{"phase":phase,"last_scenario":scenario,"runs":len(rows)})
        summary=summarize(rows,args.alpha)
        if phase=="development":
            eligible=[]
            screen=.01/(len(methods)*len(SCENARIOS))
            for name in CANDIDATES:
                cells=[r for r in summary if r["phase"]==phase and r["method"]==name]
                if all(r["violation_pvalue"]>=screen for r in cells):
                    eligible.append((float(np.mean([r["mean_queries"] for r in cells])),name))
            if not eligible:
                raise RuntimeError("All CELEUS candidates failed the development coverage screen")
            selected=min(eligible)[1]
            print("FROZEN SELECTION:",selected,flush=True)
            write_json(out/"development_selection.json",{"name":selected,"settings":asdict(CANDIDATES[selected]),"ranking":sorted(eligible)})
    summary=summarize(rows,args.alpha)
    audit=[r for r in summary if r["phase"]=="audit"]
    audit_pass=all(r["violation_pvalue"]>=.01/len(audit) for r in audit)
    selected_config={"schema_version":1,"name":selected,"settings":asdict(CANDIDATES[selected]),
                     "selection_source":"synthetic_development_only","audit_pass":audit_pass,
                     "core_sha256":core_hash(),"protocol":protocol,
                     "scope":"Best of tested candidates on this synthetic suite only; no guarantee of superiority on a real model"}
    write_json(out/"selected_celeus.json",selected_config)
    write_json(out/"summary.json",summary)
    write_json(out/"trials.json",rows)
    with (out/"trials.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=["# Synthetic validation report", "", f"Frozen CELEUS selection: **{selected}**. Audit screen: **{'PASS' if audit_pass else 'FAIL'}**.",
           "", "Synthetic only: 0 model calls, 0 extra LLM calls. Development selects; audit never re-ranks.",
           "Coverage is checked over every reported interval through census, including times after first precision stopping.",
           "Monte Carlo checks can detect failures; they do not prove validity. CP intervals quantify simulation uncertainty.",
           "", "| Phase | Method | Scenario | Runs | Mean first-stop queries | Path coverage | Failure 95% CP |",
           "|---|---|---|---:|---:|---:|---|" ]
    for r in summary:
        lines.append(f"| {r['phase']} | {r['method']} | {r['scenario']} | {r['runs']} | {r['mean_queries']:.1f} | {r['path_coverage']:.3f} | {r['failure_95pct_CP'][0]:.4f}, {r['failure_95pct_CP'][1]:.4f} |")
    lines += ["", "Timing includes full-path offline auditing after first stop, not production early-stop runtime.",
              "The logistic synthetic feature dimension is 7; real structural/hash features have dimension 52. Transfer is unverified.",
              "Do not treat no detected excess failures as evidence that an arbitrary API has fixed, order-independent outcomes."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Saved {out}/selected_celeus.json; audit_pass={audit_pass}",flush=True)
    return audit_pass

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,default=Path("validation/new"))
    p.add_argument("--n",type=int,default=800)
    p.add_argument("--dev-seeds",type=int,default=12)
    p.add_argument("--audit-seeds",type=int,default=100)
    p.add_argument("--alpha",type=float,default=.0125)
    p.add_argument("--epsilon",type=float,default=.05)
    p.add_argument("--check-every",type=int,default=10)
    a=p.parse_args()
    if min(a.n,a.dev_seeds,a.audit_seeds,a.check_every)<1 or a.n<50 or not 0<a.alpha<1 or not 0<a.epsilon<=.5:
        p.error("Invalid simulation settings")
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1): passed=run(a)
    except ImportError:
        passed=run(a)
    raise SystemExit(0 if passed else 2)

if __name__=="__main__":main()
