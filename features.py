"""CPU-only public features. This module never reads gold, row IDs or responses."""
from __future__ import annotations
import hashlib
import json
import math
import re
import numpy as np

FEATURE_VERSION = "public-structural-hash-v1"
NAMES = ["intercept", "log_chars", "log_words", "digits", "non_ascii", "math", "code", "newlines",
         "negation", "quotes", "log_A", "log_B", "length_gap", "AB_overlap", "log_evidence", "log_candidate",
         "candidate_evidence_overlap", "number_mismatch", "log_messages", "tool_messages", "tool_errors",
         "step_fraction", "log_option_mean", "option_std_ratio"]
NAMES += [f"category_hash_{i}" for i in range(12)] + [f"word_hash_{i}" for i in range(16)]

def flat(value):
    if isinstance(value,str): return value
    if value is None: return ""
    return json.dumps(value,ensure_ascii=False,sort_keys=True)

def words(value):
    return re.findall(r"\w+",flat(value).lower())

def overlap(a,b):
    aa,bb=set(words(a)),set(words(b))
    return len(aa&bb)/max(1,len(aa|bb))

def hashed(values,dim):
    out=np.zeros(dim)
    for token in values:
        h=int.from_bytes(hashlib.blake2b(token.encode(),digest_size=8).digest(),"little")
        out[h%dim] += 1 if h & (1<<63) else -1
    return out/max(1.,float(np.linalg.norm(out)))

def vector(state, question, subset):
    # Signature intentionally has no gold/item-id argument. Options are public,
    # even if one of them happens to be the correct answer.
    text=flat(state)+" "+flat(question.get("criteria",{}))
    tokens=words(text)
    n=max(1,len(text))
    a=flat(state.get("candidate_A_conversation",""))
    b=flat(state.get("candidate_B_conversation",""))
    evidence=flat(state.get("evidence",""))
    candidate=flat(state.get("candidate",""))
    nums_c=set(re.findall(r"\d+",candidate)); nums_e=set(re.findall(r"\d+",evidence))
    messages=state.get("messages",[])
    if not isinstance(messages,list): messages=[]
    match=re.search(r"zero-based message index (\d+)",question.get("instructions",""))
    index=int(match.group(1)) if match else 0
    lengths=np.array([len(flat(x)) for x in question.get("criteria",{}).values()] or [0])
    structural=[1.,math.log1p(n),math.log1p(len(tokens)),sum(c.isdigit() for c in text)/n,
       sum(ord(c)>127 for c in text)/n,sum(c in "+=*/<>^∑∫√" for c in text)/n,
       sum(c in "{}[]`;" for c in text)/n,text.count("\n")/n,
       sum(t in {"not","no","never","none","false"} for t in tokens)/max(1,len(tokens)),
       text.count('"')/n,math.log1p(len(a)),math.log1p(len(b)),abs(len(a)-len(b))/max(1,len(a)+len(b)),
       overlap(a,b),math.log1p(len(evidence)),math.log1p(len(candidate)),overlap(candidate,evidence),
       len(nums_c-nums_e)/max(1,len(nums_c)),math.log1p(len(messages)),
       sum(m.get("role")=="tool" for m in messages)/max(1,len(messages)),
       sum("error" in flat(m).lower() for m in messages if m.get("role")=="tool")/max(1,len(messages)),
       index/max(1,len(messages)-1),math.log1p(float(lengths.mean())),float(lengths.std())/max(1,float(lengths.mean()))]
    categories=["subset="+str(subset),"language="+str(state.get("language",""))]
    return np.concatenate([structural,hashed(categories,12),hashed(tokens,16)])

def build_matrix(pool):
    # Explicit SQL projection prevents target/answer/ID leakage into the learner.
    rows=pool.execute("SELECT i.idx,i.question,i.subset,c.state FROM items i JOIN contexts c ON i.context_id=c.id ORDER BY i.idx")
    raw=[]
    for row in rows:
        if row["idx"] != len(raw): raise ValueError("Pool indices must be contiguous")
        raw.append(vector(json.loads(row["state"]),json.loads(row["question"]),row["subset"]))
    X=np.asarray(raw,dtype=np.float64)
    mean=X.mean(axis=0); scale=X.std(axis=0)
    scale=np.maximum(scale,.05)
    X[:,1:]=np.clip((X[:,1:]-mean[1:])/scale[1:],-5,5)
    X[:,0]=1
    return X,{"version":FEATURE_VERSION,"names":NAMES,"mean":mean.tolist(),"scale":scale.tolist(),
              "gold_access":False,"api_calls":0,"dimension":X.shape[1]}
