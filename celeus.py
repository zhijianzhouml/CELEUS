"""No-extra-LLM, finite-pool CELEUS. Only NumPy/SciPy; no API or gold-pool access.

All predictions, proposals, supports and stakes are frozen BEFORE querying.
CS targets binary LOSS; the pipeline displays its reflected accuracy interval.
See docs/METHOD_REVIEW.md for the proof and numerical contract.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.special import expit
from scipy.linalg import cho_factor, cho_solve
from evalue import EmptyConfidenceSet

@dataclass(frozen=True)
class Settings:
    acquisition: str = "uniform"
    beta: float = 0.4
    ridge: float = 1.0
    refit_every: int = 16
    warmup: int = 32
    half_life: float = 64.0
    cap: float = 0.9
    surrogate: bool = True

    def __post_init__(self):
        if self.acquisition not in {"uniform", "mixture", "waterfill", "guarded_ig"}:
            raise ValueError("Unknown acquisition")
        if not 0 < self.beta <= 1 or not 0 < self.cap < 1:
            raise ValueError("Invalid beta/cap")
        if self.ridge <= 0 or self.refit_every < 1 or self.warmup < 0 or self.half_life <= 0:
            raise ValueError("Invalid learner settings")

    def dictionary(self):
        return asdict(self)


class LogisticSurrogate:
    """Regularized logistic MAP + Laplace covariance, with intercept in X[:,0].

    Labels supplied to fit() must be queried losses only. Unweighted conditional
    likelihood is a working model, not a claim of design-unbiased fitting.
    Misspecification affects efficiency but cannot invalidate the corrected CS.
    """
    def __init__(self, dimension, ridge=1.0):
        self.precision = np.full(dimension, ridge, dtype=float)
        self.precision[0] = 0.25
        self.theta = np.zeros(dimension)
        self.covariance = np.diag(1 / self.precision)

    def fit(self, X, y):
        if len(y) == 0:
            return
        theta = self.theta.copy()
        def objective(w):
            z = X @ w
            return float(np.sum(np.logaddexp(0, z) - y*z) + .5*np.dot(self.precision*w, w))
        for _ in range(25):
            p = expit(X @ theta)
            grad = X.T @ (p-y) + self.precision*theta
            if np.max(np.abs(grad)) < 1e-6:
                break
            H = (X.T * (p*(1-p))) @ X + np.diag(self.precision)
            step = cho_solve(cho_factor(H), grad)
            old = objective(theta)
            rate = 1.0
            for _ in range(18):
                trial = theta-rate*step
                if objective(trial) <= old + 1e-12:
                    theta = trial
                    break
                rate *= .5
            else:
                break
        p = expit(X @ theta)
        H = (X.T*(p*(1-p))) @ X + np.diag(self.precision)
        self.theta = theta
        self.covariance = cho_solve(cho_factor(H), np.eye(len(theta)))

    def predict(self, X):
        z = X @ self.theta
        latent = np.maximum(np.einsum("ij,ij->i", X @ self.covariance, X), 0)
        # Logistic-normal approximation; NOT an exact posterior or guarantee.
        g = np.clip(expit(z/np.sqrt(1+math.pi*latent/8)), .005, .995)
        return g, latent


def waterfill(scores, beta):
    """Minimize sum scores_i^2/q_i subject to q_i >= beta/n.

    This is the floor-constrained optimum for a WORKING residual second moment.
    Uniform mixture is feasible but is generally not this constrained optimum.
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n == 0 or not np.all(np.isfinite(scores)) or np.any(scores < 0) or not 0 < beta <= 1:
        raise ValueError("Invalid waterfill inputs")
    floor = beta/n
    if beta == 1 or scores.sum() == 0:
        return np.full(n, 1/n)
    q = np.full(n, floor)
    active = np.ones(n, dtype=bool)
    while True:
        mass = 1-floor*np.count_nonzero(~active)
        proposal = mass*scores[active]/scores[active].sum()
        below = proposal < floor
        if not below.any():
            q[active] = proposal
            break
        active[np.flatnonzero(active)[below]] = False
    # Correct only final roundoff while preserving the floor.
    q[np.argmax(q)] += 1-float(q.sum())
    return q


def support(g, q, total_loss, N):
    center = (total_loss + float(np.sum(g)))/N
    a = center-float(np.max(g/(N*q)))
    b = center+float(np.max((1-g)/(N*q)))
    # A small outward pad is conservative for IEEE arithmetic.
    pad = 1e-12*(1+abs(a)+abs(b))
    return center, a-pad, b+pad


def stakes(h, a, b, u, c):
    plus = np.minimum(h, c/np.maximum(np.asarray(u)-a, 1e-300))
    minus = np.minimum(h, c/np.maximum(b-np.asarray(u), 1e-300))
    return plus, minus


class BettingCS:
    """Two monotone one-sided wealths; outward continuous bisection.

    No coarse parameter grid; each old support/stake stays frozen. Rejecting
    both tails at 2/alpha gives a union bound. Completion gives exact census.
    Empty intersections raise, never reset to [0,1].
    """
    def __init__(self, N, alpha=.05, epsilon=.02, half_life=64, cap=.9):
        if N < 1 or not 0 < alpha < 1 or not 0 < epsilon <= .5 or not 0 < cap < 1:
            raise ValueError("Invalid CS parameters")
        self.N, self.alpha, self.epsilon = int(N), float(alpha), float(epsilon)
        self.half_life, self.cap = half_life, cap
        self.t, self.total_loss = 0, 0
        self.lo, self.hi = 0., 1.
        self.history = np.empty((N, 5), dtype=float) # z,a,b,h,remaining_before
        self.threshold = math.log(2/alpha)
        self.roundoff_guard = 1e-9

    @property
    def interval(self):
        return self.lo, self.hi

    @property
    def accuracy_interval(self):
        return 1-self.hi, 1-self.lo

    @property
    def width(self):
        return self.hi-self.lo

    def base_stake(self):
        n = self.N-self.t
        if not self.t:
            variance = .25
        else:
            z, _, _, _, ns = self.history[:self.t].T
            w = np.exp2(-np.arange(self.t-1,-1,-1)/self.half_life)
            mu = np.dot(w,z)/w.sum()
            variance = ((n/self.N)**2*.25 + np.dot(w*(n/ns)**2,(z-mu)**2))/(w.sum()+1)
        # EW64D is a predictable heuristic; depletion is NOT an exact variance
        # correction when the residual distribution / learner changes.
        return self.epsilon/(variance+self.epsilon**2)

    def log_wealth(self, u, side):
        z,a,b,h,_ = self.history[:self.t].T
        lp,lm = stakes(h,a,b,u,self.cap)
        factors = lp*(z-u) if side == "plus" else -lm*(z-u)
        if np.any(factors <= -1):
            raise ArithmeticError("Illegal betting factor; audit support")
        return float(np.log1p(factors).sum())

    def update(self, z, a, b, h, loss, *, invert=True):
        if self.t >= self.N or loss not in (0,1):
            raise ValueError("Invalid observation")
        if not all(math.isfinite(v) for v in (z,a,b,h)) or not a <= z <= b or h <= 0:
            raise ValueError("Signal violates predictable support")
        self.history[self.t] = [z,a,b,h,self.N-self.t]
        self.t += 1
        self.total_loss += int(loss)
        if invert:
            # Monotonicity: K+(u) decreases, K-(u) increases. Retain the
            # rejected-side endpoint of each bracket, i.e. outward rounding.
            th = self.threshold+self.roundoff_guard
            if self.log_wealth(self.hi,"plus") >= th or self.log_wealth(self.lo,"minus") >= th:
                raise EmptyConfidenceSet("CELEUS wealth excluded the entire running interval")
            if self.log_wealth(self.lo,"plus") >= th:
                left,right = self.lo,self.hi
                while right-left > 1e-10:
                    mid=(left+right)/2
                    if self.log_wealth(mid,"plus") >= th: left=mid
                    else: right=mid
                self.lo=left
            if self.log_wealth(self.hi,"minus") >= th:
                left,right=self.lo,self.hi
                while right-left > 1e-10:
                    mid=(left+right)/2
                    if self.log_wealth(mid,"minus") >= th: right=mid
                    else: left=mid
                self.hi=right
        self.lo=max(self.lo,self.total_loss/self.N)
        self.hi=min(self.hi,(self.total_loss+self.N-self.t)/self.N)
        if self.lo > self.hi:
            raise EmptyConfidenceSet("CELEUS running intersection is empty; do not reset")
        return self.interval


@dataclass
class Plan:
    index: int
    q: float
    g: float
    center: float
    a: float
    b: float
    stake: float
    q_min: float
    q_max: float
    mix_weight: float
    predicted_variance: float

    def dictionary(self):
        return asdict(self)


class CELEUS:
    """Planner and certificate. observe(plan,loss) is the ONLY label channel."""
    def __init__(self, X, settings=None, *, alpha=.05, epsilon=.02, seed=0, check_every=10):
        X = np.asarray(X,dtype=float)
        if X.ndim != 2 or len(X)==0 or not np.all(np.isfinite(X)):
            raise ValueError("X must be a finite nonempty matrix")
        self.X=X
        self.N=len(X)
        self.settings=settings or Settings()
        self.cs=BettingCS(self.N,alpha,epsilon,self.settings.half_life,self.settings.cap)
        self.rng=np.random.default_rng(seed)
        self.remaining=np.ones(self.N,dtype=bool)
        self.paid=[]
        self.labels=[]
        self.model=LogisticSurrogate(X.shape[1],self.settings.ridge)
        self.g=np.full(self.N,.5)
        self.latent=np.zeros(self.N)
        self.fit_count=-1
        self.check_every=check_every
        self.pending=None

    def proposal(self):
        t=len(self.paid)
        if t == self.N: raise ValueError("Census completed")
        s=self.settings
        if s.surrogate and t>=s.warmup and (self.fit_count<0 or t-self.fit_count>=s.refit_every):
            self.model.fit(self.X[self.paid],np.asarray(self.labels))
            self.g,self.latent=self.model.predict(self.X)
            self.fit_count=t
        ids=np.flatnonzero(self.remaining)
        g=self.g[ids] if s.surrogate else np.zeros(len(ids))
        n=len(ids)
        uniform=np.full(n,1/n)
        q=uniform.copy()
        h=self.cs.base_stake()
        mix=0.
        v=g*(1-g)
        if s.acquisition != "uniform" and t>=s.warmup and s.surrogate:
            score=np.sqrt(np.maximum(v,1e-6))
            if s.acquisition == "mixture":
                q=s.beta/n+(1-s.beta)*score/score.sum()
                mix=1.
            elif s.acquisition == "waterfill":
                q=waterfill(score,s.beta)
                mix=1.
            else:
                # An approximation inspired by the draft, NOT its A9-IG1
                # implementation: actual supports for each candidate proposal,
                # five segment candidates, no common-envelope floors.
                latent=self.latent[ids]
                epi=v*v*latent
                share=min(1.,float(epi.mean()/max(v.mean(),1e-12)))
                ig=np.log1p(v*latent)
                direction=(1-share)*score/score.sum()+share*(ig/ig.sum() if ig.sum()>0 else uniform)
                endpoint=waterfill(direction,s.beta)
                center,_,_=support(g,uniform,self.cs.total_loss,self.N)
                probes=np.clip([center-self.cs.epsilon,center+self.cs.epsilon],self.cs.lo,self.cs.hi)
                def growth(proposal):
                    center,a,b=support(g,proposal,self.cs.total_loss,self.N)
                    z0=center-g/(self.N*proposal)
                    z1=center+(1-g)/(self.N*proposal)
                    lp,_=stakes(h,a,b,probes[0],s.cap)
                    _,lm=stakes(h,a,b,probes[1],s.cap)
                    plus=np.dot(proposal,(1-g)*np.log1p(lp*(z0-probes[0]))+g*np.log1p(lp*(z1-probes[0])))
                    minus=np.dot(proposal,(1-g)*np.log1p(-lm*(z0-probes[1]))+g*np.log1p(-lm*(z1-probes[1])))
                    return np.array([plus,minus])
                base=growth(uniform)
                best=1e-8
                for weight in (.25,.5,.75,1.):
                    candidate=(1-weight)*uniform+weight*endpoint
                    improvement=float(np.min(growth(candidate)-base))
                    if improvement>best:
                        best=improvement
                        q=candidate
                        mix=weight
        q[np.argmax(q)]+=1-float(q.sum())
        if np.min(q) < s.beta/n-1e-12 or np.any(q<=0):
            raise ArithmeticError("Invalid probability floor")
        return ids,g,q,h,mix

    def plan(self):
        if self.pending is not None:
            return self.pending
        ids,g,q,h,mix=self.proposal()
        center,a,b=support(g,q,self.cs.total_loss,self.N)
        # One random variate; never select based on the unseen loss.
        cdf=np.cumsum(q); cdf[-1]=1.
        j=int(np.searchsorted(cdf,self.rng.random(),side="right"))
        self.pending=Plan(int(ids[j]),float(q[j]),float(g[j]),center,a,b,h,float(q.min()),float(q.max()),mix,
                          float(np.sum(g*(1-g)/q)/self.N**2))
        return self.pending

    def observe(self, plan, loss):
        if plan is not self.pending or not self.remaining[plan.index] or loss not in (0,1):
            raise ValueError("Observe exactly the pending, distinct queried item")
        z=plan.center+(loss-plan.g)/(self.N*plan.q)
        self.remaining[plan.index]=False
        self.paid.append(plan.index)
        self.labels.append(int(loss))
        self.pending=None
        self.cs.update(z,plan.a,plan.b,plan.stake,loss,invert=len(self.paid)%self.check_every==0 or len(self.paid)==self.N)
        return z
