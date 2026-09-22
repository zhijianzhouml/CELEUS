"""Exact hypergeometric PPR confidence sequence for uniform WOR binary scores.

Only this module implements statistical inference. No SAVE/CELEUS dependency.
A uniform prior on the UNKNOWN number K of successes in a fixed pool of N gives
    E_t(K) = 1 / ((t+1) * HypergeomPMF(S_t; N, K, t)).
E_0=1; under the true K it is a nonnegative test supermartingale (support can
shrink). Invert with Ville, intersect over time, and keep integer K endpoints.

Reference: Waudby-Smith & Ramdas, Confidence sequences for sampling without
replacement, NeurIPS 2020, Section 2, https://arxiv.org/abs/2006.04347.

Floating log-gamma values ONLY propose endpoints. Integer arithmetic verifies
both sides of each endpoint. A discrepancy invokes exact integer bisection.
Consequently numerical approximation cannot silently remove a valid candidate.
"""
from __future__ import annotations

import math
from fractions import Fraction


class EmptyConfidenceSet(RuntimeError):
    """A possible alpha-probability exclusion event; never replace by a fake CI."""


class BinaryWORCS:
    def __init__(self, N: int, alpha: str | float | Fraction = "0.05"):
        if isinstance(N, bool) or not isinstance(N, int) or N < 1:
            raise ValueError("N must be a positive integer")
        self.N = N
        self.alpha = alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        self.t = self.s = 0
        self.lo, self.hi = 0, N
        self._nchoose = 1  # C(N,t), maintained exactly
        self.log_threshold = math.log(float(1 / self.alpha))

    def update(self, x: int, *, invert: bool = True) -> tuple[float, float]:
        if x not in (0, 1) or isinstance(x, float) and not x.is_integer():
            raise ValueError("BinaryWORCS only accepts binary correctness, not probabilities")
        if self.t >= self.N:
            raise ValueError("Cannot observe more than N distinct pool items")
        self.t += 1
        self.s += int(x)
        self._nchoose = self._nchoose * (self.N - self.t + 1) // self.t
        # Deterministic feasibility bounds remain valid between inversion checkpoints.
        self.lo = max(self.lo, self.s)
        self.hi = min(self.hi, self.N - self.t + self.s)
        if invert or self.t == self.N:
            a, b = self.current_interval_counts()
            self.lo, self.hi = max(self.lo, a), min(self.hi, b)
        if self.lo > self.hi:
            raise EmptyConfidenceSet("Running confidence set became empty; stop and audit, do not report a zero-width success.")
        return self.interval

    @property
    def interval(self) -> tuple[float, float]:
        return self.lo / self.N, self.hi / self.N

    @property
    def width(self) -> float:
        return (self.hi - self.lo) / self.N

    def width_at_most(self, width: str | float) -> bool:
        w = Fraction(str(width))
        return Fraction(self.hi - self.lo, self.N) <= w

    @staticmethod
    def _logcomb(n: int, r: int) -> float:
        if r < 0 or r > n:
            return -math.inf
        return math.lgamma(n + 1) - math.lgamma(r + 1) - math.lgamma(n - r + 1)

    def log_evalue(self, K: int) -> float:
        """For diagnostics/proposals only, never the final rejection decision."""
        if not self.s <= K <= self.N - self.t + self.s:
            return math.inf
        return (self._logcomb(self.N, self.t) - self._logcomb(K, self.s)
                - self._logcomb(self.N-K, self.t-self.s) - math.log(self.t+1))

    def accepts(self, K: int) -> bool:
        """Exact decision E_t(K) < 1/alpha. Equality is rejected conservatively."""
        if not self.s <= K <= self.N - self.t + self.s:
            return False
        numerator = self.alpha.numerator * self._nchoose
        denominator = (self.alpha.denominator * (self.t + 1)
                       * math.comb(K, self.s) * math.comb(self.N-K, self.t-self.s))
        return numerator < denominator

    def current_interval_counts(self) -> tuple[int, int]:
        """Invert the current e-value in O(log N) proposals + exact validation."""
        if not self.t:
            return 0, self.N
        left, right = self.s, self.N-self.t+self.s
        # Hypergeometric likelihood is log-concave in integer K; this is a mode.
        mode = min(right, max(left, self.s * (self.N+1) // self.t))
        if not self.accepts(mode):
            raise EmptyConfidenceSet("No candidate K accepted by the current exact e-value")

        def bisect_left(predicate):
            lo, hi = left, mode
            while lo < hi:
                mid = (lo+hi)//2
                if predicate(mid):
                    hi = mid
                else:
                    lo = mid+1
            return lo

        def bisect_right(predicate):
            lo, hi = mode, right
            while lo < hi:
                mid = (lo+hi+1)//2
                if predicate(mid):
                    lo = mid
                else:
                    hi = mid-1
            return lo

        approx = lambda k: self.log_evalue(k) < self.log_threshold
        a, b = bisect_left(approx), bisect_right(approx)
        # Exact adjacency checks plus log-concavity certify the entire interval.
        if not (self.accepts(a) and (a == left or not self.accepts(a-1))):
            a = bisect_left(self.accepts)
        if not (self.accepts(b) and (b == right or not self.accepts(b+1))):
            b = bisect_right(self.accepts)
        return a, b
