"""Exact checks of the uniform without-replacement e-value baseline (evalue.py)."""
from fractions import Fraction
from itertools import combinations
import math
import unittest
from unittest.mock import patch
from evalue import BinaryWORCS, EmptyConfidenceSet


class StatisticsTests(unittest.TestCase):
    def test_exact_inversion_against_all_integer_candidates(self):
        for N in range(1,35):
            for t in range(1,N+1):
                for s in range(t+1):
                    cs=BinaryWORCS(N,"0.05")
                    cs.t,cs.s,cs._nchoose=t,s,math.comb(N,t)
                    accepted=[k for k in range(N+1) if cs.accepts(k)]
                    self.assertTrue(accepted)
                    self.assertEqual(cs.current_interval_counts(),(min(accepted),max(accepted)))

    def test_float_proposals_cannot_override_integer_validation(self):
        cs=BinaryWORCS(100,"0.05")
        cs.t,cs.s,cs._nchoose=37,23,math.comb(100,37)
        expected=cs.current_interval_counts()
        # Deliberately corrupt every floating-point proposal in both directions.
        with patch.object(cs,"log_evalue",return_value=1e30):
            self.assertEqual(cs.current_interval_counts(),expected)
        with patch.object(cs,"log_evalue",return_value=-1e30):
            self.assertEqual(cs.current_interval_counts(),expected)

    def test_uniform_mixture_identity(self):
        for N in range(1,12):
            for t in range(N+1):
                for s in range(t+1):
                    mixture=sum((Fraction(math.comb(k,s)*math.comb(N-k,t-s),math.comb(N,t))
                                 for k in range(s,N-t+s+1)),Fraction())/(N+1)
                    self.assertEqual(mixture,Fraction(1,t+1))

    def test_test_supermartingale_exactly(self):
        def capital(N,K,t,s):
            if not s<=K<=N-t+s: return None
            return Fraction(math.comb(N,t),(t+1)*math.comb(K,s)*math.comb(N-K,t-s))
        for N in range(1,14):
            for K in range(N+1):
                for t in range(N):
                    for s in range(max(0,t-(N-K)),min(t,K)+1):
                        p=Fraction(K-s,N-t)
                        nxt=Fraction()
                        if p: nxt+=p*capital(N,K,t+1,s+1)
                        if p<1: nxt+=(1-p)*capital(N,K,t+1,s)
                        self.assertLessEqual(nxt,capital(N,K,t,s))

    def test_exhaustive_anytime_coverage_all_binary_orders(self):
        # Exact probabilities over every ordering of each fixed small population.
        for alpha in (Fraction(1,20),Fraction(1,5)):
            for N in range(1,11):
                for K in range(N+1):
                    failed=0
                    for ones in combinations(range(N),K):
                        ones=set(ones);cs=BinaryWORCS(N,alpha);bad=False
                        for t in range(N):
                            try: cs.update(int(t in ones))
                            except EmptyConfidenceSet: bad=True;break
                            if not cs.lo<=K<=cs.hi: bad=True
                        failed+=bad
                    self.assertLessEqual(Fraction(failed,math.comb(N,K)),alpha)

    def test_feasibility_and_census(self):
        for xs in ([0]*12,[1]*12,[1,0]*6):
            cs=BinaryWORCS(len(xs),"0.01")
            for x in xs: cs.update(x,invert=False)
            self.assertEqual(cs.lo,sum(xs));self.assertEqual(cs.hi,sum(xs))
            self.assertTrue(cs.width_at_most("0"))

    def test_bounds_are_nested(self):
        cs=BinaryWORCS(1000,"0.01");lo,hi=0,1000
        for t in range(200):
            cs.update(int(t%3==0),invert=(t+1)%10==0)
            self.assertGreaterEqual(cs.lo,lo);self.assertLessEqual(cs.hi,hi)
            lo,hi=cs.lo,cs.hi

    def test_rejects_wrong_metric(self):
        for x in (-1,.2,float("nan"),2):
            with self.assertRaises(ValueError): BinaryWORCS(10).update(x)
        for N in (0,-1,3.2,True):
            with self.assertRaises(ValueError): BinaryWORCS(N)

    def test_naive_wor_martingale_counterexample(self):
        # N=2 outcomes {0,1}, lambda_1=0 and lambda_2=sign(.5-X_1).
        # Raw centered iid-style betting produces capital 1.5 under either order.
        for first in (0,1):
            second=1-first;lam=1 if first==0 else -1
            self.assertEqual(1+lam*(second-.5),1.5)


if __name__ == "__main__":
    unittest.main()
