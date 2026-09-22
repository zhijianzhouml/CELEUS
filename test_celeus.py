"""Statistical invariants of the CELEUS certificate: exact identities, supports,
water-filling optimality, monotone wealth, exhaustive small-pool coverage."""
import copy
from fractions import Fraction
from itertools import product
import unittest
import numpy as np
from celeus import CELEUS, Settings, BettingCS, support, stakes, waterfill
from evalue import EmptyConfidenceSet


class MathematicalTests(unittest.TestCase):
    def test_exact_conditional_identity_adversarial_surrogates(self):
        # Rational arithmetic: fixed unobserved labels, arbitrary misspecification,
        # nonuniform positive probabilities and a paid prefix.
        N=7;total=2
        y=[0,1,1,0]
        g=[Fraction(99,100),Fraction(1,100),Fraction(1,2),Fraction(4,5)]
        q=[Fraction(1,10),Fraction(2,10),Fraction(3,10),Fraction(4,10)]
        center=(total+sum(g))/N
        z=[center+(yy-gg)/(N*qq) for yy,gg,qq in zip(y,g,q)]
        mean=sum(qq*zz for qq,zz in zip(q,z))
        self.assertEqual(mean,Fraction(total+sum(y),N))
        residual=[yy-gg for yy,gg in zip(y,g)]
        variance=sum(qq*(zz-mean)**2 for qq,zz in zip(q,z))
        self.assertEqual(variance,(sum(rr**2/qq for rr,qq in zip(residual,q))-sum(residual)**2)/N**2)

    def test_support_and_true_candidate_martingale_at_adaptive_histories(self):
        rng=np.random.default_rng(183)
        for _ in range(50):
            N=17; t=5;n=N-t
            y=rng.integers(0,2,n); total=int(rng.integers(0,t+1))
            g=rng.uniform(0,1,n);q=waterfill(rng.random(n),.15)
            m,a,b=support(g,q,total,N)
            z=m+(y-g)/(N*q)
            truth=(total+y.sum())/N
            self.assertTrue(np.all((z>=a)&(z<=b)))
            self.assertAlmostEqual(float(q@z),truth,12)
            plus,minus=stakes(4.2,a,b,truth,.9)
            self.assertAlmostEqual(float(q@(1+plus*(z-truth))),1.,12)
            self.assertAlmostEqual(float(q@(1-minus*(z-truth))),1.,12)

    def test_waterfill_KKT_and_mixture_not_optimum(self):
        scores=np.array([.001,.05,.25,.5,.9]);beta=.4
        q=waterfill(scores,beta)
        self.assertAlmostEqual(float(q.sum()),1,14)
        self.assertGreaterEqual(float(q.min()),beta/len(q)-1e-14)
        active=q>beta/len(q)+1e-12
        np.testing.assert_allclose(scores[active]/q[active],np.full(active.sum(),(scores[active]/q[active])[0]))
        mix=beta/len(q)+(1-beta)*scores/scores.sum()
        self.assertLessEqual(float(np.sum(scores*scores/q)),float(np.sum(scores*scores/mix))+1e-12)
        np.testing.assert_allclose(waterfill(np.zeros(5),.4),.2)

    def test_monotone_wealth_and_outward_inversion(self):
        cs=BettingCS(500,.05,.04)
        rng=np.random.default_rng(6)
        for t in range(100):
            h=cs.base_stake();z=float(rng.binomial(1,.3))
            cs.update(z,-.3,1.4,h,int(z),invert=False)
        grid=np.linspace(0,1,101)
        self.assertTrue(np.all(np.diff([cs.log_wealth(u,"plus") for u in grid])<=1e-10))
        self.assertTrue(np.all(np.diff([cs.log_wealth(u,"minus") for u in grid])>=-1e-10))

    def test_enumerate_all_adaptive_small_pool_paths(self):
        # Sum exact path probabilities over all permutations, using fractional
        # proposals that depend on past queried losses. Enumerate all binary
        # fixed pools, not iid labels. Numerical inversion is checked separately.
        N=5;alpha=.2
        for labels in product((0,1),repeat=N):
            truth=sum(labels)/N
            failure_probability=Fraction()
            def visit(ids,paid,total,cs,prob,bad):
                nonlocal failure_probability
                if not ids:
                    if bad:failure_probability+=prob
                    return
                weights=[1+(i+total)%3 for i in ids]
                qs=[Fraction(w,sum(weights)) for w in weights]
                # Deliberately wrong, history-dependent prediction.
                g=np.array([.9 if (i+total)%2 else .1 for i in ids])
                q=np.array([float(x) for x in qs])
                m,a,b=support(g,q,total,N);h=cs.base_stake()
                for j,i in enumerate(ids):
                    child=copy.deepcopy(cs);failure=bad
                    try:
                        child.update(m+(labels[i]-g[j])/(N*q[j]),a,b,h,labels[i])
                        failure |= not child.lo-1e-9<=truth<=child.hi+1e-9
                    except EmptyConfidenceSet:failure=True
                    visit(ids[:j]+ids[j+1:],paid+1,total+labels[i],child,prob*qs[j],failure)
            visit(list(range(N)),0,0,BettingCS(N,alpha,.05),Fraction(1),False)
            self.assertLessEqual(float(failure_probability),alpha+1e-12)

    def test_freezing_no_duplicate_and_census(self):
        X=np.column_stack([np.ones(25),np.linspace(-2,2,25)])
        labels=np.array([0,1]*12+[0])
        engine=CELEUS(X,Settings(warmup=2,refit_every=1),alpha=.001,epsilon=.00001,check_every=1)
        seen=set()
        for _ in range(25):
            plan=engine.plan()
            self.assertIs(engine.plan(),plan)
            self.assertNotIn(plan.index,seen);seen.add(plan.index)
            before=plan.dictionary().copy()
            z=engine.observe(plan,int(labels[plan.index]))
            self.assertEqual(before,plan.dictionary())
            self.assertAlmostEqual(z,before["center"]+(labels[plan.index]-before["g"])/(25*before["q"]))
        self.assertEqual(engine.cs.interval,(float(labels.mean()),float(labels.mean())))

    def test_empty_is_not_repaired(self):
        cs=BettingCS(3,.05,.02)
        cs.lo=.8;cs.hi=.9
        with self.assertRaises(EmptyConfidenceSet):cs.update(0,0,1,1,0,invert=False)


if __name__ == "__main__":
    unittest.main()
