#!/usr/bin/env python3
"""Certify a model's accuracy on a fixed benchmark pool with CELEUS.

Runs offline on a simulated pool, so it needs no API key. To use your own
benchmark, replace `simulated_pool()` with your frozen list of cases and
`ask_model()` with a call to the model you evaluate:

    X = feature_matrix(cases)                 # public information only
    engine, status = certify(X, ask_model)    # ask_model(i) -> True if correct
    print(engine.cs.accuracy_interval)

Every case is asked at most once. The interval holds with probability at least
1 - alpha at every stopping time, as long as each case has a fixed answer.
"""
import argparse
import numpy as np
from celeus import CELEUS, Settings
from features import vector

# Configuration selected on the synthetic development suite (validate_synthetic.py).
SELECTED = Settings(acquisition="waterfill", beta=0.8)


def feature_matrix(cases):
    """One row per case from PUBLIC fields only, never the reference answer.

    `cases` holds (state, question, subset) triples: `state` is the case content
    as a dict, `question` a dict with the answer options under "criteria". Columns
    are standardized like features.build_matrix; column 0 is the intercept.
    """
    X = np.asarray([vector(state, question, subset) for state, question, subset in cases], dtype=float)
    mean, scale = X.mean(axis=0), np.maximum(X.std(axis=0), .05)
    X[:, 1:] = np.clip((X[:, 1:] - mean[1:]) / scale[1:], -5, 5)
    X[:, 0] = 1
    return X


def certify(X, ask_model, *, alpha=.05, epsilon=.02, seed=0, settings=SELECTED, max_questions=None):
    """Ask one case at a time until the accuracy interval is at most ±epsilon wide.

    X: (N, d) feature matrix of the whole frozen pool, intercept in column 0.
    ask_model(i): True if the model answers case i correctly, else False.
    Returns the engine (interval in engine.cs.accuracy_interval) and a status.
    """
    engine = CELEUS(X, settings, alpha=alpha, epsilon=epsilon, seed=seed)
    cap = engine.N if max_questions is None else min(engine.N, max_questions)
    while engine.cs.width > 2 * epsilon:
        if len(engine.paid) >= cap:
            return engine, "question_cap"
        plan = engine.plan()                  # chosen and frozen before asking
        correct = ask_model(plan.index)       # the only place answers enter
        engine.observe(plan, 0 if correct else 1)   # CELEUS tracks the loss
    return engine, "precision_reached"


def simulated_pool(N, seed):
    """Features plus hidden correctness; harder cases are more often wrong."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(N, 5))
    p_wrong = 1 / (1 + np.exp(-(-1.2 + 1.5 * Z[:, 0] - Z[:, 1])))
    correct = rng.random(N) >= p_wrong
    X = np.column_stack([np.ones(N), Z])
    return X, correct


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=20000, help="pool size")
    p.add_argument("--alpha", type=float, default=.05, help="1 - confidence")
    p.add_argument("--epsilon", type=float, default=.02, help="target half-width")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    X, correct = simulated_pool(a.n, a.seed)
    engine, status = certify(X, lambda i: bool(correct[i]), alpha=a.alpha, epsilon=a.epsilon, seed=a.seed)
    lo, hi = engine.cs.accuracy_interval
    asked = len(engine.paid)
    print(f"status: {status}")
    print(f"questions asked: {asked:,} of {engine.N:,} ({100 * asked / engine.N:.1f}% of the pool)")
    print(f"{100 * (1 - a.alpha):g}% accuracy interval: [{lo:.3f}, {hi:.3f}]")
    print(f"true pool accuracy (known only in simulation): {correct.mean():.3f}")


if __name__ == "__main__":
    main()
