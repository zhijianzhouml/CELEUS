# CELEUS

**Certify a model's accuracy on a fixed benchmark with far fewer questions.**

CELEUS asks the model one benchmark case at a time and keeps an accuracy interval that is valid at every step (an anytime-valid confidence sequence). It stops as soon as the interval is narrow enough. A cheap CPU predictor, built only from the text of each case, steers it towards the most informative cases. It needs no extra LLM and no GPU.

Paper: <https://arxiv.org/pdf/2606.20820>

## What you get

For a fixed pool of `N` cases, where each answer is right or wrong:

- an interval that contains the model's accuracy **on the whole pool** with probability at least `1 - alpha`, simultaneously at every stopping time;
- the number of questions actually asked, usually a small fraction of `N`.

Each case is asked at most once. The guarantee needs each case to have a fixed outcome, or a random outcome that could be assigned in advance, independent of the sampling order. It is a statement about the frozen pool, not about future deployment data.

## Files

| File | What it is |
|---|---|
| `celeus.py` | The method: surrogate predictor, water-filling sampling, residual-corrected betting confidence sequence. NumPy and SciPy only. |
| `features.py` | 52 cheap text features per case, computed from public fields only (never the reference answer). |
| `evalue.py` | Baseline: uniform sampling without replacement with an exact hypergeometric e-value confidence sequence. |
| `example.py` | Runnable walkthrough, plus `certify()` and `feature_matrix()` helpers for your own benchmark. |
| `validate_synthetic.py` | Offline selection of the configuration on development seeds, then an audit on disjoint seeds. |
| `audit_selected.py` | Stress check of the frozen configuration: 52 features, larger pools, ±2 points. |
| `test_celeus.py`, `test_evalue.py` | Exact and exhaustive checks of the statistics (16 tests). |
| `docs/METHOD_REVIEW.md` | The method, the proof sketch and the numerical contract. |
| `validation/` | The saved synthetic validation results for this exact code. |

## Install

Python 3.10 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

```bash
python example.py
```

This runs offline on a simulated pool of 20,000 cases:

```text
status: precision_reached
questions asked: 2,220 of 20,000 (11.1% of the pool)
95% accuracy interval: [0.668, 0.708]
true pool accuracy (known only in simulation): 0.692
```

## Use it on your own benchmark

1. **Fix the pool first.** Pin the dataset revision, write down the filtering rules, and list every eligible case once, before asking the model anything.
2. **One case, one question with a fixed answer.** Score 1 if the model's answer matches the reference, otherwise 0. Settle any random choice (such as answer order) with a fixed seed in advance. If the model refuses a case, count it as wrong; never skip it.
3. **Build features from public information only.** Use `feature_matrix()` for text cases, or supply your own matrix with column 0 equal to 1.
4. **Run CELEUS until the target precision.**

```python
from example import certify, feature_matrix

# cases: a list of (state, question, subset), where state is the case content as
# a dict and question["criteria"] maps each answer label to its description.
X = feature_matrix(cases)

def ask_model(i):
    """Ask the model case i; return True if its answer is correct."""
    ...

engine, status = certify(X, ask_model, alpha=0.05, epsilon=0.02, seed=0)
print(status, len(engine.paid), engine.cs.accuracy_interval)
```

`epsilon` is the half-width: `0.02` stops once the interval is at most ±2 percentage points wide. For one claim over several benchmarks, split `alpha` between them; for example, 95% for four benchmarks together means `alpha = 0.0125` each.

5. **Report the interval, the number of questions and the cost.** Save every answer, so that a rerun never pays for the same question twice. Look up a saved answer only after CELEUS has chosen the case, so the cache cannot influence which cases are chosen.

For a paid API, save the plan (`engine.plan()`) and every answer durably before continuing. Then an interrupted run can be resumed by replaying them in order.

## Selected configuration

```python
Settings(acquisition="waterfill", beta=0.8, ridge=1.0, refit_every=16,
         warmup=32, half_life=64.0, cap=0.9)   # interval check every 10 answers
```

The first 32 cases are drawn uniformly at random. After that, the predictor is refit every 16 answers, and each case is drawn with probability at least `0.8/n`, where `n` is the number of cases left.

## Synthetic validation

```bash
python validate_synthetic.py --out validation/release --n 800 --dev-seeds 12 --audit-seeds 100 --epsilon 0.05
python audit_selected.py
```

The first command selects the configuration on 12 development seeds per scenario, then checks it on 100 disjoint audit seeds, with pools of 800 cases, ±5 points and `alpha = 0.0125`. The audit never changes the selection. Results for six scenarios (`validation/release/REPORT.md`):

| Method | Mean questions to reach ±5 points | Path coverage per scenario |
|---|---:|---|
| CELEUS, water-filling β 0.8 (selected) | 308.6 | 0.98–1.00 |
| CELEUS, uniform sampling | 316.8 | 0.98–1.00 |
| Corrected, no predictor | 351.0 | 0.98–1.00 |
| Uniform e-value baseline | 404.1 | 0.98–1.00 |

The selected configuration needed 23.7% fewer questions than the uniform baseline.

The stress check (`validation/transfer_stress/`) uses 1,600-case pools, 52 features, ±2 points and 20 seeds per scenario. There the selected configuration averaged 986.6 questions, against 1,176.7 for the uniform baseline (16.2% fewer). Of the 360 stress paths, 4 ended with an empty interval, which counts as a failure: 2 for uniform CELEUS, 1 for the selected configuration and 1 for the exact uniform baseline. They are kept in the results.

Monte Carlo checks can reveal problems but cannot prove validity, and synthetic savings do not guarantee savings on a real model.

## Tests

```bash
python -m unittest test_celeus test_evalue
```

These include an exact rational-arithmetic check of the corrected signal, and an enumeration of every adaptive sampling path on small pools that confirms the failure probability stays below `alpha`.

## Assumptions and limits

- Frozen pool, prompt and model. Each case must have a fixed outcome. If a model's answers drift with load, history or version, that process is not certified.
- Coverage does not depend on the predictor being good; a poor predictor only costs more questions.
- CELEUS minimizes the number of questions, not token cost. For a very cheap model, CPU time for refitting the predictor can matter, so measure wall time.
- The confidence sequence is a conservative float64 implementation with outward rounding, not a formally verified arbitrary-precision proof. The uniform baseline verifies its endpoints exactly.

More detail: [`docs/METHOD_REVIEW.md`](docs/METHOD_REVIEW.md).
