# CELEUS: method and numerical contract

CELEUS certifies the **loss rate of a fixed, finite pool** of benchmark cases: the fraction of cases the evaluated model gets wrong. A loss interval `[L, U]` is reported as the accuracy interval `[1-U, 1-L]`. It makes no claim about future deployment data.

## 1. Design choices

| Topic | What CELEUS does |
|---|---|
| No extra LLM | Only CPU features, a regularized logistic MAP fit and a Laplace covariance. |
| What is predicted | Whether the evaluated model will be **wrong** on a case (its loss). It does not predict the reference answer, and never queries the model in advance. |
| Order of each round | Predict, then query, then train. The predictor is refit after every 16 queried cases. The first 32 cases use prediction 0.5 and uniform sampling, and they count toward the cost. |
| Validity with a bad predictor | Coverage holds whatever the predictor's quality, provided the actual sampling probabilities `q`, the frozen predictions, the support bounds and predictable stakes are used. The predictor need not be calibrated. |
| `sqrt(g(1-g))` | A square-root proxy for the residual second moment under a Bernoulli working model, not an oracle residual. |
| Mixture vs water-filling | `beta/n + (1-beta)*score/sum(score)` is a feasible proposal, not the optimum. The second-moment optimum under a probability floor is water-filling. Both are candidates. |
| Choice of `beta` | Selected on an independent synthetic development suite: water-filling with `beta = 0.8`. This applies to that candidate set and suite only. |
| Epistemic variance | Not added. If `g` is the posterior predictive Bernoulli mean, `g(1-g)` already includes predictive uncertainty; adding an epistemic term may count it twice. Information gain is used only in a separate acquisition heuristic. |
| Depletion factor in the stake | A predictable betting heuristic, not an exact variance correction when the proposal, the remaining pool and the predictor all change. Its accuracy plays no role in the coverage proof. |
| Point estimate | The mean over adaptively drawn cases is descriptive only. The certified point is the interval midpoint; no claim is made that a stopped estimate is unbiased. |

Finite-pool inference allows cases to be dependent. It does **not** allow the model to drift over time: each case must have a fixed outcome, or a random outcome that can be assigned in advance, independently of the sampling order. If a model's answers change with load, call history or version, CELEUS does not certify the expected risk of that changing process.

## 2. The surrogate predictor

`features.py` reads only the public case content (`state`), the answer options (`question["criteria"]`) and the subset name. It never reads reference answers, case IDs, group IDs or past responses.

The feature vector has 52 dimensions: 24 structural features (including the intercept), 12 hashed category features and 16 normalized hashed word counts. It is neither TF-IDF nor a semantic embedding. The structural features cover:

- pairwise comparisons: lengths of candidates A and B, their length gap and word overlap;
- grounded answers: evidence and candidate lengths, their word overlap, and numbers in the candidate that are missing from the evidence;
- multiple choice: mean and spread of the option lengths;
- agent trajectories: number of messages, share of tool messages, visible tool errors, and the position of the judged step;
- generic text statistics: length, digits, non-ASCII, math and code characters, newlines, negations, quotes.

Standardizing with the mean and standard deviation of the public features over the whole pool involves no target loss. Each pool gets its own predictor, fitted only on the losses of cases that were selected and successfully queried. The logistic-normal predictive integral and the Laplace covariance are approximations; they affect efficiency, never validity.

Under adaptive sampling, the unweighted conditional likelihood is a working predictor. If the model is misspecified, the shifting training distribution can hurt prediction quality. Certification comes from the residual correction below, not from the predictor.

## 3. Corrected signal and support

Let `N` be the pool size, `T` the total loss already observed, `J` the set of remaining cases and `n = |J|`. Before each query, fix predictions `g_i ∈ [0,1]` and probabilities `q_i > 0` on `J`, and set

$$
m_t=\frac{T+\sum_{i\in J}g_i}{N},\qquad
Z_t=m_t+\frac{\ell_{I_t}-g_{I_t}}{Nq_{I_t}},\qquad I_t\sim q .
$$

Conditional on the fixed pool, the public features and the past:

$$
\mathbb E[Z_t\mid\mathcal F_{t-1}]
=\frac{T+\sum_J g_i+\sum_J(\ell_i-g_i)}{N}=R_N ,
$$

the loss rate of the whole pool. The exact conditional variance is

$$
\operatorname{Var}(Z_t\mid\mathcal F_{t-1})
=\frac1{N^2}\left[\sum_J\frac{(\ell_i-g_i)^2}{q_i}-\Big(\sum_J(\ell_i-g_i)\Big)^2\right].
$$

So prediction accuracy and sampling efficiency are different questions. Without clear heterogeneity in the residuals, adaptive sampling gains little over uniform; if small `q` values amplify rare residuals, it can do worse.

The support must cover every possible draw in the round:

$$
a_t=m_t-\max_{i\in J}\frac{g_i}{Nq_i},\qquad
b_t=m_t+\max_{i\in J}\frac{1-g_i}{Nq_i}.
$$

`Z_t` can fall outside `[0,1]`. **The code never clips `Z`, and never uses the uniform-sampling bounds for an adaptive signal.** A small outward floating-point tolerance is added to the support.

## 4. Betting confidence sequence

For each candidate loss rate `u`, two one-sided wealth processes are tracked:

$$
K_t^+(u)=\prod_{s\le t}\big[1+\lambda_s^+(u)(Z_s-u)\big],\qquad
K_t^-(u)=\prod_{s\le t}\big[1-\lambda_s^-(u)(Z_s-u)\big].
$$

The positive stake is `min(h, c/(u-a))` and the negative stake is `min(h, c/(b-u))`, with `c = 0.9`; when a denominator is not positive, that side needs no cap and the stake is `h`. Every past `h`, `a` and `b` is frozen permanently: old wealth is never recomputed with a newer predictor.

At the true value `u = R_N`, each factor has conditional mean 1 and is at least `1-c > 0`, so both wealth processes are nonnegative test martingales. Rejecting each side at `2/alpha` and combining Ville's inequality with a union bound gives coverage of at least `1-alpha` simultaneously at every reported time, on every path.

`K+(u)` decreases in `u` and `K-(u)` increases. The code inverts them by continuous bisection and keeps the outer endpoint of each bracket, with no coarse grid. The threshold gets an extra `1e-9` and bisection keeps about `1e-10` of outward slack. This is a conservatively designed float64 implementation, **not a formally verified arbitrary-precision proof**. The uniform baseline in `evalue.py` verifies its endpoints with integer and rational arithmetic instead.

After every observation the interval is intersected with the deterministic completion bound `[T/N, (T+N-t)/N]`. By default the wealth is inverted every 10 observations; in between only the completion bound is updated, which keeps the sequence valid at any time. A full census shrinks the interval to the exact value. An empty intersection stops the run and is reported; it is never reset to `[0,1]` or turned into a false zero-width success.

`epsilon` always means the **half-width**: a run stops once the full width is at most `2*epsilon`.

## 5. Acquisition candidates and selection

- `celeus_uniform`: the same online surrogate and certificate, with uniform sampling without replacement. It isolates the gain from acquisition.
- `celeus_mix04`: mixture of the square-root working variance with uniform, `beta = 0.4`.
- `celeus_water04` / `celeus_water08`: `q_i = max(beta/n, score_i/tau)`, with `tau` chosen so the probabilities sum to 1.
- `celeus_guarded_ig`: mixes the working variance with an information-gain direction, compares the expected two-sided log growth at five points between uniform and that direction, and moves only if both sides improve. Each candidate uses its own true support. It is not a global optimum or a multi-step lookahead.
- `corrected_no_surrogate` (validation only): `g = 0`, uniform, same betting certificate. It separates the gain from the surrogate from the gain from changing the certificate.
- `evalue`: the uniform without-replacement hypergeometric PPR baseline, with no predictor.

`validate_synthetic.py` picks the default configuration on development seeds only; the disjoint audit seeds are only checked and reported, never used to re-rank. `validation/release/selected_celeus.json` stores a hash of the core code, so any change to the statistics, feature or validation code requires validating again.

The cross-scenario average is a pre-declared development target, not a claim about real benchmark distributions. `audit_selected.py` checks the transfer from the small synthetic feature dimension to the real 52 dimensions; it does not guarantee savings on a real model either.

## 6. Running it on a real pool

1. Freeze the dataset revision, the prompt, the model version and the eligible pool.
2. Build `X` from public fields only.
3. Predict, form the actual `q`, draw one case, and save the whole plan before asking anything.
4. Query only that case and save its response and usage. A failure stays on the same case; it is never skipped.
5. Form `Z` from the pre-query plan and update the interval. Only then may the predictor be retrained.
6. Stop at the target precision, a budget guard, a question cap, or an error.

To resume after an interruption, replay the history in order: every step trains only on data from before that step, and the saved sampling plan is checked item by item. Never fit on all saved losses first and then reinterpret the old draws.

Saved answers may be reused across runs (an identical request is paid once), but a cache must be consulted only after a case has been selected and its plan frozen. The sampler then never sees stored answers of unselected cases, so reuse cannot steer selection. This makes the "each case has a fixed outcome" assumption literal. Each run still counts every case it uses as a query.

CELEUS minimizes the **number of queries**; it does not implement sampling that is optimal for heterogeneous token costs. When the evaluated model is very cheap, CPU time for training the predictor can offset the latency saved, so measure wall time.

## 7. Multiplicity and model randomness

When one family claim covers several pools, split `alpha` across them (for example `alpha = 0.05` over four pools gives `alpha_b = 0.0125` each). Two methods shown together do not share one family claim unless `alpha` is split again. Selecting the configuration on synthetic data needs no share of the real run's `alpha`, but never run several real seeds and report the best one.

If a model's outputs are random or drift, a separate full run is not the ground truth for the same fixed pool of outcomes. Overlap checks can reveal some problems, but cannot prove that the outcomes are fixed.

## References

- CELEUS paper: <https://arxiv.org/pdf/2606.20820>
- Waudby-Smith & Ramdas, [Confidence sequences for sampling without replacement](https://arxiv.org/abs/2006.04347), NeurIPS 2020: the PPR baseline.
- Waudby-Smith & Ramdas, [Estimating means of bounded random variables by betting](https://arxiv.org/abs/2010.09686): the betting confidence-sequence framework.
