# Synthetic validation report

Frozen CELEUS selection: **celeus_water08**. Audit screen: **PASS**.

Synthetic only: 0 model calls, 0 extra LLM calls. Development selects; audit never re-ranks.
Coverage is checked over every reported interval through census, including times after first precision stopping.
Monte Carlo checks can detect failures; they do not prove validity. CP intervals quantify simulation uncertainty.

| Phase | Method | Scenario | Runs | Mean first-stop queries | Path coverage | Failure 95% CP |
|---|---|---|---:|---:|---:|---|
| audit | celeus_uniform | clustered | 100 | 414.0 | 0.990 | 0.0003, 0.0545 |
| audit | celeus_uniform | heterogeneous | 100 | 255.0 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_uniform | linear | 100 | 294.3 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_uniform | nonlinear | 100 | 438.3 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_uniform | rare_errors | 100 | 87.0 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_uniform | uninformative | 100 | 412.4 | 0.980 | 0.0024, 0.0704 |
| audit | celeus_water08 | clustered | 100 | 411.6 | 0.990 | 0.0003, 0.0545 |
| audit | celeus_water08 | heterogeneous | 100 | 218.1 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_water08 | linear | 100 | 277.9 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_water08 | nonlinear | 100 | 437.9 | 1.000 | 0.0000, 0.0362 |
| audit | celeus_water08 | rare_errors | 100 | 93.2 | 0.980 | 0.0024, 0.0704 |
| audit | celeus_water08 | uninformative | 100 | 412.6 | 0.990 | 0.0003, 0.0545 |
| audit | corrected_no_surrogate | clustered | 100 | 431.9 | 0.990 | 0.0003, 0.0545 |
| audit | corrected_no_surrogate | heterogeneous | 100 | 320.7 | 1.000 | 0.0000, 0.0362 |
| audit | corrected_no_surrogate | linear | 100 | 429.3 | 0.990 | 0.0003, 0.0545 |
| audit | corrected_no_surrogate | nonlinear | 100 | 432.3 | 1.000 | 0.0000, 0.0362 |
| audit | corrected_no_surrogate | rare_errors | 100 | 87.5 | 0.990 | 0.0003, 0.0545 |
| audit | corrected_no_surrogate | uninformative | 100 | 404.5 | 0.980 | 0.0024, 0.0704 |
| audit | evalue | clustered | 100 | 497.8 | 1.000 | 0.0000, 0.0362 |
| audit | evalue | heterogeneous | 100 | 355.8 | 1.000 | 0.0000, 0.0362 |
| audit | evalue | linear | 100 | 493.7 | 1.000 | 0.0000, 0.0362 |
| audit | evalue | nonlinear | 100 | 495.3 | 0.980 | 0.0024, 0.0704 |
| audit | evalue | rare_errors | 100 | 109.1 | 1.000 | 0.0000, 0.0362 |
| audit | evalue | uninformative | 100 | 473.2 | 1.000 | 0.0000, 0.0362 |
| development | celeus_guarded_ig | clustered | 12 | 408.3 | 1.000 | 0.0000, 0.2646 |
| development | celeus_guarded_ig | heterogeneous | 12 | 226.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_guarded_ig | linear | 12 | 277.5 | 0.917 | 0.0021, 0.3848 |
| development | celeus_guarded_ig | nonlinear | 12 | 441.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_guarded_ig | rare_errors | 12 | 106.2 | 0.917 | 0.0021, 0.3848 |
| development | celeus_guarded_ig | uninformative | 12 | 410.8 | 1.000 | 0.0000, 0.2646 |
| development | celeus_mix04 | clustered | 12 | 410.0 | 1.000 | 0.0000, 0.2646 |
| development | celeus_mix04 | heterogeneous | 12 | 230.8 | 1.000 | 0.0000, 0.2646 |
| development | celeus_mix04 | linear | 12 | 290.0 | 1.000 | 0.0000, 0.2646 |
| development | celeus_mix04 | nonlinear | 12 | 443.3 | 1.000 | 0.0000, 0.2646 |
| development | celeus_mix04 | rare_errors | 12 | 94.8 | 1.000 | 0.0000, 0.2646 |
| development | celeus_mix04 | uninformative | 12 | 411.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_uniform | clustered | 12 | 406.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_uniform | heterogeneous | 12 | 256.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_uniform | linear | 12 | 291.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_uniform | nonlinear | 12 | 439.2 | 1.000 | 0.0000, 0.2646 |
| development | celeus_uniform | rare_errors | 12 | 94.2 | 1.000 | 0.0000, 0.2646 |
| development | celeus_uniform | uninformative | 12 | 406.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water04 | clustered | 12 | 421.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water04 | heterogeneous | 12 | 240.0 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water04 | linear | 12 | 310.8 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water04 | nonlinear | 12 | 445.8 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water04 | rare_errors | 12 | 108.3 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water04 | uninformative | 12 | 412.5 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water08 | clustered | 12 | 408.3 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water08 | heterogeneous | 12 | 219.2 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water08 | linear | 12 | 284.2 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water08 | nonlinear | 12 | 441.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water08 | rare_errors | 12 | 92.7 | 1.000 | 0.0000, 0.2646 |
| development | celeus_water08 | uninformative | 12 | 410.0 | 1.000 | 0.0000, 0.2646 |
| development | corrected_no_surrogate | clustered | 12 | 424.2 | 1.000 | 0.0000, 0.2646 |
| development | corrected_no_surrogate | heterogeneous | 12 | 324.2 | 1.000 | 0.0000, 0.2646 |
| development | corrected_no_surrogate | linear | 12 | 442.5 | 1.000 | 0.0000, 0.2646 |
| development | corrected_no_surrogate | nonlinear | 12 | 433.3 | 1.000 | 0.0000, 0.2646 |
| development | corrected_no_surrogate | rare_errors | 12 | 105.0 | 1.000 | 0.0000, 0.2646 |
| development | corrected_no_surrogate | uninformative | 12 | 400.8 | 1.000 | 0.0000, 0.2646 |
| development | evalue | clustered | 12 | 503.3 | 1.000 | 0.0000, 0.2646 |
| development | evalue | heterogeneous | 12 | 349.2 | 1.000 | 0.0000, 0.2646 |
| development | evalue | linear | 12 | 509.2 | 1.000 | 0.0000, 0.2646 |
| development | evalue | nonlinear | 12 | 510.0 | 1.000 | 0.0000, 0.2646 |
| development | evalue | rare_errors | 12 | 122.5 | 1.000 | 0.0000, 0.2646 |
| development | evalue | uninformative | 12 | 470.8 | 1.000 | 0.0000, 0.2646 |

Timing includes full-path offline auditing after first stop, not production early-stop runtime.
The logistic synthetic feature dimension is 7; real structural/hash features have dimension 52. Transfer is unverified.
Do not treat no detected excess failures as evidence that an arbitrary API has fixed, order-independent outcomes.
