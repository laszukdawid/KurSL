# MCMC Baseline

- Date: 2026-07-29
- Implementation: `944b22a` (`KurslMCMC`, emcee 2.2.1)
- Runtime: Python 3.13.1, NumPy 2.2.2, SciPy 1.15.1
- CPU: Intel Core i7-8750H, 6 cores / 12 threads
- Command: `uv run --frozen python -m benchmarks.benchmark_mcmc`
- Case: 2 uncoupled oscillators, 100 samples, 20 walkers, 20 iterations, seed 42

| Median | Minimum | Maximum |
| ---: | ---: | ---: |
| 4.254 s | 4.202 s | 4.361 s |

Known parameters:

```text
[[15.0, 0.2, 1.0, 0.0],
 [35.0, 2.0, 3.0, 0.0]]
```

Recovered parameters:

```text
[[15.2824984, 0.3063236, 1.3172548,  0.0186864],
 [35.0752625, 2.0770461, 3.0346681, -0.0036522]]
```

Relative reconstruction error: `0.0315284`.
