# MCMC Baseline

- Date: 2026-07-29
- Implementation: `944b22a` (`KurslMCMC`, emcee 2.2.1)
- Runtime: Python 3.13.1, NumPy 2.2.2, SciPy 1.15.1
- CPU: Intel Core i7-8750H, 6 cores / 12 threads
- Command: `uv run --frozen python -m benchmarks.benchmark_mcmc`
- Repetitions: 3

| Case | Measured iterations | Median | Minimum | Maximum | Projected target |
| --- | ---: | ---: | ---: | ---: | ---: |
| Smoke: 2 oscillators, 100 samples, 20 walkers | 20 of 20 | 4.406 s | 4.273 s | 4.432 s | 0.1 min |
| Epilepsy-scale: 4 oscillators, 4,097 samples, 72 walkers | 1 of 100 | 89.173 s | 87.216 s | 91.774 s | 148.6 min |

The representative case matches the signal length, oscillator count, harmonic order, and iteration count configured in
`examples/kursl_epilepsy.py`. Its projection multiplies the measured complete-iteration median by 100; the full
100-iteration run was not executed.

## Smoke Correctness

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

The representative case uses an independently generated, uncoupled four-oscillator signal and validates parameter
recovery, output shape, iteration count, and relative reconstruction error after every measured iteration.
