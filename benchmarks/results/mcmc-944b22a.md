# MCMC Baseline

- Date: 2026-07-30
- Implementation: `944b22a` (`KurslMCMC`, emcee 2.2.1)
- Runtime: Python 3.13.1, NumPy 2.2.2, SciPy 1.15.1
- CPU: Intel Core i7-8750H, 6 cores / 12 threads
- Command: `uv run --frozen python -m benchmarks.benchmark_mcmc`
- Repetitions: 3
- Verification: parameter absolute error `< 1e-6`; relative reconstruction energy `< 1e-10`

Each timing measures one complete MCMC iteration. The projection multiplies its median by 100.

| Case | Median | Minimum | Maximum | Max reconstruction error | 100-iteration projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| `known-01-o2-h1-s64` | 0.224 s | 0.223 s | 0.226 s | 1.375e-29 | 0.4 min |
| `known-02-o2-h1-s96` | 0.331 s | 0.324 s | 0.335 s | 1.380e-28 | 0.6 min |
| `known-03-o2-h2-s128` | 0.769 s | 0.763 s | 0.778 s | 7.141e-20 | 1.3 min |
| `known-04-o2-h3-s160` | 1.486 s | 1.426 s | 1.523 s | 5.034e-20 | 2.5 min |
| `known-05-o3-h1-s72` | 0.454 s | 0.453 s | 0.489 s | 1.081e-19 | 0.8 min |
| `known-06-o3-h1-s112` | 0.705 s | 0.695 s | 0.731 s | 8.273e-29 | 1.2 min |
| `known-07-o3-h2-s144` | 1.864 s | 1.860 s | 1.900 s | 6.338e-28 | 3.1 min |
| `known-08-o3-h2-s192` | 2.515 s | 2.494 s | 2.636 s | 4.611e-27 | 4.2 min |
| `known-09-o3-h3-s224` | 4.843 s | 4.630 s | 4.863 s | 1.171e-20 | 8.1 min |
| `known-10-o4-h1-s80` | 0.802 s | 0.786 s | 0.816 s | 1.772e-19 | 1.3 min |
| `known-11-o4-h1-s120` | 1.203 s | 1.196 s | 1.211 s | 2.062e-19 | 2.0 min |
| `known-12-o4-h2-s168` | 3.723 s | 3.639 s | 3.774 s | 7.632e-20 | 6.2 min |
| `known-13-o4-h2-s216` | 5.010 s | 4.760 s | 5.102 s | 4.633e-23 | 8.4 min |
| `known-14-o4-h3-s256` | 9.734 s | 9.696 s | 10.288 s | 7.625e-27 | 16.2 min |
| `known-15-o5-h1-s96` | 1.929 s | 1.922 s | 2.010 s | 5.335e-20 | 3.2 min |
| `known-16-o5-h2-s144` | 5.691 s | 5.465 s | 6.089 s | 3.017e-27 | 9.5 min |
| `known-17-o5-h2-s200` | 7.401 s | 7.301 s | 8.159 s | 2.294e-27 | 12.3 min |
| `known-18-o5-h3-s272` | 18.945 s | 18.582 s | 19.193 s | 2.028e-19 | 31.6 min |
| `known-19-o6-h2-s320` | 17.742 s | 17.302 s | 17.938 s | 4.508e-26 | 29.6 min |
| `known-20-epilepsy-scale-o4-h2-s4097` | 110.376 s | 109.786 s | 110.935 s | 5.340e-18 | 184.0 min |

The largest case matches the signal length, oscillator count, harmonic order, and target iteration count configured in
`examples/kursl_epilepsy.py`. The full 100-iteration runs were not executed.

All cases use independently generated uncoupled signals. Walkers start within a tiny nonzero neighborhood of the known
parameters, making this a strict local-correctness and scaling baseline rather than a convergence-from-poor-priors test.
