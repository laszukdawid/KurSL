# KurSL
[![codecov](https://codecov.io/gh/laszukdawid/KurSL/branch/master/graph/badge.svg)](https://codecov.io/gh/laszukdawid/KurSL)

## Overview

*KurSL* is a mathematical model, which provides explanation for interacting oscillatory patterns. It is constructed of two components: Kuramoto coupling and Sturm-Liouville eigenvalue equation. The first one explains coupling between oscillators, and the latter provides a general oscillatory model. To read more about the method 

## Examples

The example below was produced using the *KurSL* model. It was generated assuming three oscillators with up to second harmonic coupling.
For the code on how this example was generated please see `examples/kursl_example.py`.

<img src="./examples/KurSL_example.png" alt="Example of the KurSL model" width="500px"/>

## MCMC Benchmark

Run `task benchmark` (or `uv run python -m benchmarks.benchmark_mcmc`) to measure the legacy MCMC implementation on 20
deterministic signals with known parameters. Cases vary oscillator count, harmonic order, sample count, duration,
frequency, phase, and amplitude. The largest workload matches the 4,097-sample, four-oscillator, second-order
configuration in `examples/kursl_epilepsy.py`.

Each case measures complete MCMC iterations and projects a 100-iteration runtime. Every recovered parameter must be
within `1e-6` of its known value and relative reconstruction energy must be below `1e-10`; otherwise the benchmark fails
instead of recording a timing. Walkers start very close to the known parameters, so this is a strict local-correctness
and scaling benchmark, not a convergence benchmark from poor initial estimates.
