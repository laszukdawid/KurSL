# KurSL
[![codecov](https://codecov.io/gh/laszukdawid/KurSL/branch/master/graph/badge.svg)](https://codecov.io/gh/laszukdawid/KurSL)

## Overview

*KurSL* is a mathematical model, which provides explanation for interacting oscillatory patterns. It is constructed of two components: Kuramoto coupling and Sturm-Liouville eigenvalue equation. The first one explains coupling between oscillators, and the latter provides a general oscillatory model. To read more about the method 

## Examples

The example below was produced using the *KurSL* model. It was generated assuming three oscillators with up to second harmonic coupling.
For the code on how this example was generated please see `examples/kursl_example.py`.

<img src="./examples/KurSL_example.png" alt="Example of the KurSL model" width="500px"/>

## MCMC Benchmark

Run `task benchmark` (or `uv run python -m benchmarks.benchmark_mcmc`) to measure the legacy MCMC implementation on
deterministic signals with known parameters. The suite includes a quick fit and a representative workload matching the
4,097-sample, four-oscillator, second-order configuration in `examples/kursl_epilepsy.py`. The representative case
measures complete MCMC iterations and projects the configured 100-iteration runtime. Every run is validated; incorrect
parameter recovery, shape, iteration count, or reconstruction error fails the benchmark instead of recording a timing.
