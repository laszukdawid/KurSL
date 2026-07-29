import argparse
import os
import platform
import statistics
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import scipy

from benchmarks.signals import KNOWN_MCMC_SIGNALS, KnownMCMCSignal
from kursl import KurSL, KurslMCMC


@dataclass(frozen=True)
class MCMCResult:
    theta: np.ndarray
    reconstruction: np.ndarray
    iterations: int


def fit(case: KnownMCMCSignal) -> MCMCResult:
    previous_directory = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as temporary_directory:
            os.chdir(temporary_directory)
            np.random.seed(case.seed)
            mcmc = KurslMCMC(
                case.theta_init,
                case.theta_std,
                nwalkers=case.nwalkers,
                niter=case.measured_iterations,
            )
            mcmc.set_threshold(0.0)
            mcmc.set_sampler(case.time, case.values)
            mcmc.sampler.random_state = np.random.RandomState(case.seed).get_state()
            theta = mcmc.run()
            iterations = mcmc.get_lnprob().shape[0]
    finally:
        os.chdir(previous_directory)

    reconstruction = np.sum(KurSL(theta).generate(case.time)[2], axis=0)
    return MCMCResult(theta, reconstruction, iterations)


def validate(case: KnownMCMCSignal, result: MCMCResult) -> None:
    np.testing.assert_equal(result.theta.shape, case.theta.shape)
    np.testing.assert_equal(result.reconstruction.shape, case.values.shape)
    np.testing.assert_equal(result.iterations, case.measured_iterations)
    np.testing.assert_allclose(result.theta[:, 0], case.theta[:, 0], rtol=0, atol=case.frequency_tolerance)
    np.testing.assert_allclose(result.theta[:, 1], case.theta[:, 1], rtol=0, atol=case.phase_tolerance)
    np.testing.assert_allclose(result.theta[:, 2], case.theta[:, 2], rtol=0, atol=case.amplitude_tolerance)
    np.testing.assert_allclose(result.theta[:, 3:], case.theta[:, 3:], rtol=0, atol=case.coupling_tolerance)

    residual = result.reconstruction - case.values
    relative_error = np.sum(residual * residual) / np.sum(case.values * case.values)
    np.testing.assert_array_less(relative_error, case.max_relative_error)


def measure(case: KnownMCMCSignal, repetitions: int) -> tuple[str, float, float, float]:
    durations = []
    for _ in range(repetitions):
        started_at = time.perf_counter()
        result = fit(case)
        durations.append(time.perf_counter() - started_at)
        validate(case, result)

    return case.name, statistics.median(durations), min(durations), max(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the deterministic KurSL MCMC baseline.")
    parser.add_argument("--repetitions", type=int, default=3, help="Measured runs per benchmark (default: 3).")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")

    print(f"Python {platform.python_version()} | NumPy {np.__version__} | SciPy {scipy.__version__}")
    print(f"{'benchmark':<28} {'iter':>6} {'median':>10} {'min':>10} {'max':>10} {'projected':>12}")
    for case in KNOWN_MCMC_SIGNALS:
        name, median, minimum, maximum = measure(case, args.repetitions)
        projected = median * case.target_iterations / case.measured_iterations
        print(
            f"{name:<28} {case.measured_iterations:>6} {median:>9.3f}s {minimum:>9.3f}s "
            f"{maximum:>9.3f}s {projected / 60:>10.1f} min"
        )


if __name__ == "__main__":
    main()
