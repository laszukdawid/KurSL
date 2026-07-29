from benchmarks.benchmark_mcmc import fit, validate
from benchmarks.signals import KNOWN_MCMC_SIGNALS


def test_known_mcmc_signal():
    case = KNOWN_MCMC_SIGNALS[0]
    validate(case, fit(case))


def test_representative_signal_configuration():
    case = KNOWN_MCMC_SIGNALS[1]
    assert case.name == "epilepsy-scale-4x4097"
    assert case.theta.shape == (4, 9)
    assert case.time.size == 4_097
    assert case.nwalkers == 72
    assert case.measured_iterations == 1
    assert case.target_iterations == 100
