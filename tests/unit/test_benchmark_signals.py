import pytest

from benchmarks.benchmark_mcmc import fit, validate
from benchmarks.signals import KNOWN_MCMC_SIGNALS


@pytest.mark.parametrize("case", KNOWN_MCMC_SIGNALS, ids=lambda case: case.name)
def test_known_mcmc_signal(case):
    validate(case, fit(case))
