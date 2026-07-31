import pytest

from benchmarks.benchmark_mcmc import fit, validate
from benchmarks.signals import KNOWN_MCMC_SIGNALS


def test_corpus_contains_twenty_cases():
    assert len(KNOWN_MCMC_SIGNALS) == 20


@pytest.mark.parametrize("case", KNOWN_MCMC_SIGNALS, ids=lambda case: case.name)
def test_known_mcmc_signal(case):
    validate(case, fit(case))
