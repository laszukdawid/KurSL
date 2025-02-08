import numpy as np
from utils import random_theta

from kursl import KurSL, KurslMCMC, ModelWrapper


def test_mcmc_default():
    oscN, nH = 3, 2
    paramN = 3 + nH * (oscN - 1)
    theta_init = random_theta(oscN, nH)
    mcmc = KurslMCMC(theta_init)

    assert np.all(mcmc.theta_init == theta_init)
    assert mcmc.oscN == oscN
    assert mcmc.paramN == paramN
    assert mcmc.nH == nH

    assert mcmc.ndim == theta_init.size
    assert mcmc.nwalkers == 2 * theta_init.size
    assert mcmc.niter == 100

    assert mcmc.init_pos.shape == (2 * theta_init.size, theta_init.size)

    assert mcmc.threads == 1
    assert mcmc.save_iter == 10
    assert mcmc.THRESHOLD == 0.05
    assert mcmc.SAVE_INIT_POS is True

    assert isinstance(mcmc.model, ModelWrapper), (
        f"Default model should be a wrapper ({ModelWrapper.__name__}), but received {type(mcmc.model)} type"
    )
    assert isinstance(mcmc.model.model, KurSL), (
        f"Default model within wrapper should be KurSL ({KurSL.__name__}), but received {type(mcmc.model.model)} type"
    )


def test_run_default():
    def _cos(val):
        return val[2] * np.cos(val[0] * t + val[1])

    theta = [
        [15, 0, 1, 0],
        [35, 2, 3, 0],
    ]
    theta_std = [
        [1.0, 0.1, 0.8, 0.01],
        [1.5, 0.1, 1.5, 0.01],
    ]
    theta, theta_std = np.array(theta), np.array(theta_std)
    t = np.linspace(0, 1, 100)
    c1 = _cos(theta[0])
    c2 = _cos(theta[1])
    S = c1 + c2

    theta_init = np.array(theta, dtype=np.float64)
    theta_init[:, 0] += 2 - np.random.random(2) * 0.5
    theta_init[:, 2] += 1 - np.random.random(2) * 0.5
    mcmc = KurslMCMC(theta_init, theta_std, nwalkers=40, niter=200)
    mcmc.set_threshold(0.001)
    mcmc.set_sampler(t, S)
    mcmc.run()

    # After simulation is finished check for correctness
    theta_computed = mcmc.get_theta()
    assert np.allclose(theta, theta_computed, atol=0.5), "Expected:\n{}\nReceived:\n{}".format(theta, theta_computed)


def test_run_start_from_solution():
    def _cos(val):
        return val[2] * np.cos(val[0] * t + val[1])

    theta = np.array(
        [
            [15, 0, 1, 0],
            [35, 2, 3, 0],
        ],
        dtype=np.float64,
    )
    t = np.linspace(0, 1, 100)
    c1 = _cos(theta[0])
    c2 = _cos(theta[1])
    S = c1 + c2

    # Initial guess is far off. Just for an example.
    theta_init = np.array(theta, dtype=np.float64)
    theta_init += np.random.random(theta_init.shape) * 3
    mcmc = KurslMCMC(theta_init)
    mcmc.set_sampler(t, S)

    # However, execution starts close to solution.
    pos = np.tile(theta.flatten(), (mcmc.nwalkers, 1))
    pos += np.random.random(pos.shape) * 0.3
    mcmc.run(pos=pos)
    niter_executed = mcmc.get_lnprob().shape[0]
    assert niter_executed < mcmc.niter, (
        "Starting close solution should allow for quick convergence and not all iterations would be executed."
    )
