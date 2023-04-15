import numpy as np
import pytest

from kursl import KurSL, KurslMCMC, ModelWrapper

QUICK_TEST = True


def random_theta(oscN, nH):
    W = np.random.random((oscN, 1)) * 20 + 10
    P = np.random.random((oscN, 1)) * 6.28
    R = np.random.random((oscN, 1)) * 5 + 1
    K = np.random.random((oscN, nH * (oscN - 1))) * 2 - 1
    return np.hstack((W, P, R, K))


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
    assert mcmc.SAVE_INIT_POS == True

    assert isinstance(mcmc.model, ModelWrapper), "Default model should be a wrapper ({}), but received {} type".format(
        ModelWrapper.__name__, type(mcmc.model)
    )
    assert isinstance(
        mcmc.model.model, KurSL
    ), "Default model within wrapper should be KurSL ({}), but received {} type".format(
        KurSL.__name__, type(mcmc.model.model)
    )


def test_init_walkers():
    theta_init = np.array(
        [
            [115, 0, 3, 0, 1, 2, 0, -1, -3],
            [120, 1, 2, -1, 0, 1, 2, -3, 2],
            [135, 2, 9, -2, -2, -2, 0, -2, 0],
        ],
        dtype=np.float64,
    )
    oscN, paramN = theta_init.shape
    nH = int((paramN - 3) / (oscN - 1))
    assert nH == 3
    theta_std = np.random.random((oscN, paramN))

    mcmc = KurslMCMC(theta_init)
    mcmc._init_walkers(theta_init, theta_std)
    nwalkers = mcmc.nwalkers

    assert mcmc.init_pos.shape == (nwalkers, oscN * paramN)
    assert np.all(
        mcmc.init_pos[0] == theta_init.flatten()
    ), f"First walker should have the same values as passed. First walkers:\n{mcmc.init_pos[0]}\nPassed:\n{theta_init}"


def test_init_walkers_incorrect_theta_std():
    oscN, nH = 4, 3
    theta_init = random_theta(oscN, nH)
    theta_std = random_theta(oscN, nH + 1)
    mcmc = KurslMCMC(theta_init)
    with pytest.raises(ValueError) as error:
        mcmc._init_walkers(theta_init, theta_std)
    assert "Incorrect shape" in str(error)


def test_init_walkers_proper_scaling_with_K():
    """Whenever sum of coupling factors is higher than intrinsic frequency
    it should be scaled appropriately as otherwise imaginary numbers appear."""
    theta_init = np.array(
        [
            [10, 0, 1, 5, 5],
            [15, 0, 2, 5, 5],
            [20, 0, 2, 5, 30],
        ],
        dtype=np.float64,
    )

    mcmc = KurslMCMC(theta_init)
    theta_mcmc = mcmc.init_pos[0].reshape(theta_init.shape)
    assert theta_mcmc[0, 0] == 10, "Intrinsic freq stays the same"
    assert np.allclose(
        theta_mcmc[0, 3:], [4.9, 4.9]
    ), "If W==sum(K) all couplings are scaled by 0.98*W, same as for W<sum(K)."
    assert np.allclose(theta_mcmc[1, 3:], [5, 5]), "For W>sum(K) nothing should be changed."
    assert np.allclose(theta_mcmc[2, 3:], [2.8, 16.8]), "If W<sum(K) all couplings are scaled by 0.98*W"


def test_neg_log():
    x = np.array([0.999, 2.011, 10, np.exp(0)])
    y_exp = -3
    y_out = KurslMCMC.neg_log(x)
    assert abs(y_out - y_exp) < 1e-3


def test_set_threshold():
    theta_init = random_theta(5, 1)
    mcmc = KurslMCMC(theta_init)
    assert mcmc.THRESHOLD == 0.05
    assert mcmc.model.THRESHOLD == 0.05
    mcmc.set_threshold(0.1)
    assert mcmc.THRESHOLD == 0.1
    assert mcmc.model.THRESHOLD == 0.1


def test_set_model():
    """Assign newly defined model to use in MCMC.
    Model has to have `oscN` and `nH` properties."""

    class NewModel:
        def __init__(self):
            self.oscN = 2  # Number of oscillators
            self.nH = 2  # Number of coupling harmonics

    theta_init = np.random.random((3, 10))
    mcmc = KurslMCMC(theta_init)
    assert isinstance(mcmc.model, ModelWrapper)
    assert isinstance(mcmc.model.model, KurSL)

    new_model = NewModel()
    mcmc.set_model(new_model)
    assert isinstance(mcmc.model.model, NewModel)
    assert mcmc.model.THRESHOLD == 0.05


def test_set_sampler():
    t = np.linspace(0, 1, 100)
    S = np.random.random(t.size)
    s = S[:-1]
    s_var = np.sum((s - s.mean()) * (s - s.mean()))
    theta_init = random_theta(3, 2)

    mcmc = KurslMCMC(theta_init)
    assert mcmc.model.s_var == 1, "Default var is 1"
    assert mcmc.sampler is None, "Without explicit assignment there should be no sampler."

    mcmc.set_sampler(t, S)
    assert (
        type(mcmc.sampler).__name__ == "EnsembleSampler"
    ), "Executing `set_sampler` should create EnsembleSampler `sampler`"
    assert mcmc.model.s_var == s_var, "Updated var for the model"


@pytest.mark.skipif(QUICK_TEST, reason="Computation intensive and takes time")
def test_run_default():
    _cos = lambda l: l[2] * np.cos(l[0] * t + l[1])
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
    _cos = lambda l: l[2] * np.cos(l[0] * t + l[1])
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
        "Starting close solution should allow for quick convergence " "and not all iterations would be executed."
    )


def test_run_with_incorrect_pos():
    t = np.linspace(0, 1, 100)
    S = np.random.random(t.size)
    mcmc = KurslMCMC(random_theta(3, 2))
    mcmc.set_sampler(t, S)
    with pytest.raises(ValueError) as error:
        mcmc.run(pos=random_theta(3, 2))
    assert "pos.shape" in str(error)


def test_run_without_model():
    mcmc = KurslMCMC(random_theta(3, 2))
    mcmc.model = None  # This situation shouldn't even be possible
    with pytest.raises(AttributeError) as error:
        mcmc.run()
    assert "Model not selected" in str(error)


def test_run_without_sampler():
    mcmc = KurslMCMC(random_theta(3, 2))
    with pytest.raises(AttributeError) as error:
        mcmc.run()
    assert "Sampler not defined" in str(error)
    assert "set_sampler" in str(error)


def test_get_theta_without_running_sampler():
    theta_init = random_theta(3, 2)
    mcmc = KurslMCMC(theta_init)

    # After simulation is finished check for correctness
    with pytest.raises(AttributeError) as error:
        mcmc.get_theta()
    assert "theta" in str(error)
    assert "run()" in str(error)


def test_theta_computed_boolean():
    theta_init = random_theta(2, 1)
    mcmc = KurslMCMC(theta_init, niter=2)
    assert not mcmc._theta_computed(), "Without run should fail"
    mcmc.set_sampler(np.linspace(0, 1, 100), np.random.random(100))
    mcmc.run()
    assert mcmc._theta_computed(), "After run should return true"


def test_lnlikelihood():
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)
    theta = random_theta(3, 2)
    kursl = KurSL(theta)
    model = ModelWrapper(kursl)
    lnlikelihood = KurslMCMC.lnlikelihood(theta, t, S[:-1], model)
    assert lnlikelihood < 0, "Any negative value is good"


def test_lnlikelihood_zero():
    t = np.arange(0, 1, 0.01)
    S = 2 * np.cos(10 * t + 0) + 6 * np.cos(20 * t + 4)
    theta = np.array(
        [
            [10, 0, 2, 0],
            [20, 4, 6, 0],
        ]
    )
    kursl = KurSL(theta)
    model = ModelWrapper(kursl)
    lnlikelihood = KurslMCMC.lnlikelihood(theta, t, S[:-1], model)
    assert np.round(lnlikelihood, 10) == 0, "Exact reconstruction should return 0"
    assert model.THRESHOLD_OBTAINED, "0 is below any threshold"


def test_lnprior():
    t = np.arange(0, 1, 0.01)
    theta = random_theta(3, 2)
    kursl = KurSL(theta)
    model = ModelWrapper(kursl)
    lnprob = KurslMCMC.lnprior(theta, model)

    # Given default uniform probablities it's either 0 or -np.inf
    assert np.round(lnprob, 10) == 0, "Theta within max ranges results in 0"


def test_lnprior_theta_outside_ranges():
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)
    theta = random_theta(3, 2)
    kursl = KurSL(theta)
    model = ModelWrapper(kursl)

    new_theta = theta.copy()
    new_theta[0, 2] = model.MIN_R - 1
    assert KurslMCMC.lnprior(new_theta, model) == -np.inf
    new_theta[0, 2] = model.MAX_R + 1
    assert KurslMCMC.lnprior(new_theta, model) == -np.inf

    new_theta = theta.copy()
    new_theta[1, 0] = model.MIN_W - 1
    assert KurslMCMC.lnprior(new_theta, model) == -np.inf
    new_theta[1, 0] = model.MAX_W + 1
    assert KurslMCMC.lnprior(new_theta, model) == -np.inf

    # Check for W_i < sum_j (|k_ij|)
    new_theta = theta.copy()
    new_theta[2, 0] = np.sum(new_theta[2, 3:]) * 0.6
    assert KurslMCMC.lnprior(new_theta, model) == -np.inf


def test_lnprob():
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)
    theta = random_theta(4, 3)
    kursl = KurSL(theta)
    model = ModelWrapper(kursl)
    assert KurslMCMC.lnprob(theta, t, S[:-1], model) < 0, "Any good theta should return negative value"


def test_lnprob_theta_outside_range():
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)
    theta = random_theta(4, 3)
    kursl = KurSL(theta)
    model = ModelWrapper(kursl)

    theta[0, 0] = model.MIN_W - 1
    assert KurslMCMC.lnprob(theta, t, S[:-1], model) == -np.inf, "Inherit behaviour of lnprob and lnlikelihood"
