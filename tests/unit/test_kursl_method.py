import pytest

import numpy as np

from kursl import KurslMethod


def test_default_init():
    kursl = KurslMethod()
    assert kursl.nH == 1
    assert kursl.max_osc == -1
    assert kursl.ptype == "norm"
    assert kursl.energy_ratio == 0.1
    assert kursl.f_min == 0
    assert kursl.f_max == 1e10

    assert kursl.nwalkers == 40
    assert kursl.niter == 100

    assert kursl.theta_init is None
    assert kursl.samples is None
    assert kursl.lnprob is None


def test_set_options():
    kursl = KurslMethod()
    assert kursl.niter == 100

    options = {"niter": 200, "missing": None}
    kursl.set_options(options)
    assert kursl.niter == 200


@pytest.mark.skip("Functional test")
def test_compute_prior_default():
    """Test default setting for computing prior parameters.
    Assumes single order for KurSL."""
    t = np.arange(0, 5, 0.001)
    c1 = 2 * np.cos(2 * 2 * np.pi * t + 0)
    c2 = 1.1 * np.cos(5 * 2 * np.pi * t + 1)
    S = c1 + c2

    kursl = KurslMethod()
    kursl.compute_prior(t, S)
    params = kursl.theta_init
    params[:, 1] = (params[:, 1] + 2 * np.pi) % (2 * np.pi)

    # Testing for number
    assert kursl.oscN == 2, "2 oscillators"
    assert kursl.paramN == 4, "4 params per oscillator"
    assert params.shape == (2, 4), "Two oscillators (W, ph, A, K)"

    # Testing for frequency
    assert abs(params[0, 0] - 5 * 2 * np.pi) < 0.05, f"Expected {10*np.pi} rad/s, Got {params[0,0]} rad/s"
    assert abs(params[1, 0] - 2 * 2 * np.pi) < 0.05, f"Expected {4*np.pi} rad/s, Got {params[1,0]} rad/s"

    # Testing for phase
    assert abs(params[0, 1] - 1) < 0.001, f"Expected phase {1}, Got {params[0, 1]}."
    assert abs(params[1, 1] - 0) < 0.001, f"Expected phase {0}, Got {params[1, 1]}."

    # Testing for amplitude
    assert abs(params[0, 2] - 1.1) < 0.1, f"Expected amp {1.1}, Got {params[0, 2]}."
    assert abs(params[1, 2] - 2) < 0.1, f"Expected amp {2}, Got {params[1, 2]}."

    # Testing for coupling
    assert params[0, 3] == 0, "First->Second coupling should be 0"
    assert params[1, 3] == 0, "Second->First coupling should be 0"


@pytest.mark.skip("Functional test")
def test_compute_prior_high_order():
    t = np.arange(0, 5, 0.001)
    c1 = 2 * np.cos(2 * 2 * np.pi * t + 0)
    c2 = 1.1 * np.cos(5 * 2 * np.pi * t + 1)
    c3 = 4.3 * np.cos(8 * 2 * np.pi * t + 5)
    S = c1 + c2 + c3

    nH = 4
    oscN, paramN = 3, 3 + nH * 2
    kursl = KurslMethod(nH)
    kursl.compute_prior(t, S)
    params = kursl.theta_init

    assert kursl.nH == nH
    assert kursl.oscN == 3, "Three oscillators"
    assert params.shape == (oscN, paramN), "Expected number of parameters: oscN*( 3 + nH*(oscN-1) )"


def test_set_prior_default_setting():
    "Sets priors for default kursl settings"
    kursl = KurslMethod()
    assert kursl.theta_init is None
    oscN, nH = 3, 1
    params = np.random.random((oscN, 3 + nH * (oscN - 1)))
    kursl.set_prior(params)
    assert np.all(kursl.theta_init == params), "Same array is expected"


def test_set_prior_nondefault_setting():
    oscN, nH = 3, 3
    kursl = KurslMethod(nH)
    params = np.random.random((oscN, 3 + nH * (oscN - 1)))
    kursl.set_prior(params)
    assert np.all(kursl.theta_init == params)
    assert kursl.oscN == oscN
    assert kursl.nH == nH
    assert kursl.paramN == params.shape[1]


def test_set_prior_incorrect_shape_new_theta():
    oscN, nH = 3, 3
    kursl = KurslMethod(nH)
    params = np.random.random((oscN, oscN))
    with pytest.raises(ValueError) as error:
        kursl.set_prior(params)
    assert "incorrect shape" in str(error)


def test_set_prior_incorrect_shape_update_theta():
    oscN, nH, paramN = 3, 1, 5
    kursl = KurslMethod(nH)
    params = np.random.random((oscN, paramN))
    kursl.set_prior(params)

    new_params = np.random.random((oscN, paramN + 2))
    with pytest.raises(ValueError) as error:
        kursl.set_prior(new_params)
    assert "incorrect shape" in str(error)

    assert np.all(kursl.theta_init == params), "Unsuccessful update shouldn't modify theta"


def test_detrend_default():
    "Default detrend is mean"
    S = np.random.random(100) * 2 + 1
    S_default = KurslMethod.detrend(S)
    assert np.round(np.mean(S_default), 7) == 0, "Default detrend is mean removing"


def test_detrend_mean():
    S = np.random.random(100) * 2 - 1
    S_nomean = KurslMethod.detrend(S, remove_type="mean")
    assert np.round(np.mean(S_nomean), 7) == 0, "Removing mean should return mean 0"


def test_detrend_cubic():
    t = np.arange(0, 1, 0.005)
    S = t**3 + 2 * (t - 0.5) ** 2 + 5
    S_nocubic = KurslMethod.detrend(S, remove_type="cubic")
    assert np.round(np.mean(S_nocubic), 7) == 0, "Removing mean should return mean 0"


def test_cost_lnprob_zeros():
    (X, Y) = np.zeros((2, 100))
    like = KurslMethod.cost_lnprob(X, Y)
    assert like == 0


def test_cost_lnprob_same():
    X = np.random.random(100)
    Y = X.copy()
    like = KurslMethod.cost_lnprob(X, Y)
    assert like == 0, "Same processes produce 0 lnprob"


def test_cost_lnprob():
    t = np.arange(0, 1, 0.005)
    X = t - 3
    Y = t * t - 1.0 / (t + 1)
    like = KurslMethod.cost_lnprob(X, Y)
    expected_like, eps = -459.6, 1e-1
    assert abs(like - expected_like) < eps, "Expected likelihood ({} +- {}), but got {}".format(
        expected_like, eps, like
    )


@pytest.mark.skip("Functional test")
def test_cost_function():
    """Simple case when there is no coupling"""
    t = np.arange(0, 1, 0.005)
    params = np.array(
        [
            [10, 0, 1, 0, 0],
            [18, 0, 2, 0, 0],
            [26, 0, 3, 0, 0],
        ]
    )

    # Each oscilaltor is params[osc, :]
    c1 = 1 * np.cos(10 * t + 0)
    c2 = 2 * np.cos(18 * t + 0)
    c3 = 3 * np.cos(26 * t + 0)
    S = c1 + c2 + c3

    kursl = KurslMethod()
    cost = kursl.cost_function(t, params, S)
    assert np.round(cost, 6) == 0, "Cost should be 0. Any difference could be due to ODE precision"


@pytest.mark.skip("Functional test")
def test_run_optimize_default():
    """Tests optmization with default scipy optimization method,
    which is L-BFGS and so it takes a while.
    """
    t = np.arange(0, 1, 0.01)
    theta = np.array(
        [
            [10, 0, 1, 0],
            [18, 0, 2, 0],
        ]
    )
    c1 = theta[0, 2] * np.cos(theta[0, 0] * t + theta[0, 1])
    c2 = theta[1, 2] * np.cos(theta[1, 0] * t + theta[1, 1])
    S = c1 + c2

    theta_init = theta.copy()
    theta_init[0, 0] -= 1
    theta_init[1, 0] += 1

    kursl = KurslMethod()
    _theta = kursl.run_optimize(t, S, theta_init=theta_init, maxiter=20)
    assert _theta.shape == theta.shape, "Results in same shape"
    assert np.allclose(theta, _theta, rtol=1e-1, atol=1e-1), "Expecting fit to be similar to theta initial value"


def test_run_mcmc_default():
    """Tests for correctness in execution, not for results correctness.
    For the latter see KurslMCMC testing."""
    t = np.arange(0, 1, 0.01)
    theta = np.array(
        [
            [10, 0, 1, 0],
            [18, 0, 2, 0],
        ]
    )
    c1 = theta[0, 2] * np.cos(theta[0, 0] * t + theta[0, 1])
    c2 = theta[1, 2] * np.cos(theta[1, 0] * t + theta[1, 1])
    S = c1 + c2

    theta_init = theta.copy()
    theta_init[0, 0] -= 1
    theta_init[1, 0] += 1

    options = {"niter": 10, "nwalkers": 20}
    kursl = KurslMethod(**options)
    _theta = kursl.run_mcmc(t, S, theta_init=theta_init)
    assert _theta.shape == theta.shape, "Results in same shape"


def test_run():
    "Tests execution of run. Other tests should check correctness."
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)

    oscN, nH = 2, 1
    paramN = 3 + nH * (oscN - 1)
    options = {"niter": 10, "nwalkers": 20}
    kursl = KurslMethod(nH=nH, max_osc=oscN, **options)
    assert not kursl.PREOPTIMIZE, "No preoptmize"
    assert not kursl.POSTOPTIMIZE, "No postoptmize"

    theta = kursl.run(t, S)
    assert theta.shape == (oscN, paramN)
    assert kursl.oscN == oscN
    assert kursl.nH == nH
    assert np.all(
        theta == kursl.theta_init
    ), "After computing make sure it is assigned.\n" "Received\n{}\nGot\n{}".format(theta, kursl.theta_init)


def test_run_custom_theta():
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)

    oscN, nH = 2, 2
    theta_init = np.random.random((oscN, 3 + nH * (oscN - 1)))
    theta_init = np.array(
        [
            [10, 0, 2, 0, 0],
            [20, 0, 5, 0, 1],
        ]
    )
    options = {"niter": 10, "nwalkers": 20}
    kursl = KurslMethod(nH=nH, **options)
    theta = kursl.run(t, S, theta_init=theta_init)

    assert theta.shape == theta_init.shape, "Shape shouldn't change"
    assert kursl.nH == nH
    assert kursl.oscN == theta_init.shape[0]
    assert kursl.paramN == theta_init.shape[1]
