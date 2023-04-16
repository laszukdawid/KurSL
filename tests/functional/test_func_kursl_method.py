import pytest

import numpy as np

from kursl import KurslMethod


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

    oscN = kursl.oscN
    paramN = kursl.paramN
    # Testing for number
    assert oscN == 2, "2 oscillators"
    assert paramN == 4, "4 params per oscillator"
    assert params.shape == (2, 4), "Two oscillators (W, ph, A, K)"

    # Testing for frequency
    assert abs(params[0, 0] - 5 * 2 * np.pi) < 0.05, "Expected {} rad/s, Got {} [rad/s]".format(
        5 * 2 * np.pi, params[0, 0]
    )
    assert abs(params[1, 0] - 2 * 2 * np.pi) < 0.05, "Expected {} rad/s, Got {} [rad/s]".format(
        2 * 2 * np.pi, params[1, 0]
    )

    # Testing for phase
    assert abs(params[0, 1] - 1) < 0.001, "Expected phase {}, Got {}.".format(1, params[0, 1])
    assert abs(params[1, 1] - 0) < 0.001, "Expected phase {}, Got {}.".format(0, params[1, 1])

    # Testing for amplitude
    assert abs(params[0, 2] - 1.1) < 0.1, "Expected amp {}, Got {}.".format(1.1, params[0, 2])
    assert abs(params[1, 2] - 2) < 0.1, "Expected amp {}, Got {}.".format(2, params[1, 2])

    # Testing for coupling
    assert params[0, 3] == 0, "First->Second coupling should be 0"
    assert params[1, 3] == 0, "Second->First coupling should be 0"


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


def test_run_optimize_no_prior():
    "Tests for exception when there is no default theta_init"
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)
    kursl = KurslMethod()
    with pytest.raises(ValueError) as error:
        kursl.run_optimize(t, S)
    assert "No prior parameters were assigned." in str(error)


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
    assert kursl.oscN, oscN
    assert kursl.nH, nH
    assert np.all(
        theta == kursl.theta_init
    ), "After computing make sure it is assigned.\n" "Received\n{}\nGot\n{}".format(theta, kursl.theta_init)


def test_run_pre_post_optimizations():
    "Tests execution of run. Other tests should check correctness."
    t = np.arange(0, 1, 0.01)
    S = np.random.random(t.size)

    oscN, nH = 2, 1
    paramN = 3 + nH * (oscN - 1)
    opt_maxiter = 10
    options = {
        "niter": 10,
        "nwalkers": 20,
        "PREOPTIMIZE": True,
        "POSTOPTIMIZE": True,
        "opt_maxiter": opt_maxiter,
    }
    kursl = KurslMethod(nH=nH, max_osc=oscN, **options)
    assert kursl.PREOPTIMIZE, "With preoptmize"
    assert kursl.POSTOPTIMIZE, "With postoptmize"
    assert kursl.opt_maxiter == opt_maxiter

    theta = kursl.run(t, S)
    assert theta.shape == (oscN, paramN)
    assert kursl.oscN, oscN
    assert kursl.nH, nH
    assert np.all(
        theta == kursl.theta_init
    ), "After computing make sure it is assigned.\n" "Received\n{}\nGot\n{}".format(theta, kursl.theta_init)
