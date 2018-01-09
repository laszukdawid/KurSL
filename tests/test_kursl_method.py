#!/usr/bin/python
import numpy as np
import unittest

from kursl.kursl_method import KurslMethod

QUICK_TEST = False

class TestKurslMethod(unittest.TestCase):

    def test_default_init(self):
        kursl = KurslMethod()
        self.assertEqual(kursl.nH, 1)
        self.assertEqual(kursl.max_osc, -1)
        self.assertEqual(kursl.ptype, "norm")
        self.assertEqual(kursl.energy_ratio, 0.1)
        self.assertEqual(kursl.f_min, 0)
        self.assertEqual(kursl.f_max, 1e10)

        self.assertEqual(kursl.nwalkers, 20)
        self.assertEqual(kursl.niter, 50)

        self.assertEqual(kursl.theta_init, None)
        self.assertEqual(kursl.samples, None)
        self.assertEqual(kursl.lnprob, None)

    def test_set_options(self):
        kursl = KurslMethod()
        self.assertEqual(kursl.niter, 50)

        options = {"niter": 100, "missing": None}
        kursl.set_options(options)
        self.assertEqual(kursl.niter, 100)

    def test_compute_prior_default(self):
        """Test default setting for computing prior parameters.
        Assumes single order for KurSL."""
        t = np.arange(0, 5, 0.001)
        c1 = 2*np.cos(2*2*np.pi*t + 0)
        c2 = 1.1*np.cos(5*2*np.pi*t + 1)
        S = c1 + c2

        kursl = KurslMethod()
        kursl.compute_prior(t, S)
        params = kursl.theta_init
        params[:, 1] = (params[:,1] + 2*np.pi) % (2*np.pi)

        oscN = kursl.oscN
        paramN = kursl.paramN
        # Testing for number
        self.assertEqual(oscN, 2, "2 oscillators")
        self.assertEqual(paramN, 4, "4 params per oscillator")
        self.assertEqual(params.shape, (2,4), "Two oscillators (W, ph, A, K)")

        # Testing for frequency
        self.assertTrue(abs(params[0,0]-5*2*np.pi)<0.05,
            "Expected {} rad/s, Got {} [rad/s]".format(5*2*np.pi, params[0,0]))
        self.assertTrue(abs(params[1,0]-2*2*np.pi)<0.05,
            "Expected {} rad/s, Got {} [rad/s]".format(2*2*np.pi, params[1,0]))

        # Testing for phase
        self.assertTrue(abs(params[0,1]-1)<0.001,
                "Expected phase {}, Got {}.".format(1, params[0,1]))
        self.assertTrue(abs(params[1,1]-0)<0.001,
                "Expected phase {}, Got {}.".format(0, params[1,1]))

        # Testing for amplitude
        self.assertTrue(abs(params[0,2]-1.1)<0.1,
                "Expected amp {}, Got {}.".format(1.1, params[0,2]))
        self.assertTrue(abs(params[1,2]-2)<0.1,
                "Expected amp {}, Got {}.".format(2, params[1,2]))

        # Testing for coupling
        self.assertEqual(params[0,3], 0, "First->Second coupling should be 0")
        self.assertEqual(params[1,3], 0, "Second->First coupling should be 0")

    def test_compute_prior_high_order(self):
        t = np.arange(0, 5, 0.001)
        c1 = 2*np.cos(2*2*np.pi*t + 0)
        c2 = 1.1*np.cos(5*2*np.pi*t + 1)
        c3 = 4.3*np.cos(8*2*np.pi*t + 5)
        S = c1 + c2 + c3

        nH = 4
        oscN, paramN = 3, 3+nH*2
        kursl = KurslMethod(nH)
        kursl.compute_prior(t, S)
        params = kursl.theta_init

        self.assertEqual(kursl.nH, nH)
        self.assertEqual(kursl.oscN, 3, "Three oscillators")
        self.assertEqual(params.shape, (oscN, paramN),
                "Expected number of parameters: oscN*( 3 + nH*(oscN-1) )")

    def test_set_prior_default_setting(self):
        "Sets priors for default kursl settings"
        kursl = KurslMethod()
        self.assertEqual(kursl.theta_init, None)
        oscN, nH = 3, 1
        params = np.random.random((oscN, 3 + nH*(oscN-1)))
        kursl.set_prior(params)
        self.assertTrue(np.all(kursl.theta_init==params),
                "Same array is expected")

    def test_set_prior_nondefault_setting(self):
        oscN, nH = 3, 3
        kursl = KurslMethod(nH)
        params = np.random.random((oscN, 3 + nH*(oscN-1)))
        kursl.set_prior(params)
        self.assertTrue(np.all(kursl.theta_init==params))
        self.assertEqual(kursl.oscN, oscN)
        self.assertEqual(kursl.nH, nH)
        self.assertEqual(kursl.paramN, params.shape[1])

    def test_set_prior_incorrect_shape_new_theta(self):
        oscN, nH = 3, 3
        kursl = KurslMethod(nH)
        params = np.random.random((oscN, oscN))
        with self.assertRaises(ValueError) as error:
            kursl.set_prior(params)
        self.assertTrue("incorrect shape" in str(error.exception))

    def test_set_prior_incorrect_shape_update_theta(self):
        oscN, nH, paramN = 3, 1, 5
        kursl = KurslMethod(nH)
        params = np.random.random((oscN, paramN))
        kursl.set_prior(params)

        new_params = np.random.random((oscN, paramN+2))
        with self.assertRaises(ValueError) as error:
            kursl.set_prior(new_params)
        self.assertTrue("incorrect shape" in str(error.exception))

        self.assertTrue(np.all(kursl.theta_init==params),
                "Unsuccessful update shouldn't modify theta")
    def test_detrend_default(self):
        "Default detrend is mean"
        S = np.random.random(100)*2 + 1
        S_default = KurslMethod.detrend(S)
        self.assertEqual(np.round(np.mean(S_default), 7), 0,
                "Default detrend is mean removing")

    def test_detrend_mean(self):
        S = np.random.random(100)*2 - 1
        S_nomean = KurslMethod.detrend(S, remove_type="mean")
        self.assertEqual(np.round(np.mean(S_nomean), 7), 0,
                "Removing mean should return mean 0")

    def test_detrend_cubic(self):
        t = np.arange(0, 1, 0.005)
        S = t**3 + 2*(t-0.5)**2 + 5
        S_nocubic = KurslMethod.detrend(S, remove_type="cubic")
        self.assertEqual(np.round(np.mean(S_nocubic), 7), 0,
                "Removing mean should return mean 0")

    def test_cost_lnprob_zeros(self):
        (X, Y) = np.zeros((2, 100))
        like = KurslMethod.cost_lnprob(X, Y)
        self.assertEqual(like, 0)

    def test_cost_lnprob_same(self):
        X = np.random.random(100)
        Y = X.copy()
        like = KurslMethod.cost_lnprob(X, Y)
        self.assertEqual(like, 0, "Same processes produce 0 lnprob")

    def test_cost_lnprob(self):
        t = np.arange(0, 1, 0.005)
        X = t-3
        Y = t*t - 1./(t+1)
        like = KurslMethod.cost_lnprob(X, Y)
        expected_like, eps = -459.6, 1e-1
        self.assertTrue(abs(like - expected_like)<eps,
                    "Expected likelihood ({} +- {}), but got {}".format(
                        expected_like, eps, like))

    def test_cost_function(self):
        """Simple case when there is no coupling"""
        t = np.arange(0, 1, 0.005)
        params = np.array([[10, 0, 1, 0, 0],
                           [18, 0, 2, 0, 0],
                           [26, 0, 3, 0, 0],
                          ])

        # Each oscilaltor is params[osc, :]
        c1 = 1*np.cos(10*t + 0)
        c2 = 2*np.cos(18*t + 0)
        c3 = 3*np.cos(26*t + 0)
        S = c1 + c2 + c3

        kursl = KurslMethod()
        cost = kursl.cost_function(t, params, S)
        self.assertEqual(np.round(cost, 6), 0,
                "Cost should be 0. Any difference could be due to ODE precision")

    @unittest.skipIf(QUICK_TEST,
                "Skipping in quick runs as it takes too long to execute.")
    def test_run_optimize_default(self):
        """Tests optmization with default scipy optimization method,
        which is L-BFGS and so it takes a while.
        """
        t = np.arange(0, 1, 0.01)
        theta = np.array([[10, 0, 1, 0],
                          [18, 0, 2, 0],
                         ])
        c1 = theta[0,2]*np.cos(theta[0,0]*t + theta[0,1])
        c2 = theta[1,2]*np.cos(theta[1,0]*t + theta[1,1])
        S = c1 + c2

        theta_init = theta.copy()
        theta_init[0,0] -= 1
        theta_init[1,0] += 1

        kursl = KurslMethod()
        _theta = kursl.run_optimize(t, S, theta_init=theta_init, maxiter=20)
        self.assertEqual(_theta.shape, theta.shape, "Results in same shape")
        self.assertTrue(np.allclose(theta, _theta, rtol=1e-1, atol=1e-1),
                "Expecting fit to be similar to theta initial value")

    def test_run_optimize_no_prior(self):
        "Tests for exception when there is no default theta_init"
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)
        kursl = KurslMethod()
        with self.assertRaises(ValueError) as error:
            kursl.run_optimize(t, S)
        self.assertEqual("No prior parameters were assigned.", str(error.exception))


    @unittest.skipIf(QUICK_TEST,
                "Skipping in quick runs as it takes too long to execute.")
    def test_run_mcmc_default(self):
        """Tests for correctness in execution, not for results correctness.
        For the latter see KurslMCMC testing."""
        t = np.arange(0, 1, 0.01)
        theta = np.array([[10, 0, 1, 0],
                          [18, 0, 2, 0],
                         ])
        c1 = theta[0,2]*np.cos(theta[0,0]*t + theta[0,1])
        c2 = theta[1,2]*np.cos(theta[1,0]*t + theta[1,1])
        S = c1 + c2

        theta_init = theta.copy()
        theta_init[0,0] -= 1
        theta_init[1,0] += 1

        options = {"niter": 10, "nwalkers": 20}
        kursl = KurslMethod(**options)
        _theta = kursl.run_mcmc(t, S, theta_init=theta_init)
        self.assertEqual(_theta.shape, theta.shape, "Results in same shape")

    def test_run_mcmc_no_prior(self):
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)
        kursl = KurslMethod()
        with self.assertRaises(ValueError) as error:
            kursl.run_mcmc(t, S)

        self.assertEqual("No prior parameters were assigned.", str(error.exception))

    @unittest.skipIf(QUICK_TEST,
                "Skipping in quick runs as it takes too long to execute.")
    def test_run(self):
        "Tests execution of run. Other tests should check correctness."
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)

        oscN, nH = 2, 1
        paramN = 3 + nH*(oscN-1)
        options = {"niter":10, "nwalkers":20}
        kursl = KurslMethod(nH=nH, max_osc=oscN, **options)
        self.assertFalse(kursl.PREOPTIMIZE, "No preoptmize")
        self.assertFalse(kursl.POSTOPTIMIZE, "No postoptmize")

        theta = kursl.run(t, S)
        self.assertEqual(theta.shape, (oscN, paramN))
        self.assertTrue(kursl.oscN, oscN)
        self.assertTrue(kursl.nH, nH)
        self.assertTrue(np.all(theta==kursl.theta_init),
                "After computing make sure it is assigned.\n"
                "Received\n{}\nGot\n{}".format(theta, kursl.theta_init))

    def test_run_custom_theta(self):
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)

        oscN, nH = 2, 2
        theta_init = np.random.random((oscN, 3+nH*(oscN-1)))
        theta_init = np.array([[10, 0, 2, 0, 0],
                              [20, 0, 5, 0, 1],
                             ])
        options = {"niter":10, "nwalkers":20}
        kursl = KurslMethod(nH=nH, **options)
        theta = kursl.run(t, S, theta_init=theta_init)

        self.assertEqual(theta.shape, theta_init.shape,
                "Shape shouldn't change")
        self.assertEqual(kursl.nH, nH)
        self.assertEqual(kursl.oscN, theta_init.shape[0])
        self.assertEqual(kursl.paramN, theta_init.shape[1])

    def test_run_pre_post_optimizations(self):
        "Tests execution of run. Other tests should check correctness."
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)

        oscN, nH = 2, 1
        paramN = 3 + nH*(oscN-1)
        opt_maxiter = 10
        options = {"niter":10, "nwalkers":20,
                "PREOPTIMIZE":True, "POSTOPTIMIZE":True,
                "opt_maxiter":opt_maxiter}
        kursl = KurslMethod(nH=nH, max_osc=oscN, **options)
        self.assertTrue(kursl.PREOPTIMIZE, "With preoptmize")
        self.assertTrue(kursl.POSTOPTIMIZE, "With postoptmize")
        self.assertEqual(kursl.opt_maxiter, opt_maxiter)

        theta = kursl.run(t, S)
        self.assertEqual(theta.shape, (oscN, paramN))
        self.assertTrue(kursl.oscN, oscN)
        self.assertTrue(kursl.nH, nH)
        self.assertTrue(np.all(theta==kursl.theta_init),
                "After computing make sure it is assigned.\n"
                "Received\n{}\nGot\n{}".format(theta, kursl.theta_init))
