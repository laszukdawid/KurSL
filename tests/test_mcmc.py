import numpy as np
import unittest

from kursl import KurSL
from kursl import KurslMCMC
from kursl import ModelWrapper

QUICK_TEST = False


class TestMCMC(unittest.TestCase):

    @staticmethod
    def random_theta(oscN, nH):
        W = np.random.random((oscN,1))*20 + 10
        P = np.random.random((oscN,1))*6.28
        R = np.random.random((oscN,1))*5 + 1
        K = np.random.random((oscN, nH*(oscN-1)))*2 - 1
        return np.hstack((W, P, R, K))

    def test_mcmc_default(self):
        oscN, nH = 3, 2
        paramN = 3 + nH*(oscN-1)
        theta_init = self.random_theta(oscN, nH)
        mcmc = KurslMCMC(theta_init)

        self.assertTrue(np.all(mcmc.theta_init==theta_init))
        self.assertEqual(mcmc.oscN, oscN)
        self.assertEqual(mcmc.paramN, paramN)
        self.assertEqual(mcmc.nH, nH)

        self.assertEqual(mcmc.ndim, theta_init.size)
        self.assertEqual(mcmc.nwalkers, 2*theta_init.size)
        self.assertEqual(mcmc.niter, 100)

        self.assertTrue(mcmc.init_pos.shape, (2*theta_init.size, theta_init.size))

        self.assertEqual(mcmc.threads, 1)
        self.assertEqual(mcmc.save_iter, 10)
        self.assertEqual(mcmc.THRESHOLD, 0.05)
        self.assertEqual(mcmc.SAVE_INIT_POS, True)

        self.assertTrue(isinstance(mcmc.model, ModelWrapper),
                "Default model should be a wrapper ({}), but received "
                "{} type".format(ModelWrapper.__name__, type(mcmc.model)))
        self.assertTrue(isinstance(mcmc.model.model, KurSL),
                "Default model within wrapper should be KurSL ({}), but received "
                "{} type".format(KurSL.__name__, type(mcmc.model.model)))

    def test_init_walkers(self):
        theta_init = np.array([[115, 0, 3,  0,  1,  2, 0, -1, -3],
                               [120, 1, 2, -1,  0,  1, 2, -3,  2],
                               [135, 2, 9, -2, -2, -2, 0, -2,  0],
                              ], dtype=np.float64)
        oscN, paramN = theta_init.shape
        nH = int((paramN-3)/(oscN-1))
        self.assertEqual(nH, 3)
        theta_std = np.random.random((oscN, paramN))

        mcmc = KurslMCMC(theta_init)
        mcmc._init_walkers(theta_init, theta_std)
        nwalkers = mcmc.nwalkers

        self.assertEqual(mcmc.init_pos.shape, (nwalkers, oscN*paramN))
        self.assertTrue(np.all(mcmc.init_pos[0]==theta_init.flatten()),
                "First walker should have the same values as passed. "
                "First walkers:\n{}\nPassed:\n{}".format(mcmc.init_pos[0], theta_init))

    def test_init_walkers_incorrect_theta_std(self):
        oscN, nH = 4, 3
        theta_init = self.random_theta(oscN, nH)
        theta_std = self.random_theta(oscN, nH+1)
        mcmc = KurslMCMC(theta_init)
        with self.assertRaises(ValueError) as error:
            mcmc._init_walkers(theta_init, theta_std)
        self.assertTrue("Incorrect shape" in str(error.exception))

    def test_init_walkers_proper_scaling_with_K(self):
        """Whenever sum of coupling factors is higher than intrinsic frequency
        it should be scaled appropriately as otherwise imaginary numbers appear."""
        theta_init = np.array([[10, 0, 1, 5, 5],
                               [15, 0, 2, 5, 5],
                               [20, 0, 2, 5, 30],
                              ], dtype=np.float64)

        mcmc = KurslMCMC(theta_init)
        theta_mcmc = mcmc.init_pos[0].reshape(theta_init.shape)
        self.assertEqual(theta_mcmc[0,0], 10, "Intrinsic freq stays the same")
        self.assertTrue(np.allclose(theta_mcmc[0,3:], [4.9, 4.9]),
                "If W==sum(K) all couplings are scaled by 0.98*W, same "
                "as for W<sum(K).\nReceived: {}\nExpected: {}".format( \
                        theta_mcmc[0,3:], [4.9, 4.9]))
        self.assertTrue(np.allclose(theta_mcmc[1,3:], [5, 5]),
                "For W>sum(K) nothing should be changed."
                "\nReceived: {}\nExpected: {}".format(theta_mcmc[1,3:], [5, 5]))
        self.assertTrue(np.allclose(theta_mcmc[2,3:], [2.8, 16.8]),
                "If W<sum(K) all couplings are scaled by 0.98*W"
                "\nReceived: {}\nExpected: {}".format(theta_mcmc[2,3:], [2.8, 16.8]))

    def test_neg_log(self):
        x = np.array([0.999, 2.011, 10, np.exp(0)])
        y_exp = -3
        y_out = KurslMCMC.neg_log(x)
        self.assertTrue(abs(y_out-y_exp)<1e-3)

    def test_set_threshold(self):
        theta_init = self.random_theta(5, 1)
        mcmc = KurslMCMC(theta_init)
        self.assertEqual(mcmc.THRESHOLD, 0.05)
        self.assertEqual(mcmc.model.THRESHOLD, 0.05)
        mcmc.set_threshold(0.1)
        self.assertEqual(mcmc.THRESHOLD, 0.1)
        self.assertEqual(mcmc.model.THRESHOLD, 0.1)

    def test_set_model(self):
        """Assign newly defined model to use in MCMC.
        Model has to have `oscN` and `nH` properties."""

        class NewModel:
            def __init__(self):
                self.oscN = 2 # Number of oscillators
                self.nH = 2   # Number of coupling harmonics
        theta_init = np.random.random((3, 10))
        mcmc = KurslMCMC(theta_init)
        self.assertTrue(isinstance(mcmc.model, ModelWrapper))
        self.assertTrue(isinstance(mcmc.model.model, KurSL))

        new_model = NewModel()
        mcmc.set_model(new_model)
        self.assertTrue(isinstance(mcmc.model.model, NewModel))
        self.assertEqual(mcmc.model.THRESHOLD, 0.05)

    def test_set_sampler(self):
        t = np.linspace(0, 1, 100)
        S = np.random.random(t.size)
        s = S[:-1]
        s_var = np.sum((s-s.mean())*(s-s.mean()))
        theta_init = self.random_theta(3, 2)

        mcmc = KurslMCMC(theta_init)
        self.assertEqual(mcmc.model.s_var, 1, "Default var is 1")
        self.assertTrue(mcmc.sampler is None,
                "Without explicit assignment there should be no sampler.")

        mcmc.set_sampler(t, S)
        self.assertEqual(type(mcmc.sampler).__name__, "EnsembleSampler",
                "Executing `set_sampler` should create EnsembleSampler `sampler`")
        self.assertEqual(mcmc.model.s_var, s_var, "Updated var for the model")

    @unittest.skipIf(QUICK_TEST, "Computation intensive and takes time")
    def test_run_default(self):
        _cos = lambda l: l[2]*np.cos(l[0]*t + l[1])
        theta = [[15, 0, 1, 0],
                 [35, 2, 3, 0],
                ]
        theta_std = [[1.0, 0.1, 0.8, 0.01],
                     [1.5, 0.1, 1.5, 0.01],
                    ]
        theta, theta_std = np.array(theta), np.array(theta_std)
        t = np.linspace(0, 1, 100)
        c1 = _cos(theta[0])
        c2 = _cos(theta[1])
        S = c1 + c2

        theta_init = np.array(theta, dtype=np.float64)
        theta_init[:, 0] += 2 - np.random.random(2)*0.5
        theta_init[:, 2] += 1 - np.random.random(2)*0.5
        mcmc = KurslMCMC(theta_init, theta_std, nwalkers=40, niter=200)
        mcmc.set_threshold(0.001)
        mcmc.set_sampler(t, S)
        mcmc.run()

        # After simulation is finished check for correctness
        theta_computed = mcmc.get_theta()
        self.assertTrue(np.allclose(theta, theta_computed, atol=0.5),
                "Expected:\n{}\nReceived:\n{}".format(theta, theta_computed))

    def test_run_start_from_solution(self):
        _cos = lambda l: l[2]*np.cos(l[0]*t + l[1])
        theta = np.array([[15, 0, 1, 0],
                          [35, 2, 3, 0],
                         ], dtype=np.float64)
        t = np.linspace(0, 1, 100)
        c1 = _cos(theta[0])
        c2 = _cos(theta[1])
        S = c1 + c2

        # Initial guess is far off. Just for an example.
        theta_init = np.array(theta, dtype=np.float64)
        theta_init += np.random.random(theta_init.shape)*3
        mcmc = KurslMCMC(theta_init)
        mcmc.set_sampler(t, S)

        # However, execution starts close to solution.
        pos = np.tile(theta.flatten(), (mcmc.nwalkers, 1))
        pos += np.random.random(pos.shape)*0.3
        mcmc.run(pos=pos)
        niter_executed = mcmc.get_lnprob().shape[0]
        self.assertTrue(niter_executed<mcmc.niter,
                "Starting close solution should allow for quick convergence "
                "and not all iterations would be executed.")

    def test_run_with_incorrect_pos(self):
        t = np.linspace(0, 1, 100)
        S = np.random.random(t.size)
        mcmc = KurslMCMC(self.random_theta(3, 2))
        mcmc.set_sampler(t, S)
        with self.assertRaises(ValueError) as error:
            mcmc.run(pos=self.random_theta(3, 2))
        self.assertTrue("pos.shape" in str(error.exception))

    def test_run_without_model(self):
        mcmc = KurslMCMC(self.random_theta(3, 2))
        mcmc.model = None # This situation shouldn't even be possible
        with self.assertRaises(AttributeError) as error:
            mcmc.run()
        self.assertEqual("Model not selected", str(error.exception))

    def test_run_without_sampler(self):
        mcmc = KurslMCMC(self.random_theta(3, 2))
        with self.assertRaises(AttributeError) as error:
            mcmc.run()
        self.assertTrue("Sampler not defined" in str(error.exception))
        self.assertTrue("set_sampler" in str(error.exception))

    def test_get_theta_without_running_sampler(self):
        theta_init = self.random_theta(3, 2)
        mcmc = KurslMCMC(theta_init)

        # After simulation is finished check for correctness
        with self.assertRaises(AttributeError) as error:
            mcmc.get_theta()
        self.assertTrue("theta" in str(error.exception))
        self.assertTrue("run()" in str(error.exception))

    def test_theta_computed_boolean(self):
        theta_init = self.random_theta(2, 1)
        mcmc = KurslMCMC(theta_init, niter=2)
        self.assertFalse(mcmc._theta_computed(), "Without run should fail")
        mcmc.set_sampler(np.linspace(0,1,100), np.random.random(100))
        mcmc.run()
        self.assertTrue(mcmc._theta_computed(), "After run should return true")

    def test_lnlikelihood(self):
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)
        theta = self.random_theta(3, 2)
        kursl = KurSL(theta)
        model = ModelWrapper(kursl)
        lnlikelihood = KurslMCMC.lnlikelihood(theta, t, S[:-1], model)
        self.assertTrue(lnlikelihood<0, "Any negative value is good")

    def test_lnlikelihood_zero(self):
        t = np.arange(0, 1, 0.01)
        S = 2*np.cos(10*t + 0) + 6*np.cos(20*t + 4)
        theta = np.array([[10, 0, 2, 0],
                          [20, 4, 6, 0],
                         ])
        kursl = KurSL(theta)
        model = ModelWrapper(kursl)
        lnlikelihood = KurslMCMC.lnlikelihood(theta, t, S[:-1], model)
        self.assertEqual(np.round(lnlikelihood, 10), 0, "Exact reconstruction should return 0")
        self.assertTrue(model.THRESHOLD_OBTAINED, "0 is below any threshold")

    def test_lnprior(self):
        t = np.arange(0, 1, 0.01)
        theta = self.random_theta(3, 2)
        kursl = KurSL(theta)
        model = ModelWrapper(kursl)
        lnprob = KurslMCMC.lnprior(theta, model)

        # Given default uniform probablities it's either 0 or -np.inf
        self.assertEqual(np.round(lnprob, 10), 0, "Theta within max ranges results in 0")

    def test_lnprior_theta_outside_ranges(self):
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)
        theta = self.random_theta(3, 2)
        kursl = KurSL(theta)
        model = ModelWrapper(kursl)

        new_theta = theta.copy()
        new_theta[0, 2] = model.MIN_R - 1
        self.assertEqual(KurslMCMC.lnprior(new_theta, model), -np.inf)
        new_theta[0, 2] = model.MAX_R + 1
        self.assertEqual(KurslMCMC.lnprior(new_theta, model), -np.inf)

        new_theta = theta.copy()
        new_theta[1, 0] = model.MIN_W - 1
        self.assertEqual(KurslMCMC.lnprior(new_theta, model), -np.inf)
        new_theta[1, 0] = model.MAX_W + 1
        self.assertEqual(KurslMCMC.lnprior(new_theta, model), -np.inf)

        # Check for W_i < sum_j (|k_ij|)
        new_theta = theta.copy()
        new_theta[2, 0] = np.sum(new_theta[2, 3:])*0.6
        self.assertEqual(KurslMCMC.lnprior(new_theta, model), -np.inf)

    def test_lnprob(self):
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)
        theta = self.random_theta(4, 3)
        kursl = KurSL(theta)
        model = ModelWrapper(kursl)
        self.assertTrue(KurslMCMC.lnprob(theta, t, S[:-1], model)<0,
                "Any good theta should return negative value")

    def test_lnprob_theta_outside_range(self):
        t = np.arange(0, 1, 0.01)
        S = np.random.random(t.size)
        theta = self.random_theta(4, 3)
        kursl = KurSL(theta)
        model = ModelWrapper(kursl)

        theta[0,0] = model.MIN_W - 1
        self.assertEqual(KurslMCMC.lnprob(theta, t, S[:-1], model), -np.inf,
                "Inherit behaviour of lnprob and lnlikelihood")

