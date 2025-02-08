import logging

import numpy as np
import scipy.optimize as opt

from kursl.kursl_model import KurSL
from kursl.mcmc import KurslMCMC
from kursl.preprocess import Preprocessor


########################################
## Declaring Class
class KurslMethod(object):
    logger = logging.getLogger(__name__)

    _peak_types = ["triang", "norm", "lorentz"]

    def __init__(self, nH=1, max_osc=-1, *args, **kwargs):
        # TODO: attribute that changes size of theta
        self.nH = nH
        self.max_osc = max_osc
        self.model = KurSL()
        self.model.nH = nH

        self.ptype = "norm"
        self.name_suffix = None

        self.energy_ratio = 0.1

        self.f_min = 0
        self.f_max = 1e10

        # MCMC variables
        self.nwalkers = 40
        self.niter = 100

        self.PREOPTIMIZE = 0
        self.POSTOPTIMIZE = 0
        self.opt_maxiter = 100
        self.opt_verbose = False

        self.theta_init = None
        self.samples = None
        self.lnprob = None

        self.threads = 1

        # Set parameters
        self.set_options(kwargs)

    def set_options(self, options):
        _option_names = self.__dict__.keys()
        for key in options.keys():
            if key in _option_names:
                val = options[key]
                self.logger.debug("Setting: self.{} = {}".format(key, val))
                self.__dict__[key] = val

    def compute_prior(self, t, S):
        preprocessor = Preprocessor(
            max_osc=self.max_osc,
            nH=self.nH,
            energy_ratio=self.energy_ratio,
        )

        self.theta_init = preprocessor.compute_prior(t, S)
        self.oscN, self.paramN = self.theta_init.shape
        self.logger.debug("priorC: " + str(self.theta_init))

    def set_prior(self, theta):
        """Sets prior value for theta parameter."""
        # TODO: This could be probably replaced with @property
        theta = np.array(theta)
        if self.theta_init is None:
            oscN = theta.shape[0]
            expected_shape = (oscN, 3 + self.nH * (oscN - 1))
        else:
            expected_shape = self.theta_init.shape

        if expected_shape and (theta.shape != expected_shape):
            raise ValueError(
                "Attempting to update initial theta "
                "with incorrect shape. Got shape {}, whereas "
                "shape {} is expected".format(theta.shape, expected_shape)
            )

        self.theta_init = theta
        self.oscN = theta.shape[0]
        self.paramN = theta.shape[1]
        self.nH = int((self.paramN - 3) / (self.oscN - 1))

    @staticmethod
    def fit_cubic(S, x):
        return np.poly1d(np.polyfit(x, S, 3))(x)

    @staticmethod
    def detrend(S, remove_type="mean"):
        if remove_type == "mean":
            S -= np.mean(S)
        elif remove_type == "cubic":
            S -= KurslMethod.fit_cubic(S, np.arange(len(S)))
        else:
            raise ValueError("Incorrect detrend value")
        return S

    @staticmethod
    def cost_lnprob(X, Y):
        """Calculates negative log likelikehood assuming that difference
        between processes is Gaussian process. Such assumption simplifies
        calculation to like = -0.5*sum(abs(X-Y))."""
        diff = X - Y
        dEnergy = diff * diff
        like = -0.5 * np.sum(dEnergy)
        return like

    def cost_function(self, t, params, Y_target):
        """Returns cost of model fit under `params` to Y_target.
        It should always return value that should be minimized.
        """
        _, _, s_rec = self.model(t, params)
        s_rec_flat = np.sum(s_rec, axis=0)
        cost = np.abs(self.cost_lnprob(s_rec_flat, Y_target[:-1]))
        return cost

    def run_optimize(self, t, S, theta_init=None, maxiter=None, verbose=False):
        """Performs optmization using SciPy default method (L-BFGS)"""
        if theta_init is None:
            if self.theta_init is None:
                raise ValueError("No prior parameters were assigned.")
            # Local assigment
            theta_init = self.theta_init

        options = {
            "maxiter": maxiter if maxiter else self.opt_maxiter,
            "disp": verbose or self.opt_verbose,
        }

        # Define cost function
        def cost(p):
            return self.cost_function(t, p.reshape(theta_init.shape), S)

        # Construct optimizer
        optimal_result = opt.minimize(cost, theta_init.flatten(), options=options)

        # Run optimizer
        theta = optimal_result["x"]

        # return optmized results
        return theta.reshape(theta_init.shape)

    def run_mcmc(self, t, S, theta_init=None):
        """Use MCMC to fit KurSL model to signal S.

        Performs MCMC to search parameter space for a set that fits
        KurSL model the best to provided data S.

        Input
        -----
        t -- time array
        S -- time series
        theta_init (default: None) -- initial starting parameters
        """
        if theta_init is None:
            if self.theta_init is None:
                raise ValueError("No prior parameters were assigned.")
            # Local assigment
            theta_init = self.theta_init

        theta_init = theta_init.astype(np.float64)

        # Setting number of Walkers
        nwalkers = max(self.nwalkers, theta_init.size * 2)
        self.logger.debug("nwalkers: " + str(nwalkers))
        self.logger.debug("niter: " + str(self.niter))

        # Length of target must be t.size-1 because we lose one sample
        # whilst simulating
        S = S[: t.size - 1]

        # Detrending
        S = self.detrend(S)

        # Saving results
        saveName = "KurSL_results"
        if self.name_suffix:
            saveName += "-" + self.name_suffix
        np.savez(saveName, sInput=S, x=t, nH=self.nH)

        # Set up model params
        self.model.oscN = theta_init.shape[0]
        self.model.nH = self.nH

        # Define MCMC method
        mcmc = KurslMCMC(
            theta_init, nwalkers=nwalkers, nH=self.nH, threads=self.threads
        )
        mcmc.set_model(self.model)
        mcmc.set_sampler(t, S)
        mcmc.run(niter=self.niter)

        # Plot comparison between plots
        self.samples = mcmc.get_samples()
        self.lnprob = mcmc.get_lnprob()

        sDim = self.samples.shape[1]
        best_idx = np.argmax(self.lnprob)
        theta = self.samples[int(best_idx / sDim), best_idx % sDim, :]

        return theta.reshape(theta_init.shape)

    def run(self, t, S, theta_init=None):
        """Perform KurSL model fitting to data S.

        Fit can be performed as a many step optimization.
        Fitting is done using MCMC, although depending on
        set flags pre- and post-optimization can be added
        using L-BFGS.

        Input
        -----
        t -- time array
        S -- time series

        Returns
        -------
        theta -- Parameters that best fit the KurSL model under
                 provided conditions. Columns denote in increasing
                 order: intrisic frequency W, initial phase P, initial
                 amplitude R, and coupling factors K matrices (order),
                 i.e. theta = (W, P, R, K1=(K_...), K2=(K_...))
        """

        # Detrending
        size = min(t.size, S.size)
        t = t[:size]
        S = S[:size]
        S[:] = self.detrend(S)

        # Initial parameters
        if theta_init is None:
            self.compute_prior(t, S)
            theta_init = self.theta_init
        else:
            self.set_prior(theta_init)

        ####################################################
        ## Data presentation
        self.logger.debug("Running simulation(s) with these values:")
        self.logger.debug("oscN: " + str(self.oscN))
        self.logger.debug("nH:   " + str(self.nH))
        self.logger.debug(
            "paramN: {} (min walkers should be 2*paramN)".format(
                self.oscN * (3 + self.nH * (self.oscN - 1))
            )
        )

        # Initial parameter is our best bet
        theta = self.theta_init

        if self.PREOPTIMIZE:
            self.logger.debug("Running preoptimization. Theta:\n" + str(theta))
            self.logger.debug("Cost: " + str(self.cost_function(t, theta, S)))
            theta = self.run_optimize(t, S, theta_init=theta)

        # Run MCMC
        self.logger.debug("Running MCMC. Theta:\n" + str(theta))
        self.logger.debug("Cost: " + str(self.cost_function(t, theta, S)))
        theta = self.run_mcmc(t, S, theta_init=theta)

        if self.POSTOPTIMIZE:
            self.logger.debug("Running postoptimization. Theta:\n" + str(theta))
            self.logger.debug("Cost: " + str(self.cost_function(t, theta, S)))
            theta = self.run_optimize(t, S, theta_init=theta)

        self.theta_init = theta
        self.logger.debug("Final results" + str(theta))
        self.logger.debug("Cost: " + str(self.cost_function(t, theta, S)))

        return theta


######################################
##  MAIN PROGRAMME

if __name__ == "__main__":
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    PLOT = True
    PLOT_SPECTRUM = True
    SAVE_RESULTS = True

    ptype = ["norm", "triang", "lorentz"][0]

    dt = 0.005
    tMin, tMax = 0, 2
    T = np.arange(tMin, tMax, dt)

    ########################################
    ## Params for kursl/synth
    nH = 1
    max_osc = 2

    genH = nH
    oscN = max_osc

    options = dict(ptype=ptype, energy_ratio=0.01)
    kursl = KurslMethod(nH=nH, max_osc=max_osc, **options)

    ###################################################
    ##   Generating KurSL type signal
    f = [2, 5.0, 10.0, 13]
    W = [_f * 2 * np.pi for _f in f]
    Y0 = [1.2, 2.5, 0.0, 2.0]
    R = [1.3, 1.5, 1.8, 2.0]

    K = np.zeros((3, 4, 3))
    K[0] = [[4.0, -1.5, 1.2], [-2.0, -7.0, 5.1], [12.2, 3.2, 1.8], [5.1, 8.1, -1.9]]
    K[1] = [[-3.5, -4.2, 2.5], [9.5, 3.2, 7.3], [6.3, -1.2, -3.5], [1.9, 0.2, -2.5]]
    K[2] = [[-1.2, -0.4, -2.3], [-3.3, -1.8, 2.3], [0.1, 6.0, 1.7], [1.9, -1.5, 4.3]]

    W = np.array(W[:oscN])
    Y0 = np.array(Y0)[:oscN]
    R = np.array(R)[:oscN]
    K = np.hstack(K[:, :oscN, : oscN - 1])

    kSum = np.sum(np.abs(K), axis=1)[:, None]
    idx = (kSum > W[:, None]).flatten()
    if np.any(idx):
        K[idx] *= 0.98 * W[idx, None] / kSum[idx]

    genParams = np.column_stack((W, Y0, R, K))

    ###########################################
    ## Message
    logger.debug(
        "Generating signal with KurSL for %i oscillators with parameters" % (oscN)
    )
    logger.debug("genParams: " + str(genParams))

    if SAVE_RESULTS:
        np.savetxt("genParam.txt", genParams, fmt="%.6e")

    kursl.oscN = oscN
    kursl.nH = nH
    phi, A, sInput = kursl.model(T, genParams)

    T = T[:-1]  # sInput based on diff

    ####################################################
    ## Determining num of osc based on Fourier energy
    sInput = np.sum(sInput, axis=0)
    sInput[:] = sInput + np.random.random(sInput.size) * 0.2 * oscN
    if PLOT or PLOT_SPECTRUM:
        try:
            import pylab as plt
        except ImportError:
            logger.error("Cannot import matplotlib. Make sure you installed it.")
            sys.exit(1)

    if PLOT:
        plt.figure()
        plt.plot(T, sInput)
        plt.savefig("signal")
        plt.clf()

    if PLOT_SPECTRUM:
        plt.figure()
        freq = np.fft.fftfreq(len(T), dt)
        idx = np.r_[freq >= 0] & np.r_[freq <= 25]

        freq = freq[idx]
        F = np.abs(np.fft.fft(sInput)[idx])
        plt.plot(freq, F)

        plt.savefig("sInput_FD")
        plt.clf()

    ####################################################
    ####################################################

    logger.info("Running model")
    theta = kursl.run(T, sInput)
    logger.debug("theta: ")
    logger.debug(theta)

    if PLOT:
        _, _, rec = kursl.model(T, theta)
        plt.figure()
        plt.plot(T, sInput, "g")
        plt.plot(T[: rec.shape[1]], np.sum(rec, axis=0), "r")
        plt.savefig("signal")
        plt.clf()

    # Saving results
    if SAVE_RESULTS:
        logger.info("Saving results to 'results.npy' ... ")
        np.savez(
            "results.npy",
            samples=kursl.samples,
            lnprob=kursl.lnprob,
            sInput=sInput,
            x=T,
            nH=nH,
            genParams=genParams,
            theta=theta,
        )

    ####################################################
    ####################################################
    logger.info("----------------")
    logger.info("|Finished KurSL|")
    logger.info("----------------")
