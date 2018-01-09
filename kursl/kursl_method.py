#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Feel free to contact for any information.
from __future__ import division, print_function

#import matplotlib
#matplotlib.use("Agg")

import logging
import numpy as np
import scipy.optimize as opt
import os

from .kursl_model import KurSL
from .mcmc import KurslMCMC
from .preprocess import Preprocessor

########################################
## Declaring Class
class KurslMethod(object):

    logger = logging.getLogger(__name__)

   # _allowed_options = ["ptype", "signalType"]
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
        self.nwalkers = 20
        self.niter = 50

        self.PREOPTIMIZE = 0
        self.POSTOPTIMIZE = 0
        self.opt_maxiter = 100
        self.opt_verbose = False

        self.theta_init = None
        self.samples = None
        self.lnprob = None

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
        preprocessor = Preprocessor(max_osc=self.max_osc,
                            nH=self.nH,
                            energy_ratio=self.energy_ratio,
                            )

        self.theta_init = preprocessor.compute_prior(t, S)
        self.oscN, self.paramN = self.theta_init.shape
        self.logger.debug('priorC: ' + str(self.theta_init))

        ## Save initial parameters
        #np.savetxt('initParam.txt', self.theta_init.T, delimiter=' & ', fmt='%7.4g')

    def set_prior(self, theta):
        """Sets prior value for theta parameter."""
        #TODO: This could be probably replaced with @property
        theta = np.array(theta)
        if self.theta_init is None:
            oscN = theta.shape[0]
            expected_shape = (oscN, 3+self.nH*(oscN-1))
        else:
            expected_shape = self.theta_init.shape

        if expected_shape and (theta.shape != expected_shape):
            raise ValueError("Attempting to update initial theta "
                    "with incorrect shape. Got shape {}, whereas "
                    "shape {} is expected".format(theta.shape, expected_shape))

        self.theta_init = theta
        self.oscN = theta.shape[0]
        self.paramN = theta.shape[1]
        self.nH = int((self.paramN-3)/(self.oscN-1))

    @staticmethod
    def detrend(S, remove_type="mean"):
        if remove_type == "mean":
            S -= np.mean(S)
        elif remove_type == "cubic":
            fitCubic = lambda S, x: np.poly1d(np.polyfit(x, S, 3))(x)
            S -= fitCubic(S, np.arange(len(S)))
        else:
            raise ValueError("Incorrect detrend value")
        return S

    @staticmethod
    def cost_lnprob(X, Y):
        """Calculates negative log likelikehood assuming that difference
        between processes is Gaussian process. Such assumption simplifies
        calculation to like = -0.5*sum(abs(X-Y))."""
        diff = X-Y
        dEnergy = diff*diff
        like = -0.5*np.sum(dEnergy)
        return like

    def cost_function(self, t, params, Y_target):
        """Returns cost of model fit under `params` to Y_target.
           It should always return value that should be minimized.
        """
        _, _, s_rec = self.model(t, params)
        s_rec_flat = np.sum(s_rec, axis=0)
        diff = s_rec_flat - Y_target[:-1]
        cost = np.abs(self.cost_lnprob(s_rec_flat, Y_target[:-1]))
        return cost

    def run_optimize(self, t, S, theta_init=None, maxiter=None, verbose=False):
        """Performs optmization using SciPy default method (L-BFGS)"""
        if theta_init is None:
            if self.theta_init is None:
                raise ValueError("No prior parameters were assigned.")
            # Local assigment
            theta_init = self.theta_init

        options = {'maxiter': maxiter if maxiter else self.opt_maxiter,
                    'disp': verbose or self.opt_verbose}

        # Define cost function
        cost = lambda p: self.cost_function(t, p.reshape(theta_init.shape), S)

        # Construct optimizer
        optimal_result = opt.minimize(cost, theta_init, options=options)

        # Run optimizer
        best_param = optimal_result['x']

        # return optmized results
        return best_param.reshape(theta_init.shape)

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
        nwalkers = max(self.nwalkers, theta_init.size*2)
        self.logger.debug("nwalkers: " + str(nwalkers))
        self.logger.debug("niter: " + str(self.niter))

        # Length of target must be t.size-1 because we lose one sample
        # whilst simulating
        S = S[:t.size-1]

        # Detrending
        S = self.detrend(S)

        # Saving results
        saveName = 'KurSL_results'
        if self.name_suffix: saveName += '-'+self.name_suffix
        np.savez(saveName, sInput=S, x=t, nH=self.nH)

        # Set up model params
        self.model.oscN = theta_init.shape[0]
        self.model.nH = self.nH

        # Define MCMC method
        mcmc = KurslMCMC(theta_init, nwalkers=nwalkers, nH=self.nH)
        mcmc.set_model(self.model)
        mcmc.set_sampler(t, S)
        mcmc.run(niter=self.niter)

        # Plot comparison between plots
        self.samples = mcmc.get_samples()
        self.lnprob = mcmc.get_lnprob()

        sDim = self.samples.shape[1]
        best_idx = np.argmax(self.lnprob)
        theta = self.samples[int(best_idx/sDim), best_idx%sDim, :]

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
        self.logger.debug('Running simulation(s) with these values:')
        self.logger.debug("oscN: " + str(self.oscN))
        self.logger.debug("nH:   " + str(self.nH))
        self.logger.debug("paramN: {} (min walkers should be 2*paramN)".format(self.oscN*(3+self.nH*(self.oscN-1))))

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

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    PLOT = 0
    PLOT_SPECTRUM = 1

    # Flags
    ADD_NOISE = 0
    SET_PRIOR_LOAD = 0
    LOAD_PARAM = 0

    RANDOM_PARAM = 0
    W_MIN_DIFF = 3 # Synthetic specific; min distance in generated frequency

    epiLetter = "S"
    epiNumber = 20
    signalType = "kursl"
    ptype = ['triang', 'norm','lorentz'][0]

    fs = 173. # Hz
    dt = 1./fs
    tMin = 0 # Initial time
    tMax = 1 # Length of segment
    N = int((tMax-tMin)*fs)

    f_min, f_max = 0, 40
    saveName = "results-{}".format(signalType)

    ########################################
    ## Params for kursl/synth
    nH = 1
    max_osc = 2

    genH = nH
    oscN = max_osc

    ########################################
    ## Correct setting validation
    if not (ptype in ['triang','norm','lorentz']):
        raise ValueError("Wrong `ptype` selected")

    if not (tMin<tMax):
        raise ValueError("Unstructered time array")

    if not (f_min<f_max):
        raise ValueError("f_min < f_max")

    ########################################
    options = dict(ptype=ptype, signalType=signalType)
    options['energy_ratio'] = 0.01
    kursl = KurslMethod(nH=nH, max_osc=max_osc, **options)

    ###################################################
    ##   Generating KurSL type signal
    if signalType in ['kursl', 'synth']:
        T = np.linspace(tMin, tMax, N+1)
        t0, t1, dt = T[0], T[-1], T[1]-T[0]

        if RANDOM_PARAM:
            logger.debug("Generating parameters for KurSL")
            W_MIN, W_MAX = 3, 40
            Y_MIN, Y_MAX = 0, 2*np.pi
            R_MIN, R_MAX = 0, 5
            K_MIN, K_MAX = -5, 5

            allK = {}

            while True:
                W = np.random.random(oscN)*W_MAX + W_MIN
                if np.all(np.diff(W)>W_MIN_DIFF): break

            R = np.random.random(oscN)*R_MAX + R_MIN
            Y0 = np.random.random(oscN)*Y_MAX + Y_MIN

            for _nH in range(genH):
                allK[_nH] = np.random.random((oscN, oscN-1))*K_MAX + K_MIN

        elif LOAD_PARAM:
            load_params = np.load('testParams.npy')
            W = load_params[:,0]
            Y0 = load_params[:,1]
            R = load_params[:,2]

            shape = load_params.shape
            allK = {}
            maxH = (shape[1]-3)/(shape[0]-1)
            for _nH in range(maxH):
                allK[_nH] = load_params[:, np.arange(shape[0]-1)*maxH+_nH]

        else:
            f = [2, 5., 10., 13, 6, 19]
            W = [_f*2*np.pi for _f in f]
            Y0 = [1.2, 2.5, 0.0, 2.0, 0.5]
            R = [1.3, 1.5, 1.8, 2.0, 3.2]

            # Setting up K values.
            # These are not necessarily used to generated signal, as
            # there is normalisation condition below making w > sum(|k|)
            allK = {}
            allK[0] =[[  112.0, -1.5,   7.5,  2.0, -1.0],
                      [ -2.0, -7.0,   5.1,  9.7, -6.0],
                      [ 12.2, 13.2,  13.8,  4.3,  2.9],
                      [  5.1, 10.1,  -1.9,  0.5, -3.1],
                      [ -11.7,  3.9,   4.5, -8.5,  2.0]]
            allK[1] =[[ -3.5,  10.2,  -11.8,  4.2,  2.5],
                      [  9.5,  32.1,   3.2,  7.3,  5.7],
                      [  20.5,  6.3,  -1.2,  5.7, -3.5],
                      [  1.9, -2.1,   9.1,  0.2, -2.5],
                      [ -2.2,  2.1,  14.0,  0.1,  5.9]]
            allK[2] =[[ -1.2, -0.4,   -25.5, -2.3,  5.2],
                      [ -3.3,  4.2,  11.5, -1.8,  2.3],
                      [  7.7,  0.1,   6.0, -4.1,  1.7],
                      [  1.9, 10.7,  -1.5, -1.1,  4.3],
                      [ -1.7,  2.4,  -9.2, -1.1,  0.6]]


            W = np.array(W[:oscN])
            Y0 = np.array(Y0)[:oscN]
            R = np.array(R)[:oscN]
            K = np.zeros((oscN, genH*(oscN-1)))
            for _nH in range(genH):
                _tmpK = np.array(allK[_nH])
                K[:, np.arange(oscN-1)*genH+_nH] = _tmpK[:oscN,:oscN-1]

            kSum = np.sum(np.abs(K),axis=1)[:,None]
            idx = (kSum>W[:,None]).flatten()
            if np.any(idx): K[idx] *= 0.98*W[idx,None]/kSum[idx]
            #~ priorC[3*oscN:] = K.T.flatten()

            # Sorting params in reverse freq
            genParams = np.column_stack((W,Y0,R,K))
            genParams = genParams[np.argsort(W)[::-1]]

            ###########################################
            ## Message
            logger.debug('Generating %s type signal for %i oscillators with parameters' %(signalType,oscN))
            logger.debug('genParams: ' + str(genParams))
            logger.debug('genParams.shape: ' + str(genParams.shape))

            saveName = 'genParam-%s'%(signalType)
            np.savetxt(saveName+'.txt', genParams, fmt='%.6e')
            #plt.save(saveName, genParams)

            kursl.oscN = oscN
            kursl.nH = nH
            phi, A, sOscInput = kursl.model(T, genParams)
            sInput = sOscInput

            # Adding some noise random function
            if ADD_NOISE:
                phi += 0.01*np.random.normal(0,1, (oscN, N-1))
                A += 0.01*np.random.normal(0,1, (oscN, N-1))

            T = T[:-1] # sInput based on diff

    ####################################################
    ## Generate random signal
    if signalType == 'random':
        N = 1024*3
        T = np.linspace(0, 3, N)
        t0, t1, dt = T[0], T[-1], T[1]-T[0]
        sInput = np.random.normal(0, 1, (1,N))

    ####################################################
    ## Fetch epilepsy data
    if signalType == 'epilepsy':
        print("Loading epilepsy data {}.{}".format(epiLetter, epiNumber))

        _l = epiLetter
        dataPath = os.path.join("Data","Epilepsy",_l,_l)
        print(dataPath)

        loadData = lambda n: np.loadtxt(dataPath+"{:03}.txt".format(n))

        sInput = loadData(epiNumber)
        sInput = sInput.reshape((1,-1))

        N = sInput.size
        fs = 173.61 # Hz
        T = np.arange(N)/fs

        idx = np.r_[T>=tMin] & np.r_[T<tMax]
        T = T[idx]
        sInput = sInput[:,idx]


    ####################################################
    ## Determining num of osc based on Fourier energy
    sInput = np.sum(sInput, axis=0)
    if PLOT or PLOT_SPECTRUM:
        import pylab as plt

    if PLOT:
        plt.figure()
        plt.plot(T, sInput)
        plt.savefig("sInput")
        plt.clf()

    if PLOT_SPECTRUM:
        peaks = [ 1.96503875, 4.99999966, 6.3333319, 9.65672334, 12.36707368]

        plt.figure()

        freq = np.fft.fftfreq(len(T), dt)
        idx = np.r_[freq>=f_min] & np.r_[freq<=f_max]

        freq = freq[idx]
        F = np.abs(np.fft.fft(sInput)[idx])
        plt.plot(freq, F)

        for p in peaks: plt.axvline(p, color='red', linestyle='dashed')
        plt.savefig('sInput_FD')
        plt.clf()

    nInput = sInput.shape[0]

    ###################################################
    ## Loading prior parameters from file
    if SET_PRIOR_LOAD:
        theta_init = np.load('best_param.npy')
        theta_init = theta_init[np.argsort(theta_init[:,0])[::-1]]
        oscN = theta_init.shape[0]
        paramN = theta_init.shape[1]
    else:
        print("Parameters will be determined")

    ####################################################
    ####################################################

    print("Running model")
    best_param = kursl.run(T, sInput)
    samples = kursl.samples
    lnprob = kursl.lnprob
    print('oscN: ', oscN)
    print('nH: ', nH)
    print("best_param: ")
    print(best_param)
    best_param = best_param.reshape((-1, 3+nH*(oscN-1)))

    # Saving results
    print("Saving results to " + saveName + "... ")
    if "genParams" in locals():
        np.savez(saveName, samples=samples, lnprob=lnprob, sInput=sInput, x=T, nH=nH, genParams=genParams)
    else:
        np.savez(saveName, samples=samples, lnprob=lnprob, sInput=sInput, x=T, nH=nH)

    ####################################################
    ####################################################
    print("----------------")
    print("|Finished KurSL|")
    print("----------------")
