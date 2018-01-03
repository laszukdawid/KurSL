#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 02/12/2016
#
# Feel free to contact for any information.
from __future__ import division, print_function

import emcee
import logging
import numpy as np
import time, datetime

from kursl import KurSL
from kursl import ModelWrapper

#import warnings
#warnings.simplefilter('always', DeprecationWarning)

class KurslMCMC(object):

    logger = logging.getLogger(__name__)

    def __init__(self, theta_init, theta_std=None, nH=1, nwalkers=None, nruns=None, **kwargs):

        # Setting paramters
        self.theta_init = theta_init
        self.ndim = theta_init.size
        self.nwalkers = nwalkers if nwalkers else int(1.1*theta_init.size)*2
        self.niter = nruns if nruns else 100

        # Inner flag options
        self.skip_init_steps = 0
        self.save_iter = 10

        self.THRESHOLD = 0.05
        self.SAVE_INIT_POS = True

        # Number of harmonics
        self.nH = nH
        self.oscN = theta_init.shape[0]
        self.perN = theta_init.shape[1]

        # Setting up KurSL model, which is default (only?)
        kursl = KurSL(theta_init)
        self.set_model(kursl)

        self._init_walkers(theta_init, theta_std)

    def _init_walkers(self, theta_init, theta_std):
        """Initiating positions for walkers"""

        # Setting std for walkers
        if theta_std is None:
            theta_std = np.zeros(theta_init.shape)
            theta_std[:, 0] = 0.5*np.log(theta_init[:,0]+1) # Frequency
            theta_std[:, 1] = 1.0 # Phase
            theta_std[:, 2] = np.sqrt(np.abs(theta_init[:,2])) # Amp
            theta_std[:, 3:] = 5. # k couplings

            # Variance cannot be negative
            theta_std[theta_std<=0] = 1

        self.logger.debug('theta_std: \n' + str(theta_std))
        theta_std = theta_std.flatten()

        theta_flat = theta_init.flatten()
        self.init_pos = np.tile(theta_flat, (self.nwalkers, 1))

        # Keep first `keep` paramters and don't add any noise
        keep = 1
        for param in range(self.ndim):
            self.init_pos[keep:, param] += np.random.normal(0, 0.5*theta_std[param], self.nwalkers-keep)

        # Making amplitude positive
        # W0, y00, R0, K... , W1, y01, ...
        idx = np.arange(self.oscN)*(self.nH*(self.oscN-1)+3) + 2
        self.init_pos[:, idx] = np.abs(self.init_pos[:, idx])

        # Square root in amplitude part cannot be negative,
        # i.e. sum of all 'k' has to be smaller than 'w'
        for osc in range(self.oscN):
            W = self.init_pos[:,osc*self.perN]
            K = self.init_pos[:,osc*self.perN+3:]
            kSum = np.sum(np.abs(K),axis=1)
            idx = kSum>W
            if np.any(idx):
                R = 0.98*W[idx]/kSum[idx]
                self.init_pos[idx, osc*self.perN+3:] *= R[:,None]

        # If needed, save initial position
        if self.SAVE_INIT_POS:
            np.save('init_pos', self.init_pos)

        self.set_estimates(theta_init)

    @staticmethod
    def negLog(x):
        return -np.sum(np.log(x))

    def set_model(self, model):
        """Set reconstruction model."""
        self.model = ModelWrapper(model)

        self.model.s_var = 1.
        self.model.THRESHOLD = self.THRESHOLD
        self.model.THRESHOLD_OBTAINED = False

    def set_threshold(self, threshold):
        """ Setting up energy threshold. If ratio of reconstructed signal's
            to input signal's energy is less or equal then procedure is stopped.
        """
        self.model.THRESHOLD = threshold

    def set_sampler(self, x, y):
        y_detrend = y-y.mean()
        self.s_var = np.sum(y_detrend*y_detrend)

        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                    self.lnProb,
                                    args=(x, y, self.model),
                                    threads=4)

    def run(self, pos=None, niter=None):
        """ Runs MCMC algorithm. It starts with positions pos
            and moves each walker over N iterations.
        """
        niter = niter if niter else self.niter
        pos = pos if pos else self.init_pos

        if self.model is None:
            self.logger.error("Model not selected!")
            raise AttributeError("Model not selected")

        posArr, lnprobArr = np.zeros((niter,)+pos.shape), np.zeros((niter, pos.shape[0]))

        measure_time0 = time.time()
        self.logger.debug('pos.shape: ' + str(pos.shape))
        for n, results in enumerate(self.sampler.sample(pos,
                                        iterations=niter,
                                        storechain=True)):

            posArr[n] = np.array(results[0])
            lnprobArr[n] = np.array(results[1])

            if n % self.save_iter == 0:
                measure_time = time.time()
                dt = measure_time - measure_time0
                msg = datetime.datetime.now().strftime('%H:%M:%S')
                msg += " | iteration: {:3}/{:3}  -- time {:10.4f}s".format(n, niter, dt)
                msg += " -- best {:10.2f}".format(np.max(lnprobArr[:n+1]))
                self.logger.info(msg)

                np.save('pos', posArr)
                np.save('lnprob', lnprobArr)

                measure_time0 = time.time()

            if self.model.THRESHOLD_OBTAINED:
                measure_time = time.time()
                dt = measure_time - measure_time0
                self.logger.info("iteration: {:3}/{:3}  -- time {:10.4f}s".format(n, niter, dt))
                self.logger.info("ENERGY THRESHOLD OBTAINED. Halting computation")
                break

            if np.max(lnprobArr[:n+1])>0:
                measure_time = time.time()
                dt = measure_time - measure_time0
                self.logger.info("iteration: {:3}/{:3}  -- time {:10.4f}s".format(n, niter, dt))
                self.logger.info("Extremely small likelihood obtained. Halting computation")
                break

        posArr = posArr[:n+1]
        lnprobArr = lnprobArr[:n+1]
        np.save('pos', posArr)
        np.save('lnprob', lnprobArr)
        self.chain = posArr
        self.lnprobability = lnprobArr

        # Whatever the outcome, clear flag
        self.model.THRESHOLD_OBTAINED = False

    def get_samples(self):
        return self.chain[:, self.skip_init_steps:,:]

    def get_lnprob(self):
        #~ return self.sampler.lnprobability[:, self.skip_init_steps:]
        return self.lnprobability[:, self.skip_init_steps:]

    def set_estimates(self, theta):
        _theta = theta.reshape((self.oscN,-1))

        # Prob dist for freq
        self.wMean = _theta[:,0]
        self.wStd = np.sqrt(np.abs(self.wMean))*0.1

        # Prob dist for kCoupling
        self.kMean = _theta[:, 3:]
        self.kStd = np.sqrt(np.abs(self.kMean))

        self.logger.debug("self.wMean: " + str(self.wMean))
        self.logger.debug("self.wStd:  " + str(self.wStd))

    @staticmethod
    def lnLikelihood(theta, x, y, model):
        """ Logarithm of likelihood function.
            Related to the probability of parameters given the data.

            Theta is matrix of parameters in form of:
            Th = [ W Ph R K] , dim(Th) = N(3 + h(N-1))
            where: W - Nx1, Ph = Nx1, R = Nx1, K - hxNx(N-1)
        """

        # Determing value of N
        # if th.size == (N+2)N -> N = Sqrt(th.size +1) -1

        _theta = theta.reshape((model.oscN,-1))

        # Calculating
        _, _, S = model(x, _theta)
        d = y - np.sum(S, axis=0)

        d_energy = np.sum(d*d)
        if d_energy/model.s_var < model.THRESHOLD:
            model.THRESHOLD_OBTAINED = True

        like = -0.5*np.sum(d_energy)
        return like

    @staticmethod
    def lnPrior(theta, model):
        """ Logarithm of parameter's a priori probability function.

            Theta is matrix of parameters in form of:

            Th = [ W K Ph R] , dim(Th) = N x (N+2)
            where: W - Nx1, K - Nx(N-1), Ph = Nx1, R = Nx1
        """

        # Determing value of N
        # if th.size == (N+2)N -> N = Sqrt(th.size +1) -1
        _theta = theta.reshape((model.oscN, -1))

        W = _theta[:,0]
        Ph = _theta[:,1]
        R = _theta[:,2]
        K = _theta[:,3:]

        # Amplitude has udist_nity probability in [0, MAX_R]
        if np.any(R<=model.MIN_R) or np.any(R>model.MAX_R):
            return -np.inf
        else:
            likeR = model.dist_R(R)

        ## Freq has unity probability in [0, MAX_W]
        if np.any(W<=model.MIN_W) or np.any(W>model.MAX_W):
            return -np.inf

        # Constrain:
        # if W_i + sum_j | K_ij | < 0  -->  -inf
        if np.any(W - np.sum(np.abs(K), axis=1) <= 0):
            return -np.inf

        # Sum of all log could be changed !
        likeW = model.dist_W(W)

        # Phase probability:
        likePh = model.dist_ph(Ph)

        # K coupling prob --
        likeK = model.dist_K(K)

        # Total likelihood is a sum of all likelihoods
        #~ like = negLog(likeW)  + negLog(likeA) + negLog(likeK) + negLog(likePh)
        like = likeW + likePh + likeR + likeK

        return like

    @classmethod
    def lnProb(cls, theta, x, y, model):
        """ Log probability of obtaining y(x) data
            given parameters theta. According to Bayesian
            theorem it is
            ln P(Y|T) = ln P(T) + ln P(T|Y) - ln P(Y)
        """

        # Calculate prior of theta
        lp = cls.lnPrior(theta, model)
        if not np.isfinite(lp): return -np.inf
        val = lp + cls.lnLikelihood(theta, x, y, model)
        return val

# End of Class

######################################

# Exaple usage of program.
# 1. Prepare oscillators.
# 2. Adjust Kuramoto system via Bayes inference
# 3. Plot results

if __name__ == "__main__":

    logfile = __file__.split('.')[0] + ".log"
    #logging.basicConfig(filename=logfile, level=logging.DEBUG)
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    PLOT_RESULTS = True

    FIG_SIG = 0
    FIG_IMF = 0
    FIG_instFreq = 0
    FIG_instFreq_zoom = 0
    FIG_CORNER = 0

    MIN_R, MAX_R = 1, 5
    MIN_W, MAX_W = 10, 30

    N = 1024*2
    t = np.linspace(0, 3, N)
    t0, t1, dt = t[0], t[-1], t[1]-t[0]

    oscN = 2 # Number of oscillators
    nH = 2   # Number of harmonics

    # Initial values for system
    W = np.random.random(oscN)*(MAX_W-MIN_W) + MIN_W
    R = np.random.random(oscN)*(MAX_R-MIN_R) + MIN_R
    Phi0 = np.random.random(oscN)*2*np.pi
    kMat = np.random.random((oscN, nH*(oscN-1)))

    # P - W(Nx1) R(Nx1) Ph(Nx1) K(Nx(M(N-1))
    # P - Nx(3+M(N-1))
    P = np.zeros((oscN, 3+nH*(oscN-1)))
    P[:,0] = W
    P[:,1] = Phi0
    P[:,2] = R
    if oscN != 1: P[:,3:3+nH*(oscN-1)] = kMat
    noise = P*np.random.normal(0, 0.2)

    print("P:")
    print(P)
    print("noise:")
    print(noise)

    # Generating signal
    phase, amp, sInput = KurSL(P).generate(t)
    for i in range(oscN): sInput[i] += np.random.normal(0, 0.2*R[i], N-1)

    # Initiating theta params values
    nll = lambda *args: -lnlike(*args)
    r = lambda s, n: rand(0,s,n)/np.sqrt(s)

    # Applying MCMC
    theta_init = P + noise

    x = t
    y = np.sum(sInput, axis=0)+np.random.random(t.size-1)

    mcmc = KurslMCMC(theta_init, nH=nH, nwalkers=20, nruns=100)
    mcmc.set_sampler(x,y)
    mcmc.run()

    # Plot comparison between plots
    samples = mcmc.get_samples()
    lnprob = mcmc.get_lnprob()
    np.save("samples", samples)
    np.save("lnprob", lnprob)

    sDim = samples.shape[1]
    bestIdx = np.argmax(lnprob)
    params = samples[int(bestIdx/sDim), bestIdx%sDim, :]
    params = params.reshape((oscN, -1))
    np.save("bestParam", params)

    print("Best estimate: " + str(params))

    # Plot results
    if PLOT_RESULTS:
        import pylab as plt

        kursl = KurSL(params)
        phase, amp, rec = kursl.generate(t)

        plt.figure()
        for i in range(oscN):
            plt.subplot(oscN, 1, i+1)
            plt.plot(t[:-1], sInput[i], 'g')
            plt.plot(t[:-1], rec[i], 'r')

        plt.savefig("fit", dpi=200)
        plt.show()
