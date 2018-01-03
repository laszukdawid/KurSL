#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Feel free to contact for any information.
from __future__ import division, print_function

import matplotlib
matplotlib.use("Agg")

import logging
import numpy as np
import scipy.optimize as opt

from scipy.linalg import norm as matrix_norm

########################################
## Declaring Class
class Preprocessor(object):

    logger = logging.getLogger(__name__)

    _peak_types = ["triang", "norm", "lorentz"]

    def __init__(self, max_osc=-1, nH=None, energy_ratio=None):

        self.nH = nH if nH else 1
        self.max_osc = max_osc

        self.ptype = "norm"

        self.energy_ratio = energy_ratio if energy_ratio else 0.1

        self.fMin = 0
        self.fMax = 1e10

        self.theta_init = None

    @classmethod
    def _remove_peak(cls, t, s, ptype="norm"):
        """Fit and remove peak of a given type"""
        if ptype=="norm":
            def peak(t, *p):
                _t = (t-p[0])/p[2]
                return p[1]*np.exp(-_t*_t)

            _wd = 0.5
            _amp = np.max(s)
            _pos = t[s==_amp][0]

        elif ptype=="triang":
            def peak(t, *p):
                s = 1-np.abs((t-p[0])/p[2])
                s[s<0] = 0
                return p[1]*s

            _wd = 1.0
            _amp = np.max(s)
            _pos = t[s==_amp][0]

        elif ptype=="lorentz":
            def peak(t, *p):
                _t = (t-p[0])/(0.5*p[2])
                return p[1]/(_t*_t + 1)

            _wd = 0.2
            _amp = np.max(s)
            _pos = t[s==np.max(s)][0]

        else:
            raise ValueError("Incorect ptype value. Passed "+str(ptype))

        init_guess = ([_pos, _amp, _wd])
        bound_min = (max(_pos-2., t[0]), _amp/2, max(_wd-1., 0.01))
        bound_max = (min(_pos+2., t[-1]), _amp*2, _wd+1.)
        bounds = (bound_min, bound_max)
        popt, _ = opt.curve_fit(peak, t, s, init_guess, bounds=bounds)
        peakS = peak(t, *popt)

        return peakS, popt

    @classmethod
    def remove_energy(cls, t, S, energy_ratio=0.1, max_peaks=-1, ptype="norm"):
        """Decrease input's energy by removing peaks.

        Iteratively fits and removes peaks from provided signal.
        Returns input without subtracted peaks and parameters of fitted
        peaks, i.e. position, amplitude and width.

        Use case for the method is to determine oscillation peaks in
        provided Fourier spectrum.
        """

        energy = matrix_norm(S)
        _S = S.copy()

        param = []
        while(True):
            _peakY, _param = cls._remove_peak(t, _S, ptype)
            _S[:] = _S - _peakY
            # Trim negative part after peak removal
            _S[_S<0] = 0

            param.append(_param)

            new_energy = matrix_norm(_S)
            current_ratio = new_energy/energy
            cls.logger.debug("new_energy = {}, (r = {} )".format(new_energy, current_ratio))

            # Break if energy ratio is reached
            if current_ratio <= energy_ratio:
                break

            # Break if reached maximum number of peaks
            if max_peaks > 0 and len(param) >= max_peaks:
                break

        return _S, np.array(param)

    def determine_params(self, t, S, energy_ratio=0.1, max_peaks=-1, ptype="norm"):
        """Determine oscillation parameters of time series.

        Extracts parameters of most influential oscillations by converting
        time series into Fourier spectrum and identifying the most pronounce
        peaks. Number of identified oscillations depends on energy ratio threshold.
        Oscillators are sorted in decreasing order.

        Return
        ------
        param -- Parameters for identified oscillators in increasing frequency order.
            Numpy array in shape (osc x 4), where fields are:
            params[:, 0] -- mean frequencies
            params[:, 1] -- amplitudes
            params[:, 2] -- error bars
            params[:, 3] -- initial phases
        """

        freq = np.fft.fftfreq(t.size, t[1]-t[0])
        idx = np.r_[freq>=self.fMin] & np.r_[freq<self.fMax]
        F = np.fft.fft(S)

        fourierS, param = self.remove_energy(freq[idx], np.abs(F[idx]),
                                    energy_ratio=energy_ratio,
                                    max_peaks=max_peaks,
                                    ptype=ptype)

        param = param[param[:,0].argsort()[::-1]]
        param = param.tolist()

        for i, p in enumerate(param):
            # Extracting phase
            minIdx = np.argmin(np.abs(p[0]-freq))
            param[i] = np.append(p, np.angle(F[minIdx]))

            # Scaling amplitude
            param[i][1] = np.sqrt(param[i][1]/len(fourierS))

        return np.array(param)

    def compute_prior(self, t, S):
        """Computes estimates for KurSL prior parameters.

        Return
        ------
        theta -- Initial parameters in form of 2D Numpy array,
                 where columns are (W, Y0, R, K_).
                 Note that K_ matrix doesn't have (i,i) elements, as they are zero.

        """
        self.param = self.determine_params(t, S,
                            energy_ratio=self.energy_ratio,
                            ptype=self.ptype,
                            max_peaks=self.max_osc)

        self.logger.debug("Determined prior parameters: ")
        for p in self.param:
            self.logger.debug(p)

        if np.any(self.param[:,:2]<0):
            msg = "Something went weirdly wrong. Either frequency or amplitude " \
                  "was estimated to be negative. What's the sense behind that?\n" \
                  "Estimates:" + str(self.param)
            raise AssertionError(msg)

        # There's no point in analysing
        if(self.param.shape[0]<2):
            raise Exception("Single oscillator detected. No very interesting case.")

        self.oscN = self.param.shape[0]
        self.paramN = 3+self.nH*(self.oscN-1)

        # Extract freq in decreasing order
        # WARNING! This is fine now, because we're not estimating 'k'
        #          Otherwise: Swap at the same time rows and cols in K matrix.
        W_sort_idx = np.argsort(self.param[:,0])[::-1]

        W = self.param[W_sort_idx, 0]*6.28
        R = self.param[W_sort_idx, 1]
        Y0 = (self.param[W_sort_idx, -1]+2*np.pi)%(2*np.pi)

        # Until better idea pops, just start with no coupling
        K = np.zeros((self.oscN, self.nH*(self.oscN-1)))

        ## Reconstructing signal
        self.theta_init = np.column_stack((W, Y0, R, K))
        self.logger.debug('theta_init: ' + str(self.theta_init))

        return self.theta_init


######################################
##  MAIN PROGRAMME

if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    PLOT = 0
    PLOT_SPECTRUM = 1

    # Flags
    RANDOM_PARAM = 0
    W_MIN_DIFF = 3 # Synthetic specific; min distance in generated frequency

    ptype = ['triang', 'norm','lorentz'][0]

    fs, dt = 200., 0.005
    tMin = 0 # Initial time
    tMax = 1 # Length of segment

    fMin, fMax = 0, 40
    saveName = "prior-{}".format(signalType)

    ########################################
    ## Params for kursl/synth
    max_osc = 2
    oscN = max_osc

    ########################################
    options = dict(ptype=ptype, signalType=signalType)
    options['energy_ratio'] = 0.01
    preprocessor = Preprocessor(nH=nH, max_osc=max_osc, **options)

    ###################################################
    ##   Generating KurSL type signal
    T = np.arange(tMin, tMax, dt)
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
        allK = np.zeros((nH, oscN, oscN-1))
        allK[0] = np.array([[  112.0, -1.5,   7.5,  2.0, -1.0],
                            [ -2.0, -7.0,   5.1,  9.7, -6.0],
                            [ 12.2, 13.2,  13.8,  4.3,  2.9],
                            [  5.1, 10.1,  -1.9,  0.5, -3.1],
                            [ -11.7,  3.9,   4.5, -8.5,  2.0]
                            ])
        allK[1] = np.array([[ -3.5,  10.2,  -11.8,  4.2,  2.5],
                            [  9.5,  32.1,   3.2,  7.3,  5.7],
                            [  20.5,  6.3,  -1.2,  5.7, -3.5],
                            [  1.9, -2.1,   9.1,  0.2, -2.5],
                            [ -2.2,  2.1,  14.0,  0.1,  5.9],
                            ])
        allK[2] = np.array([[ -1.2, -0.4,   -25.5, -2.3,  5.2],
                            [ -3.3,  4.2,  11.5, -1.8,  2.3],
                            [  7.7,  0.1,   6.0, -4.1,  1.7],
                            [  1.9, 10.7,  -1.5, -1.1,  4.3],
                            [ -1.7,  2.4,  -9.2, -1.1,  0.6],
                            ])

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
        plt.figure()

        freq = np.fft.fftfreq(len(T), dt)
        idx = np.r_[freq>=fMin] & np.r_[freq<=fMax]

        freq = freq[idx]
        F = np.abs(np.fft.fft(sInput)[idx])
        plt.plot(freq, F)

        for p in peaks: plt.axvline(p, color='red', linestyle='dashed')
        plt.savefig('sInput_FD')
        plt.clf()

    nInput = sInput.shape[0]
    ####################################################

    best_param = kursl.run(T, sInput)
    logger.debug(best_param)
    best_param = best_param.reshape((-1, 3+nH*(oscN-1)))

    # Saving results
    logger.debug("Saving results to " + saveName + "... ")
    np.save(saveName, best_param)

    ####################################################
    ####################################################
    logger.debug("----------------")
    logger.debug("Finished preprocess")
    logger.debug("----------------")
