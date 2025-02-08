import logging

import numpy as np
import scipy.optimize as opt
from scipy.linalg import norm as matrix_norm


########################################
## Declaring Class
class Preprocessor(object):
    logger = logging.getLogger(__name__)

    _peak_types = ["triang", "norm", "lorentz"]

    def __init__(self, max_osc=-1, nH=1, energy_ratio=0.1):
        self.nH = nH
        self.max_osc = max_osc

        self.ptype = "norm"
        self.energy_ratio = energy_ratio

        self.f_min = 0
        self.f_max = 1e10

        self.theta_init = None

    @classmethod
    def _remove_peak(cls, t, s, ptype="norm"):
        """Fit and remove peak of a given type"""
        if ptype == "norm":

            def peak(t, *p):
                _t = (t - p[0]) / p[2]
                return p[1] * np.exp(-_t * _t)

            _wd = 0.5
            _amp = np.max(s)
            _pos = t[s == _amp][0]

        elif ptype == "triang":

            def peak(t, *p):
                s = 1 - np.abs((t - p[0]) / p[2])
                s[s < 0] = 0
                return p[1] * s

            _wd = 1.0
            _amp = np.max(s)
            _pos = t[s == _amp][0]

        elif ptype == "lorentz":

            def peak(t, *p):
                _t = (t - p[0]) / (0.5 * p[2])
                return p[1] / (_t * _t + 1)

            _wd = 0.2
            _amp = np.max(s)
            _pos = t[s == np.max(s)][0]

        else:
            raise ValueError("Incorect ptype value. Passed " + str(ptype))

        init_guess = [_pos, _amp, _wd]
        bound_min = (max(_pos - 2.0, t[0]), _amp / 2, max(_wd - 1.0, 0.01))
        bound_max = (min(_pos + 2.0, t[-1]), _amp * 2, _wd + 1.0)
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
        while True:
            _peakY, _param = cls._remove_peak(t, _S, ptype)
            _S[:] = _S - _peakY
            # Trim negative part after peak removal
            _S[_S < 0] = 0

            param.append(_param)

            new_energy = matrix_norm(_S)
            current_ratio = new_energy / energy
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

        freq = np.fft.fftfreq(t.size, t[1] - t[0])
        idx = np.r_[freq >= self.f_min] & np.r_[freq < self.f_max]
        F = np.fft.fft(S)

        fourierS, param = self.remove_energy(
            freq[idx], np.abs(F[idx]), energy_ratio=energy_ratio, max_peaks=max_peaks, ptype=ptype
        )

        param = param[param[:, 0].argsort()[::-1]]
        param = param.tolist()

        for i, p in enumerate(param):
            # Extracting phase
            min_idx = np.argmin(np.abs(p[0] - freq))
            param[i] = np.append(p, np.angle(F[min_idx]))

            # Scaling amplitude
            param[i][1] = param[i][1] / len(fourierS)

        return np.array(param)

    def compute_prior(self, t, S):
        """Computes estimates for KurSL prior parameters.

        Return
        ------
        theta -- Initial parameters in form of 2D Numpy array,
                 where columns are (W, Y0, R, K_).
                 Note that K_ matrix doesn't have (i,i) elements, as they are zero.

        """
        self.param = self.determine_params(
            t, S, energy_ratio=self.energy_ratio, ptype=self.ptype, max_peaks=self.max_osc
        )

        self.logger.debug("Determined prior parameters: ")
        self.logger.debug("\n".join([str(p) for p in self.param]))

        if np.any(self.param[:, :2] < 0):
            msg = (
                "Something went weirdly wrong. Either frequency or amplitude "
                "was estimated to be negative. What's the sense behind that?\n"
                "Estimates:\n" + str(self.param)
            )
            raise AssertionError(msg)

        # There's no point in analysing
        if self.param.shape[0] < 2:
            raise ValueError("Single oscillator detected. No very interesting case.")

        self.oscN = self.param.shape[0]
        self.paramN = 3 + self.nH * (self.oscN - 1)

        # Extract freq in decreasing order
        # WARNING! This is fine now, because we're not estimating 'k'
        #          Otherwise: Swap at the same time rows and cols in K matrix.
        W_sort_idx = np.argsort(self.param[:, 0])[::-1]

        W = self.param[W_sort_idx, 0] * 6.28
        R = self.param[W_sort_idx, 1]
        Y0 = (self.param[W_sort_idx, -1] + 2 * np.pi) % (2 * np.pi)

        # Until better idea pops, just start with no coupling
        K = np.zeros((self.oscN, self.nH * (self.oscN - 1)))

        ## Reconstructing signal
        self.theta_init = np.column_stack((W, Y0, R, K))
        self.logger.debug("theta_init: " + str(self.theta_init))

        return self.theta_init


######################################
##  MAIN PROGRAMME

if __name__ == "__main__":
    import sys

    import pylab as plt

    from kursl import KurSL

    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(__file__)

    ###################################################
    ## Signal generation specific

    # Min distance in generated frequency
    W_MIN_DIFF = 7

    # Type of peak that is fit
    ptype = ["norm", "triang", "lorentz"][0]

    # Time array
    tMin, tMax, dt = 0, 5, 0.005
    t = np.arange(tMin, tMax, dt)

    # Number of oscillators
    oscN = 6
    nH = 2

    ###################################################
    ## Generating KurSL type signal
    logger.debug("Generating parameters for KurSL")
    W_MIN, W_MAX = 9, 100
    Y_MIN, Y_MAX = 0, 2 * np.pi
    R_MIN, R_MAX = 0, 5
    K_MIN, K_MAX = -5, 5

    # Making sure that there's W_MIN_DIFF between all W
    while True:
        W = np.random.random(oscN) * W_MAX + W_MIN
        if np.all(np.diff(W) > W_MIN_DIFF):
            break

    R = np.random.random(oscN) * R_MAX + R_MIN
    Y0 = np.random.random(oscN) * Y_MAX + Y_MIN
    K = np.random.random((oscN, nH * (oscN - 1))) * K_MAX + K_MIN
    K[:] = 0.8 * W[:, None] * K / np.sum(K, axis=1)[:, None]

    theta = np.column_stack((W, Y0, R, K))

    kursl = KurSL(theta)
    _, _, s_gen = kursl.generate(t)
    s_gen = np.sum(s_gen, axis=0)
    t = t[:-1]

    ####################################################
    ## Estimate peaks present in signal
    preprocess = Preprocessor(max_osc=oscN, nH=nH)
    theta_init = preprocess.compute_prior(t, s_gen)
    peaks = theta_init[:, 0] * 0.5 / np.pi

    ####################################################
    ## Plot extracted frequencies in spectrum
    plt.figure()

    freq = np.fft.fftfreq(len(t), dt)
    f_min = max(0, W_MIN / 6.28 - 2)
    f_max = min(W_MAX / 6.28 + 2, 1 / (2 * np.pi * dt))
    idx = np.r_[freq > f_min] & np.r_[freq < f_max]

    freq = freq[idx]
    F = np.abs(np.fft.fft(s_gen)[idx])
    plt.plot(freq, F)
    for p in peaks:
        plt.axvline(p, color="red", linestyle="dashed")
    plt.xlim((f_min, f_max))
    plt.savefig("s_gen_spectrum")
    plt.clf()

    ####################################################
    ####################################################
    logger.debug("----------------")
    logger.debug("Finished preprocess")
    logger.debug("----------------")
