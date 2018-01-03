#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
#
# Feel free to contact for any information.
from __future__ import print_function
import numpy as np

from scipy.integrate import ode

class KurSL(object):
    """
        KurSL assigns model with parameters P.
        These parameters has to convertable to NumPy 2D array.
        It is assumed that P is (oscN, oscN*(3+nH(oscN-1))) size,
        where for each oscillators parameters are: W, ph, A, k(1)(1), ..., k(oscN-1)(nH).
    """
    def __init__(self, P=None):

        if P is not None:
            self.set_params(P)

        self.kODE = ode(self.kuramoto_ODE)
        self.kODE.set_integrator("dopri5")

    def set_params(self, P):
        # Assert conditions
        P = np.array(P)

        # Extracting model parameters
        self.oscN, self.p = oscN, p = P.shape
        self.nH = nH = int((p-3)/(oscN-1))
        self.P = P

        self.W = P[:,0]
        self.Y = P[:,1]
        self.R = P[:,2]
        self.K = P[:,3:]

        # Convert K array to include self coupling (K_ii = 0)
        mask = ~np.eye(oscN, dtype=bool)
        kMat = np.zeros((nH, oscN, oscN))
        for _nH in range(nH):
            _k = self.K[:,_nH*(oscN-1):(_nH+1)*(oscN-1)]
            kMat[_nH][mask] = _k.flatten()
        self.K = kMat.copy()

    def generate(self, t):
        """Solves Kuramoto ODE for time series `t` with initial
        parameters passed when initiated object.
        """

        # Set parameters into model
        self.kODE.set_initial_value(self.Y, t[0])
        self.kODE.set_f_params((self.W, self.K))

        phase = np.empty((self.oscN, len(t)))

        # Run ODE integrator
        for idx, _t in enumerate(t[1:]):
            phase[:,idx] = self.kODE.y
            self.kODE.integrate(_t)

        phase[:,-1] = self.kODE.y
        dPhi = np.diff(phase)
        phase = phase[:,:-1]
        dPhi0 = dPhi[:,0][:,None]

        amp = np.sqrt(dPhi0/dPhi)
        amp *= (self.R/np.max(amp, axis=1))[:,None]
        P = np.cos(phase)
        S = amp*P

        return phase, amp, S

    def __call__(self, t, P):
        # Assert conditions
        P = np.array(P)
        assert P.ndim == 2
        assert P.shape[0]>1

        # Extracting model parameters
        oscN, p = P.shape
        nH = int((p-3)/(oscN-1))

        W = P[:,0]
        Y = P[:,1]
        R = P[:,2]
        K = P[:,3:]

        # Convert K array to include self coupling (K_ii = 0)
        mask = ~np.eye(oscN, dtype=bool)
        kMat = np.zeros((nH, oscN, oscN))
        for _nH in range(nH):
            _k = K[:,_nH*(oscN-1):(_nH+1)*(oscN-1)]
            kMat[_nH][mask] = _k.flatten()
        K = kMat.copy()

        kODE = ode(self.kuramoto_ODE)
        kODE.set_integrator("dopri5")
        kODE.set_initial_value(Y, t[0])
        kODE.set_f_params((W, K))

        phase = np.empty((oscN, len(t)))

        # Run ODE integrator
        for idx, _t in enumerate(t[1:]):
            phase[:,idx] = kODE.y
            kODE.integrate(_t)

        phase[:,-1] = kODE.y
        dPhi = np.diff(phase)
        phase = phase[:,:-1]
        dPhi0 = dPhi[:,0][:,None]

        amp = np.sqrt(dPhi0/dPhi)
        amp *= (R/np.max(amp, axis=1))[:,None]
        P = np.cos(phase)
        S = amp*P

        return phase, amp, S

    @staticmethod
    def kuramoto_ODE(t, y, arg):
        """General Kuramoto ODE of m'th harmonic order.
           Argument `arg` = (w, k), with
            w -- iterable frequency
            k -- 3D coupling matrix, unless 1st order
            """

        w, k = arg
        w = np.array(w, dtype=np.float64)
        k = np.array(k, dtype=np.float64)
        yt = y[:,None]
        dy = y-yt
        phase = w
        for m, _k in enumerate(k):
            phase += np.sum(_k*np.sin((m+1)*dy),axis=1)

        return phase

    @staticmethod
    def kuramoto_ODE_jac(self, t, y, arg):
        """Kuramoto's Jacobian passed for ODE solver."""

        _, k, n_osc = arg
        yt = y[:,None]
        dy = y-yt

        phase = [m*k[m-1]*np.cos(m*dy) for m in range(1,1+self.nH)]
        phase = np.sum(phase, axis=0)

        for i in range(n_osc):
            phase[i,i] = -np.sum(phase[:,i])

        return phase

############################################
##  MAIN PROGRAMME
if __name__ == "__main__":

    #######################################
    ## Flags

    # Plotting
    FIG_REC_TIME = 0 # Plot reconstructed time series
    FIG_REC_FREQ = 0 # Plot signal's Fourier spectrum
    FIG_REC_ALL  = 0 # Plot all components and their FT
    FIG_REC_DIFF = 0 # Plot components and differents to cos(wt)
    SHOW_PLOTS   = 0 # Show plots

    if np.any((FIG_REC_TIME, FIG_REC_FREQ, FIG_REC_ALL, FIG_REC_DIFF)):
        import pylab as plt

    # Assign parameters randomly
    RANDOM = True

    #######################################
    # Number of oscillators
    oscN = 3
    nH = 3

    dt = 0.01
    tMin, tMax = 0, 5
    fMin, fMax = 0, 25

    T = np.linspace(tMin, tMax, (tMax-tMin)/dt)
    N = T.size

    #######################################
    if RANDOM:
        W_MIN, W_MAX = 6, 90
        Y_MIN, Y_MAX = 0, 2*np.pi
        R_MIN, R_MAX = 0, 5
        K_MIN, K_MAX = -3.5, 7.5

        W = np.random.random(oscN)*W_MAX + W_MIN
        Y0 = np.random.random(oscN)*Y_MAX + Y_MIN
        R = np.random.random(oscN)*R_MAX + R_MIN
        K = np.random.random((oscN, (oscN-1)*nH))*K_MAX + K_MIN
        K = (K.T*0.1*W).T

    # Some exemplary values
    else:
        W = [2, 5., 10., 13, 6, 19]
        Y0 = [1.2, 2.5, 0, 2, 0.5]
        R = [1, 1.5, 1, 1, 0.5]
        K = [[ 2.0,  1.5,  .5,   2., -1.0, -3.1],
             [  -2, -7.0, 4.1,   2., -6.0,  2.8],
             [ 2.2,  3.2, 3.8,  0.0,  2.9,  1.1],
             [ 0.1, 10.1, 0.0,  0.5, -3.1, -9.7],
             [-1.7,  0.1, 4.5, -8.5,  2.0,  5.5]]

        W = np.array(W[:oscN])*2.*np.pi
        Y0 = np.array(Y0)[:oscN]
        R = np.array(R)[:oscN]
        K = np.array(K)[:oscN, :(oscN-1)*nH]

    #######################################
    # Merge data and sort in reverse freq order
    genParams = np.column_stack((W,Y0,R,K))
    genParams = genParams[np.argsort(W)[::1]]

    labels_list = ["W","Y0","R"] + ['k^{%i}_{%i}'%(i,j) for j in range(1,1+nH) for i in range(1,oscN)]
    labels_str = ' & '.join(labels_list)+"\n"
    labels_byte = bytes(labels_str.encode("ASCII"))

    f = open('genParams.txt','wb')
    f.seek(0)
    f.truncate() # Wipe file
    f.write(labels_byte)
    np.savetxt(f, genParams, fmt='%.2f',delimiter=' & ')
    f.close()
    print("genParams: ", genParams)

    #######################################
    ## Start model
    kurSL = KurSL(genParams)
    phi, amp, sOscInput = kurSL.generate(T)
    sInput = np.sum(sOscInput, axis=0)

    T = T[:-1] # sInput based on diff

    saveName = 'kursl-model'
    np.savez(saveName, genParams=genParams, sOscInput=sOscInput,
                A=amp, phi=phi, T=T)

    #######################################
    # Plotting results
    freq = np.fft.fftfreq(sInput.size, dt)
    idx = np.r_[freq>fMin] & np.r_[freq<fMax]

    eachFT = np.abs(np.fft.fft(sOscInput)[:,idx])
    FT = np.abs(np.fft.fft(sInput)[idx])
    freq = freq[idx]


    ####################
    if FIG_REC_ALL:
        fig = plt.figure(figsize=(10,3*oscN))

        for n in range(oscN):
            # Time domain
            ax = fig.add_subplot(oscN, 2, 2*n+1)
            plt.plot(T, sOscInput[n])
            plt.plot(T, -amp[n],'r')
            plt.plot(T,  amp[n],'r')

            yMax = np.max(np.abs(amp[n]))
            plt.ylim((-yMax*1.05, yMax*1.05))
            plt.locator_params(axis='y', nbins=4)
            if(n==0): plt.title("Time series")
            if(n==oscN-1): plt.xlabel("Time [s]")
            if(n!=oscN-1): plt.gca().axes.get_xaxis().set_ticks([])

            # Frequency domain
            ax = fig.add_subplot(oscN, 2, 2*n+2)
            plt.plot(freq, eachFT[n]/np.max(eachFT[n]))
            plt.locator_params(axis='y', nbins=3)

            plt.gca().axes.get_yaxis().set_ticks([])
            if(n==0): plt.title("Fourier spectrum")
            if(n==oscN-1): plt.xlabel("Frequency [Hz]")
            if(n!=oscN-1): plt.gca().axes.get_xaxis().set_ticks([])

        #~ plt.suptitle("All comp TF Dist")
        plt.tight_layout()
        plt.savefig('KurSL_eachTFD_or{}'.format(nH), dpi=120)

    ####################
    if FIG_REC_DIFF:
        fig = plt.figure()

        for n in range(oscN):
            # Time domain
            ax = fig.add_subplot(oscN, 2, 2*n+1)
            plt.plot(T, sOscInput[n])
            plt.plot(T, -amp[n],'r')
            plt.plot(T,  amp[n],'r')

            yMax = np.max(np.abs(amp[n]))
            plt.ylim((-yMax*1.05, yMax*1.05))
            plt.locator_params(axis='y', nbins=4)
            if(n!=oscN-1): plt.gca().axes.get_xaxis().set_ticks([])

            # Time domain - diff to no coupling
            ax = fig.add_subplot(oscN, 2, 2*n+2)
            plt.plot(T, sOscInput[n]-R[n]*np.cos(W[n]*T+Y0[n]))

            plt.locator_params(axis='y', nbins=4)
            if(n!=oscN-1): plt.gca().axes.get_xaxis().set_ticks([])

        plt.suptitle("All comps and cmp to no coupling comps")
        plt.tight_layout()
        plt.savefig('KurSL_diffTD_or{}'.format(nH))

    ####################
    if FIG_REC_TIME:
        plt.figure()
        plt.plot(T, sInput)
        plt.title("Time series")
        plt.ylabel("Amplitude")
        plt.xlabel("Time [s]")
        plt.savefig('KurSL_TD_or{}'.format(nH))

    ####################
    if FIG_REC_FREQ:
        plt.figure()
        plt.plot(freq, FT/np.max(FT))
        plt.xlim((fMin, fMax))
        plt.title("Fourier spectrum")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")
        plt.savefig('KurSL_FD_or{}'.format(nH))

    ####################
    if SHOW_PLOTS:
        plt.show()

    ########################################
    ######      this is the end       ######
    ########################################
