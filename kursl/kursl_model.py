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
        #amp *= R[:,None]
        amp[:] = amp*(R/np.max(amp, axis=1))[:,None]
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

    import pylab as plt

    # Plotting
    FIG_REC_TIME = 1 # Plot reconstructed time series
    FIG_REC_FREQ = 1 # Plot signal's Fourier spectrum
    FIG_REC_ALL  = 1 # Plot all components and their FT
    SHOW_PLOTS   = 1 # Show plots

    #######################################
    # Number of oscillators
    oscN = 4
    nH = 2

    t_min, t_max, dt = 0, 5, 0.005
    f_min, f_max = 0, 30

    T = np.arange(t_min, t_max, dt)

    #######################################
    W_MIN, W_MAX = 6, 150
    Y_MIN, Y_MAX = 0, 2*np.pi
    R_MIN, R_MAX = 0, 5
    K_MIN, K_MAX = -3.5, 7.5

    W = np.random.random(oscN)*W_MAX + W_MIN
    Y0 = np.random.random(oscN)*Y_MAX + Y_MIN
    R = np.random.random(oscN)*R_MAX + R_MIN
    K = np.random.random((oscN, (oscN-1)*nH))*K_MAX + K_MIN
    K = 0.1*W[:,None]*K/np.sum(np.abs(K), axis=1)[:,None]

    genParams = np.column_stack((W,Y0,R,K))
    np.savetxt("genParams.txt", genParams)
    print("genParams: ", genParams)

    #######################################
    ## Start model
    kurSL = KurSL(genParams)
    phi, amp, s_osc = kurSL.generate(T)
    s_flat = np.sum(s_osc, axis=0)
    T = T[:-1] # signal based on diff

    saveName = 'kursl-model'
    np.savez(saveName, genParams=genParams, s_input=s_osc,
                A=amp, phi=phi, T=T)

    #######################################
    # Plotting results
    freq = np.fft.fftfreq(s_flat.size, dt)
    idx = np.r_[freq>f_min] & np.r_[freq<f_max]

    eachFT = np.abs(np.fft.fft(s_osc)[:,idx])
    FT = np.abs(np.fft.fft(s_flat)[idx])
    freq = freq[idx]

    ####################
    if FIG_REC_ALL:
        fig = plt.figure(figsize=(10,3*oscN))

        for n in range(oscN):
            # Time domain
            ax = fig.add_subplot(oscN, 2, 2*n+1)
            plt.plot(T, s_osc[n])
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
        plt.savefig('KurSL_eachTFD', dpi=120)

    ####################
    if FIG_REC_TIME:
        plt.figure()
        plt.plot(T, s_flat)
        plt.title("Time series")
        plt.ylabel("Amplitude")
        plt.xlabel("Time [s]")
        plt.savefig('KurSL_TD')

    ####################
    if FIG_REC_FREQ:
        plt.figure()
        plt.plot(freq, FT/np.max(FT))
        plt.xlim((f_min, f_max))
        plt.title("Fourier spectrum")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequency [Hz]")
        plt.savefig('KurSL_FD')

    ####################
    if SHOW_PLOTS:
        plt.show()

    ########################################
    ######      this is the end       ######
    ########################################

