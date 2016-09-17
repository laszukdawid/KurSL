#!/usr/bin/python
# coding: UTF-8
#
# Author:  Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 16/09/2016
#
# Implemented with Python 2.7 in mind.
# Feel free to contact for any information.
from __future__ import print_function
import numpy as np

from numpy import cos, pi
from scipy.integrate import odeint

class KurSL:
    """
        KurSL assigns model with parameters P.
        These parameters has to convertable to NumPy 2D array. 
        It is assumed that P is (oscN, oscN*(3+nH(oscN-1))) size,
        where for each oscillators parameters are: W, ph, A, k(1)(1), ..., k(oscN-1)(nH).
    """
    def __init__(self, P):
        
        # Assert conditions
        P = np.array(P)
        assert(P.ndim == 2)
        assert(P.shape[0]>1)
        
        # Extracting model parameters
        self.oscN, self.p = oscN, p = P.shape
        self.nH = nH = (p-3)/(oscN-1)
        self.P = P
        
        self.W = W = P[:,0]
        self.Y = Y = P[:,1]
        self.R = R = P[:,2]
        self.K = K = P[:,3:]

        # Convert K array to include self coupling (K_ii = 0)  
        kMat = np.zeros((oscN, nH*oscN))
        for _nH in xrange(nH):
            _k = K[:,_nH*(oscN-1):(_nH+1)*(oscN-1)]
            kMat[:, _nH*oscN+1:(_nH+1)*oscN] += np.triu(_k)
            kMat[:, _nH*oscN:(_nH+1)*oscN-1] += np.tril(_k,-1)
        
        self.kMat = kMat
        
    def generateKurSL(self, t):

        # Solve ODE with initial conditions (Y0) and 
        # model parameters (W, K).       
        arg = (self.W,self.kMat)
        phi = odeint(self.kuramoto_ODE, Y0, t, args=arg)
        
        # Transpose such that phase and its derivative
        # has dim(S) = (osc, t.size)
        phi = phi.T
        dPhi = np.diff(phi)/dt
        
        # Check for zero division
        assert(not np.any(phi)==0)
        assert(not np.any(dPhi)==0)
    
        # Apply phase relation to obtain amplitude (ampC)
        # and phase (phiC) components.
        ampC = 1./np.sqrt(dPhi/dPhi[:,0][:,None])
        phiC = cos(phi[:,:-1])
        S = R[:,None]*ampC*phiC
    
        return phiC, ampC, S
    
    def kuramoto_ODE(self, y, t, w, k):
        """Function passed for ODE solver.
           In this case it is frequency Kuramoto model.
        """
    
        w = np.array(w)
        k = np.array(k)
        n_osc = len(w)
        y = np.array(y)
    
        yt = y[:,None]
        dy = y-yt
        phase = w
        for i in xrange(nH):
            phase += np.sum(k[:,i*n_osc:(i+1)*n_osc]*np.sin((i+1)*dy),axis=1)
        
        return phase

############################################
##  MAIN PROGRAMME 
if __name__ == "__main__":
    import pylab as py

    #######################################
    # Flags
    FIG_REC_TIME = 0
    FIG_REC_FREQ = 0
    FIG_REC_ALL  = 1
    FIG_REC_DIFF = 0
    
    SHOW_PLOTS   = 1
    REAL_DATA = 0
    ADD_NOISE = 0
    RANDOM = 0
    
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
        Y_MIN, Y_MAX = 0, 2*pi
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
        
        W = np.array(W[:oscN])*2.*pi
        Y0 = np.array(Y0)[:oscN]
        R = np.array(R)[:oscN]
        K = np.array(K)[:oscN, :(oscN-1)*nH]
    
    #######################################
    # Merge data and sort in reverse freq order
    genParams = np.column_stack((W,Y0,R,K))
    genParams = genParams[np.argsort(W)[::1]]
    
    np.savetxt('genParams.txt', genParams, fmt='%.2f',delimiter=' & ')
    print("genParams: ", genParams)

    #######################################
    ## Start model
    kurSL = KurSL(genParams) 
    phi, A, sOscInput = kurSL.generateKurSL(T)
    sInput = np.sum(sOscInput, axis=0)
        
    T = T[:-1] # sInput based on diff

    saveName = 'kursl-model'
    np.savez(saveName, genParams=genParams, sOscInput=sOscInput, 
                A=A, phi=phi, T=T)
                
    #######################################
    # Plotting results
    freq = np.fft.fftfreq(sInput.size, dt)
    idx = np.r_[freq>fMin] & np.r_[freq<fMax]
    
    eachFT = np.abs(np.fft.fft(sOscInput)[:,idx])
    FT = np.abs(np.fft.fft(sInput)[idx])
    freq = freq[idx]


    ####################
    if FIG_REC_ALL:
        fig = py.figure(figsize=(10,3*oscN))
        
        for n in xrange(oscN):
            # Time domain
            ax = fig.add_subplot(oscN, 2, 2*n+1)
            py.plot(T, sOscInput[n])
            py.plot(T, -A[n],'r')
            py.plot(T,  A[n],'r')
            
            yMax = np.max(np.abs(A[n]))
            py.ylim((-yMax*1.05, yMax*1.05))
            py.locator_params(axis='y', nbins=4)
            if(n!=oscN-1): py.gca().axes.get_xaxis().set_ticks([])

            # Frequency domain
            ax = fig.add_subplot(oscN, 2, 2*n+2)
            py.plot(freq, eachFT[n]/np.max(eachFT[n]))
            py.locator_params(axis='y', nbins=3)

            py.gca().axes.get_yaxis().set_ticks([])
            if(n!=oscN-1): py.gca().axes.get_xaxis().set_ticks([])
        
        #~ py.suptitle("All comp TF Dist")
        py.tight_layout()
        py.savefig('KurSL_eachTFD_or{}'.format(nH), dpi=120)

    ####################
    if FIG_REC_DIFF:
        fig = py.figure()
        
        for n in xrange(oscN):
            # Time domain
            ax = fig.add_subplot(oscN, 2, 2*n+1)
            py.plot(T, sOscInput[n])
            py.plot(T, -A[n],'r')
            py.plot(T,  A[n],'r')
            
            yMax = np.max(np.abs(A[n]))
            py.ylim((-yMax*1.05, yMax*1.05))
            py.locator_params(axis='y', nbins=4)
            if(n!=oscN-1): py.gca().axes.get_xaxis().set_ticks([])

            # Time domain - diff to no coupling
            ax = fig.add_subplot(oscN, 2, 2*n+2)
            py.plot(T, sOscInput[n]-R[n]*np.cos(W[n]*T+Y0[n]))

            py.locator_params(axis='y', nbins=4)
            if(n!=oscN-1): py.gca().axes.get_xaxis().set_ticks([])

        py.suptitle("All comps and cmp to no coupling comps")
        py.tight_layout()
        py.savefig('KurSL_diffTD_or{}'.format(nH))

    ####################
    if FIG_REC_TIME:
        py.figure()
        py.plot(T, sInput)
        py.savefig('KurSL_TD_or{}'.format(nH))

    ####################
    if FIG_REC_FREQ:
        py.figure()
        py.plot(freq, FT/np.max(FT))
        py.xlim((fMin, fMax))
        py.savefig('KurSL_FD_or{}'.format(nH))
    
    ####################
    if SHOW_PLOTS:
        py.show()

    ########################################
    ######      this is the end       ######
    ########################################