#!/usr/bin/python
# coding: UTF-8
#
# Author: Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 21/07/2016
#
# Feel free to contact for any information.
import numpy as np
import pylab as py

from numpy import sin, cos, pi
from scipy.integrate import odeint
import scipy.optimize as opt
import os, sys, time

from mcmc import MCMC

############################################
## Function declaration

def movingAverage(s, r=2):

    N = s.size
    S = np.zeros(N+r)
    S[:r] = s[0]
    for i in xrange(r): S[i:N+i] += s

    return S[:N]/r

def removePeak(t, s, ptype="norm"):
    if ptype=="norm":
        def peak(t, *p):
            return p[1]*np.exp(-((p[0]-t)/p[2])**2)
            #~ return p[1]*np.exp(-((p[0]-t)/2)**2)

        _amp = np.max(s)
        _pos = t[s==_amp][0]
        _wd = 0.2

        initGuess = ([_pos, _amp, _wd])
        #~ initGuess = ([_pos, _amp])

    elif ptype=="triang":
        def peak(t, *p):
            #~ s = 1-np.abs(t-p[0])
            s = 1-np.abs((t-p[0])/p[2])
            s[s<=0] = 0
            return p[1]*s

        _amp = np.max(s)
        _pos = t[s==_amp][0]
        _wd = 0.2
        initGuess = ([_pos, _amp, _wd])
        #~ initGuess = [_pos, _amp]

    elif ptype=="lorentz":
        def peak(t, *p):
            #~ return p[1]/( (t-p[0])*(t-p[0]) + 1)
            return p[1]/( (t-p[0])*(t-p[0]) + p[2]*p[2])

        _amp = np.max(s)
        _pos = t[s==_amp][0]
        _wd = 0.2
        initGuess = ([_pos, _amp, _wd])
        #~ initGuess = ([_pos, _amp])

    #~ pdb.set_trace()
    _param = opt.curve_fit(peak, t, s, initGuess)
    _param = map(abs, _param)
    peakS = peak(t, *_param[0])

    print "Init: ", initGuess
    print "Fit:  ", _param[0]
    print "Diff: ", np.array(_param[0])-np.array(initGuess)

    return peakS, _param[0]

def removePeaks(t, s, nPeaks=2, ptype="norm"):

    S = s.copy()
    for _p in xrange(nPeaks):

        _peakS, _param = removePeak(t, S, ptype=ptype)

        print "\n  _p = ", _p
        print _param[0]
        S = S - _peakS

        #~ S = np.abs(S)
        print "Norm S: ", np.sum(np.abs(S)**2)

    return S, _param

def removeEnergy(t, s, ratio=0.5, maxPeaks=5, ptype="norm"):

    norm = lambda x: np.sum(np.abs(x)**2)

    E = norm(s)
    S = s.copy()

    param = []
    while(1):
        _peakS, _param = removePeak(t, S, ptype)
        S = S.copy() - _peakS
        S[S<0] = 0

        _param[1] = np.sum(_peakS)*8
        param.append(_param)

        nE = norm(S)
        r = nE/E

        print "nE = {}, (r = {} )".format(nE, r)

        if( (r<=ratio) or (len(param)>=maxPeaks)): break

    return S, param

def removeFourierEnergy(t, s, ratio=0.5, maxPeaks=5, ptype="norm"):
    freq = np.fft.fftfreq(t.size, t[1]-t[0])

    idx = np.r_[freq>0] & np.r_[freq<100]
    print 's.shape: ', s.shape
    freq = freq[idx]
    F = np.abs(np.fft.fft(s)[idx])

#     F = movingAverage(F, 3)
    #~ F = movingAverage(F, 10)
    py.plot(freq, F)
    py.savefig("in_spectrum")
    py.close()

    return removeEnergy(freq, F, ratio=ratio, maxPeaks=maxPeaks, ptype=ptype)

def recKurSL(t, C):

    t0, t1, dt = t[0], t[-1], t[1]-t[0]

    oscN = C.shape[0]
    W  = C[:,0]
    Y0 = C[:,1]
    R  = C[:,2]
    K  = C[:,3:3+nH*(oscN-1)]

    kMat = np.zeros((oscN, nH*oscN))
    for _nH in xrange(nH):
        _k = K[:,_nH*(oscN-1):(_nH+1)*(oscN-1)]
        kMat[:, _nH*oscN+1:(_nH+1)*oscN] += np.triu(_k)
        kMat[:, _nH*oscN:(_nH+1)*oscN-1] += np.tril(_k,-1)

    arg = (W,kMat)
    phi = odeint(kuramoto_ODE, Y0, t, args=arg)

    phi = phi.T
    dPhi = np.diff(phi)/dt

    A = 1./np.sqrt(dPhi/dPhi[0])
    A *= (R/np.max(A, axis=1))[:,None]
    P = cos(phi[:,:-1])
    S = A*P

    return phi[:,:-1], A, S

def kuramoto_ODE(y, t, w, k):
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


def determineParams(t, s, ratio=0.001, ptype="triang", maxPeaks=10):

    freq = np.fft.fftfreq(t.size, t[1]-t[0])

    idx = np.r_[freq>=fMin] & np.r_[freq<fMax]

    freq = freq[idx]
    F = np.fft.fft(s)[idx]

    #~ maF = movingAverage(np.abs(F), 3)
    maF = np.abs(F)
    py.plot(freq, maF)
    py.savefig("in_spectrum")
    py.close()

    fourierS, param = removeEnergy(freq, maF, ratio=ratio, maxPeaks=maxPeaks, ptype=ptype)

    param = np.array(param)
    param = param[param[:,0].argsort()]
    param = param.tolist()

    for i in xrange(len(param)):

        p = param[i]

        # Extracting phase
        minIdx = np.argmin(np.abs(p[0]-freq))
        param[i] = np.append(p, np.angle(F[minIdx]))

        # Scaling amplitude
        param[i][1] = np.sqrt(param[i][1]/len(fourierS))
        #~ param[i][1] = param[i][1]/np.sqrt(param[i][0]/6.14)
        #~ param[i][1] = [3,2,2][i]


    return param

######################################
##  MAIN PROGRAMME

if __name__ == "__main__":

    # MCMC variables
    nWalkers = 300
    nRuns = 200

    MATCH_THRESHOLD = 0.01


    global nH
    nH = 3
    maxOsc = 6

    # Flags
    ADD_NOISE = 0
    SET_PRIOR = 0
    SET_PRIOR_LOAD = 0
    LOAD_PARAM = 0

    REMOVE_MEAN = 1
    RANDOM_PARAM = 0
    W_MIN_DIFF = 3 # Synthetic specific; min distance in generated frequency

    signalType = ["synth","kursl","random","epilepsy"][3]
    ptype = ['triang', 'norm','lorentz'][0]

    fs = 300 # Hz
    tMin = 0 # Initial time
    tSeg = 5 # Length of segment
    tStep = 1 # Length of step
    segNum = [0,1] # Range: from 1

    fMin, fMax = 0, 40

    tMax = tMin+tSeg+(segNum[1]-segNum[0])*tStep
    N = int((tMax-tMin)*fs)
    nBeg = int(tMin*fs)
    nSeg = int(tSeg*fs)
    nStep = int(tStep*fs)

    ########################################
    ## Params for kursl/synth
    genH = 3
    oscN = maxOsc

    epiLetter = "S"
    epiNumber = 88

    ########################################
    ## Correct setting validation
    assert(signalType in ['synth','kursl', 'random', 'epilepsy'])
    assert(ptype in ['triang','norm','lorentz'])

    assert(tMin<tMax)
    assert(fMin<fMax)

    assert((tMin+nStep*tSeg)>tMax)

    ###################################################
    ##   Generating KurSL type signal
    if signalType in ['synth','kursl']:

        if oscN < 2: raise Exception('Number of oscillators has to be at least 2.')

        T = np.linspace(tMin, tMax, N+1)
        t0, t1, dt = T[0], T[-1], T[1]-T[0]

        if RANDOM_PARAM:
            print "Generating parameters for KurSL"
            W_MIN, W_MAX = 3, 40
            Y_MIN, Y_MAX = 0, 2*pi
            R_MIN, R_MAX = 0, 5
            K_MIN, K_MAX = -5, 5

            allK = {}

            while True:
                W = np.random.random(oscN)*W_MAX + W_MIN
                if np.all(np.diff(W)>W_MIN_DIFF): break

            R = np.random.random(oscN)*R_MAX + R_MIN
            Y0 = np.random.random(oscN)*Y_MAX + Y_MIN

            for _nH in xrange(genH):
                allK[_nH] = np.random.random((oscN, oscN-1))*K_MAX + K_MIN
        elif LOAD_PARAM:
            loadParams = np.load('testParams.npy')
            W = loadParams[:,0]
            Y0 = loadParams[:,1]
            R = loadParams[:,2]

            shape = loadParams.shape
            allK = {}
            maxH = (shape[1]-3)/(shape[0]-1)
            for _nH in xrange(maxH):
                allK[_nH] = loadParams[:, np.arange(shape[0]-1)*maxH+_nH]

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
        #~ K = np.array(K)[:oscN, :(oscN-1)*nH]
        K = np.zeros((oscN, genH*(oscN-1)))
        if signalType != 'synth':
            for _nH in xrange(genH):
                _tmpK = np.array(allK[_nH])
                K[:, np.arange(oscN-1)*genH+_nH] = _tmpK[:oscN,:oscN-1]


        kSum = np.sum(np.abs(K),axis=1)[:,None]
        idx = (kSum>W[:,None]).flatten()
        if np.any(idx): K[idx] *= 0.98*W[idx,None]/kSum[idx]
        #~ priorC[3*oscN:] = K.T.flatten()

        # Sorting params in reverse freq
        genParams = np.column_stack((W,Y0,R,K))
        genParams = genParams[np.argsort(W)[::-1]]
        #~ initTheta = genParams

        ###########################################
        ## Message
        print 'Generating %s type signal for %i oscillators with parameters' %(signalType,oscN)
        print 'genParams: ', genParams
        print 'genParams.shape: ', genParams.shape

        saveName = 'genParam-%s'%(signalType)
        np.savetxt(saveName+'.txt', genParams, fmt='%.6e')
        py.save(saveName, genParams)

        phi, A, sOscInput = recKurSL(T, genParams)
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
        print "Loading epilepsy data {}.{}".format(epiLetter, epiNumber)

        _l = epiLetter
        dataPath = os.path.join("..","..","Data","Epilepsy",_l,_l)
        print dataPath

        loadData = lambda x: np.loadtxt(dataPath+"{:03}.txt".format(x))

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
    py.figure()
    py.plot(T, sInput)
    py.savefig("sInput")
    py.clf()

    #~ s = sInput
    nInput = sInput.shape[0]

    ###################################################
    ## Loading prior parameters from file
    if SET_PRIOR_LOAD:
        initTheta = np.load('bestParam.npy')
        initTheta = initTheta[np.argsort(initTheta[:,0])[::-1]]

    ## Setting up priors (possibly generated before, but incomplete)
    elif SET_PRIOR:
        initTheta = genParams.copy()
        oscN = initTheta.shape[0]
        paramN = 3+nH*(oscN-1)

        while(initTheta.size < oscN*paramN):
            initTheta = np.column_stack((initTheta, np.zeros(oscN)))

    ## Prior for
    else:
        #~ t = T[nBeg:nBeg+nSeg]
        param = determineParams(T, sInput, ratio=0.01, ptype=ptype, maxPeaks=maxOsc)
        param = np.array(param)

        print "Params: "
        for p in param: print p


        if np.any(param[:,:2]<0):
            print "*"*30
            print "\n SOMETHING IS WRONG !! \n"
            print " some parameters are estimated as  negative! "
            print " we're making them positive! "
            print "*"*30
            param = np.abs(param)

        oscN = param.shape[0]
        paramN = 3+nH*(oscN-1)
        W = param[:, 0]*6.28
        R = param[:, 1]
        Y0= (param[:, -1]+2*pi)%(2*pi)

        print 'W: ', W
        priorC = np.zeros(oscN*paramN)
        priorC[      :  oscN] = W
        priorC[  oscN:2*oscN] = Y0
        priorC[2*oscN:3*oscN] = R

        # There's no point in analysing
        if(oscN<2):
            raise Exception("Single oscillator detected. No very interesting case.")

        # Prior K
        if oscN!=1:
            K = (np.random.random((oscN, nH*(oscN-1)))-0.5)*5.
            kSum = np.sum(np.abs(K),axis=1)[:,None]
            idx = (kSum>W[:,None]).flatten()
            if np.any(idx): K[idx] *= 0.98*W[idx,None]/kSum[idx]
            priorC[3*oscN:] = K.T.flatten()
            #~ priorC = priorC[:, None]
        print 'priorC: ', priorC.reshape((-1, oscN)).T

        ####################################################
        ## Reconstructing signal
        initTheta = np.column_stack((W,Y0,R,K))

        # Sorting params in reverse freq
        initTheta = initTheta[np.argsort(W)[::-1]]

        # Save initial parameters
        np.savetxt('initParam.txt', initTheta.T, delimiter=' & ', fmt='%7.4g')
        sys.exit()

    ####################################################

    ####################################################
    ## Data presentation
    print 'Running simulation(s) with these values:'
    print "oscN: ", oscN
    print "nH:   ", nH
    print "paramN: {} (min walkers should be 2*paramN)".format(oscN*(3+nH*(oscN-1)))


    ####################################################
    ##  For each segment
    for seg in xrange(segNum[0], segNum[1]):

        # Calculate initial indices
        idxStart = nBeg + seg*nStep
        idxEnd = idxStart + nSeg

        t = T[idxStart:idxEnd]
        s = sInput[idxStart:idxEnd]

        #######################################################
        ## If not first segment, get priors from previous simulation
        if seg:
            # Priors are set by reading a file with results from previous simulation.
            # This obviously can be skipped since we already know those results.
            # However, if one wants to start in the middle it is easier this way.
            # For slight optimisation one could remove reading file and pass data via reference.

            # Loading results
            loadName = 'KurSL_results-%s-%i.npz'%(signalType,seg-1)
            print "Loading params from files: ", loadName
            data = np.load(loadName)

            samples = data['samples']
            lnprob = data['lnprob']
            data.close()

            # If previous simulation was halted due to good fitting (below threshold),
            # then some values will be unassigned (zeros). Check how many samples
            # were used previously.
            curIdx = sum(samples[:,0,0]!=0)
            print 'curIdx: ', curIdx
            print "Based on %i percent"%(int(curIdx*100./samples.shape[0]))
            samples = samples[:curIdx,:,:]
            lnprob = lnprob[:curIdx,:]

            sDim = paramN = samples.shape[1]

            # Info on input
            if 'oscN' not in locals():
                oscN = int((np.sqrt((3-nH)*(3-nH)+4*nH*paramN)-(3-nH))/(2*nH))

            print "[Start] Estimation on all values"

            bestIdx = np.argmax(lnprob)
            print "bestIdx: ", bestIdx
            bestParam = samples[ int(bestIdx/sDim), bestIdx%sDim, :]
            bestParam = bestParam.reshape((oscN, 3+nH*(oscN-1)))

            initTheta = bestParam

        ####################################################
        ####################################################

        ####################################################
        ##  MCMC part

        # Setting number of Walkers
        nWalkers = max( nWalkers, int(1.1*initTheta.size)*2)

        x = t.copy()
        y = s[:x.size-1].copy()

        if REMOVE_MEAN: y -= np.mean(y)

        print "nWalkers: ", nWalkers
        print "nRuns: ", nRuns
        print "x.shape: ", x.shape
        print "y.shape: ", y.shape

        # Check everything is fine
        assert(nWalkers>2*initTheta.size)
        assert((x.size-1)==y.size)
        # Saving results
        saveName = 'KurSL_results-%s-%i'%(signalType,seg)
        if(signalType in ['synth', 'kursl']):
            np.savez(saveName, sInput=y, x=x, nH=nH, genParams=genParams)
        else:
            np.savez(saveName, sInput=y, x=x, nH=nH)


        mcmc = MCMC(initTheta, nwalkers=nWalkers, nH=nH)
        if seg: mcmc.setPriorDist(samples)
        mcmc.setThreshold(MATCH_THRESHOLD)
        mcmc.setModel(recKurSL)
        mcmc.setSampler(x,y)
        mcmc.run(N=nRuns)

        # Plot comparison between plots
        samples = mcmc.getSamples()
        lnprob = mcmc.getLnprob()

        sDim = samples.shape[1]
        bestIdx = np.argmax(lnprob)
        print "bestIdx: ", bestIdx
        bestParam = samples[int(bestIdx/sDim), bestIdx%sDim, :]

        print 'oscN: ', oscN
        print 'nH: ', nH
        print "bestParam: "
        print bestParam
        bestParam = bestParam.reshape((-1, 3+nH*(oscN-1)))

        # Saving results
        print "Saving results to " + saveName + "... "
        if(signalType in ['synth', 'kursl']):
            np.savez(saveName, samples=samples, lnprob=lnprob, sInput=y, x=x, nH=nH, genParams=genParams)
        else:
            np.savez(saveName, samples=samples, lnprob=lnprob, sInput=y, x=x, nH=nH)

        ####################################################
        ####################################################


    print "----------------"
    print "|Finished KurSL|"
    print "----------------"
