#!/usr/bin/python
# coding: UTF-8
#
# Author:  Dawid Laszuk
# Contact: laszukdawid@gmail.com
#
# Last update: 30/07/2016
#
# Feel free to contact for any information.

# Used colour schema:
CMAP = 'red'      # MAP value
CMED = 'blue'     # Median value
CBEST = 'yellow'  # Global MAP value
CMEAN = 'magenta' # Mean value
CTRUE = 'black'   # True value

import matplotlib
font = {#'family' : 'normal',
        #~ 'weight' : 'bold',
        'size'   : 10}

matplotlib.rc('font', **font)
import numpy as np
import pylab as py

from numpy import sin, cos, pi
from scipy.integrate import odeint
from scipy.stats import gaussian_kde
import os, sys, time

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
    
def kdeFunc(samples):
    wNum, sNum, pNum = samples.shape

    kdeNum = 50
    xKde = np.zeros((pNum, kdeNum))
    yKde = np.zeros((pNum, kdeNum))


    for i in xrange(pNum):
        y = samples[:,:,i].flatten()
        xKde[i] = np.linspace(np.min(y), np.max(y), kdeNum)
        yKde[i] = gaussian_kde(y, bw_method="silverman")(xKde[i])

    return xKde, yKde


def mapFunc(xKde, yKde):
    """ Returns xKde value for which yKde is the biggest.
    """
    M = np.argmax(yKde, axis=-1)

    #~ return xKde[yKde==np.max(yKde, axis=-1)[:,None]]
    return np.array([xKde[i, M[i]] for i in xrange(M.size)])


def plotRec(t, S, saveName):

    F = np.fft.fft(S)
    freq = np.fft.fftfreq(S.size, t[1]-t[0])

    idx = freq>0
    freq = freq[idx]
    F = np.abs(F[idx])

    py.figure(figsize=(14,7))

    # Plotting time series
    py.subplot(121)
    py.plot(t[:-1], S)

    py.xlim((min(t), max(t)))
    py.locator_params(axis='y', tight=None, nbins=3)
    py.locator_params(axis='x', nbins=5)
    
    py.savefig(saveName+'_TD', dpi=120)

    # Plotting frequency
    py.subplot(122)
    py.plot(freq, F)
    py.xlim((fMin, fMax))

    py.savefig(saveName+'_FD', dpi=120)

######################################
##  MAIN PROGRAMME

if __name__ == "__main__":

    #############################################
    # Flags
    FIG_SIG_REC  = 1
    PLOT_W_HIST  = 1
    PLOT_Ph_HIST = 1
    PLOT_A_HIST  = 1
    PLOT_K_HIST  = 1

    PLOT_CORNER_FREQ = 0
    PLOT_CORNER_Ph   = 0
    PLOT_CORNER_A    = 0
    PLOT_CORNER_K    = 0
    PLOT_CORNER_ALL  = 0
    PLOT_REC = 1
    
    PLOT_INPUT = 1
    PLOT_SPEC = 1

    REAL_DATA = 0
    ADD_NOISE = 0

    MEDIAN = 0
    MAP    = 0
    MEAN   = 0
    BEST   = 1
    TRUE   = 1

    COMPARE_TRUTH = 1

    _discardSamples = 20

    fMin, fMax = 0, 30
    binSize = 30

    #############################################
    LIM_VIEW = 1
    xLimW = [(75,85),(55,65),(35,43),(28,35),(8,15)]
    xLimA = [(1,3),(1,2.5),(2,4),(1,2.5),(0.5,2)]
    
    xLimK = [[( 0, 7), (-1, 6), ( 1, 5), (-3, 2)],
             [(-3, 0), (-4, 0), (-3, 2), (-4, 1)],
             [( 1, 5), (-3, 2), (-4, 2), (-3, 3)],
             [(-3, 2), (-2, 2), (-3, 2), (-3, 2)],
             [(-3, 1), (-2, 3), (-1, 4), (-5, 1)]
             ]
    #############################################

    # Files are saved by method and by number of seg.
    # If not dynamic then segNum = 0. 
    signalType = ['synth', 'kursl'][1]
    segNum = 0 

    # Loading results
    data = np.load('KurSL_results-%s-%i.npz' %(signalType,segNum))

    try:
        samples = data['samples']
        lnprob = data['lnprob']
    except KeyError:
        samples = np.load('pos.npy')
        lnprob = np.load('lnprob.npy')

        curIdx = sum(samples[:,0,0]!=0)
        print 'curIdx: ', curIdx
        print "Based on %i percent"%(int(curIdx*100./samples.shape[0]))
        samples = samples[:curIdx,:,:]
        lnprob = lnprob[:curIdx,:]

    sInput = data['sInput']
    t = data['x']
    nH = data['nH']
    genParams = data['genParams']

    ##############################################

    size = min(t.size, sInput.shape[-1])
    t = t[:size]

    print 't.shape: ', t.shape
    print 'sInput.shape: ', sInput.shape

    if len(sInput.shape)!=1:
        sInputFlat = np.sum(sInput, axis=0)
    else:
        sInputFlat = sInput
        sInput = sInput.reshape((1,-1))

    sInput = sInput[:, :size]
    nInput = sInput.shape[0]
    
    L = len(samples[0,0,:])
    oscN = int((np.sqrt((3-nH)*(3-nH)+4*nH*L)-(3-nH))/(2*nH))
    paramN = int(3+nH*(oscN-1))

    # Discarding some values
    discIdx = min(int(0.2*samples.shape[0]), _discardSamples)
    samples = samples[discIdx:,:,:]
    lnprob = lnprob[discIdx:,:]

    print "Loaded parameters: " 
    print 'nH:     ', nH
    print 'oscN:   ', oscN
    print 'paramN: ', paramN
    print 'samples.shape: ', samples.shape

    ##################################################
    # Normalising plots
    normF = lambda s: np.sum(np.abs(s)**2, axis=1)
    mse = lambda x,y: np.mean((x-y)*(x-y))
    
    # normalise phase to 0--2pi
    _phIdx = paramN*np.arange(oscN)+1
    samples[:,:, _phIdx] = (samples[:,:, _phIdx]+2*np.pi) % (2*np.pi)

    # If needed to compare to true values
    if COMPARE_TRUTH:
        Ph, A, trueRec = recKurSL(t, genParams)

    #################################################
    ## Plot initial signal
    if PLOT_INPUT: 
        s = np.sum(sInput, axis=0)
        py.figure()
        py.plot(t, s)
        py.savefig('sInput', dpi=120)
        py.close()
        
    if PLOT_SPEC:
        s = np.sum(sInput, axis=0)
        freq = np.fft.fftfreq(s.size, t[1]-t[0])
        idx = np.r_[freq>fMin] & np.r_[freq<fMax]
        
        freq = freq[idx]
        S = np.abs(np.fft.fft(s)[idx])
        py.figure()
        py.plot(freq, S)
        py.savefig('sInputFreq', dpi=120)
        py.close()
    
    #################################################
    ## Process reconstructions
    if BEST:
        print "[Start] Estimation on all values"
        # Estimate as median

        sDim = samples.shape[1]
        bestIdx = np.argmax(lnprob)
        print "bestIdx: ", bestIdx
        
        # Double check the indices are fine
        assert(int(bestIdx/sDim)==int(np.floor(float(bestIdx)/sDim)))
        bestParam = samples[ int(bestIdx/sDim), bestIdx%sDim, :]
        bestParam = bestParam.reshape((oscN, paramN))
        np.save('bestParam', bestParam)

        print "bestParam: "
        print bestParam
        Ph, A, bestRec = recKurSL(t, bestParam)
        rec = bestRec

        bestRecSig = np.sum(bestRec, axis=0)

        if(FIG_SIG_REC):
            plotRec(t, bestRecSig, "recSig-best")

        if(oscN==nInput):
            normValS = normF(sInput)/normF(rec)
            recNormed = rec*normValS[:,None]

            # Checking error of initial guess
            err = mse(rec, sInput[:,:-1])
            normErr = mse(recNormed, sInput[:,:-1])
            print "normF(recNomred): ", normF(recNormed)
            print "normF(sInput[:,:-1]): ", normF(sInput[:,:-1])
            print "rec-sInput[:,:-1]: ", rec-sInput[:,:-1]
            print "\n" + "Best error: {} / (normed {} )".format(err, normErr) + "\n"


        #####################################
        # Plot MAP overall results from MCMC
        errArr = [0]*oscN

        py.figure()
        for i in range(oscN+1):

            py.subplot(oscN+1, 1, i+1)

            if(i==0):
                py.plot(t, sInputFlat, 'g')
                py.plot(t[:-1], bestRecSig, 'r')
                py.ylabel('rec')
            else:
                py.plot(t[:-1], rec[i-1], 'r')
                if COMPARE_TRUTH and (i-1)<genParams.shape[0]:
                    py.plot(t[:-1], trueRec[i-1], 'g')
                py.ylabel(str(i))

            py.xlim((min(t), max(t)))
            py.locator_params(axis='y', tight=None, nbins=3)

            if i != oscN: py.gca().axes.get_xaxis().set_ticks([])
            else: py.locator_params(axis='x',nbins=5)

        py.subplots_adjust(bottom=0.05,hspace=0.4)

        totErr = mse(sInputFlat[:-1], bestRecSig)
        py.suptitle("Total MSE - %.3e" %totErr)
        py.savefig("fitBest", dpi=120)
        
        print "[Finished]  Estimation on all values"
        print

        py.clf()

    #############################################
    if MEDIAN:
        print "[Start] Estimating median values"
        # Estimate as median
        medianSamples = [np.median(samples[:,:, param]) for param in xrange(samples[0,0].size)]
        medParam = np.array(medianSamples)
        medParam = medParam.reshape((oscN, paramN))

        # Sorting invrese freq
        #~ medParam = medParam[np.argsort(medParam[:,0])[::-1]]

        print "medParam: "
        print medParam
        ph, A, medRec = recKurSL(t, medParam)
        rec = medRec

        medRecSig = np.sum(medRec, axis=0)

        if(FIG_SIG_REC):
            plotRec(t, medRecSig, "recSig-med")

        if(oscN==nInput):
            normValS = normF(sInput)/normF(rec)
            recNormed = rec*normValS[:,None]

            # Checking error of initial guess
            err = mse(rec, sInput[:,:-1])
            normErr = mse(recNormed, sInput[:,:-1])
            print "\n" + "Median error: {} / (normed {} )".format(err, normErr) + "\n"

        #####################################
        # Plot MEDIAN results from MCMC
        errArr = [0]*oscN
        py.figure()
        for i in range(oscN+1):

            py.subplot(oscN+1, 1, i+1)

            if(i==0):
                py.plot(t, sInputFlat, 'g')
                py.plot(t[:-1], medRecSig, 'r')
                py.ylabel('rec')
            else:

                py.plot(t[:-1], rec[i-1], 'r')
                if COMPARE_TRUTH and (i-1)<genParams.shape[0]:
                    py.plot(t[:-1], trueRec[i-1], 'g')
                py.ylabel(str(i))

            py.xlim((min(t), max(t)))
            py.locator_params(axis='y', tight=None, nbins=3)

            if i != oscN-1: py.gca().axes.get_xaxis().set_ticks([])
            else: py.locator_params(axis='x',nbins=5)

        py.subplots_adjust(bottom=0.05,hspace=0.4)

        totErr = mse(sInputFlat[:-1], medRecSig)
        py.suptitle("Total MSE - %.3e" %totErr)
        py.savefig("fitMedian", dpi=120)

        print "[Finished] Estimating median values"
        print

    #############################################
    if MAP:
        print "[Start] Estimating MAP values"

        print ' Estimating prob dist'
        xKde, yKde = kdeFunc(samples)
        mapParam = mapFunc(xKde, yKde).reshape((oscN,paramN))
        #~ mapParam = mapParam[np.argsort(mapParam[:,0])[::-1]]

        print "Map Param: "
        print mapParam
        Ph, A, mapRec = recKurSL(t, mapParam)
        rec = mapRec

        mapRecSig = np.sum(mapRec, axis=0)

        if(FIG_SIG_REC):
            plotRec(t, mapRecSig, "recSig-map")

        if(oscN==nInput):
            # Checking error of final guess
            recNorm = normF(rec)
            normValS = normF(sInput)/recNorm
            recNormed = rec*normValS[:,None]

            # Checking error of initial guess
            err = mse(rec, sInput[:,:-1])
            normErr = mse(recNormed, sInput[:,:-1])
            print "\n" + "MAP error: {} / (normed {} )".format(err, normErr) + "\n"

            print 'normValS: ', normValS
            optAmp = mapParam[:,-1]
            mapParam[:,-1] *= normValS

        a,b,rec = recKurSL(t, mapParam)
        mapRec = rec

        #####################################
        # Plot MAP results from MCMC
        errArr = [0]*oscN

        py.figure()
        for i in range(oscN+1):

            py.subplot(oscN+1, 1, i+1)

            if(i==0):
                py.plot(t, sInputFlat, 'g')
                py.plot(t[:-1], mapRecSig, 'r')
                py.ylabel('rec')
            else:
                py.plot(t[:-1], rec[i-1], 'r')
                if COMPARE_TRUTH and (i-1)<genParams.shape[0]:
                    py.plot(t[:-1], trueRec[i-1], 'g')
                py.ylabel(str(i))

            py.xlim((min(t), max(t)))
            py.locator_params(axis='y', tight=None, nbins=3)

            if i != oscN-1: py.gca().axes.get_xaxis().set_ticks([])
            else: py.locator_params(axis='x',nbins=5)


        py.subplots_adjust(bottom=0.05,hspace=0.4)

        totErr = mse(sInputFlat[:-1], mapRecSig)
        py.suptitle("Total MSE - %.3e" %totErr)
        py.savefig("fitMAP", dpi=120)
        #~ py.show()

        print "[Finished] Estimating MAP values"
        print

    #############################################
    if MEAN:
        print "[Start] Estimating mean values"

        meanParam = np.mean(np.mean(samples, axis=0), axis=0)
        meanParam = meanParam.reshape((oscN,paramN))
        #~ meanParam = meanParam[np.argsort(meanParam[:,0])[::-1]]

        print "Mean Param: "
        print meanParam
        Ph, A, meanRec = recKurSL(t, meanParam)
        rec = meanRec

        meanRecSig = np.sum(meanRec, axis=0)

        if(FIG_SIG_REC):
            plotRec(t, meanRecSig, "recSig-mean")

        if(oscN==nInput):
            # Checking error of final guess
            recNorm = normF(rec)
            normValS = normF(sInput)/recNorm
            recNormed = rec*normValS[:,None]

            # Checking error of initial guess
            err = mse(rec, sInput[:,:-1])
            normErr = mse(recNormed, sInput[:,:-1])
            print "\n" + "MEAN error: {} / (normed {} )".format(err, normErr) + "\n"

            print 'normValS: ', normValS
            optAmp = meanParam[:,-1]
            meanParam[:,-1] *= normValS

        a,b,rec = recKurSL(t, meanParam)
        meanRec = rec

        #####################################
        # Plot MEAN results from MCMC
        errArr = [0]*oscN

        py.figure()
        for i in range(oscN+1):

            py.subplot(oscN+1, 1, i+1)

            if(i==0):
                py.plot(t, sInputFlat, 'g')
                py.plot(t[:-1], meanRecSig, 'r')
                py.ylabel('rec')
            else:
                py.plot(t[:-1], rec[i-1], 'r')
                if COMPARE_TRUTH and (i-1)<genParams.shape[0]:
                    py.plot(t[:-1], trueRec[i-1], 'g')
                py.ylabel(str(i))

            py.xlim((min(t), max(t)))
            py.locator_params(axis='y', tight=None, nbins=3)

            if i != oscN-1: py.gca().axes.get_xaxis().set_ticks([])
            else: py.locator_params(axis='x',nbins=5)


        py.subplots_adjust(bottom=0.05,hspace=0.4)

        totErr = mse(sInputFlat[:-1], meanRecSig)
        py.suptitle("Total MSE - %.3e" %totErr)
        py.savefig("fitMean", dpi=120)
        #~ py.show()

        print "[Finished] Estimating mean values"
        print

    #############################################
    if PLOT_REC:
        print "Plotting reconstuction... ",
        sIn = np.sum(sInput, axis=0)[:-1]
        T = t[:-1]

        absMax = lambda x: np.max(np.abs(x))
        
        yLim = absMax(sIn)
        if MEDIAN: yLim = max( yLim, absMax(np.sum(medRec, axis=0)))
        if MAP:    yLim = max( yLim, absMax(np.sum(mapRec, axis=0)))
        if MEAN:   yLim = max( yLim, absMax(np.sum(meanRec, axis=0)))
        if BEST:   yLim = max( yLim, absMax(np.sum(bestRec, axis=0)))
        
        def plotOver(sIn, sRec, typeName):
            py.plot(T, sIn, 'g')
            py.plot(T, sRec, 'r')
            py.xlim((min(t), max(t)))
            py.ylim((-yLim, yLim))

            py.ylabel("ER %.3f" %(mse(sIn, sRec)/mse(sIn,0)))

            py.title("Input and reconstruction (%s)"%typeName)
            py.locator_params(nbins=5)
        
        def plotDiff(sIn, sRec, typeName):            
            py.plot(T, sIn-sRec)
            py.xlim((min(t), max(t)))
            py.ylim((-yLim, yLim))
            
            py.ylabel("MSE %.3e" %mse(sIn, sRec))
            py.title("Pointwise difference (%s)" %typeName)
            py.locator_params(nbins=5)
            
        nrows = MEDIAN + MAP + MEAN + BEST
        n = 0

        py.clf()

        if MEDIAN:
            sRec = np.sum(medRec, axis=0)

            # Plotting MEDIAN and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(sIn, sRec, "MED")

            # Plotting difference between MEDIAN and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(sIn, sRec, "MED")

            # Increase plotting index
            n += 2

        if MEAN:
            sRec = np.sum(meanRec, axis=0)

            # Plotting MEAN and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(sIn, sRec, "MEAN")
            
            # Plotting difference between MEAN and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(sIn, sRec, "MEAN")

            # Increase plotting index
            n += 2

        if MAP:
            sRec = np.sum(mapRec, axis=0)

            # Plotting MAP and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(sIn, sRec, "MAP")

            # Plotting difference between MAP and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(sIn, sRec, "MAP")

            # Increase plotting index
            n += 2

        if BEST:
            sRec = np.sum(bestRec, axis=0)

            # Plotting MAP and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver( sIn, sRec, "BEST")

            # Plotting difference between MAP and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff( sIn, sRec, "BEST")

            # Increase plotting index
            n += 2

        py.tight_layout()
        py.savefig('recCompare')
        print "Done."

    #############################################
    if PLOT_W_HIST:
        print "Plotting W histogram... ",
        labW = ["W%i"%(i+1) for i in range(oscN)]
        labels = labW

        N = oscN
        rows = int(np.floor(np.sqrt(N)))
        cols = int(np.ceil(float(N)/rows))

        offset = 0

        fig = py.figure(figsize=(2*cols,2*rows))
        for p in xrange(N):
            ax = fig.add_subplot(rows, cols, p+1)
            aSamples = samples[:,:, offset+p*paramN].flatten()
            if LIM_VIEW:
                aSamples = aSamples[np.r_[aSamples>xLimW[p][0]] & np.r_[aSamples<xLimW[p][1]]]

            py.hist(aSamples, binSize, normed=True, color='c')

            if MEDIAN:
                py.axvline(medParam[p,offset], color=CMED)

            if MAP:
                # Calculating MAP
                _x = xKde[offset+p*paramN]
                _y = yKde[offset+p*paramN]
                py.plot(_x, _y,'r-')
                py.axvline(mapParam[p, offset], color=CMAP)

            if MEAN:
                py.axvline(meanParam[p, offset], color=CMEAN)

            if BEST:
                py.axvline(bestParam[p, offset], color=CBEST)

            if TRUE:
                py.axvline(genParams[p, offset], color=CTRUE)

            py.title(labels[p])
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            py.locator_params(nbins=5)

            # Custom limits
            if LIM_VIEW: py.xlim(xLimW[p])

        py.tight_layout()
        py.savefig('hist_w',dpi=200)
        print "Done."

    #############################################
    if PLOT_Ph_HIST:
        print "Plotting phase histograms... ",
        labPh = ["Ph%i"%(i+1) for i in range(oscN)]
        labels = labPh

        N = oscN
        rows = int(np.floor(np.sqrt(N)))
        cols = int(np.ceil(float(N)/rows))

        offset = 1

        fig = py.figure(figsize=(2*cols,2*rows))
        for p in xrange(N):
            ax = fig.add_subplot(rows, cols, p+1)
            aSamples = samples[:,:, offset+p*paramN].flatten()
            aSamples = aSamples % (2*np.pi)

            py.hist(aSamples, binSize, normed=True, color='c')

            if MEDIAN:
                py.axvline(medParam[p, offset] % (2*np.pi), color=CMED)

            if MAP:
                # Calculating MAP
                _x = xKde[offset+p*paramN]
                _y = yKde[offset+p*paramN]
                py.plot(_x, _y,'r-')
                py.axvline(mapParam[p, offset] % (2*np.pi), color=CMAP)

            if MEAN:
                py.axvline(meanParam[p, offset] % (2*np.pi), color=CMEAN)

            if BEST:
                py.axvline(bestParam[p, offset] % (2*np.pi), color=CBEST)

            if TRUE:
                py.axvline(genParams[p, offset] % (2*np.pi), color=CTRUE)
                
            py.title(labels[p])
            py.xticks([0,np.pi,2*np.pi], ['0', '$\pi$', '$2\pi$'])
            py.xlim((0,2*np.pi))

        py.tight_layout()
        py.savefig('hist_Ph',dpi=200)

        print "Done."

    #############################################
    if PLOT_A_HIST:
        print "Plotting amplitude histograms... ",
        labA = ["A%i"%(i+1) for i in range(oscN)]
        labels = labA

        N = oscN
        rows = int(np.floor(np.sqrt(N)))
        cols = int(np.ceil(float(N)/rows))

        offset = 2

        fig = py.figure(figsize=(2*cols,2*rows))
        for p in xrange(N):
            ax = fig.add_subplot(rows, cols, p+1)
            aSamples = samples[:,:, offset+p*paramN].flatten()
            if LIM_VIEW:
                aSamples = aSamples[np.r_[aSamples>xLimA[p][0]] & np.r_[aSamples<xLimA[p][1]]]

            py.hist(aSamples, binSize, normed=True, color='c')
            
            if MEDIAN:
                py.axvline(medParam[p, offset], color=CMAP)

            if MAP:
                # Calculating MAP
                _x = xKde[offset+p*paramN]
                _y = yKde[offset+p*paramN]
                py.plot(_x, _y,'r-')
                py.axvline(mapParam[p, offset], color=CMAP)

            if MEAN:
                py.axvline(meanParam[p, offset], color=CMEAN)

            if BEST:
                py.axvline(bestParam[p, offset], color=CBEST)

            if TRUE:
                py.axvline(genParams[p, offset], color=CTRUE)
                
            py.title(labels[p])
            ax.tick_params(axis='x', labelsize=7)
            ax.tick_params(axis='y', labelsize=7)
            py.locator_params(nbins=5)
            
            # Custom limits
            if LIM_VIEW: py.xlim(xLimA[p])

        py.tight_layout()
        py.savefig('hist_A',dpi=200)

        print "Done."

    #############################################
    if PLOT_K_HIST and oscN>1:
        print "Plotting K histograms... ",

        N = oscN
        rows = N
        cols = N-1

        # For each harmonic
        for _nH in xrange(nH):
            
            labK = [["K(%i,%i)"%(i+1,j+1) for i in range(oscN) if i!=j] for j in range(oscN)]
            labels = labK

            fig = py.figure(figsize=(2*cols,2*rows))
            #~ begN, endN = 3+_nH*(oscN-1), 3+(_nH+1)*(oscN-1)
            begN, endN = 0, oscN-1
            
            plotN = 0
            for kIdx in range(begN, endN):
                for p in xrange(N):
                    offset = 3 + kIdx*nH + _nH
                    print 'offset+p*paramN: ', offset+p*paramN
                    
                    ax = fig.add_subplot(rows, cols, p*(N-1)+kIdx+1)
                    if LIM_VIEW:
                        aSamples = aSamples[np.r_[aSamples>xLimK[p][kIdx-begN][0]] & np.r_[aSamples<xLimK[p][kIdx-begN][1]]]
                    aSamples = samples[:,:, offset+p*paramN].flatten()

                    py.hist(aSamples, binSize, normed=True, color='c')
                    
                    if MEDIAN:
                        py.axvline(medParam[p,offset], color=CMED)
    
                    if MAP:
                        # Calculating MAP
                        _x = xKde[offset+p*paramN]
                        _y = yKde[offset+p*paramN]
                        py.plot(_x, _y,'r-')
                        py.axvline(mapParam[p, offset], color=CMAP)
    
                    if MEAN:
                        py.axvline(meanParam[p, offset], color=CMEAN)
    
                    if BEST:
                        py.axvline(bestParam[p, offset], color=CBEST)
    
                    if TRUE:
                        py.axvline(genParams[p, offset], color=CTRUE)
    
                    py.title(labels[p][kIdx-begN])
                    ax.tick_params(axis='x', labelsize=7)
                    ax.tick_params(axis='y', labelsize=7)
                    py.locator_params(nbins=5)
                    
                    if LIM_VIEW: py.xlim(xLimK[p][kIdx])
    
            py.tight_layout()
            py.savefig('hist_K_h'+str(_nH),dpi=200)
            print "done for " + str(_nH+1) + "  ",
        print "All K done."


    if PLOT_CORNER_FREQ:
        # Plotting pairwise distributions in corner shape
        # Due to huge computational task do it only if oscN<=4
        print "Plotting Corner frequency graphs"

        import corner
        labels = ["W%i"%(i+1) for i in range(oscN)]

        ndim = oscN*paramN
        _samples = samples.copy()[:,:,:]
        _samples = _samples.reshape((-1,ndim))
        _samples = _samples[:, np.arange(oscN)*paramN]

        ndim = oscN
        FIG, axes = py.subplots(ndim, ndim, figsize=(ndim, ndim))
        fig = corner.corner(_samples,
                            labels=labels,
                            truths=genParams[:,0].flatten(),
                            #~ fig=FIG
                            )
        fig.savefig("corner_freq.png",dpi=100)

    if PLOT_CORNER_Ph:
        # Plotting pairwise distributions in corner shape
        # Due to huge computational task do it only if oscN<=4
        print "Plotting Corner phase graphs"

        import corner
        labels = ["Ph%i"%(i+1) for i in range(oscN)]

        ndim = oscN*paramN
        _samples = samples.copy()[:,:,:]
        _samples = _samples.reshape((-1,ndim))
        _samples = _samples[:, np.arange(oscN)*paramN+1]

        ndim = oscN
        FIG, axes = py.subplots(ndim, ndim, figsize=(ndim, ndim))
        fig = corner.corner(_samples,
                            labels=labels,
                            truths=genParams[:,1].flatten(),
                            #~ fig=FIG
                            )
        fig.savefig("corner_Ph.png",dpi=100)

    if PLOT_CORNER_A:
        # Plotting pairwise distributions in corner shape
        # Due to huge computational task do it only if oscN<=4
        print "Plotting Corner Amplitude graphs"

        import corner
        labels = ["A%i"%(i+1) for i in range(oscN)]

        ndim = oscN*paramN
        _samples = samples.copy()[:,:,:]
        _samples = _samples.reshape((-1,ndim))
        _samples = _samples[:, np.arange(oscN)*paramN+2]

        ndim = oscN
        FIG, axes = py.subplots(ndim, ndim, figsize=(ndim, ndim))
        fig = corner.corner(_samples,
                            labels=labels,
                            truths=genParams[:,2].flatten(),
                            #~ fig=FIG
                            )
        fig.savefig("corner_A.png",dpi=100)

    #~ if PLOT_CORNER_K:
        #~ # Plotting pairwise distributions in corner shape
        #~ # Due to huge computational task do it only if oscN<=4
        #~ print "Plotting Corner Amplitude graphs"
#~ 
        #~ import corner
        #~ labK = [["K(%i,%i)"%(i+1,j+1) for i in range(oscN) if i!=j] for j in range(oscN)]
        #~ labels = [item for sublist in labK for item in sublist]
#~ 
        #~ ndim = oscN*paramN
        #~ _samples = samples.copy()[:,:,:]
        #~ _samples = _samples.reshape((-1,ndim))
        #~ _samples = _samples[:, np.arange(oscN)*paramN+2]
#~ 
        #~ ndim = oscN
        #~ FIG, axes = py.subplots(ndim, ndim, figsize=(ndim, ndim))
        #~ fig = corner.corner(_samples
                            #~ labels=labels,
                            #~ #truths=medParam.flatten(),
                            #~ #fig=FIG
                            #~ )
        #~ fig.savefig("corner_A.png",dpi=100)

    ##################################################
    ## Finished
    print "Analysis finished."
