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
    FIG_SIG_INPUT = 0
    FIG_SIG_REC   = 1
    FIG_REC_SEP   = 1
    
    PLOT_W_HIST  = 1
    PLOT_Ph_HIST = 1
    PLOT_A_HIST  = 1
    PLOT_K_HIST  = 1

    PLOT_CORNER_FREQ = 1
    PLOT_CORNER_Ph = 1
    PLOT_CORNER_A = 1
    PLOT_CORNER_K = 1
    PLOT_CORNER_ALL = 1
    PLOT_REC = 1
    
    PLOT_INPUT = 1
    PLOT_SPEC = 1

    REAL_DATA = 0
    ADD_NOISE = 0

    MEDIAN = 0
    MAP    = 0
    MEAN   = 0
    BEST   = 1

    COMPARE_TRUTH = 1

    _discardSamples = 20

    fMin, fMax = 0, 50
    binSize = 50
    #############################################

    # Files are saved by method and by number of seg.
    fs = 173.61 # Hz
    tMin = 0 # Initial time
    tSeg = 3 # Length of segment 
    tStep = 1 # Length of step
    segNum = [0, 4] # Range: from 1

    fMin, fMax = 0, 40

    tMax = tMin+tSeg+(segNum[1]-segNum[0])*tStep
    N = int((tMax-tMin)*fs)
    nBeg = int(tMin*fs)
    nSeg = int(tSeg*fs)
    nStep = int(tStep*fs)
 
    #############################################

    # Normalising plots
    normF = lambda s: np.sum(np.abs(s)**2, axis=1)
    mse = lambda x,y: np.mean((x-y)*(x-y))
    
    #############################################

    # Files are saved by method and by number of seg.
    # If not dynamic then nSeg = 0. 
    signalType = ['epilepsy', 'random'][0]
    initTArr = range(segNum[0], segNum[1])

    if BEST:    bestParamArr, bestRecArr = [], []
    if MAP:     mapParamArr,  mapRecArr  = [], []
    if MEAN:    meanParamArr, meanRecArr = [], []
    if MEDIAN:  medParamArr,  medRecArr  = [], []
    
    tArr = []
    sIn = []
    
    # Getting true values
#     fileName = 'KurSL_results-%s-%i.npz' %(signalType, segNum[0])
#     genParams = np.load(fileName)['genParams']
    
    # TO DO: This actually doesn't work as it should.
    # Instead of plotting whole signal it plots only the first segment.
    # Plotting Input signal
    if FIG_SIG_INPUT:
        fileName = 'KurSL_results-%s-%i.npz' %(signalType, segNum[0]) 
        data = np.load(fileName)
        S = data['sInput']
        T = data['x']
        T = np.arange(tMin, tMax, (tMax-tMin)*(T[1]-T[0]))

        samples = data['samples']
        nH = data['nH']
        L = len(samples[0,0,:])
        oscN = int((np.sqrt((3-nH)*(3-nH)+4*nH*L)-(3-nH))/(2*nH))
        paramN = int(3+nH*(oscN-1))
        
        a,b,rec = recKurSL(T, genParams)
        plotRec(T, np.sum(rec, axis=0), 'sInput')

    ###############################################
    ## For each segment in dynamical analysis
    for seg in xrange(segNum[0], segNum[1]):

        # Loading results
        fileName = 'KurSL_results-%s-%i.npz' %(signalType, seg)
        print 'fileName: ', fileName
        try:
            data = np.load(fileName)
        except IOError:
            print 'No such file: ', fileName
            print 'Breaking...'
            break
            
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
    
        #~ sInput = data['sInput']
        s = data['sInput']
        t = data['x'][:-1]
        nH = data['nH']
    
        ##############################################

        # Calculate initial indices         
        idxStart = nBeg + seg*nStep
        idxEnd = idxStart + nSeg

        #~ t = T[idxStart:idxEnd]
        #~ s = sInput[idxStart:idxEnd]
        
        print 't.size: ', t.size
        print 's.size: ', s.size
        assert (t.size == s.size), "Signal must have the same length as time"

        sIn.append(s.copy())
        tArr.append(t.copy())
        initTArr[seg] = t[0]
        
        L = len(samples[0,0,:])
        oscN = int((np.sqrt((3-nH)*(3-nH)+4*nH*L)-(3-nH))/(2*nH))
        paramN = int(3+nH*(oscN-1))
        
        # Discarding some values
        discIdx = min(int(0.2*samples.shape[0]), _discardSamples)
        samples = samples[discIdx:,:,:]
        lnprob = lnprob[discIdx:,:]

        #############################################
        if MEDIAN:
            print "[Start] Estimating median values"
            # Estimate as median
            medianSamples = [np.median(samples[:,:, param]) for param in xrange(samples[0,0].size)]
            medParam = np.array(medianSamples)
            medParam = medParam.reshape((oscN, paramN))
            medParamArr.append(medParam)
     
            print "medParam: "
            print medParam
            ph, A, medRec = recKurSL(t, medParam)
     
            medRecSig = np.sum(medRec, axis=0)
            medRecArr.append(medRecSig)
     
        #############################################
        if MEAN:
            print "[Start] Estimating mean values"
     
            meanParam = np.mean(np.mean(samples, axis=0), axis=0)
            meanParam = meanParam.reshape((oscN, paramN))
            meanParamArr.append(meanParam)
     
            print "Mean Param: "
            print meanParam
            Ph, A, meanRec = recKurSL(t, meanParam)
     
            meanRecSig = np.sum(meanRec, axis=0)
            meanRecArr.append(meanRecSig)

        #############################################
        if BEST:
            print "[Start] Estimation on all values"
            # Estimate as median
    
            sDim = samples.shape[1]
            bestIdx = np.argmax(lnprob)
            print "bestIdx: ", bestIdx
            bestParam = samples[ int(bestIdx/sDim), bestIdx%sDim, :]
            bestParam = bestParam.reshape((oscN, paramN))
            bestParamArr.append(bestParam)
    
            print "bestParam: "
            print bestParam
            Ph, A, bestRec = recKurSL(t, bestParam)
    
            bestRecSig = np.sum(bestRec, axis=0)
            bestRecArr.append(bestRecSig)

        #############################################
        if MAP:
            print "[Start] Estimating MAP values"
     
            print ' Estimating prob dist'
            xKde, yKde = kdeFunc(samples)
            mapParam = mapFunc(xKde, yKde).reshape((oscN, paramN))
            mapParamArr.append(mapParam)
     
            print "Map Param: "
            print mapParam
            Ph, A, mapRec = recKurSL(t, mapParam)
     
            mapRecSig = np.sum(mapRec, axis=0)
            mapRecArr.append(mapRecSig)

    # After loading all data, convert arrays into Numpy
    tArr = np.array(tArr)[:,:-1]
    sIn = np.array(sIn)[:,:-1]
    if MEDIAN:
        medParamArr = np.array(medParamArr)
        medRecArr = np.array(medRecArr)
    if MEAN:
        meanParamArr = np.array(meanParamArr)
        meanRecArr = np.array(meanRecArr)
    if MAP:
        mapParamArr = np.array(mapParamArr)
        mapRecArr = np.array(mapRecArr)
    if BEST:
        bestParamArr = np.array(bestParamArr)
        bestRecArr = np.array(bestRecArr)
    #############################################
    if PLOT_REC:
        print "Plotting reconstuction... ",
        absMax = lambda x: np.max(np.abs(x))
        
        yLim = absMax(sIn)
        if MEDIAN: yLim = max( yLim, absMax(np.sum(medRec, axis=0)))
        if MEAN:   yLim = max( yLim, absMax(np.sum(meanRec, axis=0)))
        if MAP:    yLim = max( yLim, absMax(np.sum(mapRec, axis=0)))
        if BEST:   yLim = max( yLim, absMax(np.sum(bestRec, axis=0)))
        
        def plotOver(tArr, sIn, sRec, typeName):
            totER = 0
            for i in xrange(tArr.shape[0]):
                totER += mse(sIn[i], sRec[i])/mse(sIn[i],0)
                py.plot(tArr[i], sIn[i], 'g')
                py.plot(tArr[i], sRec[i], 'r')
            py.xlim((np.min(tArr), np.max(tArr)))
            py.ylim((-yLim, yLim))

            avgER = totER/tArr.shape[0]
            py.ylabel("avgER %.3f" %avgER)

            py.title("Input and reconstruction (%s)"%typeName)
            py.locator_params(nbins=5)
        
        def plotDiff(tArr, sIn, sRec, typeName):       
            totMSE = 0     
            for i in xrange(tArr.shape[0]):
                totMSE += mse(sIn[i], sRec[i])
                py.plot(tArr[i], sIn[i]-sRec[i])
            py.xlim((np.min(tArr), np.max(tArr)))
            py.ylim((-yLim, yLim))
            
            avgMSE = totMSE/tArr.shape[0]
            py.ylabel("avgMSE %.3e" %avgMSE)
            py.title("Pointwise difference (%s)" %typeName)
            py.locator_params(nbins=5)
            
        nrows = MEDIAN + MAP + MEAN + BEST
        n = 0

        py.clf()

        if MEAN:
            # Plotting MAP and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(tArr, sIn, meanRecArr, "MEAN")

            # Plotting difference between MAP and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(tArr, sIn, meanRecArr, "MEAN")

            # Increase plotting index
            n += 2

        if MEDIAN:
            # Plotting MAP and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(tArr, sIn, medRecArr, "MED")

            # Plotting difference between MAP and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(tArr, sIn, medRecArr, "MED")

            # Increase plotting index
            n += 2

        if MAP:
            # Plotting MAP and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(tArr, sIn, mapRecArr, "MAP")

            # Plotting difference between MAP and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(tArr, sIn, mapRecArr, "MAP")

            # Increase plotting index
            n += 2

        if BEST:

            # Plotting MAP and INPUT
            py.subplot(nrows, 2, n+1)
            plotOver(tArr, sIn, bestRecArr, "BEST")

            # Plotting difference between MAP and INPUT
            py.subplot(nrows, 2, n+2)
            plotDiff(tArr, sIn, bestRecArr, "BEST")

            # Increase plotting index
            n += 2

        py.tight_layout()
        py.savefig('recCompare')
        print "Done."

    if FIG_REC_SEP:
        print "Plotting BEST reconstuction comparison... ",
        absMax = lambda x: np.max(np.abs(x))
        
        yLim = absMax(sIn)
        if MEDIAN: yLim = max( yLim, absMax(np.sum(medRec, axis=0)))
        if MEAN:   yLim = max( yLim, absMax(np.sum(meanRec, axis=0)))
        if MAP:    yLim = max( yLim, absMax(np.sum(mapRec, axis=0)))
        if BEST:   yLim = max( yLim, absMax(np.sum(bestRec, axis=0)))
        
        fMin, fMax = 0, 60
        
        def plotOverSingle(t, sIn, sRec, typeName):
            sDetrend = sIn - sIn.mean()
            ER = mse(sIn, sRec)/mse(sIn,0)
            py.plot(t, sIn, 'g')
            py.plot(t, sRec, 'r')
            py.xlim((np.min(tArr), np.max(tArr)))
            py.ylim((-yLim, yLim))

            py.ylabel("ER %.3f" %ER)

            py.title("Input and reconstruction (%s)"%typeName)
            py.locator_params(nbins=5)
        
        def plotFreqSingle(t, sInSingle, sRecSingle, typeName):
            dt = t[1]-t[0]
            freq = np.fft.fftfreq(sInSingle.size, dt)
            
            idx = np.r_[freq>=fMin] & np.r_[freq<fMax]
            freq = freq[idx]
            FIn = np.abs(np.fft.fft(sInSingle)[idx])
            FRec = np.abs(np.fft.fft(sRecSingle)[idx])
            
            ER = mse(sInSingle, sRecSingle)/mse(sInSingle,0)
            py.plot(freq, FIn, 'g')
            py.plot(freq, FRec, 'r')
            py.xlim((fMin, fMax))
            
            py.ylabel("ER %.3f" %ER)
            py.title("Spectra (%s)" %typeName)
            py.locator_params(nbins=5)
            
        nrows = tArr.shape[0]
        n = 0

        if MEDIAN:
            py.clf()
            py.figure()

            for n in range(nrows):
                # Plotting MAP and INPUT
                py.subplot(nrows, 2, 2*n+1)
                plotOverSingle(tArr[n], sIn[n], medRecArr[n], "MED")

                # Plotting difference between MAP and INPUT
                py.subplot(nrows, 2, 2*n+2)
                plotFreqSingle(tArr[n], sIn[n], medRecArr[n], "MED")

                py.tight_layout()
                py.savefig('recMedCompare')
            print "Median finished."            

        if MEAN:
            py.clf()
            py.figure()

            for n in range(nrows):
                # Plotting MAP and INPUT
                py.subplot(nrows, 2, 2*n+1)
                plotOverSingle(tArr[n], sIn[n], meanRecArr[n], "MED")

                # Plotting difference between MAP and INPUT
                py.subplot(nrows, 2, 2*n+2)
                plotFreqSingle(tArr[n], sIn[n], meanRecArr[n], "MED")

                py.tight_layout()
                py.savefig('recMeanCompare')
            print "Mean finished."            

        if MAP:
            py.clf()
            py.figure()

            for n in range(nrows):
                # Plotting MAP and INPUT
                py.subplot(nrows, 2, 2*n+1)
                plotOverSingle(tArr[n], sIn[n], mapRecArr[n], "MED")

                # Plotting difference between MAP and INPUT
                py.subplot(nrows, 2, 2*n+2)
                plotFreqSingle(tArr[n], sIn[n], mapRecArr[n], "MED")


                py.tight_layout()
                py.savefig('recMapCompare')
            print "MAP finished."            

        if BEST:
            py.clf()
            py.figure()

            for n in range(nrows):
                # Plotting MAP and INPUT
                py.subplot(nrows, 2, 2*n+1)
                plotOverSingle(tArr[n], sIn[n], bestRecArr[n], "BEST")

                # Plotting difference between MAP and INPUT
                py.subplot(nrows, 2, 2*n+2)
                plotFreqSingle(tArr[n], sIn[n], bestRecArr[n], "BEST")


                py.tight_layout()
                py.savefig('recBestCompare')
            print "Best finished."

        
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
        for p in xrange(oscN):
            ax = fig.add_subplot(rows, cols, p+1)

            if MEDIAN: py.plot(initTArr, medParamArr[:,p, offset], CMED)
            if MEAN:   py.plot(initTArr, meanParamArr[:,p, offset], CMEAN)
            if MAP:    py.plot(initTArr, mapParamArr[:,p, offset], CMAP)
            if BEST:   py.plot(initTArr, bestParamArr[:,p, offset], CBEST)
            
            py.title(labels[p])
            ax.tick_params(axis='x', labelsize=7)
            py.locator_params(nbins=5)

        py.tight_layout()
        py.savefig('dyn_w',dpi=200)
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
        for p in xrange(oscN):
            ax = fig.add_subplot(rows, cols, p+1)

            if MEDIAN: py.plot(initTArr, medParamArr[:,p, offset], CMED)
            if MEAN:   py.plot(initTArr, meanParamArr[:,p, offset], CMEAN)
            if MAP:    py.plot(initTArr, mapParamArr[:,p, offset], CMAP)
            if BEST:   py.plot(initTArr, bestParamArr[:,p, offset], CBEST)
            
            py.title(labels[p])
            ax.tick_params(axis='x', labelsize=7)
            py.locator_params(nbins=5)

        py.tight_layout()
        py.savefig('dyn_Ph',dpi=200)

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
        for p in xrange(oscN):
            ax = fig.add_subplot(rows, cols, p+1)

            if MEDIAN: py.plot(initTArr, medParamArr[:,p, offset], CMED)
            if MEAN:   py.plot(initTArr, meanParamArr[:,p, offset], CMEAN)
            if MAP:    py.plot(initTArr, mapParamArr[:,p, offset], CMAP)
            if BEST:   py.plot(initTArr, bestParamArr[:,p, offset], CBEST)

            py.title(labels[p])
            ax.tick_params(axis='x', labelsize=7)
            py.locator_params(nbins=5)

        py.tight_layout()
        py.savefig('dyn_A',dpi=200)

        print "Done."

    #############################################
    if PLOT_K_HIST and oscN>1:
        print "Plotting K histograms... ",

        N = oscN
        rows = N
        cols = N-1
        
        for _nH in xrange(nH):
            print str(_nH) + ", ",
            labK = [["K(%i,%i)"%(i+1,j+1) for i in range(oscN) if i!=j] for j in range(oscN)]
            labels = labK

            N = oscN
            plotN = N*(N-1)
            rows = N
            cols = N-1

            fig = py.figure(figsize=(2*cols,2*rows))
            begN, endN = 0, oscN-1
            
            plotN = 0
            for kIdx in range(begN, endN):
                for p in xrange(N):
                    offset = 3 + kIdx*nH + _nH
#                     print 'offset+p*paramN: ', offset+p*paramN
                    
                    ax = fig.add_subplot(rows, cols, p*(N-1)+kIdx+1)
                    if MEDIAN: py.plot(initTArr, medParamArr[:,p, offset], CMED)
                    if MEAN:   py.plot(initTArr, meanParamArr[:,p, offset], CMEAN)
                    if MAP:    py.plot(initTArr, mapParamArr[:,p, offset], CMAP)
                    if BEST:   py.plot(initTArr, bestParamArr[:,p, offset], CBEST)
                    
                    py.title(labels[p][kIdx-begN])
                    ax.tick_params(axis='x', labelsize=7)
                    py.locator_params(nbins=5)

            py.tight_layout()
            py.savefig('dyn_K_'+str(_nH),dpi=200)
        print "Done."
        
    #############################################
    
    # Plotting pairwise distributions in corner shape
    # Due to huge computational task do it only if oscN<=4

    if PLOT_CORNER_FREQ:
        import corner

        labels = ["W%i"%(i+1) for i in range(oscN)]

        ndim = oscN*(3+nH*(oscN-1))
        _samples = samples.copy()[:,:,:]
        _samples = _samples.reshape((-1,ndim))
        _samples = _samples[:, np.arange(oscN)*(3+nH*(oscN-1))]

        ndim = oscN
        FIG, axes = py.subplots(ndim, ndim, figsize=(ndim, ndim))
        fig = corner.corner(_samples
                            #~ labels=labels,
                            #~ truths=medParam.flatten(),
                            #~ fig=FIG
                            )
        fig.savefig("corner_freq.png",dpi=100)

    
    ##################################################
    ## Finished
    print "Analysis finished."
