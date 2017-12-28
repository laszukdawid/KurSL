# Coding: UTF-8
import logging

class ModelWrapper(object):
    """Wrapper for models used in MCMC.

    Full isolation allows to pass models between process
    which allows for multiprocessing.
    """

    logger = logging.getLogger(__name__)

    def __init__(self, model=None):

        if model is not None:
            self.set_model(model)

        self.MIN_R = 0
        self.MAX_R = 1e10

        self.MIN_W = 0
        self.MAX_W = 1e5

        self.THRESHOLD = 0.1
        self.THRESHOLD_OBTAINED = False

    def set_model(self, model):
        self.model = model
        self.oscN = model.oscN
        self.nH = model.nH

    def __call__(self, t, P):
        return self.model(t, P)

    def generate(self, t):
        return self.model.generate(t)

    def set_params(self, P):
        return self.model.set_params(P)

    def dist_W(self, x):
        return 0

    def dist_K(self, x):
        return 0

    def dist_ph(self, x):
        return 0

    def dist_R(self, x):
        return 0

#    def set_prior_dist(self, samples):
#        # TODO: Add prior_dist check to dists
#        self.prior_dist = True
#        self.logger.debug("Setting prior distrubition... ")
#        wNum, sNum, pNum = samples.shape
#
#        kdeNum = 50
#        xKde = np.zeros((pNum, kdeNum))
#        yKde = np.zeros((pNum, kdeNum))
#
#        self.logger.debug("Calulating KDE based on provided samples... ")
#        for i in range(pNum):
#            y = samples[:,:,i].flatten()
#            xKde[i] = np.linspace(np.min(y), np.max(y), kdeNum)
#            yKde[i] = gaussian_kde(y, bw_method="silverman")(xKde[i])
#
#        idx = np.arange(self.oscN)*((self.oscN-1)*self.nH+3)
#
#        # Setting W
#        offsetW = 0
#        self.xW = xKde[idx+offsetW]
#        self.histW = yKde[idx+offsetW]
#
#        # Setting Ph
#        offsetPh = 1
#        self.xPh = xKde[idx+offsetPh]
#        self.histPh = yKde[idx+offsetPh]
#
#        # Setting R
#        offsetR = 2
#        self.xR = xKde[idx+offsetR]
#        self.histR = yKde[idx+offsetR]
#
#        # Setting K for all nH
#        offset = np.arange(3,3+self.nH*(self.oscN-1))
#        self.xK = np.zeros((self.oscN, self.nH*(self.oscN-1), kdeNum))
#        self.histK = np.zeros((self.oscN, self.nH*(self.oscN-1), kdeNum))
#        for i in range(self.oscN):
#            self.xK[i] = xKde[idx[i]+offset]
#            self.histK[i] = yKde[idx[i]+offset]
#
#        self.logger.debug("Setting new distrubitions")
#        self.model.dist_W = self.kde_dist_W
#        self.model.dist_K = self.kde_dist_K
#        self.model.dist_ph = self.kde_dist_ph
#        self.model.dist_R = self.kde_dist_R
#
#    def kde_dist_W(self, W):
#        idx = np.argmin(np.abs(self.xW-W[:,None]), axis=0)
#        return self.negLog([self.histW[i][idx[i]] for i in range(self.oscN)])
#
#    def kde_dist_Ph(self, Ph):
#        idx = np.argmin(np.abs(self.xPh-Ph[:,None]), axis=0)
#        return self.negLog([self.histPh[i][idx[i]] for i in range(self.oscN)])
#
#    def kde_dist_R(self, R):
#        idx = np.argmin(np.abs(self.xR-R[:,None]), axis=0)
#        return self.negLog([self.histR[i][idx[i]] for i in range(self.oscN)])
#
#    def kde_dist_K(self, K):
#        dist_K = np.zeros(K.shape)
#        idx = np.argmin(np.abs(self.xK-K[:,:,None]), axis=-1)
#        for i in range(self.nH*(self.oscN-1)):
#            for j in range(self.oscN):
#                dist_K[j,i] = self.histK[j,i,idx[j,i]]
#        return self.negLog(dist_K)
