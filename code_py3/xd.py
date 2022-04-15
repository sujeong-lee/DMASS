#!/usr/bin/env python
import sys
from time import time
import numpy as np
from astroML.decorators import pickle_results
#from astroML.density_estimation import XDGMM
#from cmass_modules import io, DES_to_SDSS, im3shape, Cuts
#import matplotlib
#matplotlib.use('Agg')
#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from numpy import linalg
#from __future__ import print_function, division
#from ..utils import logsumexp, log_multivariate_gaussian, check_random_state
from multiprocessing import Process, Queue
#from sklearn.mixture import GMM as GaussianMixture
from sklearn.mixture import GaussianMixture
from xdgmm import XDGMM as XDGMM_Holoien

def getCMASSparam(filename = None ):
    
    file = open(filename, 'rb')  # cmass
    
    import pickle
    clf = pickle.load(file)
    mean = clf['retval'].mu
    weight = clf['retval'].alpha
    covariance = clf['retval'].V
    
    return mean, weight, covariance


def loadpickle(filename):
    import pickle
    f = open(filename, 'rb')
    pickle = pickle.load(f)
    clf = pickle['retval']
    xamp = clf.alpha
    xmean = clf.mu
    xcovar = clf.V
    ncomp = clf.n_components
    return ncomp, xamp, xmean, xcovar


def extreme_fitting( cat, n_comp = 20, xmean=None, xamp=None, xcovar=None, pickle_name = '../output/test/pickle.pkl', log='../xd_log/log', weight=None ):
    from extreme_deconvolution import extreme_deconvolution
    import pickle
    
    ydata, ycovar = mixing_color(cat)
    if weight is True : weight = cat['CMASS_WEIGHT']

    if xmean is None : 
        #from sklearn.mixture import GaussianMixture
        print('initial guess')
        gmm = GaussianMixture(n_comp, max_iter= 10, covariance_type='full',
                          random_state=1).fit(ydata)
        xmean = gmm.means_
        xamp = gmm.weights_
        xcovar = gmm.covariance_

        #return xmean, xamp, xcovar

    print('fitting started')
    
    l = extreme_deconvolution(ydata, ycovar, xamp, xmean, xcovar, projection=None, weight=weight, \
                          fixamp=None, fixmean=None, fixcovar=None, tol=1e-6, maxiter=1000000, \
                          w=0.0, logfile=log, splitnmerge=0, maxsnm=False, likeonly=False, logweight=False)
 
    
    data = XDGMM(n_comp, tol= 1e-6)
    data.V = xcovar
    data.mu = xmean
    data.alpha = xamp
    
    clf = {'retval' : data }
    output = open( pickle_name, 'wb')
    pickle.dump( clf, output )
    print('pickle saved ', pickle_name)
    
    #return clf


class XDGMM(object):
    """Extreme Deconvolution
    Fit an extreme deconvolution (XD) model to the data
    Parameters
    ----------
    n_components: integer
        number of gaussian components to fit to the data
    n_iter: integer (optional)
        number of EM iterations to perform (default=100)
    tol: float (optional)
        stopping criterion for EM iterations (default=1E-5)
    Notes
    -----
    This implementation follows Bovy et al. arXiv 0905.2979
    """
    def __init__(self, n_components, n_iter=100, tol=1E-5, verbose=False,
                 random_state = None):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

        # model parameters: these are set by the fit() method
        self.V = None
        self.mu = None
        self.alpha = None
    
    
    def fit(self, X, Xerr, init_params=None, R=None):
        """Fit the XD model to data
        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)
        R : array_like
            (TODO: not implemented)
            Transformation matrix from underlying to observed data.  If
            unspecified, then it is assumed to be the identity matrix.
        """
        
        if R is not None:
            raise NotImplementedError("mixing matrix R is not yet implemented")

        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        # initialize components via a few steps of GMM
        # this doesn't take into account errors, but is a fast first-guess
        
        if init_params is None : 
            gmm = GaussianMixture(self.n_components, max_iter=10, covariance_type='full',
                      random_state=self.random_state).fit(X)
        
        
        #print gmm.weights_
        
        if init_params is not None:
            print("init_params = True : initializing with cmass params ")
            filename = init_params
            #fix_mu, fix_alpha, fix_V = getCMASSparam(filename = 'pickle/gold_st82_20_XD_dmass.pkl')
            fix_mu, fix_alpha, fix_V = getCMASSparam(filename = filename)

            gmm = GaussianMixture(self.n_components, max_iter=1, covariance_type='full',
                        random_state=self.random_state).fit(X)

            gmm.means_ = fix_mu
            gmm.covariances_ = fix_V
            gmm.weights_= fix_alpha
        
        
        self.mu = gmm.means_
        self.alpha = gmm.weights_
        self.V = gmm.covariances_
        self.n_components = len(self.V)
        print('n components =',self.n_components)
        print('tolerance =', self.tol)


        logL = self.logL(X, Xerr)
        
        for i in range(self.n_iter):
            t0 = time()
            self._EMstep(X, Xerr)

            logL_next = self.logL(X, Xerr)
            t1 = time()
            
            #sys.stdout.write("\r" + 'total expected time: {:0.2f} min,  process: {:0.2f} %,  iteration {}/{}      '\
            #    .format((t1-t0) * (self.n_iter - i)/60., i * 1./self.n_iter * 100., i, self.n_iter))
            #sys.stdout.flush()
            
            if self.verbose:
                print(("%i: log(L) = %.10g" % (i + 1, logL_next)))
                print(("    (%.2g sec)" % (t1 - t0)))
              
            if logL_next < logL + self.tol:
                break
            logL = logL_next

            if init_params is not None:
                self.mu = fix_mu
                self.V = fix_V
                if all( s == 0. for s in self.alpha ) == True :
                    print('invalid value in GMM parameter')
                    break 
                pass

        #sys.stdout.write("\r" + 'expected time: {:0.2f} s,  process: {:0.2f} %       \n'.format(0, 100))
        #sys.stdout.write("\r" + 'expected time: {:0.2f} min,  process: {:0.2f} %,  iteration {}/{}      /n'\
        #        .format((t1-t0) * (self.n_iter - i)/60., i * 1./self.n_iter * 100., i, self.n_iter))

        sys.stdout.write("\r" + 'elapsed time: {:0.2f} min,  total iteration {}                                 \n'\
                .format((t1-t0) * i/60., i))
        
        return self


    def logprob_a(self, X, Xerr, parallel = None):
        """
        Evaluate the probability for a set of points
        Parameters
        ----------
        X: array_like
            Input data. shape = (n_samples, n_features)
        Xerr: array_like
            Error on input data.  shape = (n_samples, n_features, n_features)
        Returns
        -------
        p: ndarray
            Probabilities.  shape = (n_samples,)
        """
        X = np.asarray(X)
        Xerr = np.asarray(Xerr)
        n_samples, n_features = X.shape

        # assume full covariances of data
        assert Xerr.shape == (n_samples, n_features, n_features)

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]
        T = Xerr + self.V

        return log_multivariate_gaussian(X, self.mu, T, parallel = parallel) + np.log(self.alpha)

    def logL(self, X, Xerr, parallel = True):
        """Compute the log-likelihood of data given the model
        Parameters
        ----------
        X: array_like
            data, shape = (n_samples, n_features)
        Xerr: array_like
            errors, shape = (n_samples, n_features, n_features)
        Returns
        -------
        logL : float
            log-likelihood
        """
        return np.sum(logsumexp(self.logprob_a(X, Xerr, parallel = parallel), -1))


    def _EMstep(self, X, Xerr, init_params = None):
        """
        Perform the E-step (eq 16 of Bovy et al)
        """
        n_samples, n_features = X.shape

        X = X[:, np.newaxis, :]
        Xerr = Xerr[:, np.newaxis, :, :]

        w_m = X - self.mu
        T = Xerr + self.V

        #------------------------------------------------------------
        # compute inverse of each covariance matrix T
        
        Tshape = T.shape
        T = T.reshape([n_samples * self.n_components,
                       n_features, n_features])
        #Tinv = np.array([linalg.inv(T[i]) for i in range(T.shape[0])]).reshape(Tshape)
        """
        # split data
        n_process = 12
        T_split = np.array_split( T, n_process, axis=0)
    
        def inv_process(q, order, T):
            re = np.array([linalg.inv(T[i]) for i in range(T.shape[0])])
            q.put((order, re))
    
        q = Queue()
        Processes = [Process(target = inv_process, args=(q, z[0], z[1])) for z in zip(range(n_process), T_split)]
    
        for p in Processes: p.start()
        result = []
        for p in Processes : result.append(q.get())

        result.sort()
        Tinv = np.vstack([np.array(r[1]) for r in result ]).reshape(Tshape)
        """
        
        Tinv = invMatrixMultiprocessing(T).reshape(Tshape)
        T = T.reshape(Tshape)

        #------------------------------------------------------------
        # evaluate each mixture at each point
        
        N = np.exp(log_multivariate_gaussian(X, self.mu, T, Vinv=Tinv))
        #------------------------------------------------------------
        # E-step:
        #  compute q_ij, b_ij, and B_ij
        
        q = (N * self.alpha) / np.dot(N, self.alpha)[:, None]

        nanmask = np.ma.masked_invalid(q)
        q[nanmask.mask] = 0.0
        
        tmp = np.sum(Tinv * w_m[:, :, np.newaxis, :], -1)
        b = self.mu + np.sum(self.V * tmp[:, :, np.newaxis, :], -1)

        tmp = np.sum(Tinv[:, :, :, :, np.newaxis]
                     * self.V[:, np.newaxis, :, :], -2)
        B = self.V - np.sum(self.V[:, :, :, np.newaxis]
                            * tmp[:, :, np.newaxis, :, :], -2)

        #------------------------------------------------------------
        # M-step:
        #  compute alpha, m, V
        
        qj = q.sum(0)

        self.alpha = qj / n_samples

        self.mu = np.sum(q[:, :, np.newaxis] * b, 0) / qj[:, np.newaxis]

        m_b = self.mu - b
        tmp = m_b[:, :, np.newaxis, :] * m_b[:, :, :, np.newaxis]
        tmp += B
        tmp *= q[:, :, np.newaxis, np.newaxis]
        self.V = tmp.sum(0) / qj[:, np.newaxis, np.newaxis]



    def sample(self, size=1, random_state=None):
        if random_state is None:
            random_state = self.random_state
        rng = check_random_state(random_state)
        shape = tuple(np.atleast_1d(size)) + (self.mu.shape[1],)
        npts = np.prod(size)

        alpha_cs = np.cumsum(self.alpha)
        r = np.atleast_1d(np.random.random(size))
        r.sort()

        ind = r.searchsorted(alpha_cs)
        ind = np.concatenate(([0], ind))
        if ind[-1] != size:
            ind[-1] = size

        draw = np.vstack([np.random.multivariate_normal(self.mu[i],
                                                        self.V[i],
                                                        (ind[i + 1] - ind[i],))
                          for i in range(len(self.alpha))])

        return draw.reshape(shape)


def _SDSS_cmass_criteria(sdss, prior=None):
    
    modelmag_r = sdss['MODELMAG_R'] - sdss['EXTINCTION_R']
    modelmag_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
    modelmag_g = sdss['MODELMAG_G'] - sdss['EXTINCTION_G']
    cmodelmag_i = sdss['CMODELMAG_I'] - sdss['EXTINCTION_I']
    
    dperp = (modelmag_r - modelmag_i) - (modelmag_g - modelmag_r)/8.0
    fib2mag = sdss['FIBER2MAG_I']
    
    if prior is True:
        priorCut = ((cmodelmag_i > 17.0 ) &
                    (cmodelmag_i < 22.0) &
                    ((modelmag_r - modelmag_i ) < 2.5 )&
                    ((modelmag_g - modelmag_r ) < 2.5 )&
                    (fib2mag < 22.0 ))  #&

    elif prior is None:
        
        priorCut = (((cmodelmag_i > 17.5 ) &
                (cmodelmag_i < 19.9) &
                ((modelmag_r - modelmag_i ) < 2.0 )&
                (fib2mag < 21.5 )) & (dperp > 0.55) &
                 (cmodelmag_i < (19.86 + 1.6*(dperp - 0.8))))
    
   # star = (((sdss['PSFMAG_I'] - sdss['EXPMAG_I']) > (0.2 + 0.2*(20.0 - sdss['EXPMAG_I']))) &
   #      ((sdss['PSFMAG_Z'] - sdss['EXPMAG_Z']) > (9.125 - 0.46 * sdss['EXPMAG_Z'])))


    psfmag_i = sdss['PSFMAG_I'] - sdss['EXTINCTION_I']
    psfmag_z = sdss['PSFMAG_Z'] - sdss['EXTINCTION_Z'] 
    modelmag_z = sdss['MODELMAG_Z'] - sdss['EXTINCTION_Z']
    #modelmag_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
    star = (( ( psfmag_i - modelmag_i ) > (0.2 + 0.2 * (20.0 - modelmag_i ))) &
 		(( psfmag_z - modelmag_z ) > (9.125 - 0.46 * modelmag_z )) )

    return sdss[priorCut & star], priorCut&star #sdss[cmass]
    #return sdss[priorCut], priorCut


def _SDSS_LOWZ_criteria(sdss):
    
    modelmag_r = sdss['MODELMAG_R'] - sdss['EXTINCTION_R']
    modelmag_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
    modelmag_g = sdss['MODELMAG_G'] - sdss['EXTINCTION_G']
    cmodelmag_i = sdss['CMODELMAG_I'] - sdss['EXTINCTION_I']
    cmodelmag_r = sdss['CMODELMAG_R'] - sdss['EXTINCTION_R']
    cpar = 0.7 * (modelmag_g - modelmag_r) + 1.2 * ((modelmag_r - modelmag_i) - 0.18)
    cperp = (modelmag_r - modelmag_i) - (modelmag_g - modelmag_r)/4.0 - 0.18
    # fib2mag = sdss['FIBER2MAG_I']

    priorCut = ((cmodelmag_r > 16 ) &
                (cmodelmag_r < 19.6) &
		(cmodelmag_r < (13.5 + cpar/0.3)) &
 		( cperp < 0.2 ) &
 		( cperp > -0.2))

    star = ( sdss['PSFMAG_R'] - sdss['CMODELMAG_R'] > 0.3 )
    return sdss[priorCut ], priorCut # &star #sdss[cmass]
    #return sdss[priorCut&star ], priorCut&star



def divide_bins( cat, Tag = 'Z', min = 0.2, max = 1.2, bin_num = 5, TagIndex = None ):
    values, step = np.linspace(min, max, num = bin_num+1, retstep = True)
    
    binkeep = []
    binned_cat = []
    
    column = cat[Tag]
    if TagIndex is not None: column = cat[Tag][:,TagIndex]
    
    bin = (column < values[0])
    binned_cat.append( cat[bin] )
    binkeep.append(bin)

    for i in range(len(values)-1) :
        bin = (column >= values[i]) & (column < values[i+1])
        binned_cat.append( cat[bin] )
        binkeep.append(bin)
    
    bin_center = values[:-1] + step/2.
    return bin_center, binned_cat, binkeep



def detMatrixMultiprocessing( T ):

    # split data
    n_process = 12
    T_split = np.array_split( T, n_process, axis=0)
    
    def det_process(q, order, T):
        re = np.array([linalg.det(T[i]) for i in range(T.shape[0])])
        q.put((order, re))
    
    q = Queue()
    Processes = [Process(target = det_process, args=(q, z[0], z[1])) for z in zip(list(range(n_process)), T_split)]
    
    for p in Processes: p.start()
    result = []
    for p in Processes : result.append(q.get())
    
    result.sort()
    Tdet = np.hstack([np.array(r[1]) for r in result ]) #.reshape(Tshape)
    
    return Tdet


def invMatrixMultiprocessing( T ):
    
    # split data
    n_process = 12
    T_split = np.array_split( T, n_process, axis=0)
    
    def inv_process(q, order, T):
        re = np.array([linalg.inv(T[i]) for i in range(T.shape[0])])
        q.put((order, re))
    
    q = Queue()
    Processes = [Process(target = inv_process, args=(q, z[0], z[1])) for z in zip(list(range(n_process)), T_split)]
    
    for p in Processes: p.start()
    result = []
    for p in Processes : result.append(q.get())
    
    result.sort()
    Tinv = np.vstack([np.array(r[1]) for r in result ]) #.reshape(Tshape)
    
    return Tinv


def log_multivariate_gaussian(x, mu, V, Vinv=None, method=1, parallel = True):
    """Evaluate a multivariate gaussian N(x|mu, V)
    This allows for multiple evaluations at once, using array broadcasting
    Parameters
    ----------
    x: array_like
        points, shape[-1] = n_features
    mu: array_like
        centers, shape[-1] = n_features
    V: array_like
        covariances, shape[-2:] = (n_features, n_features)
    Vinv: array_like or None
        pre-computed inverses of V: should have the same shape as V
    method: integer, optional
        method = 0: use cholesky decompositions of V
        method = 1: use explicit inverse of V
    Returns
    -------
    values: ndarray
        shape = broadcast(x.shape[:-1], mu.shape[:-1], V.shape[:-2])
    Examples
    --------
    >>> x = [1, 2]
    >>> mu = [0, 0]
    >>> V = [[2, 1], [1, 2]]
    >>> log_multivariate_gaussian(x, mu, V)
    -3.3871832107434003
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    V = np.asarray(V, dtype=float)

    ndim = x.shape[-1]
    x_mu = x - mu

    if V.shape[-2:] != (ndim, ndim):
        raise ValueError("Shape of (x-mu) and V do not match")

    Vshape = V.shape
    V = V.reshape([-1, ndim, ndim])

    if Vinv is not None:
        assert Vinv.shape == Vshape
        method = 1

    if method == 0:
        Vchol = np.array([linalg.cholesky(V[i], lower=True)
                          for i in range(V.shape[0])])

        # we may be more efficient by using scipy.linalg.solve_triangular
        # with each cholesky decomposition
        VcholI = np.array([linalg.inv(Vchol[i])
                          for i in range(V.shape[0])])
        logdet = np.array([2 * np.sum(np.log(np.diagonal(Vchol[i])))
                           for i in range(V.shape[0])])

        VcholI = VcholI.reshape(Vshape)
        logdet = logdet.reshape(Vshape[:-2])

        VcIx = np.sum(VcholI * x_mu.reshape(x_mu.shape[:-1]
                                            + (1,) + x_mu.shape[-1:]), -1)
        xVIx = np.sum(VcIx ** 2, -1)

    elif method == 1:
        if Vinv is None:
            if parallel == None : Vinv = np.array([linalg.inv(V[i]) for i in range(V.shape[0])]).reshape(Vshape)
            else : Vinv = invMatrixMultiprocessing(V).reshape(Vshape)
        else:
            assert Vinv.shape == Vshape

        if parallel == None : logdet = np.log(np.array([linalg.det(V[i]) for i in range(V.shape[0])]))
        else: logdet = np.log(detMatrixMultiprocessing(V))

        logdet = logdet.reshape(Vshape[:-2])

        xVI = np.sum(x_mu.reshape(x_mu.shape + (1,)) * Vinv, -2)
        xVIx = np.sum(xVI * x_mu, -1)

    else:
        raise ValueError("unrecognized method %s" % method)

    return -0.5 * ndim * np.log(2 * np.pi) - 0.5 * (logdet + xVIx)



def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)



def split_samples(X, y, fractions=[0.75, 0.25], random_state=None):
    """Split samples into training, test, and cross-validation sets
    Parameters
    ----------
    X, y : array_like
        leading dimension n_samples
    fraction : array_like
        length n_splits.  If the fractions do not add to 1, they will be
        re-normalized.
    random_state : None, int, or RandomState object
        random seed, or random number generator
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y should have the same leading dimension")

    n_samples = X.shape[0]

    fractions = np.asarray(fractions).ravel().cumsum()
    fractions /= fractions[-1]
    fractions *= n_samples
    N = np.concatenate([[0], fractions.astype(int)])
    N[-1] = n_samples  # in case of roundoff errors

    random_state = check_random_state(random_state)
    indices = np.arange(len(y))
    random_state.shuffle(indices)

    X_indices = tuple(indices[N[i]:N[i + 1]]
                        for i in range(len(fractions)))
    y_indices = tuple(indices[N[i]:N[i + 1]]
                        for i in range(len(fractions)))

    return X_indices, y_indices



def logsumexp(arr, axis=None):
    """Computes the sum of arr assuming arr is in the log domain.
    Returns log(sum(exp(arr))) while minimizing the possibility of
    over/underflow.
    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> np.log(np.sum(np.exp(a)))
    9.4586297444267107
    >>> logsumexp(a)
    9.4586297444267107
    """
    # if axis is specified, roll axis to 0 so that broadcasting works below
    if axis is not None:
        arr = np.rollaxis(arr, axis)
        axis = 0
    
    # Use the max to normalize, as with the log this is what accumulates
    # the fewest errors
    vmax = arr.max(axis=axis)
    out = np.log(np.sum(np.exp(arr - vmax), axis=axis))
    out += vmax
    
    return out



def mixing_color(data, suffix = '', 
    mag = ['MAG_MODEL', 'MAG_DETMODEL'], 
    err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'], 
    filter = ['G', 'R', 'I'],
    sdss = None, cmass = None, elg=None,
    no_zband = True  ):
    
    magtag = [ m+'_'+f+suffix for m in mag for f in filter ]
    errtag = [ e+'_'+f for e in err for f in filter ]
    del magtag[0], errtag[0]
    if 'Z' in filter: del magtag[2], errtag[2]
    #print(magtag)

    X = [ data[mt] for mt in magtag ]
    Xerr = [ data[mt] for mt in errtag ]
    #reddeningtag = 'XCORR_SFD98'

    X = np.vstack(X).T
    Xerr = np.vstack(Xerr).T
    # mixing matrix
    W = np.array([
                  [1, 0, 0, 0, 0, 0],    # r mag
                  [0, 1, 0, 0, 0, 0],    # i mag
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],   # r-i
                  [0, 0, 0, 0, 1, -1]])  # i-z

    if 'Z' not in filter: W = W[:-1,:-1]
    X = np.dot(X, W.T)

    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, list(range(Xerr.shape[1])), list(range(Xerr.shape[1]))] = Xerr**2
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))
    return X, Xcov


def mixing_color_elg(data, suffix = '', 
    mag = ['MAG_MODEL', 'MAG_DETMODEL'], 
    err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'], 
    filter = ['G', 'R', 'I', 'Z'],
    full=False,
    no_zband = True  ):
    
    magtag = [ m+'_'+f+suffix for m in mag for f in filter ]
    errtag = [ e+'_'+f for e in err for f in filter ]
    #del magtag[0], errtag[0]
    if 'Z' in filter: del magtag[3], errtag[3]
    #print(magtag)

    X = [ data[mt] for mt in magtag ]
    Xerr = [ data[mt] for mt in errtag ]
    #reddeningtag = 'XCORR_SFD98'

    X = np.vstack(X).T
    Xerr = np.vstack(Xerr).T
    # mixing matrix
    W = np.array([[1, 0, 0, 0, 0, 0, 0],    # g mag
                  [0, 1, 0, 0, 0, 0, 0],    # r mag
                  [0, 0, 1, 0, 0, 0, 0],    # i mag
                  [0, 0, 0, 1, -1, 0, 0],   # g-r
                  #[0, 0, 0, 0, 1, -1, 0],   # r-i
                  #[0, 0, 0, 0, 0, 1, -1],   # i-z
                  [1, 0, 0, 0, 0, 0, -1]])  # r-z
    if full:
        W = np.array([[1, 0, 0, 0, 0, 0, 0],    # g mag
                    [0, 1, 0, 0, 0, 0, 0],    # r mag
                    [0, 0, 1, 0, 0, 0, 0],    # i mag
                    [0, 0, 0, 1, -1, 0, 0],   # g-r
                    [0, 0, 0, 0, 1, -1, 0],   # r-i
                    [0, 0, 0, 0, 0, 1, -1],   # i-z
                    [1, 0, 0, 0, 0, 0, -1]])  # r-z

    #if 'Z' not in filter: W = W[:-1,:-1]
    X = np.dot(X, W.T)

    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, list(range(Xerr.shape[1])), list(range(Xerr.shape[1]))] = Xerr**2
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))
    return X, Xcov

    
def XD_fitting( data = None, 
        pickleFileName = 'pickle/XD_fitting_test.pkl', 
        init_params = None, 
        suffix='', 
        mag = ['MAG_MODEL', 'MAG_DETMODEL'],
        err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'],
        filter = ['G', 'R', 'I'],
        n_cl = None, n_iter = 500, tol=1E-5, verbose=False ):

    from astroML.decorators import pickle_results
    @pickle_results(pickleFileName, verbose = True)
    def compute_XD(X, Xcov, init_params = None, n_iter=500, 
    verbose=False, n_cl = None, tol=1E-5):
        if init_params != None : 
            n_cl = 10

        if n_cl is None : 
            n_cl,_,_= _FindOptimalN( np.arange(2, 50, 2), X, 
            pickleFileName = pickleFileName+'.n_cluster' , suffix = '')

        clf= XDGMM(n_cl, n_iter=n_iter, tol=tol, verbose=verbose)
        clf.fit(X, Xcov, init_params = init_params)
        return clf

    if data is None: 
        import pickle 
        f = open(pickleFileName)
        pickle = pickle.load(f, 'rb')
        clf = pickle['retval']
    else:
        X, Xcov = mixing_color(data, mag=mag, err=err, filter=filter, 
        suffix = suffix, no_zband=False)
        clf = compute_XD(X, Xcov, init_params=init_params, n_cl = n_cl, 
        n_iter = n_iter, tol=tol, verbose=verbose)
    return clf
    
    
def XD_fitting_X( X = None, Xcov=None, 
        pickleFileName = 'pickle/XD_fitting_test.pkl', 
        init_params = None, 
        suffix='', 
        mag = ['MAG_MODEL', 'MAG_DETMODEL'],
        err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'],
        filter = ['G', 'R', 'I'],
        n_cl = None, n_iter = 500, tol=1E-5, verbose=False ):

    from astroML.decorators import pickle_results
    @pickle_results(pickleFileName, verbose = True)
    def compute_XD(X, Xcov, init_params = None, n_iter=500, 
    verbose=False, n_cl = None, tol=1E-5):
        if init_params != None : 
            n_cl = 10

        if n_cl is None : 
            n_cl,_,_= _FindOptimalN( np.arange(2, 50, 2), X, 
            pickleFileName = pickleFileName+'.n_cluster' , suffix = '')

        clf= XDGMM(n_cl, n_iter=n_iter, tol=tol, verbose=verbose)
        clf.fit(X, Xcov, init_params = init_params)
        return clf

    if X is None: 
        import pickle 
        f = open(pickleFileName, 'rb')
        pickle = pickle.load(f, encoding="latin1")
        clf = pickle['retval']        
        
    else:
        #X, Xcov = mixing_color(data, mag=mag, err=err, filter=filter, 
        #suffix = suffix, no_zband=False)
        clf = compute_XD(X, Xcov, init_params=init_params, n_cl = n_cl, 
        n_iter = n_iter, tol=tol, verbose=verbose)
    return clf


def XDastroML_fitting_X( X = None, Xcov=None, 
        FileName = 'pickle/XD_fitting_test.pkl', 
        init_params = None, 
        #suffix='', 
        #mag = ['MAG_MODEL', 'MAG_DETMODEL'],
        #err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'],
        #filter = ['G', 'R', 'I'],
        n_cl = None, n_iter = 500, tol=1E-5, verbose=False ):

    from astroML.decorators import pickle_results
    @pickle_results(FileName, verbose = True)
    def compute_XD(X, Xcov, init_params = None, n_iter=500, 
    verbose=False, n_cl = None, tol=1E-5):
        if init_params != None : 
            n_cl = 10

        if n_cl is None : 
            #n_cl,_,_= _FindOptimalN_with_err( np.arange(2, 50, 2), X, Xcov
            #pickleFileName = pickleFileName+'.n_cluster' , suffix = '')
            param_range=np.arange(2, 50, 2)
            n_cl,_,_= _FindOptimalN_with_err( param_range, X, Xcov, 
            FileName = FileName+'.n_cluster' , suffix = '')

        clf= XDGMM(n_cl, n_iter=n_iter, tol=tol, verbose=verbose)
        clf.fit(X, Xcov, init_params = init_params)
        return clf

    if X is None: 
        import pickle 
        f = open(FileName, 'rb')
        pickle = pickle.load(f, encoding="latin1")
        clf = pickle['retval']        
        
    else:
        #X, Xcov = mixing_color(data, mag=mag, err=err, filter=filter, 
        #suffix = suffix, no_zband=False)
        clf = compute_XD(X, Xcov, init_params=init_params, n_cl = n_cl, 
        n_iter = n_iter, tol=tol, verbose=verbose)
    return clf



def XDnew_fitting_X( X = None, Xcov=None, 
        FileName = None, 
        #init_params = None, 
        #suffix='', 
        #mag = ['MAG_MODEL', 'MAG_DETMODEL'],
        #err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'],
        #filter = ['G', 'R', 'I'],
        n_cl = None, n_iter = 500, tol=1E-5, method='Bovy', verbose=False ):

    

    try: 
        xdgmm_obj = XDGMM_Holoien(filename=FileName) 
        print ('Using precomputed results from ', FileName)
        return xdgmm_obj
    except FileNotFoundError: pass
    
    if X is None:
        # calling pre-computed model
        xdgmm_obj = XDGMM_Holoien(filename=FileName) 
        print ('Using precomputed results from ', FileName)
        return xdgmm_obj

    else: 
        if n_cl == None : 
            param_range=np.arange(2, 50, 2)
            optimal_n_comp,_,_= _FindOptimalN_with_err( param_range, X, Xcov, 
            pickleFileName = FileName+'.n_cluster' , suffix = '')
            
            #xdgmm_test = XDGMM_Holoien( n_iter=n_iter, tol=tol, method=method )
            ## Define the range of component numbers, and get ready to compute the BIC for each one:
            #param_range = np.arange(2, 50, 2)
            ## Loop over component numbers, fitting XDGMM model and computing the BIC:
            #bic, optimal_n_comp, lowest_bic = xdgmm_test.bic_test(X, Xcov, param_range)
            ##n_cl = optimal_n_comp[np.argmin(bic)]
        else: optimal_n_comp = n_cl

        #import time
        # fitting
        #initiated class
        xdgmm_obj = XDGMM_Holoien( n_components=optimal_n_comp, n_iter=n_iter, tol=tol, method=method )
        #xdgmm.n_components = optimal_n_comp
        print ('n_components=', optimal_n_comp)
        print ('fitting started. This will take for a while.')
        t1 = time.time()
        xdgmm_obj = xdgmm_obj.fit(X, Xcov)
        #t2 = time.time()
        print ('fitting finished')
        
        #t3 = time.time()
        print ('saving xdgmm object to.. ', FileName)
        xdgmm_obj.save_model(FileName)
        print ('file saved')
        #print ('saving obj. time:', (t3-t2)%60,'s')
        
        #t4 = time.time()
        xdgmm_obj = XDGMM_Holoien(filename=FileName) 
        #print ('loading obj. time:', (t4-t3)%60,'s')
        t2 = time.time()
        print ('elapsed time:', (t2-t1)/60.0,'s')
        return xdgmm_obj


def add_errors(model_data, real_data, real_covars):
    # For each of the model data points, find the nearest `real' data
    # point, assume that the errors are similar, and then add a random
    # error with that amplitude.
    # Initialize a scipy KD tree for this.
    import scipy.spatial as sp
    tree = sp.KDTree(real_data)
    (dist, ind) = tree.query(model_data)
    model_covars = real_covars[ind,:,:]
    noise = np.empty(model_data.shape)
    for i in range(model_data.shape[0]):
        noise[i,:] = (np.random.multivariate_normal(model_data[i,:]*0.,model_covars[i,:,:]))
    
    noisy_data = model_data + noise
    return noisy_data


def add_errors_multiprocessing(model_data, real_data, real_covars):

    n_process = 12
    
    X_sample_split = np.array_split(model_data, n_process, axis=0)
    
    def addingerror_process(q, order, xxx_todo_changeme):
        (model, data, cov) = xxx_todo_changeme
        re = add_errors(model, data, cov)
        q.put((order, re))
    
    inputs = [ (X_sample_split[i], real_data, real_covars) for i in range(n_process) ]
    
    q = Queue()
    Processes = [Process(target = addingerror_process, args=(q, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    
    for p in Processes: p.start()
    
    percent = 0.0
    result = []
    for p in Processes:
        result.append(q.get())
        percent += + 1./( n_process ) * 100
        sys.stdout.write("\r" + 'add noise to sample {:0.0f} %'.format( percent ))
        sys.stdout.flush()

    sys.stdout.write("\r" + 'add noise to sample {:0.0f} %\n'.format( 100 ))
    #result = [q.get() for p in Processes]
    result.sort()
    result = [r[1] for r in result ]

    noisy_X_sample = np.vstack( result[:n_process] )
    return noisy_X_sample




# ---------visualization tool ------------------------------------#

def doVisualization3( true, data1, data2, labels = None, ranges = None, nbins=100, prefix= 'default' ):
    if labels == None:
        print(" always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data.")
        stop
    else:
        ndim = len(labels)
    
    if ranges == None:
        # try to figure out the correct axis ranges.
        print("Using central 98% to set range.")
        ranges = []
        for i in range(ndim):
            ranges.append( np.percentile(real_data[:,i],[1.,99.]) )

    fig,axes = plt.subplots(nrows=ndim, ncols= ndim, figsize= (6*ndim, 6*ndim) )
    
    for i in range(ndim):
        for j in range(ndim):
            if i == j:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                axes[i][i].hist(data1[:,i],bins=xbins,normed=True,label='data1')
                axes[i][i].set_xlabel(labels[i])
                axes[i][i].hist(data2[:,i],bins=xbins,normed=True,alpha=0.5,label='data2')
                axes[i][i].hist(true[:,i],bins=xbins,normed=True,alpha=0.2,label='true')
                axes[i][i].legend(loc='best')
            else:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                ybins = np.linspace(ranges[j][0],ranges[j][1],100)
                axes[i][j].hist2d(data1[:,i], data1[:,j], bins = [xbins,ybins], normed=True)
                axes[i][j].set_xlabel(labels[i])
                axes[i][j].set_ylabel(labels[j])
                axes[i][j].set_title('data1')
                axes[j][i].hist2d(data2[:,i], data2[:,j], bins = [xbins,ybins], normed=True)
                axes[j][i].set_xlabel(labels[i])
                axes[j][i].set_ylabel(labels[j])
                axes[j][i].set_title('data2')

    filename = 'figure/'+prefix+"diagnostic_histograms3.png"
    print("writing output plot to: "+filename)
    fig.savefig(filename)



def doVisualization_1d( data=None, labels = None, 
ranges = None, name = None, weight = [None,None], 
color=['grey', 'tab:blue', 'tab:orange', 'tab:green'],
nbins=100, filename=None):


    if labels == None:
        print(" always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data.")
        stop
    else:
        ndim = len(labels)
    
    if ranges == None:
        # try to figure out the correct axis ranges.
        print("Using central 100% to set range.")
        ranges = []
        for i in range(ndim):
            ranges.append( np.percentile(data[-1][:,i],[0.,100.]) )

    
    fig,axes = plt.subplots(nrows=1, ncols= ndim, figsize= (4*ndim, 4) )
    #print weight[0], weight[1].size, data1[:,0].size 
    for i in range(ndim):
        xbins = np.linspace(ranges[i][0],ranges[i][1], nbins)
        for j in range(len(data)):
            axes[i].hist(data[j][:,i],bins=xbins, label=name[j], weights=weight[j], alpha = 0.5, color =color[j], density=True)
            #axes[i].hist(data[:,i],bins=xbins, alpha=1.0,label=name[1], weights=weight[1], histtype='step', color='k', lw=1, density=True)
        axes[i].set_xlabel(labels[i], fontsize = 20)
        #axes[i].hist(data2[:,i],bins=xbins,normed=True,alpha=0.5,label='data2')
        axes[i].get_yaxis().set_visible(False)
        axes[i].tick_params(labelsize=15)
        axes[i].legend(loc='best',fontsize = 10)

    #filename = "figure/diagnostic_histograms_1d.pdf"
    #print("writing output plot to: "+filename)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05, hspace=0.5);
    if filename != None: fig.savefig(filename)
    #plt.close(fig)


def doVisualization_1d_NperA( data1, true, area = None, labels = None, ranges = None, name = None, nbins=100, prefix= 'default' ):
    
    if labels == None:
        print(" always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data.")
        stop
    else:
        ndim = len(labels)
    
    if ranges == None:
        # try to figure out the correct axis ranges.
        print("Using central 98% to set range.")
        ranges = []
        for i in range(ndim):
            ranges.append( np.percentile(real_data[:,i],[1.,99.]) )

    fig, axes = plt.subplots(nrows=1, ncols= ndim, figsize= (6*ndim, 5) )
    
    for i in range(ndim):
        xbins, step = np.linspace(ranges[i][0],ranges[i][1],100, retstep = True)
        
        #N_t, _,_ = axes[i].hist(true[:,i],bins=xbins,normed=False, color='blue')
        #N_d, _,_ = axes[i].hist(data1[:,i],bins=xbins,normed=False, alpha=0.5,label=name[0], color = 'red')
        N_t, _ = np.histogram(true[:,i], bins=xbins)
        N_d, _ = np.histogram(data1[:,i], bins=xbins)
        axes[i].bar( xbins[:-1]+step/2., N_t/area[1], step, color = 'blue', label=name[1] )
        axes[i].bar( xbins[:-1]+step/2., N_d/area[0], step, color='red', alpha=0.5, label=name[0] )
        axes[i].set_xlim(ranges[i][0],ranges[i][1])
        axes[i].set_xlabel(labels[i])
        axes[i].set_ylabel('N/Area')
        #axes[i].hist(data2[:,i],bins=xbins,normed=True,alpha=0.5,label='data2')
        
        axes[i].legend(loc='best')

    filename = "figure/"+prefix+"diagnostic_histograms_1d.png"
    print("writing output plot to: "+filename)
    fig.savefig(filename)
    #plt.close(fig)


def doVisualization(model_data, real_data, name = ['model', 'real'], labels = None, ranges = None, nbins=100, prefix= 'default'):
    if labels == None:
        print(" always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data.")
        stop
    else:
        ndim = len(labels)
    
    if ranges == None:
        # try to figure out the correct axis ranges.
        print("Using central 98% to set range.")
        ranges = []
        for i in range(ndim):
            ranges.append( np.percentile(real_data[:,i],[1.,99.]) )

    fig,axes = plt.subplots(nrows=ndim, ncols= ndim, figsize= (6*ndim, 6*ndim) )
    
    for i in range(ndim):
        for j in range(ndim):
            if i == j:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                axes[i][i].hist(real_data[:,i],bins=xbins,normed=True,label=name[1], color='blue')
                axes[i][i].set_xlabel(labels[i])
                axes[i][i].hist(model_data[:,i],bins=xbins,normed=True,alpha=0.5,label=name[0], color='red')
                axes[i][i].legend(loc='best')
            else:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                ybins = np.linspace(ranges[j][0],ranges[j][1],100)
                axes[i][j].hist2d(real_data[:,i], real_data[:,j], bins = [xbins,ybins], normed=True)
                axes[i][j].set_xlabel(labels[i])
                axes[i][j].set_ylabel(labels[j])
                axes[i][j].set_title(name[1])
                axes[j][i].hist2d(model_data[:,i], model_data[:,j], bins = [xbins,ybins], normed=True)
                axes[j][i].set_xlabel(labels[i])
                axes[j][i].set_ylabel(labels[j])
                axes[j][i].set_title(name[0])

    filename = 'figure/'+prefix+"diagnostic_histograms.png"
    print("writing output plot to: "+filename)
    fig.savefig(filename)
    plt.close(fig)


def doVisualization2(true_data, test_data, labels = None, ranges = None, nbins=100, prefix= 'default'):
    if labels == None:
        print(" always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data.")
        stop
    else:
        ndim = len(labels)
    
    if ranges == None:
        # try to figure out the correct axis ranges.
        print("Using central 98% to set range.")
        ranges = []
        for i in range(ndim):
            ranges.append( np.percentile(test_data[:,i],[1.,99.]) )

    fig,axes = plt.subplots(nrows=ndim, ncols= ndim, figsize= (6*ndim, 6*ndim) )
    
    for i in range(ndim):
        for j in range(ndim):
            if i == j:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                axes[i][i].hist(test_data[ labels[i] ],bins=xbins,normed=True,label='real')
                axes[i][i].set_xlabel(labels[i])
                axes[i][i].hist(true_data[ labels[i] ],bins=xbins,normed=True,alpha=0.5,label='true')
                axes[i][i].legend(loc='best')
            else:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                ybins = np.linspace(ranges[j][0],ranges[j][1],100)
                axes[i][j].hist2d(test_data[labels[i]], test_data[labels[j]], bins = [xbins,ybins], normed=True)
                axes[i][j].set_xlabel(labels[i])
                axes[i][j].set_ylabel(labels[j])
                axes[i][j].set_title('test')
                axes[j][i].hist2d(true_data[labels[i]], true_data[labels[j]], bins = [xbins,ybins], normed=True)
                axes[j][i].set_xlabel(labels[i])
                axes[j][i].set_ylabel(labels[j])
                axes[j][i].set_title('true')

    filename = prefix+"diagnostic_histograms2.png"
    print("writing output plot to: "+filename)
    fig.savefig(filename)


def _FindOptimalN( N, Xdata, pickleFileName = None, suffix = None):
    #from sklearn.mixture import GMM
    #data, _ = mixing_color(data, suffix = suffix)
    @pickle_results( pickleFileName )
    def compute_GMM( N, covariance_type='full', n_iter=1000):
        models = [None for n in N]
        for i in range(len(N)):
            sys.stdout.write("\r" + 'Finding optimal number of cluster : {:0.0f} % '\
                             .format(i * 1./len(N) * 100.))
            sys.stdout.flush()
            models[i] = GaussianMixture(n_components=N[i], max_iter=n_iter,
                            covariance_type=covariance_type)
            models[i].fit(Xdata)
        return models
    
    models = compute_GMM(N)
    AIC = [m.aic(Xdata) for m in models]
    BIC = [m.bic(Xdata) for m in models]
    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    sys.stdout.write("\r" + 'Finding optimal number of cluster : {:0.0f} % '\
                     .format(100))
    print("\nbest fit converged:", gmm_best.converged_, end=' ')
    print(" n_components =  %i" % N[i_best])
    return N[i_best], AIC, BIC


def _FindOptimalN_with_err( N, Xdata, Xcov, pickleFileName = None, suffix = None):
    #from sklearn.mixture import GMM
    #data, _ = mixing_color(data, suffix = suffix)
    @pickle_results( pickleFileName )
    def compute_GMM( N, covariance_type='full', n_iter=1000):
        models = [None for n in N]
        for i in range(len(N)):
            sys.stdout.write("\r" + 'Finding optimal number of cluster : {:0.0f} % '\
                             .format(i * 1./len(N) * 100.))
            sys.stdout.flush()
            models[i] = GaussianMixture(n_components=N[i], max_iter=n_iter,
                            covariance_type=covariance_type)
            models[i].fit(Xdata, Xcov)
        return models
    
    models = compute_GMM(N)
    AIC = [m.aic(Xdata) for m in models]
    BIC = [m.bic(Xdata) for m in models]
    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    sys.stdout.write("\r" + 'Finding optimal number of cluster : {:0.0f} % '\
                     .format(100))
    print("\nbest fit converged:", gmm_best.converged_, end=' ')
    print(" n_components =  %i" % N[i_best])
    return N[i_best], AIC, BIC


def XDGMM_model(cmass, lowz, train = None, test = None, mock = None, cmass_fraction = None, spt = False, prefix = ''):

    """ 
    will be deprecated later
    """
    
    import esutil
    import numpy.lib.recfunctions as rf
    from multiprocessing import Process, Queue
    
    train, _ = priorCut(train)
    test, _ = priorCut(test)
    m_des, m_sdss = esutil.numpy_util.match( train['COADD_OBJECTS_ID'], cmass['COADD_OBJECTS_ID'] )
    true_cmass = np.zeros( train.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask = true_cmass == 1
    cmass_fraction = np.sum(cmass_mask) * 1./train.size
    if cmass_fraction is not None : cmass_fraction = cmass_fraction
    print('num of cmass in train', np.sum(cmass_mask), ' fraction ', cmass_fraction)

    # stack DES data

    #if reverse is True:
    X_cmass, Xcov_cmass = mixing_color( cmass )
    X_train, Xcov_train = mixing_color( train )
    X, Xcov = mixing_color( test )
    #X, Xcov = np.vstack((X_train,X_test)), np.vstack((Xcov_train, Xcov_test))

    y_train = np.zeros(( train.size, 2 ), dtype=int)
    y_train[:,0][cmass_mask] = 1
    #y_train[:,1][lowz_mask] = 1
    
    

    print('train/test', len(X_train), len( X ))
    
    # train sample XD convolution ----------------------------------
    
    # finding optimal n_cluster

    #from sklearn.mixture import GMM
    prefix1 = prefix # 'gold_st82_5_'
    prefix2 = prefix #'gold_st82_6_' # has only all likelihood.
    #prefix2 = prefix
    """
    N = np.arange(1, 50, 2)
    #pickleFileName_GMM = ['pickle/'+prefix+'bic_cmass.pkl', 'pickle/'+prefix+'bic_all.pkl']
    pickleFileName_GMM = ['pickle/'+prefix1+'bic_cmass.pkl', 'pickle/'+prefix1+'bic_all.pkl']
    n_cl_cmass, aic_cmass, bic_cmass = _FindOptimalN( N, X_cmass, pickleFileName = pickleFileName_GMM[0])
    rows = np.random.choice(np.arange(X_train.shape[0]), 10 * np.sum(cmass_mask))
    n_cl_all, aic_all, bic_all  = _FindOptimalN( N, X_train[rows,:], pickleFileName = pickleFileName_GMM[1])
    #n_cl_all, aic_all, bic_all  = _FindOptimalN( N, X_train, pickleFileName = pickleFileName_GMM[1])
    #n_cl_no, aic_no, bic_no = FindOptimalN( N, X_train[~cmass_mask], pickleFileName = 'GMM_bic_no.pkl')
    
    DAT = np.column_stack(( N, bic_cmass, bic_all, aic_cmass, aic_all))
    np.savetxt('BIC.txt', DAT, delimiter = ' ', header = 'N, bic_cmass, bic_all,  aic_cmass, aic_all' )
    print 'save to BIC.txt'
    """
    n_cl_cmass = 10
    n_cl_no = 25
    n_cl_all = 25
    # ----------------------------------
  
    #pickleFileName = ['pickle/'+prefix +'XD_all.pkl', 'pickle/'+prefix+'XD_dmass.pkl', 'pickle/'+prefix+'XD_no.pkl']
    #pickleFileName = ['pickle/'+prefix2 +'XD_all.pkl', 'pickle/'+prefix1+'XD_dmass.pkl', 'pickle/'+prefix+'XD_no.pkl']
    """
    def XDDFitting( X_train, Xcov_train,  init_params=None, filename = None, n_cluster = 25 ):
        clf = None
        @pickle_results(filename, verbose = True)
        def compute_XD(n_clusters=n_cluster, init_params = None, n_iter=500, verbose=False):
            clf= XDGMM(n_clusters, n_iter=n_iter, tol=1E-2, verbose=verbose)
            clf.fit(X_train, Xcov_train, init_params = init_params)
            return clf
        clf = compute_XD()
        return clf
    """
    # calculates CMASS fits
    #clf = XDDFitting( X_train, Xcov_train, filename= pickleFileName[0], n_cluster= n_cl_all)
    #clf_cmass = XDDFitting( X_cmass, Xcov_cmass, filename=pickleFileName[1], n_cluster=n_cl_cmass)
    #clf_nocmass = XDDFitting( X_train[~cmass_mask], Xcov_train[~cmass_mask], filename= pickleFileName[2], n_cluster=n_cl_no)


    clf = XD_fitting( train, pickleFileName = 'pickle/gold_st82_20_XD_all_tor.pkl', suffix = '_all', n_cl = n_cl_all )
    clf_nocmass = XD_fitting( train[~cmass_mask], pickleFileName = 'pickle/gold_st82_20_XD_no_tor.pkl', suffix='_no' , n_cl=n_cl_no)
    clf_cmass = XD_fitting( train[cmass_mask], pickleFileName = 'pickle/gold_st82_20_XD_cmass.pkl', suffix='_cmass', n_cl=n_cl_cmass )
    

    # logprob_a ------------------------------------------
    
    print("calculate loglikelihood gaussian with multiprocessing module")
    #cmass_logprob_a = logsumexp(clf_cmass.logprob_a( X, Xcov, parallel = True ), axis = 1)
    #all_logprob_a = logsumexp(clf.logprob_a( X, Xcov, parallel = True ), axis = 1)
    
    from multiprocessing import Process, Queue
    # split data
    n_process = 12
    
    #X_test_split = np.array_split(X_test, n_process, axis=0)
    #Xcov_test_split = np.array_split(Xcov_test, n_process, axis=0)
    
    X_split = np.array_split(X, n_process, axis=0)
    Xcov_split = np.array_split(Xcov, n_process, axis=0)
    
    def logprob_process(q,  classname, order, xxx_todo_changeme1):
        (data, cov) = xxx_todo_changeme1
        re = classname.logprob_a(data, cov)
        result = logsumexp(re, axis = 1)
        q.put((order, result))

    inputs = [ (X_split[i], Xcov_split[i]) for i in range(n_process) ]
    
    q_cmass = Queue()
    q_all = Queue()
    q_no = Queue()
    cmass_Processes = [Process(target = logprob_process, args=(q_cmass, clf_cmass, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    all_Processes = [Process(target = logprob_process, args=(q_all, clf, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    no_Processes = [Process(target = logprob_process, args=(q_no, clf_nocmass, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]

    #sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
    
    for p in cmass_Processes: p.start()
    
    percent = 0.0
    result = []
    for p in cmass_Processes:
        result.append(q_cmass.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()
    
    #result = [q_cmass.get() for p in cmass_Processes]
    result.sort()
    cmass_logprob_a = np.hstack([np.array(r[1]) for r in result ])


    for p in all_Processes: p.start()

    resultall = []
    for p in all_Processes:
        resultall.append(q_all.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()

    #resultall = [q_all.get() for p in all_Processes]
    resultall.sort()
    all_logprob_a = np.hstack([r[1] for r in resultall ])


    for p in no_Processes: p.start()

    resultno = []
    for p in no_Processes:
        resultno.append(q_no.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()

    resultno.sort()
    no_logprob_a = np.hstack([r[1] for r in resultno ])
    
    sys.stdout.write("\r" + 'multiprocessing {:0.0f} % \n'.format( 100 ))


    if spt == True:
        
        "Replace st82 likelihood with spt likelihood"
        #rows = np.random.choice( test.size, size=test.size/10)
        clf_patch_no = XD_fitting( test, pickleFileName = 'pickle/gold_st82_20_XD_no2.pkl', suffix = '_all2', n_cl = n_cl_all )
        
        q_patch = Queue()
        patch_Processes = [Process(target = logprob_process, args=(q_patch, clf_patch_no, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]

        for p in patch_Processes: p.start()

        result_patch = []
        for p in patch_Processes:
            result_patch.append(q_patch.get())
            percent += + 1./( n_process * 2 ) * 100
            sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
            sys.stdout.flush()

        result_patch.sort()
        patch_logprob_a = np.hstack([r[1] for r in result_patch ])

        sys.stdout.write("\r" + 'multiprocessing {:0.0f} % \n'.format( 100 ))
        no_logprob_a = patch_logprob_a.copy()
        #no_logprob_a = no_logprob_a - all_logprob_a + patch_logprob_a


    numerator =  np.exp(cmass_logprob_a) * cmass_fraction
    #denominator = np.exp(all_logprob_a)
    denominator = numerator + np.exp(no_logprob_a) * (1. - cmass_fraction)

    denominator_zero = denominator == 0
    EachProb_CMASS = np.zeros( numerator.shape )
    EachProb_CMASS[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]
    
    #EachProb_CMASS[EachProb_CMASS > 1.0] = 1.0
    #print 'EachProb max', EachProb_CMASS.max()
    
    #train = rf.append_fields(train, 'EachProb_CMASS', EachProb_CMASS[:train.size])
    #test['EachProb_CMASS'] = EachProb_CMASS
    test = rf.append_fields(test, 'EachProb_CMASS', EachProb_CMASS)
    test = rf.append_fields(test, 'cmassLogLikelihood', cmass_logprob_a)
    test = rf.append_fields(test, 'noLogLikelihood', no_logprob_a)
    #test = rf.append_fields(test, 'AllLogLikelihood', all_logprob_a)
    # -----------------------------------------------
    
    
    print('add noise to samples...')
    
    
    """
    X_sample_cmass = clf_cmass.sample(train[cmass_mask].size)
    X_sample_no = clf_nocmass.sample(train[~cmass_mask].size * 100 )
    X_sample_all = clf.sample(train.size * 100)

    if spt==True: X_sample_all2 = clf_patch.sample(test.size)

    X_sample_no_split = np.array_split(X_sample_no, n_process, axis=0)
    
    def addingerror_process(q, order, (model, data, cov)):
        re = add_errors(model, data, cov)
        q.put((order, re))
    
    inputs = [ (X_sample_no_split[i], X_train[~cmass_mask], Xcov_train[~cmass_mask]) for i in range(n_process) ]
    
    q = Queue()
    Processes = [Process(target = addingerror_process, args=(q, z[0], z[1])) for z in zip(range(n_process), inputs)]
    
    for p in Processes: p.start()

    percent = 0.0
    result = []
    for p in Processes:
        result.append(q.get())
        percent += + 1./( n_process ) * 100
        sys.stdout.write("\r" + 'add noise to sample {:0.0f} %'.format( percent ))
        sys.stdout.flush()

    sys.stdout.write("\r" + 'add noise to sample {:0.0f} %\n'.format( 100 ))
    #result = [q.get() for p in Processes]
    result.sort()
    result = [r[1] for r in result ]

    noisy_X_sample_no = np.vstack( result[:n_process] ) #result[1]
    noisy_X_sample_cmass = add_errors( X_sample_cmass, X_train[cmass_mask], Xcov_train[cmass_mask] )
    if spt == True:
        noisy_X_sample_all2 = add_errors( X_sample_all2, X, Xcov )
        fitsio.write('result_cat/'+prefix+'noisy_X_all2_sample.fits' ,noisy_X_sample_all2)
        print 'noisy sample from all2 likelihood ','result_cat/'+prefix+'noisy_X_all2_sample.fits'


    fitsio.write('result_cat/'+prefix+'clean_X_sample.fits' ,X_sample_no)
    print 'clean sample from no likelihood ','result_cat/'+prefix+'clean_X_sample.fits'
    fitsio.write('result_cat/'+prefix+'clean_X_all_sample.fits' ,X_sample_all)
    print 'clean sample from all likelihood ','result_cat/'+prefix+'clean_X_all_sample.fits'
    fitsio.write('result_cat/'+prefix+'noisy_X_sample.fits' ,noisy_X_sample_no)
    print 'error-convolved sample from no likelihood ','result_cat/'+prefix+'noisy_X_sample.fits'
    """
    return test




def XD_fitting_old( data, pickleFileName = 'pickle/XD_fitting_test.pkl', n_cl = 10, suffix=None ):

    def XDDFitting( X_train, Xcov_train,  init_params=None, filename = None, n_cluster = 25 ):
        clf = None
        @pickle_results(filename, verbose = True)
        def compute_XD(n_clusters=n_cluster, init_params = init_params, n_iter=500, verbose=False):
            clf= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
            clf.fit(X_train, Xcov_train, init_params = init_params)
            return clf
        clf = compute_XD()
        return clf


    # giving initial mean, V, amp
    
    X, Xcov = mixing_color(data)
    #pickleFileName = 'pickle/gold_st82_20_XD_no2.pkl'
    clf = XDDFitting( X, Xcov, filename= pickleFileName, init_params = True, n_cluster= n_cl)

    #X_sample = clf.sample(data.size * 10)
    #noisy_X_sample = add_errors( X_sample, X, Xcov)
    #fitsio.write('result_cat/clean_X_sample'+suffix+'.fits' ,X_sample)
    #fitsio.write('result_cat/noisy_X_sample'+suffix+'.fits' ,noisy_X_sample)
    #print 'clean sample from all likelihood ','result_cat/clean_X_sample'+suffix+'.fits'
    #print 'noisy sample from all likelihood ','result_cat/noisy_X_sample'+suffix+'.fits'
    return clf




def assignCMASSProb( test, clf_cmass, clf_nocmass, cmass_fraction = None, 
                     mag = ['MAG_MODEL', 'MAG_DETMODEL'],
                     err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'],
                     filter = ['G', 'R', 'I'],
                     suffix = None ):
    
    print("calculate loglikelihood gaussian with multiprocessing module")
    
    try: X, Xcov = mixing_color( test, mag=mag, err=err, filter=filter )
    except ValueError : X, Xcov = mixing_color( test, mag=mag, err=err, filter=filter, suffix = '')
        
    from multiprocessing import Process, Queue
    # split data
    n_process = 12
    
    X_split = np.array_split(X, n_process, axis=0)
    Xcov_split = np.array_split(Xcov, n_process, axis=0)
    
    def logprob_process(q,  classname, order, xxx_todo_changeme2):
        (data, cov) = xxx_todo_changeme2
        re = classname.logprob_a(data, cov)
        result = logsumexp(re, axis = 1)
        q.put((order, result))
    
    inputs = [ (X_split[i], Xcov_split[i]) for i in range(n_process) ]
    
    q_cmass = Queue()
    #q_all = Queue()
    q_no = Queue()
    cmass_Processes = [Process(target = logprob_process, args=(q_cmass, clf_cmass, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    #all_Processes = [Process(target = logprob_process, args=(q_all, clf, z[0], z[1])) for z in zip(range(n_process), inputs)]
    no_Processes = [Process(target = logprob_process, args=(q_no, clf_nocmass, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    
    #sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
    
    for p in cmass_Processes: p.start()
    
    percent = 0.0
    result = []
    for p in cmass_Processes:
        result.append(q_cmass.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()
    
    #result = [q_cmass.get() for p in cmass_Processes]
    result.sort()
    cmass_logprob_a = np.hstack([np.array(r[1]) for r in result ])

    for p in no_Processes: p.start()
    
    resultno = []
    for p in no_Processes:
        resultno.append(q_no.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()
    
    resultno.sort()
    no_logprob_a = np.hstack([r[1] for r in resultno ])
    
    sys.stdout.write("\r" + 'multiprocessing {:0.0f} % \n'.format( 100 ))

    numerator =  np.exp(cmass_logprob_a) * cmass_fraction
    #denominator = np.exp(all_logprob_a)
    denominator = numerator + np.exp(no_logprob_a) * (1. - cmass_fraction)
    
    denominator_zero = denominator == 0
    CMASS_PROB = np.zeros( numerator.shape )
    CMASS_PROB[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]

    try:
        test = rf.append_fields(test, 'CMASS_PROB', CMASS_PROB)
        #test = rf.append_fields(test, 'cmassLogLikelihood', cmass_logprob_a)
        #test = rf.append_fields(test, 'noLogLikelihood', no_logprob_a)
    except ValueError:
        test['CMASS_PROB'] = CMASS_PROB
        #test['cmassLogLikelihood'] = cmass_logprob_a
        #test['noLogLikelihood'] = no_logprob_a
        
    return test

def assignELGProb( test, clf_cmass, clf_nocmass, cmass_fraction = None, 
                     mag = ['MAG_MODEL', 'MAG_DETMODEL'],
                     err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'],
                     filter = ['G', 'R', 'I', 'Z'],
                     suffix = None ):
    
    print("calculate loglikelihood gaussian with multiprocessing module")
    
    try: X, Xcov = mixing_color_elg( test, mag=mag, err=err, filter=filter )
    except ValueError : X, Xcov = mixing_color_elg( test, mag=mag, err=err, filter=filter, suffix = '')
        
    from multiprocessing import Process, Queue
    # split data
    n_process = 12
    
    X_split = np.array_split(X, n_process, axis=0)
    Xcov_split = np.array_split(Xcov, n_process, axis=0)
    
    def logprob_process(q,  classname, order, xxx_todo_changeme2):
        (data, cov) = xxx_todo_changeme2
        re = classname.logprob_a(data, cov)
        result = logsumexp(re, axis = 1)
        q.put((order, result))
    
    inputs = [ (X_split[i], Xcov_split[i]) for i in range(n_process) ]
    
    q_cmass = Queue()
    #q_all = Queue()
    q_no = Queue()
    cmass_Processes = [Process(target = logprob_process, args=(q_cmass, clf_cmass, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    #all_Processes = [Process(target = logprob_process, args=(q_all, clf, z[0], z[1])) for z in zip(range(n_process), inputs)]
    no_Processes = [Process(target = logprob_process, args=(q_no, clf_nocmass, z[0], z[1])) for z in zip(list(range(n_process)), inputs)]
    
    #sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
    
    for p in cmass_Processes: p.start()
    
    percent = 0.0
    result = []
    for p in cmass_Processes:
        result.append(q_cmass.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()
    
    #result = [q_cmass.get() for p in cmass_Processes]
    result.sort()
    cmass_logprob_a = np.hstack([np.array(r[1]) for r in result ])

    for p in no_Processes: p.start()
    
    resultno = []
    for p in no_Processes:
        resultno.append(q_no.get())
        percent += + 1./( n_process * 2 ) * 100
        sys.stdout.write("\r" + 'multiprocessing {:0.0f} %'.format( percent ))
        sys.stdout.flush()
    
    resultno.sort()
    no_logprob_a = np.hstack([r[1] for r in resultno ])
    
    sys.stdout.write("\r" + 'multiprocessing {:0.0f} % \n'.format( 100 ))

    numerator =  np.exp(cmass_logprob_a) * cmass_fraction
    #denominator = np.exp(all_logprob_a)
    denominator = numerator + np.exp(no_logprob_a) * (1. - cmass_fraction)
    
    denominator_zero = denominator == 0
    ELG_PROB = np.zeros( numerator.shape )
    ELG_PROB[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]

    try:
        test = rf.append_fields(test, 'ELG_PROB', ELG_PROB)
        #test = rf.append_fields(test, 'cmassLogLikelihood', cmass_logprob_a)
        #test = rf.append_fields(test, 'noLogLikelihood', no_logprob_a)
    except ValueError:
        test['ELG_PROB'] = ELG_PROB
        #test['cmassLogLikelihood'] = cmass_logprob_a
        #test['noLogLikelihood'] = no_logprob_a
        
    return test

def keepSDSSgoodregion( data ):
    
    Tags = ['bad_field_mask', 'unphot_mask', 'bright_star_mask', 'rykoff_bright_star_mask','collision_mask', 'centerpost_mask']
    
    mask = np.ones( data.size, dtype = bool )
    
    for tag in Tags:
        print(tag)
        mask = mask * (data[tag] == 0)
    
    print('masked objects ', mask.size - np.sum(mask))
    
    data_tags = list(data.dtype.names)
    reduced_tags = []
    for tag in data_tags:
        if tag not in Tags: reduced_tags.append(tag)
    reduced_data = data[reduced_tags]
    return reduced_data[mask]





def probability_calibration( des = None, cmass_des = None, matchID = 'COADD_OBJECTS_ID', prefix = 'test' ):
    from utils import matchCatalogs
    pth_bin, pstep = np.linspace(0.0, 1., 50, retstep = True)
    pbin_center = pth_bin[:-1] + pstep/2.
    
    cmass_des, _ = matchCatalogs( des, cmass_des, tag=matchID)
    
    N_cmass, _ = np.histogram( cmass_des['EachProb_CMASS'], bins = pth_bin, density=False )
    N_all, _ = np.histogram( des['EachProb_CMASS'], bins = pth_bin, density=False )
    
    prob = N_cmass * 1./N_all
    real = N_cmass * 1./cmass_des.size
    
    #for i, j, k in zip(pbin_center, prob, N_cmass): print i, j, k
    
    fig,ax = plt.subplots(figsize = (5,4))
    ax.plot(pbin_center, prob,label='true fraction' )
    ax.plot(pbin_center, real,label='total fraction' )
    ax.plot([0,1], [0,1], 'k--')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('model prob')
    ax.set_ylabel('true fraction')
    ax.legend(loc='best')
    fig.savefig('com_pur_results/'+prefix + '_probability_calibration.png')
    print('save fig: com_pur_results/'+prefix+'_probability_calibration.png')



def _resampleWithPth( des ):
    
    pth_bin, step = np.linspace(0.01, 1.0, 100, endpoint = True, retstep = True)
    pcenter = pth_bin + step/2.
    
    dmass = [None for i in range(pcenter.size)]
    
    for i in range(pcenter.size-1):
        mask = (des['EachProb_CMASS'] >= pth_bin[i])&(des['EachProb_CMASS'] < pth_bin[i]+step)
        cat = des[mask]
        dmass[i] = np.random.choice( cat, size = np.around(pcenter[i] * cat.size) )
        print(pth_bin[i], pth_bin[i]+step, pcenter[i], dmass[i].size * 1./cat.size, dmass[i].size)
    
    
    i = i+1
    mask = (des['EachProb_CMASS'] >= pth_bin[i])
    cat = des[mask]
    dmass[i] = np.random.choice( cat, size = int( 0.75 * cat.size) )
    print(pth_bin[i], pth_bin[i]+step, 0.75, dmass[i].size * 1./cat.size, dmass[i].size)
    
    dmass = np.hstack(dmass)
    return dmass


def resamplewithCmassHist( des, cmass_des, des_st82 = None, pstart = 0.01 ):

    from utils import matchCatalogs
    cmass_des, _ = matchCatalogs( des_st82, cmass_des, tag ='COADD_OBJECTS_ID')

    pth_bin, step = np.linspace(pstart, 3.0, 300, endpoint = True, retstep = True)
    pcenter = pth_bin + step/2.
    pcenter = pcenter[:-1]
    N_cmass, _ = np.histogram( cmass_des['EachProb_CMASS'], bins = pth_bin, density=False )
    N_all, _ = np.histogram( des_st82['EachProb_CMASS'], bins = pth_bin, density=False )
    n = N_cmass * 1./N_all
    #n = n * 1./n[-1]
    
    desInd = np.digitize( des['EachProb_CMASS'], bins= pth_bin )

    fraction = [None for i in range(pcenter.size)]
    sampled_dmass = []
    for i in range(1, pth_bin.size-1):
        cat = des[ desInd == i ]
        try :
            sd = np.random.choice( cat, size = int(np.around(n[i] * cat.size)) )
            sampled_dmass.append(sd)
            fraction[i] = sd.size * 1./cat.size
        except ValueError:
            sd = np.random.choice( des, size = 0)
            sampled_dmass.append(sd)
            fraction[i] = 0.0

    sampled_dmass = np.hstack(sampled_dmass)


    """
    fig, ax = plt.subplots()
    
    #ax.plot( pcenter, pcurve2, color = 'grey', label = 'Ashley1', linestyle = '--')
    #ax.plot( np.insert(pcenter, 0, pcenter[0]), np.insert(n, 0, 0.0), color = 'grey',linestyle = '--', label='straight')
    ax.plot( pcenter, fraction, color = 'red', linestyle = '--' )
    ax.plot( pcenter, n, color = 'grey',linestyle = '--', label='cmass curve')


    pbin, st = np.linspace(0.0, 3., 100, endpoint=False, retstep = True)
    pc = pbin + st/2.
    
    #N, edge = np.histogram( des['EachProb_CMASS'], bins = pbin)
    weights = np.ones_like(des['EachProb_CMASS'])/len(des['EachProb_CMASS'])
    ax.hist( des['EachProb_CMASS'], bins = pbin, weights = weights * 20, alpha = 0.3, color = 'green')
    #ax.plot( pc[:-1], N * 1./des.size,  )
    ax.set_xlim(0,3)
    ax.set_ylim(0,1.1)
    ax.set_ylabel('fraction')
    ax.set_xlabel('p_threshold')
    ax.legend(loc='best')
    fig.savefig('figure/selection_prob_2.png')
    print 'figure to ', 'figure/selection_prob_2.png'
    """
    return sampled_dmass



def resampleWithPth__( des ):
    return 0



def resampleWithPth( des, pstart = 0.01, pmax = 1.0, pbins = 201 ):
    
    pmin = 0.1
    pmax = pmax # should be adjused with different samples
    p1stbin = 0.1
    
    
    #pth_bin, step = np.linspace(pstart, 1.0, pbins, endpoint = True, retstep = True)
    #pcenter = pth_bin[:-1] + step/2.
    
    pbinsmall, steps = np.linspace(0, 0.01, 101,retstep=1 )
    pbinbig, stepb = np.linspace(0.01, 1, 199, retstep=1 )
    pth_bin = np.hstack([pbinsmall, pbinbig[1:]])
    pcenters = pbinsmall[:-1] + steps/2.
    pcenterb = pbinbig[:-1] + stepb/2.
    pcenter = np.hstack([pcenters, pcenterb[1:]])

    # ellipse
    #pcurve = (pmax-pmin) * np.sqrt( 1 - (pcenter - 1)**2 * 1./(1-p1stbin)**2 ) + pmin
    #pcurve[ pcenter < pmin ] = 0.0
    #pcurve[ np.argmax(pcurve)+1:] = pmax
    
    # straight
    #pcurve2 = np.zeros(pth_bin.size)
    #pcurve_front = np.linspace(pstart, 1.0, 200, endpoint = True)
    pn = lambda x : x
    #pcurve2[:pcurve_front.size] = pcurve_front
    #pcurve2[pcurve_front.size:] = pmax

    straight_dmass = [None for i in range(pcenter.size)]
    st_fraction = np.zeros(pcenter.size)
    
    desInd = np.digitize( des['EachProb_CMASS'], bins= pth_bin )
    
    for i in range(pcenter.size):
        #mask = (des['EachProb_CMASS'] >= pth_bin[i]) & (des['EachProb_CMASS'] < pth_bin[i]+step)
        #cat = des[mask]
        cat = des[ desInd == i+1]
        
        try :
            #ellipse_dmass[i] = np.random.choice( cat, size = int(np.around(pcurve[i] * cat.size)) )
            #straight_dmass[i] = np.random.choice( cat, size = int(np.around(pcurve2[i] * cat.size)) )
            straight_dmass[i] = np.random.choice( cat, size = int(np.around(pn( pcenter[i] ) * cat.size)) )
            #print pth_bin[i], pcenter[i]
            #print pn(pcenter[i]) 
            #el_fraction[i] = ellipse_dmass[i].size * 1./cat.size
            st_fraction[i] = straight_dmass[i].size * 1./cat.size
        
        except ValueError:
            #ellipse_dmass[i] = np.random.choice( des, size = 0)
            straight_dmass[i] = np.random.choice( des, size = 0)
            #el_fraction[i] = 0
            st_fraction[i] = 0

        #fraction.append( np.sum(mask) * 1./des.size )
        #print pth_bin[i], pth_bin[i]+step, pcurve[i], dmass[i].size * 1./cat.size, dmass[i].size
    """
    i = i+1
    mask = (des['EachProb_CMASS'] >= pth_bin[i])
    cat = des[mask]
    ellipse_dmass[i] = np.random.choice( cat, size = int(np.around(pmax * cat.size)) )
    straight_dmass[i] = np.random.choice( cat, size = int(np.around(pmax * cat.size)) )
    #print pth_bin[i], pth_bin[i]+step, 0.75, dmass[i].size * 1./cat.size, dmass[i].size
    """
    #ellipse_dmass = np.hstack(ellipse_dmass)
    straight_dmass = np.hstack(straight_dmass)
    
    # plotting selection prob----------
    """
    fig, ax = plt.subplots()

    #ax.plot( pcenter, pcurve2, color = 'grey', label = 'Ashley1', linestyle = '--')
    ax.plot( pcenter , pcurve2, color = 'grey',linestyle = '-', label='selection probability')
    #ax.plot( np.insert(pcenter, 0, pcenter[0]), np.insert(pcurve, 0, 0.0), color = 'grey' , label = 'ellipse')
    #ax.plot( pcenter, el_fraction, color = 'red', linestyle = '--' )
    ax.plot( pcenter[:-1], st_fraction, color = 'blue',linestyle = '--' , label='sample hist' )
    
    pbin, st = np.linspace(0.0, 1.1, 100, endpoint=False, retstep = True)
    pc = pbin + st/2.
    
    #N, edge = np.histogram( des['EachProb_CMASS'], bins = pbin)
    weights = np.ones_like(des['EachProb_CMASS'])/len(des['EachProb_CMASS'])
    ax.hist( des['EachProb_CMASS'], bins = pbin, weights = weights * 20, alpha = 0.3, color = 'green')
    #ax.plot( pc[:-1], N * 1./des.size,  )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    ax.set_ylabel('fraction')
    ax.set_xlabel('membership probability')
    ax.legend(loc='best')
    fig.savefig('figure/selection_prob')
    print 'figure to ', 'figure/selection_prob.png'
    """
    return straight_dmass, None


