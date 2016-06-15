#!/usr/bin/env python
import sys
from time import time
import os
import numpy as np
from astroML.decorators import pickle_results
#from astroML.density_estimation import XDGMM
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
import numpy.lib.recfunctions as rf
from scipy import linalg
#from __future__ import print_function, division
from sklearn.mixture import GMM
#from ..utils import logsumexp, log_multivariate_gaussian, check_random_state




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

    def fit(self, X, Xerr, init_means=None, init_covars = None, init_weights = None, R=None):
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

        gmm = GMM(self.n_components, n_iter=10, covariance_type='full',
                  random_state=self.random_state).fit(X)
        
        print 'sum of weights',np.sum(gmm.weights_)
        print gmm.weights_
        
        if init_means is not None:
            gmm.means_[:init_means.shape[0],:] = init_means
            gmm.covars_[:init_covars.shape[0],:,:] = init_covars
            gmm.weights_[:init_weights.size] = init_weights

        self.mu = gmm.means_
        self.alpha = gmm.weights_
        self.V = gmm.covars_
        
        logL = self.logL(X, Xerr)

        for i in range(self.n_iter):
            t0 = time()
            self._EMstep(X, Xerr)
            logL_next = self.logL(X, Xerr)
            t1 = time()

            if self.verbose:
                print("%i: log(L) = %.5g" % (i + 1, logL_next))
                print("    (%.2g sec)" % (t1 - t0))
              
            if logL_next < logL + self.tol:
                break
            logL = logL_next

        return self

    def logprob_a(self, X, Xerr):
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

        return log_multivariate_gaussian(X, self.mu, T) + np.log(self.alpha)

    def logL(self, X, Xerr):
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
        return np.sum(logsumexp(self.logprob_a(X, Xerr), -1))

    def _EMstep(self, X, Xerr):
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
        Tinv = np.array([linalg.inv(T[i])
                         for i in range(T.shape[0])]).reshape(Tshape)
        T = T.reshape(Tshape)

        #------------------------------------------------------------
        # evaluate each mixture at each point
        N = np.exp(log_multivariate_gaussian(X, self.mu, T, Vinv=Tinv))

        #------------------------------------------------------------
        # E-step:
        #  compute q_ij, b_ij, and B_ij
        q = (N * self.alpha) / np.dot(N, self.alpha)[:, None]

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








def SDSS_cmass_criteria(sdss, prior=None):
    
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





def SDSS_LOWZ_criteria(sdss):
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



def priorCut(data, sdss=None):
    modelmag_g_des = data['MAG_DETMODEL_G'] - data['XCORR_SFD98_G']
    modelmag_r_des = data['MAG_DETMODEL_R'] - data['XCORR_SFD98_R']
    modelmag_i_des = data['MAG_DETMODEL_I'] - data['XCORR_SFD98_I']
    cmodelmag_i_des = data['MAG_MODEL_I'] - data['XCORR_SFD98_I']
    fib2mag_des = data['MAG_APER_4_I']

    cut = ((cmodelmag_i_des > 17) &
           (cmodelmag_i_des < 22.) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) &
           ((modelmag_r_des - modelmag_i_des ) > 0.0 ) &
           ((modelmag_g_des - modelmag_r_des ) > 0.0 ) &
            ((modelmag_g_des - modelmag_r_des ) < 2.5 ) &
           (fib2mag_des < 24.0 ) #&
           #(data['MAG_APER_4_I'] < 24.0 )
             )
    print 'prior cut ',np.sum(cut)
    return data[cut], cut



def divide_bins( cat, Tag = 'Z', min = 0.2, max = 1.2, bin_num = 5, TagIndex = None ):
    values, step = np.linspace(min, max, num = bin_num+1, retstep = True)
    
    binkeep = []
    binned_cat = []
    
    column = cat[Tag]
    if TagIndex is not None: column = cat[Tag][:,TagIndex]
    
    for i in range(len(values)-1) :
        bin = (column >= values[i]) & (column < values[i+1])
        binned_cat.append( cat[bin] )
        binkeep.append(bin)
    
    bin_center = values[:-1] + step/2.
    return bin_center, binned_cat, binkeep





def im3shape_galprof_mask( im3shape, fulldes ):
    import numpy.lib.recfunctions as rf
    """ add mode to distingush galaxy profiles used in im3shape """
    """ return only full des """
    
    im3galprofile = np.zeros(len(fulldes), dtype=np.int32)
    
    expcut = (im3shape['BULGE_FLUX'] == 0)
    devcut = (im3shape['DISC_FLUX'] == 0)
    
    des_exp = im3shape[expcut]
    des_dev = im3shape[devcut]
    
    expID = des_exp['COADD_OBJECTS_ID']
    devID = des_dev['COADD_OBJECTS_ID']
    fullID = fulldes['COADD_OBJECTS_ID']
    
    expmask = np.in1d(fullID, expID)
    devmask = np.in1d(fullID, devID)
    
    im3galprofile[expmask] = 1
    im3galprofile[devmask] = 2
    
    data = rf.append_fields(fulldes, 'IM3_GALPROF', im3galprofile)
    
    print np.sum(expmask), np.sum(devmask)
    return data



def spatialcheck(data, convert = None):
    
    if convert is True:
        ra = data['ALPHAWIN_J2000_DET'] + 360
        dec = data['DELTAWIN_J2000_DET']
        
        cut = ra < 180
        ra1 = ra[cut] + 360
        ra[cut] = ra1
    
    if convert is None :
        ra = data['RA']
        dec = data['DEC']
    
    fig, ax = plt.subplots(1,1,figsize = (7,7))
    ax.plot(ra, dec, 'b.', alpha = 0.5 )
    #ax.plot(data2['RA'], data2['DEC'], 'r.', alpha = 0.5 )
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    fig.savefig('spatialtest')
    print 'figsave : spatialtest.png'




def log_multivariate_gaussian(x, mu, V, Vinv=None, method=1):
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
            Vinv = np.array([linalg.inv(V[i])
                             for i in range(V.shape[0])]).reshape(Vshape)
        else:
            assert Vinv.shape == Vshape

        logdet = np.log(np.array([linalg.det(V[i])
                                  for i in range(V.shape[0])]))
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

    #X_divisions = tuple(X[indices[N[i]:N[i + 1]]]
    #                    for i in range(len(fractions)))
    #y_divisions = tuple(y[indices[N[i]:N[i + 1]]]
    #                    for i in range(len(fractions)))


    #return X_divisions, y_divisions
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




def AddingReddening(cat):
    import numpy.lib.recfunctions as rf   
    from suchyta_utils.y1a1_slr_shiftmap import SLRShift

    zpfile = '/n/des/lee.5922/data/systematic_maps/y1a1_wide_slr_wavg_zpshift2.fit'
    slrshift = SLRShift(zpfile, fill_periphery=True, balrogprint=None)
    offsets_g = slrshift.get_zeropoint_offset('g',cat['RA'],cat['DEC'],interpolate=True) * -1.
    offsets_r = slrshift.get_zeropoint_offset('r',cat['RA'],cat['DEC'],interpolate=True) * -1.
    offsets_i = slrshift.get_zeropoint_offset('i',cat['RA'],cat['DEC'],interpolate=True) * -1.
    offsets_z = slrshift.get_zeropoint_offset('z',cat['RA'],cat['DEC'],interpolate=True) * -1.

    cat = rf.append_fields(cat, 'XCORR_SFD98_G', offsets_g)    
    cat = rf.append_fields(cat, 'XCORR_SFD98_R', offsets_r)
    cat = rf.append_fields(cat, 'XCORR_SFD98_I', offsets_i)
    cat = rf.append_fields(cat, 'XCORR_SFD98_Z', offsets_z)

    return cat



def mixing_color(data, sdss = None, cmass = None):
    
    if sdss is None and cmass is None:
        des_g = data['MAG_DETMODEL_G'] - data['XCORR_SFD98_G']
        des_r = data['MAG_DETMODEL_R'] - data['XCORR_SFD98_R']
        des_i = data['MAG_DETMODEL_I'] - data['XCORR_SFD98_I']
        des_z = data['MAG_DETMODEL_Z'] - data['XCORR_SFD98_Z']
        des_ci = data['MAG_MODEL_I'] - data['XCORR_SFD98_I']
        des_cr = data['MAG_MODEL_R'] - data['XCORR_SFD98_R']
        #flux_g = data['FLUX_DETMODEL_G']
        #flux_r = data['FLUX_DETMODEL_R']
        #flux_i = data['FLUX_DETMODEL_I']
        #flux_z = data['FLUX_DETMODEL_Z']
        #fluxmodel_r = data['FLUX_MODEL_R']
        #fluxmodel_i = data['FLUX_MODEL_I']
        
        
        X = np.vstack([des_cr, des_ci, des_g, des_r, des_i, des_z ]).T
        #X = np.vstack([des_ci, des_g, des_r, des_i]).T
        Xerr = np.vstack([data['MAGERR_MODEL_R'],
                          data['MAGERR_MODEL_I'],
                          #data['FLUXERR_MODEL_R'],
                          #data['FLUXERR_MODEL_I'],
                          data['MAGERR_DETMODEL_G'],
                          data['MAGERR_DETMODEL_R'],
                          data['MAGERR_DETMODEL_I'],
                          data['MAGERR_DETMODEL_Z'],
                          #data['FLUXERR_DETMODEL_G'],
                          #data['FLUXERR_DETMODEL_R'],
                          #data['FLUXERR_DETMODEL_I'],
                          #data['FLUXERR_DETMODEL_Z'],
                          
                          ]).T

    # mixing matrix
    
    
    
    elif sdss is not None:
    
        des_g = data['MODELMAG_G'] - data['EXTINCTION_G']
        des_r = data['MODELMAG_R'] - data['EXTINCTION_R']
        des_i = data['MODELMAG_I'] - data['EXTINCTION_I']
        des_ci = data['CMODELMAG_I'] - data['EXTINCTION_I']
        
        #X = np.vstack([des_cr, des_ci, des_g, des_r, des_i, des_z]).T
        X = np.vstack([des_ci, des_g, des_r, des_i]).T
        Xerr = np.vstack([#data['MAGERR_MODEL_R'] ,
                          data['CMODELMAGERR_I'],
                          data['MODELMAGERR_G'],
                          data['MODELMAGERR_R'],
                          data['MODELMAGERR_I'],
                          #data['MAGERR_DETMODEL_Z']
                          ]).T
    
    elif cmass is not None:
        
        des_g = data['MODELMAG'][:,1] - data['EXTINCTION'][:,1]
        des_r = data['MODELMAG'][:,2] - data['EXTINCTION'][:,2]
        des_i = data['MODELMAG'][:,3] - data['EXTINCTION'][:,3]
        des_z = data['MODELMAG'][:,4] - data['EXTINCTION'][:,4]
        des_ci = data['CMODELMAG'][:,3] - data['EXTINCTION'][:,3]
        des_cr = data['CMODELMAG'][:,2] - data['EXTINCTION'][:,2]
        
        X = np.vstack([des_cr, des_ci, des_g, des_r, des_i, des_z]).T
        #X = np.vstack([des_ci, des_g, des_r, des_i]).T
        Xerr = np.vstack([data['CMODELMAGERR'][:,2] ,
                          data['CMODELMAGERR'][:,3],
                          data['MODELMAGERR'][:,1],
                          data['MODELMAGERR'][:,2],
                          data['MODELMAGERR'][:,3],
                          data['MODELMAGERR'][:,4]
                          ]).T

    # mixing matrix
    W = np.array([
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],    # i cmag
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],   # r-i
                  [0, 0, 0, 0, 1, -1]])  # i-z
                  
                
    X = np.dot(X, W.T)
    """
    if sdss is not None or cmass is not None:
        return X, Xcov
    
    if sdss is None:
    
    """
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))
    return X, Xcov




def MachineLearningClassifier( cmass, lowz,  train = None, test = None):
    
    from sklearn.neighbors import KNeighborsClassifier
    from astroML.classification import GMMBayes
    import esutil

    train, _ = priorCut(train)
    test, _ = priorCut(test)
    
    # making cmass and lowz mask
    h = esutil.htm.HTM(10)
    matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(train['RA'],train['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass = np.zeros( train.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask = true_cmass == 1
    
    m_des, m_sdss, d12 = h.match(test['RA'],test['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass = np.zeros( test.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask_test = true_cmass == 1
    
    print 'num of cmass/lowz', np.sum(cmass_mask)
    
    
    data = np.hstack(( train, test ))
    
    # stack DES data
    X, Xcov = align_catalogs_for_xdc(data)
        
    X_train = X[ :len(train), :]
    X_test = X[-len(test):, :]
    
    y_train = np.zeros( train.size, dtype=int)
    y_train[cmass_mask] = 1
    y_test = np.zeros( test.size, dtype=int)
    y_test[cmass_mask_test] = 1


    def compute_kNN( k_neighbors_max = 10 ):
        classifiers = []
        predictions = []
        kvals = np.arange(1,k_neighbors_max)
        
        print 'kNN'
        
        for k in kvals:
            classifiers.append([])
            predictions.append([])
            
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pur = np.sum(y_pred * y_test) *1./np.sum(y_pred)
            com = np.sum(y_pred * y_test) *1./np.sum(y_test)
            print 'n_neighbor, com/pur', k, com, pur
            classifiers[-1].append(clf)
            predictions[-1].append(y_pred)
        
        return classifiers, prediction
    def compute_GMMB():
        
        print 'GMMB'
        classifiers = []
        predictions = []
        Ncomp = np.arange(1,20,2)

        for nc in Ncomp:
            classifiers.append([])
            predictions.append([])
            clf = GMMBayes(nc, min_covar=1E-5, covariance_type='full')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pur = np.sum(y_pred * y_test) *1./np.sum(y_pred)
            com = np.sum(y_pred * y_test) *1./np.sum(y_test)
            print 'Ncomp, com/pur', nc, com, pur
            classifiers[-1].append(clf)
            predictions[-1].append(y_pred)
        
        return classifiers, predictions

    #_, predictions_kNN = compute_kNN(k_neighbors_max = 10)
    #_, predictions_GMMB = compute_GMMB()


    clf_kNN = KNeighborsClassifier(n_neighbors= 7)
    clf_kNN.fit(X_train, y_train)
    y_pred_kNN = clf_kNN.predict(X_test)
    
    print 'kNN  pur/com', np.sum(y_pred_kNN * y_test) *1./np.sum(y_pred_kNN), np.sum(y_pred_kNN * y_test) *1./np.sum(y_test)

    clf_GMMB = GMMBayes(11, min_covar=1E-5, covariance_type='full')
    clf_GMMB.fit(X_train, y_train)
    y_pred_GMMB = clf_GMMB.predict(X_test)

    print 'GMMB pur/com', np.sum(y_pred_GMMB * y_test) *1./np.sum(y_pred_GMMB), np.sum(y_pred_GMMB * y_test) *1./np.sum(y_test)

    return test[y_pred_kNN == 1], test[y_pred_GMMB == 1]



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
    for i in xrange(model_data.shape[0]):
        noise[i,:] = (np.random.multivariate_normal(model_data[i,:]*0.,model_covars[i,:,:]))
    
    noisy_data = model_data + noise
    return noisy_data



def initialize_deconvolve(data, covars, n_components = 50):
    #deconv_pars = {'covars':covars, # (ndata, dy, dy)
    #               'Projection':R}  # (ndata,dy, dx)
    # optional additions include:
    # -- weights: weights or log-weights per point (logweight can be turned on)
    # -- fix[amp/mean/covar]: to fix any of the GMM components
    # -- w: covariance regularization parameter (of the conjugate prior????)
    # -- splitnmerge: depth to go down the splitnmerge path
    # -- maxsm:  use the max number of split 'n' merge steps, K*(K-1)*(K-2) /2
    # --------------------------------------------------
    # --------------------------------------------------
    

    # pick some points randomly from xsim,ysim
    indices = np.random.choice(np.arange(data.shape[0]),n_components)
    mean_init = data[indices,:]
    cov_init_list = []
    for i in xrange(indices.size):
        cov_init_list.append(covars[i,:,:])
    covars_init = np.array(cov_init_list)
    amp_init = np.ones(n_components) * ( 1./n_components )
    # set the initial covars to be diagonal, equal to the reported errors.
    return amp_init, mean_init, covars_init


def XDGMM_model(cmass, lowz, p_threshold = 0.9, train = None, test = None, reverse=None, pickleFileName = [ 'XD_all_default.pkl', 'XD_cmass_default.pkl', 'XD_nocmass_default.pkl' ] ):

    import esutil
    import numpy.lib.recfunctions as rf
    from multiprocessing import Process, Queue
    
    
    if reverse is True: pass
        #train, _ = SDSS_cmass_criteria(train, prior=True)
        #test, _ = SDSS_cmass_criteria(test, prior=True)
    
    
    else:
        train, _ = priorCut(train)
        test, _ = priorCut(test)

    # making cmass and lowz mask
    h = esutil.htm.HTM(10)
    matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(train['RA'],train['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass = np.zeros( train.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask = true_cmass == 1

    m_des, m_sdss, d12 = h.match(train['RA'], train['DEC'], lowz['RA'],lowz['DEC'],matchDist,maxmatch=1)
    true_lowz = np.zeros( train.size, dtype = int)
    true_lowz[m_des] = 1
    lowz_mask = true_lowz == 1
    
    print 'num of cmass/lowz in train', np.sum(cmass_mask), np.sum(lowz_mask)
    
    
    
    # test mask
    m_des, m_sdss, d12 = h.match(test['RA'],test['DEC'], cmass['RA'], cmass['DEC'], matchDist,maxmatch=1)
    true_cmass = np.zeros( test.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask_test = true_cmass == 1
    
    m_des, m_sdss, d12 = h.match(test['RA'], test['DEC'], lowz['RA'],lowz['DEC'], matchDist,maxmatch=1)
    true_lowz = np.zeros( test.size, dtype = int)
    true_lowz[m_des] = 1
    lowz_mask_test = true_lowz == 1
    
    y_test = np.zeros(( test.size, 2 ), dtype=bool)
    y_test[:,0][cmass_mask_test] = 1
    y_test[:,1][lowz_mask_test] = 1
    
    print 'num of cmass/lowz in test', np.sum( y_test[:,0][cmass_mask_test]), np.sum( y_test[:,1][lowz_mask_test])

    data = np.hstack(( train, test ))
    
    # stack DES data

    if reverse is True:
        X, Xcov = mixing_color(data, cmass = True)

    else :
        X, Xcov = mixing_color(data)
    
    y_train = np.zeros(( train.size, 2 ), dtype=int)
    y_train[:,0][cmass_mask] = 1
    y_train[:,1][lowz_mask] = 1

    X_train = X[ :len(train), :]
    X_test = X[-len(test):, :]
    Xcov_train = Xcov[ :len(train), :]
    Xcov_test = Xcov[-len(test):, :]

    print 'train/test', len(X_train), len( X_test )
    
    # train sample XD convolution ----------------------------------
    
    # finding optimal n_cluster

    from sklearn.mixture import GMM
    
    N = np.arange(1, 50, 2)
    
    def FindOptimalN( N, data, pickleFileName = None):
        @pickle_results( pickleFileName )
        def compute_GMM( N, covariance_type='full', n_iter=1000):
            models = [None for n in N]
            for i in range(len(N)):
                sys.stdout.write('.')
                models[i] = GMM(n_components=N[i], n_iter=n_iter,
                                covariance_type=covariance_type)
                models[i].fit(data)
            return models
        models = compute_GMM(N)
        AIC = [m.aic(data) for m in models]
        BIC = [m.bic(data) for m in models]
        i_best = np.argmin(BIC)
        gmm_best = models[i_best]
        print 'end'
        print "best fit converged:", gmm_best.converged_
        print "BIC: n_components =  %i" % N[i_best]
        return N[i_best], AIC, BIC
    
    
    pickleFileName_GMM = ['GMM_bic_cmass.pkl', 'GMM_bic_all.pkl']
    if reverse is True : pickleFileName_GMM = ['reverse_GMM_bic_cmass.pkl', 'reverse_GMM_bic_all.pkl']
    n_cl_cmass, aic_cmass, bic_cmass = FindOptimalN( N, X_train[cmass_mask], pickleFileName = pickleFileName_GMM[0])
    rows = np.random.choice(np.arange(X_train.shape[0]), 10 * len(X_train[cmass_mask]))
    n_cl_all, aic_all, bic_all  = FindOptimalN( N, X_train[rows,:], pickleFileName = pickleFileName_GMM[1])
    #n_cl_no, aic_no, bic_no = FindOptimalN( N, X_train[~cmass_mask], pickleFileName = 'GMM_bic_no.pkl')
    
    DAT = np.column_stack(( N, bic_cmass, bic_all, aic_cmass, aic_all))
    np.savetxt('BIC.txt', DAT, delimiter = ' ', header = 'N, bic_cmass, bic_all,  aic_cmass, aic_all' )
    print 'save to BIC.txt'

    #n_cl_cmass = 10
    #n_cl_no = 25
    #n_cl_all = 25
    # ----------------------------------
  
    def XDDFitting( X_train, Xcov_train, init_means=None, init_covars = None, init_weights = None, filename = None, n_cluster = 25 ):
        clf = None
        @pickle_results(filename, verbose = True)
        def compute_XD(n_clusters=n_cluster, n_iter=500, verbose=True):
            clf= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
            clf.fit(X_train, Xcov_train, init_means=init_means, init_covars = init_covars, init_weights = init_weights)
            return clf
        clf = compute_XD()
        return clf


    #init_weightsC, init_meansC, init_covarsC = initialize_deconvolve(X_train[cmass_mask].data,  Xcov_train[cmass_mask], n_components = n_cl_cmass)


    # calculates CMASS fits
    clf_cmass = XDDFitting( X_train[cmass_mask], Xcov_train[cmass_mask], filename=pickleFileName[1], n_cluster=n_cl_cmass)

    cV = clf_cmass.V
    cmu = clf_cmass.mu
    calpha = clf_cmass.alpha
    #init_weights, init_means, init_covars = initialize_deconvolve(X_train.data,  Xcov_train, n_components = n_cl_all)
    
    from sklearn.mixture import GMM
    gmm = GMM(n_cl_all, n_iter=10, covariance_type='full',
                  random_state= None).fit(X_train)
    init_weights = gmm.weights_
    init_means = gmm.means_
    init_covars = gmm.covars_
    
    init_weights = np.hstack((init_weights, calpha * np.sum(cmass_mask) * 1./test.size ))
    init_weights = init_weights / np.sum(init_weights)
    init_means = np.vstack(( init_means, cmu ))
    init_covars = np.vstack(( init_covars, cV ))
    
    rows = np.random.choice(np.arange(X_train.shape[0]), 5 * len(X_train[cmass_mask]))
    clf = XDDFitting( X_train[rows, :], Xcov_train[rows,:,:], filename= pickleFileName[0], n_cluster= n_cl_all)
    #clf = XDDFitting( X_train[rows,:], Xcov_train[rows,:,:], init_means=init_means, init_covars = init_covars, init_weights = init_weights, filename= pickleFileName[0], n_cluster= n_cl_all + n_cl_cmass)


    """
    rows = np.random.choice(np.arange(X_train[~cmass_mask].shape[0]), 10 * len(X_train[cmass_mask]))
    init_weightsN, init_meansN, init_covarsN = initialize_deconvolve(X_train[~cmass_mask].data,  Xcov_train[~cmass_mask], n_components = n_cl_no)
    
    clf_nocmass = XDDFitting( X_train[~cmass_mask][rows,:], Xcov_train[~cmass_mask][rows,:,:], init_means=init_meansN, init_covars = init_covarsN, init_weights = init_weightsN, filename= pickleFileName[2], n_cluster=n_cl_no)
    """

    # logprob_a ------------------------------------------


    print "calculate loglikelihood gaussian with multiprocessing module"

    from multiprocessing import Process, Queue
    # split data
    n_process = 12
    X_test_split = np.array_split(X_test, n_process, axis=0)
    Xcov_test_split = np.array_split(Xcov_test, n_process, axis=0)
    
    
    print 'multiprocessing...',
    
    def logprob_process(q,  classname, order,(data, cov)):
        re = classname.logprob_a(data, cov)
        sys.stdout.write('...')
        result = logsumexp(re, axis = 1)
        q.put((order, result))
        
    inputs = [ (X_test_split[i], Xcov_test_split[i]) for i in range(n_process) ]
    
    q_cmass = Queue()
    q_all = Queue()
    q_no = Queue()
    cmass_Processes = [Process(target = logprob_process, args=(q_cmass, clf_cmass, z[0], z[1])) for z in zip(range(n_process), inputs)]
    all_Processes = [Process(target = logprob_process, args=(q_all, clf, z[0], z[1])) for z in zip(range(n_process), inputs)]

    #no_Processes = [Process(target = logprob_process, args=(q_no, clf_nocmass, z[0], z[1])) for z in zip(range(n_process), inputs)]
    
    for p in cmass_Processes: p.start()
    result = [q_cmass.get() for p in cmass_Processes]
    result.sort()
    cmass_logprob_a = np.hstack([np.array(r[1]) for r in result ])
    
    
    for p in all_Processes: p.start()
    resultall = [q_all.get() for p in all_Processes]
    resultall.sort()
    all_logprob_a = np.hstack([r[1] for r in resultall ])
    
    """
    for p in no_Processes: p.start()
    resultno = [q_no.get() for p in no_Processes]
    resultno.sort()
    no_logprob_a = np.hstack([r[1] for r in resultno ])
    """
    print 'end'
    

    numerator =  np.exp(cmass_logprob_a) * np.sum(y_train[:,0]) * 1.
    denominator = np.exp(all_logprob_a) * len(X_train)
    #denominator = numerator + np.exp(no_logprob_a) * (len(X_train) - np.sum(y_train[:,0]))

    denominator_zero = denominator == 0
    EachProb_CMASS = np.zeros( numerator.shape )
    EachProb_CMASS[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]
    EachProb_CMASS[EachProb_CMASS > 1.0] = 1.0
    
    print 'EachProb max', EachProb_CMASS.max()
    
    test = rf.append_fields(test, 'EachProb_CMASS', EachProb_CMASS)
    # -----------------------------------------------
    
    X_sample_cmass = clf_cmass.sample(train[cmass_mask].size)
    X_sample_all = clf.sample(train.size )
    #X_sample_no = clf_nocmass.sample(train[~cmass_mask].size)


    X_sample_all_split = np.array_split(X_sample_all, n_process, axis=0)
    #X_sample_no_split = np.array_split(X_sample_no, n_process, axis=0)
    
    print 'add noise...'

    def addingerror_process(q, order, (model, data, cov)):
        re = add_errors(model, data, cov)
        sys.stdout.write('...')
        q.put((order, re))
    
    #inputs = [(X_sample_cmass, X_train[cmass_mask], Xcov_train[cmass_mask]), (X_sample_all, X_train, Xcov_train), (X_sample_no, X_train[~cmass_mask], Xcov_train[~cmass_mask])]
    inputs = [ (X_sample_all_split[i], X_train, Xcov_train) for i in range(n_process) ]
    #inputs = inputs + [ (X_sample_no_split[i], X_train[~cmass_mask], Xcov_train[~cmass_mask]) for i in range(n_process) ]

    
    q = Queue()
    Processes = [Process(target = addingerror_process, args=(q, z[0], z[1])) for z in zip(range(n_process), inputs)]

    
    for p in Processes: p.start()
    result = [q.get() for p in Processes]
    result.sort()
    result = [r[1] for r in result ]

    #noisy_X_sample_cmass = result[0]
    noisy_X_sample_all = np.vstack( result[:n_process] ) #result[1]
    #noisy_X_sample_no = np.hstack( result[n_process:] )#result[2]
    noisy_X_sample_cmass = add_errors( X_sample_cmass, X_train[cmass_mask], Xcov_train[cmass_mask] )

    #noisy_X_sample_cmass = add_errors(X_sample_cmass, X_train[cmass_mask], Xcov_train[cmass_mask])
    #noisy_X_sample_all = add_errors(X_sample_all, X_train, Xcov_train)
    #noisy_X_sample_no = add_errors(X_sample_no, X_train[~cmass_mask], Xcov_train[~cmass_mask])

    # add eachprob column
    


    # Histogram method-----------------------------------------
    """
    print "calculate histogram number density"
    X_sample_cmass = clf_cmass.sample(100 * X.shape[0])
    X_sample_no = clf_nocmass.sample(1000 * X.shape[0])
    X_sample_lowz = clf_lowz.sample(100 * X.shape[0])
    
    # 3d number density histogram
    bin0, step0 = np.linspace(X[:,0].min(), X[:,0].max(), 101, retstep=True) # cmodel r
    bin1, step1 = np.linspace(X[:,1].min(), X[:,1].max(), 101, retstep=True) # cmodel i
    bin2, step2 = np.linspace(X[:,2].min(), X[:,2].max(), 101, retstep=True) # gr
    bin3, step3 = np.linspace(X[:,3].min(), X[:,3].max(), 101, retstep=True) # ri
    bin0 = np.append( bin0, bin0[-1]+step0)
    bin1 = np.append( bin1, bin1[-1]+step1)
    bin2 = np.append( bin2, bin2[-1]+step2)
    bin3 = np.append( bin3, bin3[-1]+step3)
    
    # cmass histogram model probability
    N_XDProb, edges = np.histogramdd(X_sample_cmass[:,1:4], bins = [bin1, bin2, bin3])
    n_CMASS = N_XDProb * 1./np.sum( N_XDProb )

    #X_sample_no = X_train[~cmass_mask].copy()
    N_XDProb, edges = np.histogramdd(X_sample_no[:,1:4], bins = [bin1, bin2, bin3])
    n_noCMASS = N_XDProb * 1./np.sum( N_XDProb )

    numerator =  n_CMASS * np.sum(y_train[:,0]) * 1.
    denominator =  numerator + n_noCMASS * (len(X_train) - np.sum(y_train[:,0]) )

    denominator_zero = denominator == 0
    modelProb_CMASS = np.zeros( numerator.shape )
    modelProb_CMASS[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]  # 3D model probability distriution
    
    # loz histogram model probability
    N_XDProb, edges = np.histogramdd(X_sample_lowz[:,(0,2,3)], bins = [bin0, bin2, bin3])
    n_LOWZ = N_XDProb * 1./np.sum( N_XDProb )

    X_sample_no = X_train[~lowz_mask].copy()
    N_XDProb, edges = np.histogramdd(X_sample_no[:,(0,2,3)], bins = [bin0, bin2, bin3])
    n_noLOWZ = N_XDProb * 1./np.sum( N_XDProb )    

    numerator =  n_LOWZ * np.sum(y_train[:,1]) * 1.
    denominator = numerator + n_noLOWZ * (len(X_train) - np.sum(y_train[:,1]) )

    denominator_zero = denominator == 0
    modelProb_LOWZ = np.zeros( numerator.shape )
    modelProb_LOWZ[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]

    # passing test sample to model probability
    inds0 = np.digitize( X_test[:,0], bin0 )-1
    inds1 = np.digitize( X_test[:,1], bin1 )-1
    inds2 = np.digitize( X_test[:,2], bin2 )-1
    inds3 = np.digitize( X_test[:,3], bin3 )-1
    inds_CMASS = np.vstack([inds1,inds2, inds3]).T
    inds_CMASS = [ tuple(ind) for ind in inds_CMASS ]
    EachProb_CMASS = np.array([modelProb_CMASS[ind] for ind in inds_CMASS ]) # Assign probability to each galaxies
    inds_LOWZ = np.vstack([inds0, inds2, inds3]).T
    inds_LOWZ = [ tuple(ind) for ind in inds_LOWZ ]
    EachProb_LOWZ = np.array([modelProb_LOWZ[ind] for ind in inds_LOWZ ])
    """
    # ---------------------------------------------------------


    
    #completeness = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0 / np.sum(y_test[:,0])
    #purity = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0 /np.sum(GetCMASS_mask)
    #contaminant = np.sum( GetCMASS_mask * np.logical_not(y_test[:,0]) )


    print "expected com/pur"
    
    GetCMASS_mask = test['EachProb_CMASS'] > p_threshold
    completeness = np.sum( test[GetCMASS_mask]['EachProb_CMASS'])/ np.sum( EachProb_CMASS )
    purity = np.sum( test[GetCMASS_mask]['EachProb_CMASS']) * 1./np.sum( GetCMASS_mask )

    print 'com/pur',completeness, purity
    # contaminant test

    #return 0
    #return train[cmass_mask], train[lowz_mask], test[GetCMASS_mask], test[GetLOWZ_mask]
    return train[cmass_mask], test[GetCMASS_mask], test[GetCMASS_mask * np.logical_not(y_test[:,0])], test, test[~GetCMASS_mask], noisy_X_sample_cmass, noisy_X_sample_all


    #return test


    # --------------------






def com_pur_arbi(cat = 'testcat.fits', prefix = '', subset = None):
    
    import fitsio

    dec = -1.0
    dec2 = 1.0
    ra = 310.0
    ra2 = 360.
    
    
    
    if subset is None:
        cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
        cmass_data = Cuts.SpatialCuts(cmass_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
        cmass = Cuts.keepGoodRegion(cmass_data)
    
    else:
        cmass = subset.copy()
        #cmass, _ = DES_to_SDSS.match( clean_cmass_data, dmass )
        #print 'dmass, subcmass', len(dmass), len(cmass)


    testAll = fitsio.read(cat)
    mask = testAll['EachProb_CMASS'] > 1.0
    testAll['EachProb_CMASS'][mask] = 1.0


    pdb.set_trace()

    truecmass, _ = DES_to_SDSS.match( testAll, cmass )
    
    p, pstep = np.linspace(0.0, 1.0, 50, retstep = True)
    pbin_center = p[:-1] + pstep/2.
    
    coms=[]
    purs=[]
    coms2 = []
    purs2 = []
    
    pps = []

    print 'expected com/pur'
    
    for pp in p:
        
        GetCMASS_mask_loop = testAll['EachProb_CMASS'] >= pp
        common, _ = DES_to_SDSS.match( truecmass, testAll[GetCMASS_mask_loop]  )
        purity_cmass = len(common) * 1./len(testAll[GetCMASS_mask_loop] )
        completeness_cmass  = len(common)*1./ len(truecmass )
        
        completeness_cmass2 = np.sum( testAll[GetCMASS_mask_loop]['EachProb_CMASS'])/ np.sum( testAll['EachProb_CMASS'] )
        purity_cmass2 = np.mean( testAll[GetCMASS_mask_loop]['EachProb_CMASS'])
        
        #print pp, completeness_cmass, purity_cmass
        pps.append(pp)
        coms.append(completeness_cmass)
        purs.append(purity_cmass)
        coms2.append(completeness_cmass2)
        purs2.append(purity_cmass2)


    pdb.set_trace()

    print len(pps), len(coms2)
    fig, (ax, ax2) = plt.subplots(1,2, figsize=(14,5))
    ax.plot( pps, coms, 'r-', label = 'completeness')
    ax.plot( pps, purs, 'b-', label = 'purity')
    ax.set_title('CMASS com/purity')
    ax.set_xlabel('p_threshold')

    ax2.plot( pps, coms2, 'r-', label = 'completeness')
    ax2.plot( pps, purs2, 'b-', label = 'purity')
    ax2.set_title('Model com/pur')
    ax2.set_xlabel('p_threshold')
    ax.legend(loc = 'best')
    ax.set_ylim(0.0, 1.0)
    ax2.legend(loc = 'best')
    ax2.set_ylim(0.0, 1.0)
    fig.savefig('com_pur_results/'+ prefix + 'com_pur_check')
    print 'save fig: com_pur_results/'+prefix+'com_pur_check.png'


    # probability calibration
    prob = []
    real = []
    for i in xrange(len(pbin_center)):
        these = ( testAll['EachProb_CMASS'] > p[i]) & ( testAll['EachProb_CMASS'] <= p[i+1])
        denozero = np.sum(these)
        common, _ = DES_to_SDSS.match( truecmass, testAll[these] )
        print 'p, common, these', p[i+1], len(common), np.sum(these)
        if denozero is not 0 and len(common) is not 0 :
            prob.append( len(common) * ( 1./np.sum(these) ))
            real.append( len(common) * (1./len(truecmass) ))
        else:
            prob.append( 0 )
            real.append( 0 )

    these = ( testAll['EachProb_CMASS'] == 1.0 )
    common, _ = DES_to_SDSS.match( truecmass, testAll[these] )
    print 'p, common/these', '1.0', len(common), np.sum(these)
    common, _ = DES_to_SDSS.match( truecmass, testAll[these] )



    fig,ax = plt.subplots()
    ax.plot(pbin_center, prob,label='true fraction' )
    ax.plot(pbin_center, real,label='total fraction' )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    fig.savefig('com_pur_results/'+prefix + 'probability_calibration.png')
    print 'save fig: com_pur_results/'+prefix+'probability_calibration.png'

    return pps, coms2, purs2


def main():
    
    # load dataset

    dec = -1.5
    dec2 = 1.5
    ra = 310.0
    ra2 = 360.

    cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    cmass_data = Cuts.SpatialCuts(cmass_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    clean_cmass_data = Cuts.keepGoodRegion(cmass_data)
    #lowz_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_LOWZ_South-photoObj.fits.gz')
    #lowz_data = Cuts.SpatialCuts(lowz_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    #clean_lowz_data = Cuts.keepGoodRegion(lowz_data)
    
    sdss_data_o = io.getSDSScatalogs(bigSample = True)
    sdss_data_o = io.getSDSScatalogs(  file = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit', bigSample = False)
    sdss = Cuts.SpatialCuts(sdss_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    sdss = Cuts.doBasicSDSSCuts(sdss)
    
    full_des_data = io.getDESY1A1catalogs(keyword = 'des_st82')
    des_data_f = Cuts.SpatialCuts(full_des_data, ra = ra, ra2=ra2, dec= dec, dec2= dec2  )
    des = Cuts.doBasicCuts(des_data_f)
    vetoed_des = keepSDSSgoodregion(des)
    
    """
    balrog_o = io.LoadBalrog(user = 'EMHUFF', truth = None)
    balrog = Cuts.SpatialCuts(balrog_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    balrog = AddingReddening(balrog)
    """

    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(vetoed_des, vetoed_des, [0.2,0.8], random_state=0)
    des_train = vetoed_des[trainInd]
    des_test = vetoed_des[testInd]
    
    
    prefix = 'big_'
    pickleFileName = [prefix +'XD_all.pkl', prefix+'XD_dmass.pkl', prefix+'XD_no.pkl']

    #prefix = ''
    #pickleFileName = [prefix +'XD_all_cluster25_5d.pkl', prefix+'XD_dmass_cluster25_5d.pkl', prefix+'XD_no_cluster25_5d.pkl']
    trainC, testC, contC, testAll, testno, modelX, modelAllX = XDGMM_model(clean_cmass_data, clean_cmass_data, train=des_train, test=des_test, p_threshold = 0.9, pickleFileName=pickleFileName )
    pp, coms, purs = com_pur_arbi(cat = prefix+'testcat.fits', prefix = prefix)
    
    
    
    fitsio.write(prefix+'truecat.fits' ,trainC)
    fitsio.write(prefix+'dmasscat.fits' ,testC)
    fitsio.write(prefix+'contcat.fits' ,contC)
    fitsio.write(prefix+'testcat.fits' ,testAll)
    #fitsio.write(prefix+'testno.fits' ,testno)
    fitsio.write(prefix+'modelX.fits' ,modelX)
    fitsio.write(prefix+'modelAllX.fits' ,modelAllX)
    #fitsio.write(prefix+'modelnoX.fits' ,modelnoX)




    # SDSS subset ----------------------------------------------
    
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(clean_cmass_data, clean_cmass_data, [0.5,0.5], random_state=0)
    sdss_train = clean_cmass_data[trainInd]
    sdss_test = clean_cmass_data[testInd]
    
    
    testC = fitsio.read(prefix+'dmasscat.fits')
    dmass = testC.copy()
    
    prefix = 'sub_'
    pickleFileName = [prefix +'XD_all.pkl', prefix+'XD_dmass.pkl', prefix+'XD_no.pkl']
    trainC, testC, contC, testAll, testno, modelX, modelAllX  = XDGMM_model(dmass, dmass, train=sdss_train, test=sdss_test, p_threshold = 0.9, pickleFileName=pickleFileName, reverse=True )
    
    pp, coms, purs = com_pur_arbi(subset = dmass, prefix = prefix)
    
    

    # repeat multiple times to get average --------------------------------------------
    """
    import fitsio
    for i in range(10):
        
        prefix = 'test_'+str(i)+'_'
        pickleFileName = [prefix + 'XD_all_cluster25.pkl', prefix+'XD_dmass_cluster25_5d.pkl', prefix+'XD_no_cluster25_5d.pkl']
        testAll = XDGMM_model(clean_cmass_data, clean_lowz_data, train=des_train, test=des_test, p_threshold = 0.5, pickleFileName = pickleFileName )
        fitsio.write(prefix+'testcat.fits' ,testAll)
    
    
    m_pp = np.zeros( (10,50 ), dtype=float )
    m_coms = np.zeros( (10,50 ), dtype=float )
    m_purs = np.zeros( (10,50 ), dtype=float )
    for i in range(m_coms.shape[0]):
        pp, coms, purs = com_pur_arbi(clean_cmass_data, prefix = 'test_'+str(i)+'_')
        m_pp[i,:]= np.array(pp)
        m_coms[i,:] = coms
        m_purs[i,:] = purs
    
    mean_coms = np.mean(m_coms, axis = 0)
    mean_purs = np.mean(m_purs, axis = 0)

    fig, ax = plt.subplots()
    ax.plot(m_pp.T, m_coms.T, color='grey', alpha = 0.5 )
    ax.plot(m_pp.T, m_purs.T, color='grey', alpha = 0.5 )
    ax.plot(pp, mean_coms, color='red')
    ax.plot(pp, mean_purs, color='red')
    fig.savefig('com_pur_results/total_com_pur.png')


    p_threshold = 0.5
    GetCMASS_mask = testAll['EachProb_CMASS'] > p_threshold
    dmass = testAll[GetCMASS_mask]

    """
    # ----------------------------------------------------------------



    # calling catalog -------------------------------------------------
    trainC = fitsio.read(prefix+'truecat.fits')
    testC = fitsio.read(prefix+'dmasscat.fits')
    contC = fitsio.read(prefix+'contcat.fits')
    testAll = fitsio.read(prefix+'testcat.fits')
    #testno = fitsio.read(prefix+'testno.fits')
    modelX = fitsio.read(prefix+'modelX.fits')
    #modelnoX = fitsio.read(prefix+'modelnoX.fits')
    modelAllX = fitsio.read(prefix+'modelAllX.fits')
    
    trainX, _ = mixing_color(trainC)
    testX, _ = mixing_color(testC)
    contX, _ = mixing_color(contC)
    testAllX, _ = mixing_color(testAll)
    #testnoX, _ = mixing_color(testno)
    
    # plotting
    
    labels = ['MAG_MODEL_R', 'MAG_MODEL_I', 'g-r', 'r-i', 'i-z']
    ranges = [[15,22], [15,22],[-1,2], [-1,2], [-1,2]]
    doVisualization(modelX, trainX, labels = labels, ranges = ranges, nbins=100, prefix= prefix+'cmass_')
    doVisualization(modelAllX, testAllX, labels = labels, ranges = ranges, nbins=100, prefix= prefix+'all_')
    #doVisualization(modelnoX, testnoX, labels = labels, ranges = ranges, nbins=100, prefix= 'no_')

    doVisualization(trainX, contX, labels = labels, ranges = ranges, nbins=100, prefix= prefix+'cont_true_')
    doVisualization(modelX, contX, labels = labels, ranges = ranges, nbins=100, prefix= prefix+'cont_model_')
    
    
    
    
    
    labels2 = ['SPREAD_MODEL_I', 'MAG_PSF_I', 'FLUX_MODEL_I']
    ranges2 = [[contC[lab].min() * 0.7 , contC[lab].max() * 1.2] for lab in labels2 ]
    doVisualization2(trainC, testC, labels = labels2, ranges = ranges2, nbins=80, prefix= prefix+'true_test_')
    doVisualization2(trainC, contC, labels = labels2, ranges = ranges2, nbins=80, prefix= prefix+'true_cont_')
    
    
    labels2 = ['FLUX_MODEL_G', 'FLUX_MODEL_R', 'FLUX_MODEL_I', 'FLUX_MODEL_Z']
    ranges2 = [[116.10883, 10641.5],
               [1704.7964, 24215.969],
               [5882.5488, 36152.707],
               [1150.884, 52163.207]]
    doVisualization2(trainC, testC, labels = labels2, ranges = ranges2, nbins=100, prefix= prefix+'true_test_flux_')
    doVisualization2(trainC, contC, labels = labels2, ranges = ranges2, nbins=100, prefix= prefix+'true_cont_flux_')


    stop
    

    contSX, _ = mixing_color(contS, sdss = contS)
    dperp_passed_mask =  (contSX[:, 2] - contSX[:, 1]/8.0) > 0.55
    icut_passed_mask = ((contSX[:,0]-19.86 )/1.6 + 0.8 < (contSX[:, 2] - contSX[:, 1]/8.0)) & ( contSX[:,0] < 19.9 ) & ( contSX[:,0] > 17.5 )


    cont_dperp = eachprobcol[~dperp_passed_mask]
    cont_icut = eachprobcol[~icut_passed_mask]
    
    fig, ax = plt.subplots(1,2, figsize = (14, 5))
    ax = ax.ravel()
    pbin = np.linspace(0,1.0, 50)
    ax[0].hist( cont_dperp['EachProb_CMASS'],  bins = pbin )
    #ax[1].hist( 1 - cont_dperp['EachProb_CMASS'],  bins = pbin )
    ax[1].hist( cont_icut['EachProb_CMASS'],  bins = pbin, color= 'red' )
    #ax[3].hist( 1 - cont_icut['EachProb_CMASS'],  bins = pbin, color = 'red' )
    for a in ax:
        a.set_ylabel('N')
        a.set_xlabel('probability')
    ax[0].set_title('dperp non passed')
    ax[1].set_title('icut non passed')
    fig.savefig('cont_prob')
    


    #dmass_kNN, dmass_GMMB = MachineLearningClassifier( clean_cmass_data, clean_lowz_data,  train = des_train, test = des_test)




    
def histogramDiagnosticSDSS( clean_cmass_data, sdss  ):


    import fitsio
    import esutil
    h = esutil.htm.HTM(10)
    
    # data from SDSS XD ------------------------
    trainSDSS = fitsio.read('truecat_Matchedsdss.fits')
    testSDSS = fitsio.read('dmasscat_Matchedsdss.fits')
    contSDSS = fitsio.read('contcat_Matchedsdss.fits')

    trainSDSSX, _ = mixing_color(trainSDSS, sdss = True)
    testSDSSX, _ = mixing_color(testSDSS, sdss = True)
    contSDSSX, _ = mixing_color(contSDSS, sdss = True)
    
    # data from DES XD ------------------------
    trainC = fitsio.read('truecat_Matched.fits')
    testC = fitsio.read('dmasscat_Matched.fits')
    contC = fitsio.read('contcat_Matched.fits')
    #testAll = fitsio.read('testcat_masked.fits')



    trainX, _ = mixing_color(trainC)
    testX, _ = mixing_color(testC)
    contX, _ = mixing_color(contC)
    
    _, trainS = DES_to_SDSS.match(trainC, clean_cmass_data )
    _, contS = DES_to_SDSS.match(contC, sdss )
    _, testS = DES_to_SDSS.match(testC, sdss )
    
    trainSX, _ = mixing_color(trainS, cmass = trainS)
    testSX, _ = mixing_color(testS, sdss = testS)
    contSX, _ = mixing_color(contS, sdss = contS)
    
    
    dperp_passed_mask_sdss =  ((contSDSSX[:, 2] - contSDSSX[:, 1]/8.0) > 0.55 )
    icut_passed_mask_sdss = (((contSDSSX[:,0]-19.86 )/1.6 + 0.8 < (contSDSSX[:, 2] - contSDSSX[:, 1]/8.0)) & ( contSDSSX[:,0] < 19.9 ) & ( contSDSSX[:,0] > 17.5 ))
    
    dperp_passed_mask =  (contSX[:, 2] - contSX[:, 1]/8.0) > 0.55
    icut_passed_mask = ((contSX[:,0]-19.86 )/1.6 + 0.8 < (contSX[:, 2] - contSX[:, 1]/8.0)) & ( contSX[:,0] < 19.9 ) & ( contSX[:,0] > 17.5 )
    
    dperp_passed_sdss = contSDSSX[dperp_passed_mask_sdss]
    dperp_nonpassed_sdss = contSDSSX[~dperp_passed_mask_sdss]
    icut_passed_sdss = contSDSSX[icut_passed_mask_sdss]
    icut_nonpassed_sdss = contSDSSX[~icut_passed_mask_sdss]
    SDSS_reject_sdss = contSDSS[dperp_passed_mask_sdss * icut_passed_mask_sdss]
        
    dperp_passed = contSX[dperp_passed_mask]
    dperp_nonpassed = contSX[~dperp_passed_mask]
    icut_passed = contSX[icut_passed_mask]
    icut_nonpassed = contSX[~icut_passed_mask]
    SDSS_reject = contS[dperp_passed_mask * icut_passed_mask]
    

    _, dperp_passed_indes_Inds, _ = h.match(contSDSS[dperp_passed_mask_sdss]['RA'], contSDSS[dperp_passed_mask_sdss]['DEC'], testS['RA'], testS['DEC'], 1./3600, maxmatch = 1 )
    _, dperp_nonpassed_indes_Inds, _ = h.match(contSDSS[~dperp_passed_mask_sdss]['RA'], contSDSS[~dperp_passed_mask_sdss]['DEC'], testS['RA'], testS['DEC'], 1./3600, maxmatch = 1 )
    
    _, dperp_passed_insdss_Inds, _ = h.match(contS[dperp_passed_mask]['RA'], contS[dperp_passed_mask]['DEC'], testSDSS['RA'], testSDSS['DEC'], 1./3600, maxmatch = 1 )
    _, dperp_nonpassed_insdss_Inds, _ = h.match(contS[~dperp_passed_mask]['RA'], contS[~dperp_passed_mask]['DEC'], testSDSS['RA'], testSDSS['DEC'], 1./3600, maxmatch = 1 )
    
    dperp_passed_indes = testSX[dperp_passed_indes_Inds]
    dperp_nonpassed_indes = testSX[dperp_nonpassed_indes_Inds]
    dperp_passed_insdss = testSDSSX[dperp_passed_insdss_Inds]
    dperp_nonpassed_insdss = testSDSSX[dperp_nonpassed_insdss_Inds]
    
    
    dperpbin = np.linspace(0, 1.5, 101)
    fig, axs = plt.subplots(2,4, figsize = (20, 10))
    axs = axs.ravel()
    cats = [dperp_passed_sdss, dperp_nonpassed_sdss, dperp_passed, dperp_nonpassed, dperp_passed_indes, dperp_nonpassed_indes, dperp_passed_insdss, dperp_nonpassed_insdss]
    
    
    # gri plane
    xx = np.linspace(0.0, 100, 100)
    yy = 0.55 + xx/8.
    for a, cat in zip(axs, cats):
        a.scatter( cat[:,1], cat[:,2], s = 10, marker = '.' )
        a.plot( xx, yy, linestyle = '--', color='red')
        a.set_xlabel('g-r')
        a.set_ylabel('r-i')
        a.set_xlim(0,4)
        a.set_ylim(0,2)
    
    fig.savefig('cont_diagnostic_dperp.png')
    
    
    # icut plane --------------------------
    
    _, icut_passed_indes_Inds, _ = h.match(contSDSS[icut_passed_mask_sdss]['RA'], contSDSS[icut_passed_mask_sdss]['DEC'], testS['RA'], testS['DEC'], 1./3600, maxmatch = 1 )
    _, icut_nonpassed_indes_Inds, _ = h.match(contSDSS[~icut_passed_mask_sdss]['RA'], contSDSS[~icut_passed_mask_sdss]['DEC'], testS['RA'], testS['DEC'], 1./3600, maxmatch = 1 )
    _, icut_passed_insdss_Inds, _ = h.match(contS[icut_passed_mask]['RA'], contS[icut_passed_mask]['DEC'], testSDSS['RA'], testSDSS['DEC'], 1./3600, maxmatch = 1 )
    _, icut_nonpassed_insdss_Inds, _ = h.match(contS[~icut_passed_mask]['RA'], contS[~icut_passed_mask]['DEC'], testSDSS['RA'], testSDSS['DEC'], 1./3600, maxmatch = 1 )

    
    icut_passed_indes = testSX[icut_passed_indes_Inds]
    icut_nonpassed_indes = testSX[icut_nonpassed_indes_Inds]
    icut_passed_insdss = testSDSSX[icut_passed_insdss_Inds]
    icut_nonpassed_insdss = testSDSSX[icut_nonpassed_insdss_Inds]
    

    fig2, axs = plt.subplots(2,4, figsize = (20, 10))
    axs = axs.ravel()

    cats = [icut_passed_sdss, icut_nonpassed_sdss, icut_passed, icut_nonpassed, icut_passed_indes, icut_nonpassed_indes, icut_passed_insdss, icut_nonpassed_insdss]
    

    xx = np.linspace(0.0, 100, 100)
    for a, cat in zip(axs, cats):
        a.scatter( cat[:, 0], cat[:,2] - cat[:,1]/8.0,  s = 10, marker = '.' )
        a.axhline( y = 0.55, linestyle = '--', color = 'red')
        a.plot( xx, (xx-19.86)/1.6 + 0.8, linestyle='--', color='grey')
        a.axvline( x = 19.9, linestyle = '--', color = 'grey')
        a.set_xlabel('cmodel_i')
        a.set_ylabel('dperp')
        a.set_xlim(18,21.5)
        a.set_ylim(0,1.5)

    fig2.savefig('cont_diagnostic_icut.png')




def histogramsDiagostic():
    # hist fibmag
    
    import fitsio
    trainC = fitsio.read('truecat.fits')
    testC = fitsio.read('dmasscat.fits')
    contC = fitsio.read('contcat.fits')
    
    trainX, _ = mixing_color(trainC)
    testX, _ = mixing_color(testC)
    contX, _ = mixing_color(contC)

    _, trainS = DES_to_SDSS.match(trainC, clean_cmass_data )
    _, contS = DES_to_SDSS.match(contC, sdss )
    _, testS = DES_to_SDSS.match(testC, sdss )

    trainSX = mixing_color(trainS, cmass = trainS)
    testSX = mixing_color(testS, sdss = testS)
    contSX = mixing_color(contS, sdss = contS)

    dperp_passed_mask =  (contSX[:, 2] - contSX[:, 1]/8.0) > 0.55
    icut_passed_mask = ((contSX[:,0]-19.86 )/1.6 + 0.8 < (contSX[:, 2] - contSX[:, 1]/8.0)) & ( contSX[:,0] < 19.9 ) & ( contSX[:,0] > 17.5 )
    
    

    import esutil
    h = esutil.htm.HTM(10)
    
    sdss_cmass, _  = SDSS_cmass_criteria(sdss)
    C, _, _ = h.match(contC['RA'],contC['DEC'], clean_cmass_data['RA'], clean_cmass_data['DEC'],1./3600,maxmatch=1)
    L, _, _ = h.match(contC['RA'],contC['DEC'], clean_lowz_data['RA'], clean_lowz_data['DEC'],1./3600,maxmatch=1)
    C, _, _ = h.match(testC['RA'],testC['DEC'], clean_cmass_data['RA'], clean_cmass_data['DEC'],1./3600,maxmatch=1)
    
    #dmass_kNN = fitsio.read('dmass_kNN.fits')
    #dmass_GMMB = fitsio.read('dmass_GMMB.fits')
    
    tags = ['MAG_APER_4_I', 'MAG_MODEL_G', 'MAG_MODEL_R','MAG_MODEL_I']
    fig, axs = plt.subplots(2,2, figsize = (14, 14))
    axs = axs.ravel()
    
    for a, tag in zip(axs, tags):
        
        bins, dd = np.linspace( testC[tag].min() - 0.5, testC[tag].max() + 0.5, 100, retstep = True)
        n_trainC, _ = np.histogram( trainC[tag], bins = bins, density=True )
        n_testC, _ = np.histogram( testC[tag], bins = bins, density=True )
        #n_dmass_kNN, _ = np.histogram( dmass_kNN[tag], bins = bins, density=True )
        n_dmass_cont, _ = np.histogram( contC[tag], bins = bins, density=True )
        
        a.plot(bins[:-1], n_trainC, 'k--', label='true')
        a.plot(bins[:-1], n_testC, '.', label='test')
        #a.plot(bins[:-1], n_dmass_kNN, '.', label='kNN')
        a.plot(bins[:-1], n_dmass_cont, 'r.', label='cont')
        
        #n = a.hist( contC[tag], bins, histtype='step', label='contaminant' )
        a.legend(loc='best')
        a.set_title(tag)

    fig.savefig('cont_diagonistic')


    fig, axs = plt.subplots(2,2, figsize = (14, 14))
    axs = axs.ravel()



    tags = ['CMODELMAG_I', 'GR', 'RI']
    for (i,a), tag in zip(enumerate(axs), tags):
        bins, dd = np.linspace( testX[:,i].min() - 0.5, testX[:,i].max() + 0.5, 50, retstep = True)
        n_trainC, _ = np.histogram( trainX[:,i], bins = bins, density=True )
        n_testC, _ = np.histogram( testX[:,i], bins = bins, density=True )
        n_dmass_cont, _ = np.histogram( contX[:,i], bins = bins, density=True )

        a.plot(bins[:-1], n_trainC, 'k--', label='true')
        a.plot(bins[:-1], n_testC, '.', label='test')
        a.plot(bins[:-1], n_dmass_cont, 'r.', label='cont')
        a.legend(loc='best')
        a.set_title(tag)
    fig.savefig('cont_diagonistic2')


    # contaminant in sloan photometry

    fig, axs = plt.subplots(2,2, figsize = (14, 14))
    axs = axs.ravel()
    tags = ['CMODELMAG_I', 'GR', 'RI']

    for (i,a), tag in zip(enumerate(axs), tags):
        bins, dd = np.linspace( testSX[:,i].min() - 0.5, testSX[:,i].max() + 0.5, 30, retstep = True)
        n_trainC, _ = np.histogram( trainSX[:,i], bins = bins, density=True )
        n_testC, _ = np.histogram( testSX[:,i], bins = bins, density=True )
        n_dmass_cont, _ = np.histogram( contSX[:,i], bins = bins, density=True )
        
        a.plot(bins[:-1], n_trainC, 'k--', label='true')
        a.plot(bins[:-1], n_testC, '.', label='test')
        a.plot(bins[:-1], n_dmass_cont, 'r.', label='cont')
        a.legend(loc='best')
        a.set_xlabel(tag)
    
    bins, dd = np.linspace( 0.2, 1.5, 30, retstep = True)
    n_trainC, _ = np.histogram( trainSX[:,2]-trainSX[:,1]/8.0, bins = bins, density=True )
    n_testC, _ = np.histogram( testSX[:,2]-testSX[:,1]/8.0, bins = bins, density=True )
    n_dmass_cont, _ = np.histogram( contSX[:,2] - contSX[:,1]/8.0, bins = bins, density=True )

    axs[-1].plot(bins[:-1], n_trainC, 'k--', label='true')
    axs[-1].plot(bins[:-1], n_testC, '.', label='test')
    axs[-1].plot(bins[:-1], n_dmass_cont, 'r.', label='cont')
    axs[-1].legend(loc='best')
    axs[-1].set_xlabel('DPERP')

    fig.savefig('cont_diagonistic_sdssphoto_dot')



    

    #  bin0, step0 = np.linspace(X[:,0].min(), X[:,0].max(), 101, retstep=True) # cmodel r
    bin1, step1 = np.linspace(testSX[:,0].min() - 0.5, testSX[:,0].max() +0.5, 101, retstep=True) # cmodel i
    bin2, step2 = np.linspace(testSX[:,1].min() - 0.5, testSX[:,1].max()+0.5, 101, retstep=True) # gr
    bin3, step3 = np.linspace(testSX[:,2].min() - 0.5, testSX[:,2].max()+0.5, 101, retstep=True) # ri
    #bin0 = np.append( bin0, bin0[-1]+step0)
    #bin1 = np.append( bin1, bin1[-1]+step1)
    #bin2 = np.append( bin2, bin2[-1]+step2)
    #bin3 = np.append( bin3, bin3[-1]+step3)



    
    dperpbin = np.linspace(0, 1.5, 101)
    fig2, axs = plt.subplots(2,2, figsize = (10, 10))

    axs = axs.ravel()
    #axs[0].hist2d(trainX[:, 1], trainX[:,2], bins = [bin2, bin3] )
    #axs[0].hist2d( dperp_passed[:,1],  dperp_passed[:,2], bins = [bin2, bin3])
    #axs[0].hist2d( icut_passed[:,1],  icut_passed[:,2], bins = [bin2, bin3])
    axs[0].scatter( icut_passed[:,1],  icut_passed[:,2], s = 10, color = 'red')
    xx = np.linspace(0.0, 100, 100)
    yy = 0.55 + xx/8.
    axs[0].plot( xx, yy, linestyle = '--', color='red')
    axs[0].set_xlabel('g-r')
    axs[0].set_ylabel('r-i')
    axs[0].set_xlim(0,4)
    axs[0].set_ylim(0,2)
    #axs[1].hist2d(trainX[:, 0], trainX[:,2] - trainX[:,1]/8.0, bins = [bin1, dperpbin])
    #axs[1].hist2d( dperp_passed[:,0],  dperp_passed[:,2] - dperp_passed[:,1]/8.0, bins = [bin1, dperpbin])
    axs[1].scatter( dperp_passed[:,0],  dperp_passed[:,2] - dperp_passed[:,1]/8.0, s = 10, color = 'blue')
    axs[1].plot( xx, (xx-19.86)/1.6 + 0.8, linestyle='--', color='grey')
    axs[1].axhline( y = 0.55, linestyle = '--', color = 'red')
    axs[1].axvline( x = 20.0, ymin=dperpbin[0], ymax = dperpbin[-1], linestyle = '--', color = 'grey')
    axs[1].set_xlim(18,21.5)
    axs[1].set_ylim(0,1.5)

    axs[1].set_xlabel('cmodel_i')
    axs[1].set_ylabel('dperp')
    axs[2].hist2d(contX[:, 1], contX[:,2], bins = [bin2, bin3] )
    axs[2].scatter( dperp_passed[:,1],  dperp_passed[:,2], s = 10, color = 'blue')
    axs[2].plot( xx, yy, linestyle = '--', color='red')
    axs[2].set_xlabel('g-r')
    axs[2].set_ylabel('r-i')
    axs[3].hist2d(contX[:, 0], contX[:,2] - contX[:,1]/8.0, bins = [bin1, dperpbin])
    axs[3].scatter( icut_passed[:,0],  icut_passed[:,2]- icut_passed[:,1]/8.0 , s = 10, color = 'red')
    axs[3].plot( xx, (xx-19.86)/1.6 + 0.8, linestyle='--', color='grey')
    axs[3].axhline( y = 0.55, linestyle = '--', color = 'red')
    axs[3].axvline( x = 20.0, ymin=dperpbin[0], ymax = dperpbin[-1], linestyle = '--', color = 'grey')
    axs[3].set_xlabel('cmodel_i')
    axs[3].set_ylabel('dperp')

    fig2.savefig('cont_diagonostic_2dhist_sdss.png')




    fig, axs = plt.subplots(2,2, figsize = (14, 14))
    axs = axs.ravel()
    tags = ['FIBER2MAG_I']


    
    for (i,a), tag in zip(enumerate(axs), tags):
        bins, dd = np.linspace( testS[tag].min() - 0.5, testS[tag].max() + 0.5, 30, retstep = True)
        n_trainC, _ = np.histogram( trainS[tag], bins = bins, density=True )
        n_SDSS_reject, _ = np.histogram( SDSS_reject[tag], bins = bins, density=True )
        
        a.plot(bins[:-1], n_trainC, 'k--', label='true')
        a.plot(bins[:-1], n_SDSS_reject, 'r.', label='cont')
        a.legend(loc='best')
        a.set_xlabel(tag)
        a.set_title('Objects rejected by Sloan')
    fig.savefig('cont_diagonostic_1dhist')


def sloan_vetomask( des, ra = 350, ra2 = 360, dec = -1.0, dec2 = 1.0 ):
    
    import os
    import fitsio
    
    path = '/n/des/lee.5922/data/sdss_veto_mask/'
    
    
    vetoed = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and 'vetoe_des' in i:
            print 'vetoed catalog exists : {}'.format(i)
            vetoed.append(i)
            des = fitsio.read(path +i)

    if len(vetoed) is 0:

        print 'start masking des catalog with sdss veto mask'
        import esutil
        h = esutil.htm.HTM(10)
        #read maps
        tables = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and 'fits' in i and not 'boss_survey' in i and not 'boss_geometry' in i and not 'vetoed' in i:
                tables.append(path+i)
                print i

        for m in tables:
            #map = np.loadtxt(m)
            map = fitsio.read(m)
            print 'cutting'
            cutmap = map[ (ra < map['RA']) &  (ra2 > map['RA']) &  (dec < map['DEC']) & (dec2 > map['DEC']) ]
            print 'matching'
            desInd, vetoInd, _ = h.match(des['RA'],des['RA'], cutmap['DEC'], cutmap['DEC'], 1./3600,maxmatch=1)
            des = des[desInd]
            veto = cutmap[vetoInd]
            vetomask = veto['MASK'] == 0
            des = des[vetomask]

        fitsio.write('vetoed_des_{}_{}.fits'.format(ra, ra2) ,des)

    return des


def text_to_fits( tables ):

    import os
    import fitsio
    
    path = '/n/des/lee.5922/data/sdss_veto_mask/'

    for m in tables:
        fitsfile = np.zeros((len(map), ), dtype=[('RA','f8'),('DEC','f8'), ('MASK', 'f8')] )
        print 'fitsfile shape', fitsfile.shape
        fitsfile['RA'] = map[:,0]
        fitsfile['DEC'] = map[:,1]
        fitsfile['MASK'] = map[:,2]
        fitsio.write(path+m+'.fits', fitsfile)




def doVisualization(model_data, real_data, labels = None, ranges = None, nbins=100, prefix= 'default'):
    if labels == None:
        print " always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data."
        stop
    else:
        ndim = len(labels)

    if ranges == None:
        # try to figure out the correct axis ranges.
        print "Using central 98% to set range."
        ranges = []
        for i in xrange(ndim):
            ranges.append( np.percentile(real_data[:,i],[1.,99.]) )

    fig,axes = plt.subplots(nrows=ndim, ncols= ndim, figsize= (6*ndim, 6*ndim) )
    
    for i in xrange(ndim):
        for j in xrange(ndim):
            if i == j:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                axes[i][i].hist(real_data[:,i],bins=xbins,normed=True,label='real')
                axes[i][i].set_xlabel(labels[i])
                axes[i][i].hist(model_data[:,i],bins=xbins,normed=True,alpha=0.5,label='model')
                axes[i][i].legend(loc='best')
            else:
                xbins = np.linspace(ranges[i][0],ranges[i][1],100)
                ybins = np.linspace(ranges[j][0],ranges[j][1],100)
                axes[i][j].hist2d(real_data[:,i], real_data[:,j], bins = [xbins,ybins], normed=True)
                axes[i][j].set_xlabel(labels[i])
                axes[i][j].set_ylabel(labels[j])
                axes[i][j].set_title('real')
                axes[j][i].hist2d(model_data[:,i], model_data[:,j], bins = [xbins,ybins], normed=True)
                axes[j][i].set_xlabel(labels[i])
                axes[j][i].set_ylabel(labels[j])
                axes[j][i].set_title('model')

    filename = prefix+"diagnostic_histograms.png"
    print "writing output plot to: "+filename
    fig.savefig(filename)

def doVisualization2(true_data, test_data, labels = None, ranges = None, nbins=100, prefix= 'default'):
    if labels == None:
        print " always label your axes! you must populate the 'labels' keyword with one entry for each dimension of the data."
        stop
    else:
        ndim = len(labels)
    
    if ranges == None:
        # try to figure out the correct axis ranges.
        print "Using central 98% to set range."
        ranges = []
        for i in xrange(ndim):
            ranges.append( np.percentile(test_data[:,i],[1.,99.]) )

    fig,axes = plt.subplots(nrows=ndim, ncols= ndim, figsize= (6*ndim, 6*ndim) )
    
    for i in xrange(ndim):
        for j in xrange(ndim):
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
    print "writing output plot to: "+filename
    fig.savefig(filename)



def keepSDSSgoodregion( data ):

    Tags = ['bad_field_mask', 'unphot_mask', 'bright_star_mask', 'rykoff_bright_star_mask','collision_mask', 'centerpost_mask']

    mask = np.ones( data.size, dtype = bool )
    
    for tag in Tags:
        print tag
        mask = mask * (data[tag] == 0)
    
    print 'masked objects ', mask.size - np.sum(mask)
    return data[mask]


def Cut_test(contS, sdss):
    

    contSX, _ = mixing_color(contS, sdss = contS)
    
    dperp_passed_mask =  (contSX[:, 2] - contSX[:, 1]/8.0) > 0.55
    icut_passed_mask = ((contSX[:,0]-19.86 )/1.6 + 0.8 < (contSX[:, 2] - contSX[:, 1]/8.0)) & ( contSX[:,0] < 19.9 ) & ( contSX[:,0] > 17.5 )
    
    SDSS_reject = contS[dperp_passed_mask * icut_passed_mask]

    # cutting check : SDSS rejected

    starcuts = (((SDSS_reject['PSFMAG_I'] - SDSS_reject['EXPMAG_I']) > (0.2 + 0.2*(20.0 - SDSS_reject['EXPMAG_I']))) &
          ((SDSS_reject['PSFMAG_Z'] - SDSS_reject['EXPMAG_Z']) > (9.125 - 0.46 * SDSS_reject['EXPMAG_Z'])))

    # quality cut ( Reid et al. 2016 Section 2.2 )
    exclude = 2**1 + 2**5 + 2**7 + 2**11 + 2**19 # BRIGHT, PEAK CENTER, NO PROFILE, DEBLENDED_TOO_MANY_PEAKS, NOT_CHECKED
    # blended object
    blended = 2**3
    nodeblend = 2**6
    # obejct not to be saturated
    saturated = 2**18
    saturated_center = 2**(32+11)
    
    
    use =  (
            (SDSS_reject['CLEAN'] == 1 ) & (SDSS_reject['FIBER2MAG_I'] < 22.5) &
            (SDSS_reject['TYPE'] == 3) &
            ( ( SDSS_reject['FLAGS'] & exclude) == 0) &
            ( ((SDSS_reject['FLAGS'] & saturated) == 0) | (((SDSS_reject['FLAGS'] & saturated) >0) & ((SDSS_reject['FLAGS'] & saturated_center) == 0)) ) &
            ( ((SDSS_reject['FLAGS'] & blended) ==0 ) | ((SDSS_reject['FLAGS'] & nodeblend) ==0) ) )


    print 'total', len(contS)
    print 'd cut', np.sum(dperp_passed_mask)
    print 'icut', np.sum(icut_passed_mask)
    print 'd, icut', len(SDSS_reject)
    print 's-g sepa', np.sum(starcuts)
    print 'clean, fib, type', np.sum(SDSS_reject['CLEAN'] == 1), np.sum((SDSS_reject['FIBER2MAG_I'] < 22.5) ), np.sum((SDSS_reject['TYPE'] == 3) )
    print 'excluded', np.sum(( SDSS_reject['FLAGS'] & exclude) == 0 )
    print 'saturated', np.sum(  ( ((SDSS_reject['FLAGS'] & saturated) == 0) | (((SDSS_reject['FLAGS'] & saturated) >0) & ((SDSS_reject['FLAGS'] & saturated_center) == 0)) ) )
    print 'blended', np.sum(( ((SDSS_reject['FLAGS'] & blended) ==0 ) | ((SDSS_reject['FLAGS'] & nodeblend) ==0) ))
    print 'all', np.sum(use & starcuts)
    # Here stop
    
    DAT = np.column_stack(( SDSS_reject[use & starcuts]['RA'], SDSS_reject[use & starcuts]['DEC'] ))
    np.savetxt( 'SDSS_rejectedObj.txt', DAT, delimiter = ' ', header = ' ra, dec' )

    DAT = np.column_stack(( C['RA'], C['DEC'] ))
    np.savetxt( 'MatchedPlausible_cmassObj.txt', DAT, delimiter = ' ', header = ' ra, dec' )
    
    
    sdss_cmass, _ = SDSS_cmass_criteria(sdss)
    plausible_cmass, _ = DES_to_SDSS.match( SDSS_reject[use & starcuts], sdss_cmass )
    DAT = np.column_stack(( plausible_cmass['RA'], plausible_cmass['DEC'] ))
    np.savetxt( 'plausible_cmassObj.txt', DAT, delimiter = ' ', header = ' ra, dec' )

    return 0

    #sdss_match, cmass_match = DES_to_SDSS.match(sdss, clean_cmass_data)
    #C, _, _ = h.match(sdss_cmass['RA'],testC['DEC'], clean_cmass_data['RA'], clean_cmass_data['DEC'],1./3600,maxmatch=1)
    #Indmask = np.in1d(clean_cmass_data['OBJID'], clean_cmass_data['OBJID'] )
  


    trueCMASS, _, _ = h.match(testC['RA'], testC['DEC'], clean_cmass_data['RA'], clean_cmass_data['DEC'], 1./3600, maxmatch = 1 )
    trueCMASSInd, _, _ = h.match(clean_cmass_data['RA'], clean_cmass_data['DEC'], testC['RA'], testC['DEC'], 1./3600, maxmatch = 1 )

    trueCMASSmask = np.zeros( clean_cmass_data.size, dtype=bool)
    trueCMASSmask[trueCMASSInd] = 1
    missingCMASS = clean_cmass_data[~trueCMASSmask]





    stop






if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        #main(sys.argv)
        main()
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)




    # -------------------------------------
    
    
    
 

def misc():
    
    # testing dmass with angular correlation function ------------------------------------------------
    dmass = DMASSALL.copy()
    #rand_catD = Balrog_DMASS.copy()
    import fitsio
    cmass_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_South.fits.gz')
    rand_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz')
    
    cmass_catS = Cuts.SpatialCuts(cmass_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catS = Cuts.SpatialCuts(rand_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    
    dmass_cat = Cuts.SpatialCuts(dmass, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catD = rand_catS.copy()
    
    weight_rand_SGC = rand_cat_SGC['WEIGHT_FKP']
    weight_data_SGC = cmass_cat_SGC['WEIGHT_FKP'] * cmass_cat_SGC['WEIGHT_STAR'] * ( cmass_cat_SGC['WEIGHT_CP']+cmass_cat_SGC['WEIGHT_NOZ'] -1. )
    weight_randS = rand_catS['WEIGHT_FKP']
    weight_dataS = cmass_catS['WEIGHT_FKP'] * cmass_catS['WEIGHT_STAR'] * ( cmass_catS['WEIGHT_CP']+cmass_catS['WEIGHT_NOZ'] -1. )
    theta_SGC, w_SGC, _ = angular_correlation(cmass_cat_SGC, rand_cat_SGC, weight = [weight_data_SGC, weight_rand_SGC])
    thetaS, wS, werrS = angular_correlation(cmass_catS, rand_catS, weight = [weight_dataS, weight_randS])
    thetaD, wD, werrD = angular_correlation(dmass, rand_catD)

    # jk errors

    from astroML.plotting import setup_text_plots
    setup_text_plots(fontsize=20, usetex=True)

    njack = 10
    raTag = 'RA'
    decTag = 'DEC'
    _, jkerr_SGC = jk_error( cmass_cat_SGC, njack = njack , target = angular_correlation, jkargs=[cmass_cat_SGC, rand_cat_SGC], jkargsby=[[raTag, decTag],[raTag, decTag]], raTag = raTag, decTag = decTag )
    _, Sjkerr = jk_error( cmass_catS, njack = njack , target = angular_correlation, jkargs=[cmass_catS, rand_catS], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )
    _, Djkerr = jk_error( dmass, njack = njack , target = angular_correlation, jkargs=[dmass, rand_catD], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )

    #    DAT = np.column_stack(( theta_NGC, w_NGC, jkerr_NGC, theta_SGC, w_SGC, jkerr_SGC, thetaS, wS, Sjkerr  ))
    #np.savetxt( 'cmass_acf.txt', DAT, delimiter = ' ', header = ' theta_NGC  w_NGC  jkerr_NGC  theta_SGC  w_SGC   jkerr_SGC   thetaS  wS  Sjkerr ' )

    fig, ax = plt.subplots(1,1, figsize = (7, 7))

    #ax.errorbar( theta_SGC*0.95, w_SGC, yerr = jkerr_SGC, fmt = '.', label = 'SGC')
    ax.errorbar( thetaS* 1.05, wS, yerr = Sjkerr, fmt = '.', label = 'CMASS')
    ax.errorbar( thetaD, wD, yerr = Djkerr, fmt = '.', label = 'DMASS')

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')








