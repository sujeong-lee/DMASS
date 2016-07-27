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
import fitsio
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

def classifier_test(sdss_data, des_data, suffixTag = '_dev'):
    
    des_data, sdss_data = DES_to_SDSS.match(des_data, sdss_data)
    
    try :
        if suffixTag == '_exp': use_profile = (des_data['IM3_GALPROF'] == 1)
        elif suffixTag == '_dev': use_profile = (des_data['IM3_GALPROF'] == 2)
        elif suffixTag == '_all': use_profile = (des_data['IM3_GALPROF'] < 3 )

        des_data = des_data[ use_profile ]
        sdss_data = sdss_data[ use_profile ]
    

    except ValueError :
        if 'IM3_GALPROF' not in des_data.dtype.names:
            print 'no profile info'
        else : pass
    
    """
    classifier_tags = ['mag_detmodel','magerr_detmodel','mag_psf','magerr_psf','mag_hybrid','mag_model',
                     'mag_auto','magerr_auto','mag_petro','spread_model','spreaderr_model',
                     'mag_aper_3','mag_aper_4','mag_aper_5','mag_aper_6','mag_aper_7',
                     'mag_aper_8','mag_aper_9','mag_aper_10','wavgcalib_mag_psf',
                     'flux_radius','mu_mean_model','mu_max_model','mu_eff_model']
    """
    filters = ['g', 'r', 'i', 'z']
    classifier_tags = ['flux_model' ] #,'mag_detmodel','mag_psf','mag_model','mag_auto','mag_hybrid','mag_petro','mu_mean_model','mu_max_model' ]
    corrected_tags = ['modelmag', 'cmodelmag']
    sdsscolor_tags = ['mag_detmodel','mag_model','mag_auto']
    
    #i3_tags = ['i3_radius','i3_chi2_pixel','i3_e1','i3_e2','petro_radius'] #'i3_disc_flux','i3_bulge_flux',]
    
    sdsscolor_tags = [tag+'_'+f+'_sdss' for tag in sdsscolor_tags for f in filters]
    corrected_tags = [tag+'_'+f+'_des' for tag in corrected_tags for f in filters]
    classifier_tags = [tag+'_'+f for tag in classifier_tags for f in filters]
    
    des_tags =  sdsscolor_tags + ['fiber2mag_I_DES'] #+ i3_tags
    #des_tags = corrected_tags_g + corrected_tags_r + corrected_tags_i + ['fiber2mag_I_DES']
    des_tags = [ tag.upper() for tag in des_tags ]
    
    print des_tags
                                 
    # sdss tag
    sdss_classifier_tags = ['modelMag','modelMagErr','fiber2Mag','cModelMag','cModelMagErr',
                          'fracDeV','extinction','skyIvar','psffwhm',
                          'expMag','expMagErr','deVMag','deVMagErr','psfMag']

    sdss_tags_g = [tag +'_g' for tag in sdss_classifier_tags]
    sdss_tags_r = [tag +'_r' for tag in sdss_classifier_tags]
    sdss_tags_i = [tag +'_i' for tag in sdss_classifier_tags]
    sdss_tags_z = [tag +'_z' for tag in sdss_classifier_tags]
    sdss_tags = sdss_tags_g + sdss_tags_r + sdss_tags_i + sdss_tags_z + ['type','clean']

    sdss_tags = [ tag.upper() for tag in sdss_tags ]

    print 'total (sdss/des) :', len(sdss_data) ,len(des_data)
    
    # ----------------------
    
    des_prior_cut = cmass_criteria(des_data, sdss = None)
    des_data = des_data[des_prior_cut]
    sdss_data = sdss_data[des_prior_cut]
    
    train_size = len(des_data)*3/5
    
    cl, predict_mask, good_mask = classifier3(sdss_data, des_data, des_tags = des_tags, sdss_tags = sdss_tags, train_size = train_size)
    
    print predict_mask.size, des_data.size
    #cl, predict_mask, good_mask = classifier(sdss_data, des_data, des_tags = des_tags, sdss_tags = sdss_tags)
    #predicted_des_data = des_data[predict_mask]
    # good_des_data = des_data[good_mask]
    #predicted_des_data = des_data[predict_mask]
    #good_des_data = des_data[good_mask]
    print 'des_cmass :', np.sum(predict_mask), np.sum(good_mask)
    #return predicted_des_data, good_des_data
    return des_data[predict_mask]


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
    dperp = modelmag_r_des - modelmag_i_des - (modelmag_g_des - modelmag_r_des)/8.0

    cut = ((cmodelmag_i_des > 17) &
           (cmodelmag_i_des < 22.) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) &
           ((modelmag_r_des - modelmag_i_des ) > 0.0 ) &
           ((modelmag_g_des - modelmag_r_des ) > 0.5 ) &
            ((modelmag_g_des - modelmag_r_des ) < 2.5 ) &
           (fib2mag_des < 21.0 )# &
           #(dperp > 0.4)
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
    
    try :
        expID = des_exp['COADD_OBJECTS_ID']
        devID = des_dev['COADD_OBJECTS_ID']
        fullID = fulldes['COADD_OBJECTS_ID']
        expmask = np.in1d(fullID, expID)
        devmask = np.in1d(fullID, devID)
    
    except ValueError:

        import esutil
        h = esutil.htm.HTM(10)
        expmask, _, _ = h.match( fulldes['RA'], fulldes['DEC'], des_exp['RA'], des_exp['DEC'], 1./3600, maxmatch=1 )
        devmask, _, _ = h.match( fulldes['RA'], fulldes['DEC'], des_dev['RA'], des_dev['DEC'], 1./3600, maxmatch=1 )


    im3galprofile[expmask] = 1
    im3galprofile[devmask] = 2
    
    data = rf.append_fields(fulldes, 'IM3_GALPROF', im3galprofile)
    
    print np.sum(expmask), np.sum(devmask), np.sum( (~expmask) * (~devmask))
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
        #X = np.vstack([flux_g, flux_r, flux_i, flux_z]).T
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
        des_z = data['MODELMAG_Z'] - data['EXTINCTION_Z']
        des_ci = data['CMODELMAG_I'] - data['EXTINCTION_I']
        des_cr = data['CMODELMAG_R'] - data['EXTINCTION_R']
        
        X = np.vstack([des_cr, des_ci, des_g, des_r, des_i, des_z]).T
        #X = np.vstack([des_ci, des_g, des_r, des_i]).T
        Xerr = np.vstack([data['CMODELMAGERR_R'] ,
                          data['CMODELMAGERR_I'],
                          data['MODELMAGERR_G'],
                          data['MODELMAGERR_R'],
                          data['MODELMAGERR_I'],
                          data['MODELMAGERR_Z']
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




def MachineLearningClassifier( cmass, lowz,  train = None, test = None, sub=None):
    
    from sklearn.neighbors import KNeighborsClassifier
    from astroML.classification import GMMBayes
    import esutil

    if sub is None:
        train, _ = priorCut(train)
        test, _ = priorCut(test)
    
    elif sub is True: #pass
        
        train, _ = SDSS_cmass_criteria(train, prior=True)
        test, _ = SDSS_cmass_criteria(test, prior=True)

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
    
    print 'num of cmass/lowz in train', np.sum(cmass_mask)
    print 'num of cmass/lowz in test', np.sum(cmass_mask_test)
    
    data = np.hstack(( train, test ))
    
    # stack DES data
    X, Xcov = mixing_color(data, sdss = sub)
        
    X_train = X[ :len(train), :]
    X_test = X[-len(test):, :]
    
    y_train = np.zeros( train.size, dtype=int)
    y_train[cmass_mask] = 1
    y_test = np.zeros( test.size, dtype=int)
    y_test[cmass_mask_test] = 1


    def compute_kNN( k_neighbors_max = 10 ):
        classifiers = []
        predictions = []
        kvals = np.arange(1,k_neighbors_max, 10)
        
        print 'kNN'
        
        for k in kvals:
            classifiers.append([])
            predictions.append([])
            
            clf = KNeighborsClassifier(n_neighbors=k,n_jobs = -1, weights='distance')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pur = np.sum(y_pred * y_test) *1./np.sum(y_pred)
            com = np.sum(y_pred * y_test) *1./np.sum(y_test)
            print 'n_neighbor, com/pur', k, com, pur
            classifiers[-1].append(clf)
            predictions[-1].append(y_pred)
        
        return classifiers, predictions

    def compute_GMMB(Ncomp_max = 20):
        
        print 'GMMB'
        classifiers = []
        predictions = []
        Ncomp = np.arange(1,Ncomp_max,10)

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

    _, predictions_kNN = compute_kNN(k_neighbors_max = 100)
    _, predictions_GMMB = compute_GMMB(Ncomp_max = 100)


    clf_kNN = KNeighborsClassifier(n_neighbors= 80, n_jobs = -1, weights='distance')
    clf_kNN.fit(X_train, y_train)
    y_pred_kNN = clf_kNN.predict(X_test)
    
    kNN_CMASS = np.zeros(y_pred_kNN.size, dtype=int)
    kNN_CMASS[y_pred_kNN == 1] = 1
    print kNN_CMASS
    
    test = rf.append_fields(test, 'KNN_CMASS', kNN_CMASS )
    
    print 'kNN  pur/com', np.sum(y_pred_kNN * y_test) *1./np.sum(y_pred_kNN), np.sum(y_pred_kNN * y_test) *1./np.sum(y_test)
    
    return test[y_pred_kNN==1]
    
    """
    clf_GMMB = GMMBayes(11, min_covar=1E-5, covariance_type='full')
    clf_GMMB.fit(X_train, y_train)
    y_pred_GMMB = clf_GMMB.predict(X_test)

    print 'GMMB pur/com', np.sum(y_pred_GMMB * y_test) *1./np.sum(y_pred_GMMB), np.sum(y_pred_GMMB * y_test) *1./np.sum(y_test)

    return test[y_pred_kNN == 1], test[y_pred_GMMB == 1]
    """


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




def make_truthtable(sample_all, sample_cmass):
    import esutil
    h = esutil.htm.HTM(10)
    matchDist = 1./3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(sample_all['RA'],sample_all['DEC'], sample_cmass['RA'], sample_cmass['DEC'],matchDist,maxmatch=1)
    cmass = np.in1d( np.arange(sample_all.size), m_des)
    return cmass




def reweight( train = None, test = None):

    """
    train : catalog that will be cut
    test : catalog used for assigning weight (= model)
    """
    train, _ = priorCut(train)

    #from matplotlib.colors import LogNorm
    import matplotlib.colors as colors
    # parameter normalization
    X_train, _  = mixing_color(train, cmass=None)
    X_test, _  = mixing_color(test, cmass=None)
    params = [  X_train[:,1], X_train[:,1], X_train[:,2], X_train[:,3]]# , X_train[:,4]]
    params_t = [  X_test[:,1], X_test[:,1], X_test[:,2], X_test[:,3]]#, X_test[:,4]]

    ms, mts = [ p/p.max() for p in params ], [ p/p.max() for p in params_t ]
    
    # same volume
    """
        Radius = 0.02
        weight = []
        for i in range(training.size):
        
        m_alpha1 = m1[i]
        m_alpha2 = m2[i]
        distance = np.sqrt( (m1-m_alpha1)**2 + (m2 - m_alpha2)**2 )
        distance_test = np.sqrt( (mt1-m_alpha1)**2 + (mt2 - m_alpha2)**2 )
        
        N = len(distance[distance < Radius])
        N_test = len(distance_test[distance_test < Radius])
        w = N_test * 1.0 / N / test.size
        weight.append(w)
    """
    # same number
    alpha = 5
    weight = []
    
    
    # multiprocessing
    from multiprocessing import Process, Queue
    n_process = 12
    range_split = np.array_split(range(train.size), n_process, axis=0)
    
    
    
    from time import time
    t1 = time()
    
    print 'multiprocessing...',

    def reweight_forloop(rangelist):
        ds = []
        ds_test = []
        for i in rangelist:
            distance = np.sqrt( np.sum([ (ms[j]-ms[j][i])**2 for j in range(len(params))], axis = 0) )
            distance_test = np.sqrt( np.sum([ (mts[j]-ms[j][i])**2 for j in range(len(params))], axis =0) )
            ds.append( np.partition(distance, alpha)[alpha] )
            ds_test.append( np.partition(distance_test, alpha)[alpha] )
        
        weight = (np.array(ds)/np.array(ds_test))**(len(params)) / test.size
        return weight
    
    def reweight_process(q, order, rangelist ):
        weight = reweight_forloop( rangelist )
        sys.stdout.write('...')
        q.put((order, weight))
    
    q_re = Queue()
    reweight_Processes = [Process( target = reweight_process, args=(q_re, z[0], z[1])) for z in zip(range(n_process), range_split)]
    for p in reweight_Processes : p.start()
    result = [q_re.get() for p in reweight_Processes]
    result.sort()
    weight = np.hstack(np.array([r[1] for r in result]))
    
    print 'time ', time() - t1, 's'

    norm_weight = weight/np.sum(weight)
    print 'sum of weight', np.sum(norm_weight)

    tags = train.dtype.names
    if 'REWEIGHT' in tags: train['REWEIGHT'] = norm_weight
    else:train = rf.append_fields(train, 'REWEIGHT', norm_weight)

    return train


def makePlotReweight(train = None, test = None, cmass = None ):

    weight = train['REWEIGHT']
    resample = train[ weight > np.median(weight)*3500 ]
    #resample = np.random.choice(train, size=test.size, p = weight)
    print test.size, resample.size
    
    X_train, _  = mixing_color(train, cmass=cmass)
    X_test, _  = mixing_color(test, cmass=cmass)
    X = np.vstack((X_train,X_test))
    icmod, dperp = X[:,1], X[:,3]-X[:,2]/8.0
    
    X_re, _  = mixing_color(resample, cmass=cmass)
    mr1, mr2 = X_re[:,1], X_re[:,3] - X_re[:,2]/8.0

    # plotting
    bins1 = np.linspace(18.0, 20.3, 50)
    bins2 = np.linspace(0.0, 1.6, 50)
    
    fig, axes = plt.subplots(2,2,figsize = (13,10))
    axes = axes.ravel()
    H = [None, None, None, None]
    _, _, _, H[0] = axes[0].hist2d(icmod[:train.size], dperp[:train.size], bins = [bins1,bins2], normed=True)
    _, _, _, H[1] = axes[1].hist2d(icmod[train.size:], dperp[train.size:], bins = [bins1,bins2], normed=True)
    _, _, _, H[3] = axes[3].hist2d(mr1, mr2, bins = [bins1,bins2], normed=True)

    idx = np.log10(weight).argsort()
    x = icmod[:train.size][idx]
    y = dperp[:train.size][idx]
    z = np.log10(weight)[idx]
    H[2] = axes[2].scatter(x,y, color=z, s = 30, edgecolor='')

    titles = ['DES', 'CMASS', 'weight', 'Resampled']
    #clabel = [ 'log10(N/Ntot)', 'log10(N/Ntot)', 'log10(weight)', 'log10(N/Ntot)' ]
    clabel = [ 'N', 'N', 'log10(weight)', 'N' ]
    for i, a in enumerate(axes):
        a.set_xlabel('cmod_i')
        a.set_ylabel('dperp')
        a.legend(loc='best')
        a.set_title( titles[i] )
        cbar = fig.colorbar( H[i], ax = a )
        cbar.set_label( clabel[i] )

    axes[2].axis([18.0, 20.3, 0.0, 1.6])
    fig.savefig('figure/reweight')
    print 'figure/reweight.png'
    return 0

    


def doVisualization3( true, data1, data2, labels = None, ranges = None, nbins=100, prefix= 'default' ):
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

    filename = prefix+"diagnostic_histograms3.png"
    print "writing output plot to: "+filename
    fig.savefig(filename)



def doVisualization_1d( true, data1, labels = None, ranges = None, nbins=100, prefix= 'default' ):
    
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

    fig,axes = plt.subplots(nrows=1, ncols= ndim, figsize= (6*ndim, ndim) )
    
    for i in xrange(ndim):
        xbins = np.linspace(ranges[i][0],ranges[i][1],100)
        axes[i].hist(data1[:,i],bins=xbins,normed=True,label='data1')
        axes[i].set_xlabel(labels[i])
        #axes[i].hist(data2[:,i],bins=xbins,normed=True,alpha=0.5,label='data2')
        axes[i].hist(true[:,i],bins=xbins,normed=True,alpha=0.5,label='true')
        axes[i].legend(loc='best')

    filename = "figure/"+prefix+"diagnostic_histograms_1d.png"
    print "writing output plot to: "+filename
    fig.savefig(filename)



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

                
                






def XDGMM_model(cmass, lowz, train = None, test = None, mock = None, reverse=None, prefix = ''):

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
    
    
    if mock is None:
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

    # stack DES data

    #if reverse is True:
    X_cmass, Xcov_cmass = mixing_color( cmass )
    X_train, Xcov_train = mixing_color(train, cmass = reverse)
    X_test, Xcov_test = mixing_color(test, cmass = reverse)
    X, Xcov = np.vstack((X_train,X_test)), np.vstack((Xcov_train, Xcov_test))


    y_train = np.zeros(( train.size, 2 ), dtype=int)
    y_train[:,0][cmass_mask] = 1
    y_train[:,1][lowz_mask] = 1
    if mock is None : y = np.vstack((y_train, y_test))
    
    

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
    
    pickleFileName_GMM = ['pickle/'+prefix+'bic_cmass.pkl', 'pickle/'+prefix+'bic_all.pkl']
    if reverse is True : pickleFileName_GMM = ['pickle/'+'reverse_GMM_bic_cmass.pkl', 'pickle/'+'reverse_GMM_bic_all.pkl']
    n_cl_cmass, aic_cmass, bic_cmass = FindOptimalN( N, X_cmass, pickleFileName = pickleFileName_GMM[0])
    rows = np.random.choice(np.arange(X_train.shape[0]), 10 * len(X_train[cmass_mask]))
    n_cl_all, aic_all, bic_all  = FindOptimalN( N, X_train[rows,:], pickleFileName = pickleFileName_GMM[1])
    #n_cl_no, aic_no, bic_no = FindOptimalN( N, X_train[~cmass_mask], pickleFileName = 'GMM_bic_no.pkl')
    
    DAT = np.column_stack(( N, bic_cmass, bic_all, aic_cmass, aic_all))
    np.savetxt('BIC.txt', DAT, delimiter = ' ', header = 'N, bic_cmass, bic_all,  aic_cmass, aic_all' )
    print 'save to BIC.txt'

    #n_cl_cmass = 5
    #n_cl_no = 25
    #n_cl_all = 25
    # ----------------------------------
  
    pickleFileName = ['pickle/'+prefix +'XD_all.pkl', 'pickle/'+'small_'+'XD_dmass.pkl', 'pickle/'+prefix+'XD_no.pkl']
    def XDDFitting( X_train, Xcov_train, init_means=None, init_covars = None, init_weights = None, filename = None, n_cluster = 25 ):
        clf = None
        @pickle_results(filename, verbose = True)
        def compute_XD(n_clusters=n_cluster, n_iter=500, verbose=True):
            clf= XDGMM(n_clusters, n_iter=n_iter, tol=1E-2, verbose=verbose)
            clf.fit(X_train, Xcov_train, init_means=init_means, init_covars = init_covars, init_weights = init_weights)
            return clf
        clf = compute_XD()
        return clf

    # calculates CMASS fits
    clf_cmass = XDDFitting( X_cmass, Xcov_cmass, filename=pickleFileName[1], n_cluster=n_cl_cmass)

    # giving initial mean, V, amp
    """
    cV = clf_cmass.V
    cmu = clf_cmass.mu
    calpha = clf_cmass.alpha
    
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
    """
    
    clf = XDDFitting( X_train, Xcov_train, filename= pickleFileName[0], n_cluster= n_cl_all)
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
    
    X_split = np.array_split(X, n_process, axis=0)
    Xcov_split = np.array_split(Xcov, n_process, axis=0)
    
    print 'multiprocessing...',
    
    def logprob_process(q,  classname, order,(data, cov)):
        re = classname.logprob_a(data, cov)
        sys.stdout.write('...')
        result = logsumexp(re, axis = 1)
        q.put((order, result))
        
    #inputs = [ (X_test_split[i], Xcov_test_split[i]) for i in range(n_process) ]
    inputs = [ (X_split[i], Xcov_split[i]) for i in range(n_process) ]
    
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
    
    
    train = rf.append_fields(train, 'EachProb_CMASS', EachProb_CMASS[:train.size])
    test = rf.append_fields(test, 'EachProb_CMASS', EachProb_CMASS[train.size:])
    # -----------------------------------------------
    
    
    print 'add noise to samples...'
    """
    X_sample_cmass = clf_cmass.sample(train[cmass_mask].size)
    X_sample_all = clf.sample(train.size )
    #X_sample_no = clf_nocmass.sample(train[~cmass_mask].size)

    X_sample_all_split = np.array_split(X_sample_all, n_process, axis=0)
    


    def addingerror_process(q, order, (model, data, cov)):
        re = add_errors(model, data, cov)
        sys.stdout.write('...')
        q.put((order, re))
    
    inputs = [ (X_sample_all_split[i], X_train, Xcov_train) for i in range(n_process) ]
    
    q = Queue()
    Processes = [Process(target = addingerror_process, args=(q, z[0], z[1])) for z in zip(range(n_process), inputs)]
    
    for p in Processes: p.start()
    result = [q.get() for p in Processes]
    result.sort()
    result = [r[1] for r in result ]

    noisy_X_sample_all = np.vstack( result[:n_process] ) #result[1]
    noisy_X_sample_cmass = add_errors( X_sample_cmass, X_train[cmass_mask], Xcov_train[cmass_mask] )
    """

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
    if mock is None:
        result = np.hstack((train, test))
        print "calculate comp/pur"
        completeness_purity( test, test[y_test[:,0]], prefix = prefix)
        #GetCMASS_mask = test['EachProb_CMASS'] > p_threshold
        return result

    if mock is not None:
        return test

    # --------------------



def completeness_purity( testAll, true, prefix = ''):

    #truecmass, _ = DES_to_SDSS.match( testAll, true )
    truecmass = true.copy()
    p, pstep = np.linspace(0.0, 1.0, 30, retstep = True)
    pbin_center = p[:-1] + pstep/2.
    
    coms=[]
    purs=[]
    coms2 = []
    purs2 = []
    
    pps = []
    
    truth_mask = make_truthtable(testAll, truecmass)
    
    for pp in p:
        
        GetCMASS_mask_loop = testAll['EachProb_CMASS'] >= pp
        #common, _ = DES_to_SDSS.match( truecmass, testAll[GetCMASS_mask_loop]  )
        #purity_cmass = len(common) * 1./len(testAll[GetCMASS_mask_loop] )
        purity_cmass = np.sum( truth_mask * GetCMASS_mask_loop ) * 1./np.sum(GetCMASS_mask_loop)
        completeness_cmass =  np.sum( truth_mask * GetCMASS_mask_loop ) * 1./np.sum(truth_mask)
        #completeness_cmass  = len(common)*1./ len(truecmass )
        completeness_cmass2 = np.sum( testAll[GetCMASS_mask_loop]['EachProb_CMASS'])/ np.sum( testAll['EachProb_CMASS'] )
        purity_cmass2 = np.mean( testAll[GetCMASS_mask_loop]['EachProb_CMASS'])
        print 'p, com/pur', pp, completeness_cmass, purity_cmass
        #print pp, completeness_cmass, purity_cmass
        pps.append(pp)
        coms.append(completeness_cmass)
        purs.append(purity_cmass)
        coms2.append(completeness_cmass2)
        purs2.append(purity_cmass2)


    DAT = np.column_stack(( pps, coms, purs, coms2, purs2 ))
    np.savetxt('com_pur_results/'+prefix+'com_pur.txt', DAT, delimiter = ' ', header = 'pps, true com, true pur, est_com, est_pur')
    print 'writing txt to com_pur_results/'+prefix+'com_pur.txt'
    
    
    # probability calibration
    prob = []
    real = []
    for i in xrange(len(pbin_center)):
        these = ( testAll['EachProb_CMASS'] > p[i]) & ( testAll['EachProb_CMASS'] <= p[i+1])
        denozero = np.sum(these)
        #common, _ = DES_to_SDSS.match( truecmass, testAll[these] )
        print 'p, common, these', p[i+1], np.sum( truth_mask * these ), np.sum(these)
        if denozero is not 0 and  np.sum( truth_mask * these ) is not 0 :
            prob.append( np.sum( truth_mask * these )  * ( 1./np.sum(these) ))
            real.append( np.sum( truth_mask * GetCMASS_mask_loop )  * (1./len(truth_mask) ))
        else:
            prob.append( 0 )
            real.append( 0 )

    these = ( testAll['EachProb_CMASS'] == 1.0 )
    print 'p, common/these', '1.0', np.sum( truth_mask * these ), np.sum(these)
    DAT2 = np.column_stack(( pbin_center, prob, real))
    np.savetxt('com_pur_results/'+prefix+'prob_calib.txt', DAT2, delimiter = ' ', header = 'pcenter, prob, real')
    print 'writing txt to com_pur_results/'+prefix+'prob_calib.txt'

def keepSDSSgoodregion( data ):
    
    Tags = ['bad_field_mask', 'unphot_mask', 'bright_star_mask', 'rykoff_bright_star_mask','collision_mask', 'centerpost_mask']
    
    mask = np.ones( data.size, dtype = bool )
    
    for tag in Tags:
        print tag
        mask = mask * (data[tag] == 0)
    
    print 'masked objects ', mask.size - np.sum(mask)
    
    data_tags = list(data.dtype.names)
    reduced_tags = []
    for tag in data_tags:
        if tag not in Tags: reduced_tags.append(tag)
    reduced_data = data[reduced_tags]
    return reduced_data[mask]


def MakePlots(prefix = ''):

    cp = np.loadtxt('com_pur_results/'+prefix+'com_pur.txt')
    pps = cp[:,0]
    coms = cp[:,1]
    purs = cp[:,2]
    coms2 = cp[:,3]
    purs2 = cp[:,4]

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


    calib = np.loadtxt('com_pur_results/'+prefix+'prob_calib.txt')
    pbin_center = calib[:,0]
    prob = calib[:,1]
    real = calib[:,2]

    fig,ax = plt.subplots()
    ax.plot(pbin_center, prob,label='true fraction' )
    ax.plot(pbin_center, real,label='total fraction' )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    fig.savefig('com_pur_results/'+prefix + 'probability_calibration.png')
    print 'save fig: com_pur_results/'+prefix+'probability_calibration.png'


def com_pur_arbi(cat = None, true = None, prefix = '', subset = None):
    
    import fitsio

    dec = -1.0
    dec2 = 1.0
    ra = 310.0
    ra2 = 360.
    
    
    """
    if subset is None:
        cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
        cmass_data = Cuts.SpatialCuts(cmass_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
        cmass = Cuts.keepGoodRegion(cmass_data)
    
    else:
        cmass = subset.copy()
        #cmass, _ = DES_to_SDSS.match( clean_cmass_data, dmass )
        #print 'dmass, subcmass', len(dmass), len(cmass)

    """
    cmass = true
    
    testAll = cat
    #mask = testAll['EachProb_CMASS'] > 1.0
    #testAll['EachProb_CMASS'][mask] = 1.0

    #truecmass, _ = DES_to_SDSS.match( testAll, cmass )

    import esutil
    h = esutil.htm.HTM(10)
    matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    m_cmass, _, _ = h.match(testAll['RA'],testAll['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass = np.zeros( testAll.size, dtype = bool)
    true_cmass[m_cmass] = 1
    

    p, pstep = np.linspace(0.0, 1.0, 26, retstep = True)
    pbin_center = p[:-1] + pstep/2.
    
    coms=[]
    purs=[]
    coms2 = []
    purs2 = []
    
    pps = []

    print 'expected com/pur'

    for pp in p:
        
        GetCMASS_mask_loop = testAll['EachProb_CMASS'] >= pp
        #common, _ = DES_to_SDSS.match( truecmass, testAll[GetCMASS_mask_loop]  )
        #purity_cmass = len(common) * 1./len(testAll[GetCMASS_mask_loop] )
        purity_cmass = np.sum( true_cmass * GetCMASS_mask_loop ) * 1./ np.sum(GetCMASS_mask_loop )
        completeness_cmass = np.sum( true_cmass * GetCMASS_mask_loop ) *1./np.sum(true_cmass)
        
        #completeness_cmass  = len(common) * 1./ len(truecmass)
        
        completeness_cmass2 = np.sum( testAll[GetCMASS_mask_loop]['EachProb_CMASS'])/ np.sum( testAll['EachProb_CMASS'] )
        purity_cmass2 = np.mean( testAll[GetCMASS_mask_loop]['EachProb_CMASS'])
        print np.sum( GetCMASS_mask_loop ), np.sum( true_cmass * GetCMASS_mask_loop ), np.sum(true_cmass)
        print pp, completeness_cmass, purity_cmass
        pps.append(pp)
        coms.append(completeness_cmass)
        purs.append(purity_cmass)
        coms2.append(completeness_cmass2)
        purs2.append(purity_cmass2)



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


    stop
    
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





def angularcorr( dmass = None, subCMASS = None, cmass = None, ra = 340, ra2 = 360, dec=-1.0, dec2 = 1.0, suffix = ''):
    
    from systematics import angular_correlation, jk_error
    # CMASS SGC, CMASS LOCAL, subCMASS, DMASS
    
    # testing dmass with angular correlation function ------------------------------------------------
    #rand_catD = Balrog_DMASS.copy()
    import fitsio
    cmass_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_South.fits.gz')
    cmass_cat_SGC = Cuts.keepGoodRegion(cmass_cat_SGC)
    rand_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz')
    rand_cat_SGC = Cuts.keepGoodRegion(rand_cat_SGC)
    
    cmass_catS = cmass.copy()
    if cmass is None: cmass_catS = fitsio.read('noblend_clean_cmass_data.fits')
    cmass_catS = Cuts.SpatialCuts(cmass_catS, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catS = Cuts.SpatialCuts(rand_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    
    dmass_cat = Cuts.SpatialCuts(dmass, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catD = rand_catS.copy()
    subCMASS_cat = Cuts.SpatialCuts(subCMASS, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catsub = rand_catS.copy()

    
    weight_rand_SGC = rand_cat_SGC['WEIGHT_FKP']
    weight_data_SGC = cmass_cat_SGC['WEIGHT_FKP'] * cmass_cat_SGC['WEIGHT_STAR'] * ( cmass_cat_SGC['WEIGHT_CP']+cmass_cat_SGC['WEIGHT_NOZ'] -1. )
    weight_randS = rand_catS['WEIGHT_FKP']
    weight_dataS = cmass_catS['WEIGHT_FKP'] * cmass_catS['WEIGHT_STAR'] * ( cmass_catS['WEIGHT_CP']+cmass_catS['WEIGHT_NOZ'] -1. )
    #weight_sub = subCMASS_cat['REWEIGHT']
    #weight_randsub = rand_catsub['WEIGHT_FKP']
    #weight_D = dmass_cat['EachProb_CMASS'] * 1./np.sum(dmass_cat['EachProb_CMASS'])
    #weight_randD = rand_catD['WEIGHT_FKP']
    
    theta_SGC, w_SGC, _ = angular_correlation(cmass_cat_SGC, rand_cat_SGC, weight = [weight_data_SGC, weight_rand_SGC])
    thetaS, wS, werrS = angular_correlation(cmass_catS, rand_catS, weight = [weight_dataS, weight_randS])
    thetaD, wD, werrD = angular_correlation(dmass_cat, rand_catD ) #, weight = [weight_D, weight_randD])
    thetasub, wsub, werrsub = angular_correlation(subCMASS_cat, rand_catsub ) #, weight = [weight_sub, weight_randsub ])
    # jk errors
    
    #from astroML.plotting import setup_text_plots
    #setup_text_plots(fontsize=20, usetex=True)
    
    njack = 30
    raTag = 'RA'
    decTag = 'DEC'
    _, jkerr_SGC = jk_error( cmass_cat_SGC, njack = njack , target = angular_correlation, jkargs=[cmass_cat_SGC, rand_cat_SGC], jkargsby=[[raTag, decTag],[raTag, decTag]], raTag = raTag, decTag = decTag )
    _, Sjkerr = jk_error( cmass_catS, njack = njack , target = angular_correlation, jkargs=[cmass_catS, rand_catS], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )
    _, Djkerr = jk_error( dmass_cat, njack = njack , target = angular_correlation, jkargs=[dmass_cat, rand_catD], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )
    _, subjkerr = jk_error( subCMASS_cat, njack = njack , target = angular_correlation, jkargs=[subCMASS_cat, rand_catsub], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )
    
    DAT = np.column_stack((thetaS, wS, Sjkerr, thetaD, wD, Djkerr, thetasub, wsub, subjkerr,  theta_SGC, w_SGC, jkerr_SGC,   ))
    np.savetxt( 'acf_comparison'+suffix+'.txt', DAT, delimiter = ' ', header = 'thetaS  wS  Sjkerr, thetaD, wD, Djkerr, thetasub, wsub, subjkerr  theta_SGC  w_SGC   jkerr_SGC    ' )
    print 'writing txt to acf_comparison'+suffix+'.txt'
    
    
    """
    fig, ax = plt.subplots(1,1, figsize = (7, 7))
    
    ax.errorbar( theta_SGC*0.95, w_SGC, yerr = jkerr_SGC, fmt = '.', label = 'SGC')
    ax.errorbar( thetaS* 1.05, wS, yerr = Sjkerr, fmt = '.', label = 'CMASS local')
    ax.errorbar( thetaD, wD, yerr = Djkerr, fmt = '.', label = 'DMASS')
    ax.errorbar( thetasub, wsub, yerr = subjkerr, fmt = '.', label = 'subCMASS')
    
    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')
    fig.savefig('acf_comparison'+suffix+'.png')
    print 'writing plot to acf_comparison'+suffix+'.png'
    """


def _resampleWithPth( des ):
    
    pth_bin, step = np.linspace(0.01, 1.0, 200, endpoint = True, retstep = True)
    pcenter = pth_bin + step/2.
    
    dmass = [None for i in range(pcenter.size)]
    
    for i in range(pcenter.size-1):
        mask = (des['EachProb_CMASS'] >= pth_bin[i])&(des['EachProb_CMASS'] < pth_bin[i]+step)
        cat = des[mask]
        dmass[i] = np.random.choice( cat, size = np.around(pcenter[i] * cat.size) )
        print pth_bin[i], pth_bin[i]+step, pcenter[i], dmass[i].size * 1./cat.size, dmass[i].size
    
    
    i = i+1
    mask = (des['EachProb_CMASS'] >= pth_bin[i])
    cat = des[mask]
    dmass[i] = np.random.choice( cat, size = int( 0.75 * cat.size) )
    print pth_bin[i], pth_bin[i]+step, 0.75, dmass[i].size * 1./cat.size, dmass[i].size
    
    dmass = np.hstack(dmass)
    return dmass


def makeTagUppercase(cat):
    names = list(cat.dtype.names)
    names = [ n.upper() for n in names ]
    cat.dtype.names = names
    return cat

def resampleWithPth( des ):
    
    pmin = 0.1
    pmax = 0.75
    p1stbin = 0.1
    pth_bin, step = np.linspace(0.0, 1.0, 200, endpoint = True, retstep = True)
    pcenter = pth_bin + step/2.
    
    # ellipse
    pcurve = (pmax-pmin) * np.sqrt( 1 - (pcenter - 1)**2 * 1./(1-p1stbin)**2 ) + pmin
    pcurve[ pcenter < pmin ] = 0.0
    pcurve[ np.argmax(pcurve)+1:] = pmax
    
    # straight
    pcurve2 = np.zeros(pth_bin.size)
    pcurve_front = np.linspace(0.0, 1.0, 200, endpoint = True)
    pcurve2[:pcurve_front.size] = pcurve_front
    pcurve2[pcurve_front.size:] = pmax

    ellipse_dmass = [None for i in range(pth_bin.size)]
    straight_dmass = [None for i in range(pth_bin.size)]
    el_fraction, st_fraction = np.zeros(pth_bin.size),  np.zeros(pth_bin.size)
    for i in range(pth_bin.size-1):
        mask = (des['EachProb_CMASS'] >= pth_bin[i])&(des['EachProb_CMASS'] < pth_bin[i]+step)
        cat = des[mask]
        ellipse_dmass[i] = np.random.choice( cat, size = np.around(pcurve[i] * cat.size) )
        straight_dmass[i] = np.random.choice( cat, size = np.around(pcurve2[i] * cat.size) )
        el_fraction[i], st_fraction[i] = ellipse_dmass[i].size * 1./cat.size, straight_dmass[i].size * 1./cat.size
        #fraction.append( np.sum(mask) * 1./des.size )
        #print pth_bin[i], pth_bin[i]+step, pcurve[i], dmass[i].size * 1./cat.size, dmass[i].size
    
    i = i+1
    mask = (des['EachProb_CMASS'] >= pth_bin[i])
    cat = des[mask]
    ellipse_dmass[i] = np.random.choice( cat, size = int(pmax * cat.size) )
    straight_dmass[i] = np.random.choice( cat, size = int(1.0 * cat.size) )
    #print pth_bin[i], pth_bin[i]+step, 0.75, dmass[i].size * 1./cat.size, dmass[i].size
    
    ellipse_dmass, straight_dmass = np.hstack(ellipse_dmass), np.hstack(straight_dmass)
    
    
    # plotting selection prob----------
    """
    fig, ax = plt.subplots()

    #ax.plot( pcenter, pcurve2, color = 'grey', label = 'Ashley1', linestyle = '--')
    ax.plot( np.insert(pcenter, 0, pcenter[0]), np.insert(pcurve2, 0, 0.0), color = 'grey',linestyle = '--', label='Ashley1')
    ax.plot( np.insert(pcenter, 0, pcenter[0]), np.insert(pcurve, 0, 0.0), color = 'grey' , label = 'Ashley2')
    ax.plot( pcenter, el_fraction, color = 'red', linestyle = '--' )
    ax.plot( pcenter, st_fraction, color = 'blue',linestyle = '--'  )
    
    pbin, st = np.linspace(0.01, 1.1, 100, endpoint=False, retstep = True)
    pc = pbin + st/2.
    
    #N, edge = np.histogram( des['EachProb_CMASS'], bins = pbin)
    weights = np.ones_like(des['EachProb_CMASS'])/len(des['EachProb_CMASS'])
    ax.hist( des['EachProb_CMASS'], bins = pbin, weights = weights * 20, alpha = 0.3, color = 'green')
    #ax.plot( pc[:-1], N * 1./des.size,  )
    ax.set_xlim(0,1)
    ax.set_ylim(0,1.1)
    ax.set_ylabel('fraction')
    ax.set_xlabel('p_threshold')
    ax.legend(loc='best')
    fig.savefig('figure/selection_prob')
    print 'figure to ', 'figure/selection_prob.png'
    """
    return straight_dmass, ellipse_dmass


def addphotoz(des, im3shape):
    
    import esutil
    h = esutil.htm.HTM(10)
    m_re, m_im3,_ = h.match( des['RA'], des['DEC'], im3shape['RA'], im3shape['DEC'], 1./3600, maxmatch=1)
    desdm_zp = np.zeros(des.size, dtype=float)
    desdm_zp[m_re] = im3shape[m_im3]['DESDM_ZP']
    result = rf.append_fields(des, 'DESDM_ZP', desdm_zp)
    
    
    return result

def SDSSaddphotoz(cmass):
    # add photoz to cmass cat
    photoz = fitsio.read('/n/des/lee.5922/data/cmass_cat/cmass_photoz_radec.fits')
    sortIdx, sortIdx2 = cmass['OBJID'].sort(), photoz['OBJID'].sort()
    cmass, photoz = cmass[sortIdx].ravel(), photoz[sortIdx2].ravel()
    mask = np.in1d(cmass['OBJID'], photoz['OBJID'])
    mask2 = np.in1d(photoz['OBJID'], cmass['OBJID'][mask])
    photozline = np.zeros( cmass.size, dtype=float )
    photozline[mask] = photoz['Z'][mask2]
    cmass = rf.append_fields(cmass, 'PHOTOZ', photozline)
    return cmass


def EstimateContaminantFraction( dmass = None, cmass = None ):
    #  check difference is smaller than stat error
    
    import esutil
    m_dmass, m_true, _ = esutil.htm.HTM(10).match( dmass['RA'], dmass['DEC'], cmass['RA'], cmass['DEC'], 1./3600, maxmatch = 1)
    true_mask = np.zeros( dmass.size, dtype = bool )
    true_mask[m_dmass] = 1
    
    cross_data = np.loadtxt('data_txt/acf_cross_Ashley.txt')
    r_tc, w_tc, err_tc = cross_data[:,0], cross_data[:,1], cross_data[:,2]
    
    auto_data = np.loadtxt('data_txt/acf_comparison_Ashley.txt')
    cont_data = np.loadtxt('data_txt/acf_comparison_Ashley_cont.txt')
    r, w, err, w_t, err_t,  w_c, err_c = auto_data[:,0], auto_data[:,4],  auto_data[:,5],  auto_data[:,7],  auto_data[:,8], cont_data[:,4], cont_data[:,5]
    
    f_c = np.sum(~true_mask) * 1./dmass.size
    print "current f_c = {:>0.2f} %".format( f_c * 100)
    ang_L = 1. + w - (1. -f_c)**2 * (1. + w_t)
    ang_R = 2. * f_c * (1. -f_c)*(1. + w_tc) + f_c**2 * (1+w_c)
    
    keep = (r > 0.01) & (r < 1)
    fractionL = 1. - np.sqrt( (1. + w - ang_L) * 1./(1. + w_t) )
    fractionR = (-(1.+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* ang_R) ) * 1./(w_c-1.-2* w_tc)
    m_fL, m_fR = np.mean(fractionL), np.mean(fractionR) # current max cont fraction
    
    #print "current f_cont (LHS, RHS) : ", m_fL, m_fR
    
    # ideal fraction
    eps82 = np.mean(err[keep]/w[keep])/np.sqrt(np.sum(keep))
    eps = eps82  * np.sqrt(100./1000)
    i_fractionL = np.abs(1. - np.sqrt( (1 + w - eps82) * 1./(1 + w_t) ))
    i_fractionR = np.abs((-(1+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* eps82) ) * 1./(w_c-1.-2* w_tc))
    
    """
    # calculate everything for each point, and then remove invalid values (negative or higher
    # values than 1) and get mean.
    """
    
    m_i_fractionL, m_i_fractionR = np.mean(np.ma.masked_invalid(i_fractionL[keep])), np.mean(np.ma.masked_invalid(i_fractionR[keep]))
    
    #square weight
    we = 1./(err_t**2 + err_c**2 )
    weight = we/np.sum(we[keep])
    i_f_weighted = weight * i_fractionR
    f_weighted = np.sum(np.ma.masked_invalid(i_f_weighted[keep]))
    
    print "angular >> "
    print "- amp of systematic terms ", np.mean(np.abs(ang_R[keep]))
    print "- stat err (y1a1, st82) ", eps, eps82
    print "- ideal f_c (LHS, RHS): {:>0.4f} %, {:>0.4f} %".format(m_i_fractionL * 100, m_i_fractionR * 100)
    print "- f_weighted : {:>0.4f} %".format(f_weighted*100)
    
    filename = 'data_txt/fcont_estimation.txt'
    DAT = np.column_stack((r, i_fractionL, i_fractionR, weight, i_f_weighted, err, err_t, err_c))
    np.savetxt(filename, DAT[keep], delimiter = ' ', header = 'r, i_fractionL, i_fractionR, weight, f_weighted, err, err_t, err_c')
    print 'data file writing to ',filename
    
    
    dmass_lensing = np.loadtxt('data_txt/Ashley_lensing.txt')
    true_lensing = np.loadtxt('data_txt/Ashley_true_lensing.txt')
    cont_lensing = np.loadtxt('data_txt/Ashley_cont_lensing.txt')
    
    r_p, LS, err_LS, LS_t, err_LS_t, LS_c, err_LS_c = dmass_lensing[:,0], dmass_lensing[:,1], dmass_lensing[:,2], true_lensing[:,1], true_lensing[:,2], cont_lensing[:,1], cont_lensing[:,2]
    
    keep = (r_p > 0.0) & (~(LS == 0)) & (~(LS_c == 0)) & (~(LS_t == 0))
    len_L = LS - (1 - f_c) * LS_t
    len_R = LS_c * f_c
    
    fractionL = 1 - (LS - len_L)/LS_t
    fractionR = len_R/LS_c
    m_fL_len, m_fR_len = np.mean(fractionL[keep]), np.mean(fractionR[keep])
    #print "current f_cont (LHS, RHS) : ", m_fL_len, m_fR_len
    
    # ideal fraction
    eps_len82 = np.mean(err_LS[keep]/LS[keep])/np.sqrt(np.sum(keep))
    eps_len = eps_len82 * np.sqrt(100./1000)
    i_fractionL_len = 1. - (LS - eps_len)/LS_t
    i_fractionR_len = eps_len82/LS_c
    
    maskL = np.ma.getmask(np.ma.masked_inside(i_fractionL_len, 0.0, 1.0))
    maskR = np.ma.getmask(np.ma.masked_inside(i_fractionR_len, 0.0, 1.0))
    
    m_i_fractionL_len, m_i_fractionR_len = np.mean(i_fractionL_len[maskL * keep]), np.mean(i_fractionR_len[maskR * keep])
    
    wel = 1./( err_LS_t**2 + err_LS_c**2 )
    weight_len = wel /np.sum(wel[keep * maskR])
    i_f_weighted_len = weight_len * i_fractionR_len
    f_weighted_len = np.sum(i_f_weighted_len[keep * maskR])

    print "Lensing >>"
    print "- amp of systematic terms ", np.mean(np.abs(len_R[maskR * maskL * keep]))
    print "- stat err (y1a1, st82) ", eps_len, eps_len82
    print "- ideal f_c (LHS, RHS): {:>0.4f} %, {:>0.4f} %".format(m_i_fractionL_len * 100, m_i_fractionR_len * 100)
    print "- f_weighted : {:>0.4f} %".format(f_weighted_len*100 )

    filename2 = 'data_txt/fcont_estimation_lense.txt'
    DAT2 = np.column_stack((r_p, i_fractionL_len, i_fractionR_len, weight_len, i_f_weighted_len, err_LS,err_LS_t, err_LS_c))
    np.savetxt(filename2, DAT2[keep * maskR], delimiter = ' ', header = 'r_p, i_fractionL_len, i_fractionR_len, weight_len, f_weighted_len, err_LS, err_LS_t, err_LS_c')
    print 'data file writing to ', filename2

    """
        

    filename = 'data_txt/fcont_estimation.txt'
    filename2 = 'data_txt/fcont_estimation_lense.txt'
    ang = np.loadtxt(filename)
    lensing = np.loadtxt(filename2)
    
    r, i_fractionL, i_fractionR, weight, i_f_weighted, err, err_t,err_c = [ang[:,i] for i in range(8)]
    r_p, i_fractionL_len, i_fractionR_len, weight_len, i_f_weighted_len, err_LS, err_LS_t, err_LS_c = [lensing[:,i] for i in range(8)]
    
    err_tot = np.sqrt( (err_t**2 + err_c**2)/2. )
    err_LS_tot = np.sqrt( (err_LS_t**2 + err_LS_c**2)/2. )
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (17,7))
    ax.errorbar(r, i_fractionR, yerr = err_tot, label='fraction')
    ax.errorbar(r*1.1, i_f_weighted, yerr = weight * err_tot, label='weighted_fraction')
    ax.legend(loc='best')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel('f_c')
    ax.set_title('ideal fraction of contaminant for angular corr')
    ax2.errorbar(r_p, i_fractionR_len, yerr = err_LS_tot, label='fraction')
    ax2.errorbar(r_p*1.1, i_f_weighted_len, yerr = weight_len *err_LS_tot, label='weighted_fraction')
    ax2.legend(loc='best')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('r_p')
    ax2.set_ylabel('f_c')
    ax2.set_xlim(20, 2e+4)
    ax2.set_title('ideal fraction of contaminant for lensing')
    fig.savefig('figure/fcont_estimation.png')
    
    fig, ax = plt.subplots(1,2, figsize = (17,7))
    err_LS_tot = np.sqrt( (err_LS_t**2 + err_LS_c**2)/2. )
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (17,7))
    ax.errorbar(r, 1. - i_fractionR, yerr = err_tot, label='purity')
    ax.errorbar(r*1.1, 1.-i_f_weighted, yerr = weight * err_tot, label='weighted_purity')
    ax.legend(loc='best')
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel('r')
    ax.set_ylabel('purity')
    ax.set_ylim(0.8,1.01)
    ax.set_title('ideal purity for angular corr')
    ax2.errorbar(r_p, 1.-i_fractionR_len, yerr = err_LS_tot, label='purity')
    ax2.errorbar(r_p*1.1, 1.-i_f_weighted_len, yerr = weight_len *err_LS_tot, label='weighted_purity')
    ax2.legend(loc='best')
    ax2.set_xscale('log')
    #ax2.set_yscale('log')
    ax2.set_xlabel('r_p')
    ax2.set_ylabel('purity')
    ax2.set_xlim(20, 2e+4)
    ax2.set_ylim(0.9,1.01)
    ax2.set_title('ideal purity for lensing')
    fig.savefig('figure/purity_estimation.png')


    """

def main():
    
    # load dataset

    ra = 320.
    ra2 = 360
    dec = -1
    dec2 = 1

    full_des_data = io.getDESY1A1catalogs(keyword = 'des_st82')
    des_data_clean = keepSDSSgoodregion(Cuts.doBasicCuts(full_des_data, object = 'galaxy'))
    des = Cuts.SpatialCuts(des_data_clean, ra = ra, ra2=ra2, dec= dec, dec2= dec2  )
    
    cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    cmass_photo = io.getSDSScatalogs(file = '/n/des/lee.5922/data/bosstile-final-collated-boss2-boss38-photoObj.fits.gz')
    #cmass_data_o = io.getSDSScatalogs(file ='/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-combined.fits.gz')
    
    clean_cmass_data = Cuts.keepGoodRegion(cmass_data_o)
    cmass_data = Cuts.SpatialCuts(clean_cmass_data,ra =ra, ra2=ra2 , dec=dec, dec2=dec2 )
    noblend_clean_cmass_data, noblend_clean_cmass_data_des = DES_to_SDSS.match( cmass_data, des_data_clean)
    
    """
    #lowz_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_LOWZ_South-photoObj.fits.gz')
    #lowz_data = Cuts.SpatialCuts(lowz_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    #clean_lowz_data = Cuts.keepGoodRegion(lowz_data)
    """
    
    sdss_data_o = io.getSDSScatalogs(bigSample = True)
    sdss_data_clean = Cuts.doBasicSDSSCuts(sdss_data_o)
    sdss = Cuts.SpatialCuts(sdss_data_clean,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    #cmass_sdss = SDSS_cmass_criteria(sdss)

    import esutil
    des_im3_o = esutil.io.read( '/n/des/huff.791/Projects/CMASS/Scripts/main_combined.fits',  columns = ['COADD_OBJECTS_ID', 'DESDM_ZP', 'RA', 'DEC', 'INFO_FLAG' ], upper=True)
    
    des_im3 = des_im3_o[des_im3_o['INFO_FLAG'] == 0]
    
    des_im3_o = io.getDEScatalogs(file = '/n/des/lee.5922/data/im3shape_st82.fits')
    des_im3 = Cuts.SpatialCuts(des_im3_o ,ra =320, ra2=360 , dec=-1, dec2= 1 )
    #des = im3shape_galprof_mask(des_im3, des) # add galaxy profile mode
    
    
    balrog_o = io.LoadBalrog(user = 'JELENA.BALROG_Y1A1_S82', truth = None)
    balrog = Cuts.keepGoodRegion( balrog_o, balrog=True)
    print "alphaJ2000, deltaJ2000  -->  ra, dec"
    balrogname = list( balrog.dtype.names)
    alphaInd = balrogname.index('ALPHAWIN_J2000_DET')
    deltaInd = balrogname.index('DELTAWIN_J2000_DET')
    balrogname[alphaInd], balrogname[deltaInd] = 'RA', 'DEC'
    balrog.dtype.names = tuple(balrogname)
    balrog = AddingReddening(balrog)
    balrog = Cuts.doBasicCuts(balrog, object = 'galaxy')
    
    
    # start XD
    prefix = 'small_'
    
    #devide samples
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(des, des, [0.2,0.8], random_state=0)
    des_train = des[trainInd]
    des_test = des[testInd]


    result = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des_train, test=des_test, prefix = prefix, mock=True )
    result = addphotoz(result, des_im3)
    
    result_y1a1 = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des_train, test=y1a1, prefix = prefix, mock = True )
    result_y1a1 = addphotoz(result_y1a1, des_im3_o)

    result_balrog = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des, test=balrog, prefix = prefix, mock = True )

    #fitsio.write('result_cat/result_fullst82.fits', result)
    fitsio.write('result1.fits', result)
    #fitsio.write('result_y1a1.fits', result_y1a1)
    #fitsio.write('result_balrog.fits', result_balrog)
    
    # Ashley - match distribution and visulaization ----------------------------
    
    #calling catalogs
    noblend_clean_cmass_data_des = fitsio.read('result_cat/noblend_clean_cmass_data_des.fits')
    noblend_clean_cmass_data = fitsio.read('result_cat/noblend_clean_cmass_data.fits')
    #noblend_clean_cmass_data = SDSSaddphotoz(noblend_clean_cmass_data)
    result = fitsio.read('result_cat/result1.fits')
    #result = fitsio.read('result_cat/result_fullst82.fits')
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')

    
    #result_balrog = fitsio.read('result_cat/result_balrog.fits')
    
    As_dmass, As_dmass2 = resampleWithPth(result_y1a1)
    
    
    #As_dmass, As_dmass2 = addphotoz( As_dmass, des_im3 ), addphotoz( As_dmass2, des_im3 )
    As_X, _ = mixing_color( As_dmass )
    As2_X, _ = mixing_color( As_dmass2 )
    Xtrue,_ = mixing_color( noblend_clean_cmass_data_des )
    labels = ['MAG_MODEL_R', 'MAG_MODEL_I', 'g-r', 'r-i', 'i-z']
    ranges =  [[17,22.5], [17,22.5], [0,2], [-.5,1.5], [0.0,.8]]
    doVisualization_1d( Xtrue, As_X, labels = labels, ranges = ranges, nbins=100, prefix='Ashley_')
    doVisualization_1d( Xtrue, As2_X, labels = labels, ranges = ranges, nbins=100, prefix='Ashley2_')

    cmassIndmass, _ = DES_to_SDSS.match( noblend_clean_cmass_data, As_dmass)
    angularcorr( dmass = As_dmass, subCMASS = cmassIndmass, cmass =noblend_clean_cmass_data, ra = ra, ra2 = ra2, dec=dec, dec2 = dec2, suffix = '_Ashley_y1a1')
    cmassIndmass, _ = DES_to_SDSS.match( noblend_clean_cmass_data,  As_dmass2)
    angularcorr( dmass = As_dmass2, subCMASS = cmassIndmass, cmass =noblend_clean_cmass_data, ra = ra, ra2 = ra2, dec=dec, dec2 = dec2, suffix = '_Ashley2_y1a1')
    
    
    # corr visualzation
    suffix = '_Ashley_y1a1'
    corr_txt = np.loadtxt('acf_comparison'+suffix+'.txt')
    thetaS, wS, Sjkerr = corr_txt[:,0], corr_txt[:,1], corr_txt[:,2]
    thetaD, wD, Djkerr = corr_txt[:,3], corr_txt[:,4], corr_txt[:,5]
    thetasub, wsub, subjkerr = corr_txt[:,6], corr_txt[:,7], corr_txt[:,8]
    theta_SGC, w_SGC, jkerr_SGC = corr_txt[:,9], corr_txt[:,10], corr_txt[:,11]
    
    #cross_txt = np.loadtxt('data_txt/acf_cross'+suffix+'.txt')
    #thetaCC, wCC, CCjkerr = cross_txt[:,0], cross_txt[:,1], cross_txt[:,2]
    
    fig,ax = plt.subplots()
    
    ax.errorbar( theta_SGC, w_SGC, yerr = jkerr_SGC, fmt = '.', label = 'SGC')
    ax.errorbar( thetaS* 0.95, wS, yerr = Sjkerr, fmt = '.', label = 'CMASS local')
    ax.errorbar( thetaD, wD, yerr = Djkerr, fmt = '.', label = 'DMASS')
    #ax.errorbar( thetasub* 1.05, wsub, yerr = subjkerr, fmt = '.', label = 'CMASS in dmass')
    #ax.errorbar( thetaCC* 1.05, wCC, yerr = CCjkerr, fmt = '.', label = 'cross')
    
    ax.set_xlim(1e-3, 10)
    ax.set_ylim(1e-4, 10)
    #ax.set_ylim(1e-2,2)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')
    fig.savefig('figure/acf_comparison'+suffix+'.png')
    print 'writing plot to figure/acf_comparison'+suffix+'.png'
    
    
    # corr comparison
    labels = [ 'Ashley_y1a1', 'Ashley2_y1a1']
    corr_txt = [np.loadtxt('acf_comparison_'+s+'.txt') for s in labels]
    thetaS, wS, Sjkerr = corr_txt[0][:,0], corr_txt[0][:,1], corr_txt[0][:,2]
    
    fig, (ax, ax2) = plt.subplots(2,1, figsize = (10,15))
    ax.errorbar( thetaS, wS, yerr = Sjkerr, label = 'CMASS local', color='black', alpha = 0.5)
    ax2.errorbar( thetaS, wS-wS, yerr = Sjkerr, label = 'CMASS local', color='black', alpha = 0.9)
    
    for i in range(len(suffix)):
        thetaD, wD, Djkerr = corr_txt[i][:,3], corr_txt[i][:,4], corr_txt[i][:,5]
        ax.errorbar( thetaD* (0.90 + 0.04*i), wD, yerr = Djkerr, fmt = '.', label = labels[i])
        ax2.errorbar( thetaD* (0.90 + 0.04*i), wD - wS, yerr = Djkerr, fmt = '.', label = labels[i])
    
    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')

    ax2.set_xlim(1e-2, 10)
    ax2.set_ylim(-.5, .5)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\theta(deg)$')
    ax2.set_ylabel(r'${w(\theta)}$ - ${w_{\rm{true}}(\theta) }$')
    ax2.legend(loc='best')
    figname = 'figure/acf_comparison_y1a1.png'
    fig.savefig(figname)
    print 'writing plot to ', figname
    

    result_y1a1 = [fitsio.read('result_cat/result_y1a1_'+str(i+1)+'.fits') for i in range(4)]
    cat = [resampleWithPth(re)[0] for re in result_y1a1]

    # redshift histogram
    z_bin = np.linspace(0.1, 1.0, 200)
    labels = ['dmass Ashley', 'dmass Ashley2']
    cat = [As_dmass, As_dmass2]
    #labels = ['region '+str(i+1) for i in range(4)]

    #noblend_clean_cmass_data_des = addphotoz(noblend_clean_cmass_data_des, des_im3)
    fig, axes = plt.subplots( len(cat),1, figsize = (8,5*len(cat)))
    for i in range(len(cat)):
        axes[i].hist( photoz_des['DESDM_ZP'], bins = z_bin,facecolor = 'green', normed = True, label = 'cmass')
        axes[i].hist( cat[i]['DESDM_ZP'], bins = z_bin, facecolor = 'red',alpha = 0.35, normed = True, label = labels[i])
        axes[i].set_xlabel('photo_z')
        axes[i].set_ylabel('N(z)')
        #ax.set_yscale('log')
        axes[i].legend(loc='best')

    axes[0].set_title('y1a1 redshift hist')
    figname ='figure/hist_z_y1a12.png'
    fig.savefig(figname)
    print 'saving fig to ',figname
    


    # contaminant test --------------------------------------
    from systematics import LensingSignal, cross_angular_correlation
    
    rand_o = fitsio.read('/n/des/lee.5922/data/random0_DR11v1_CMASS_South.fits.gz')
    rand_clean = Cuts.keepGoodRegion(rand_o)
    cat_cmass_rand82 = Cuts.SpatialCuts(rand_clean,ra =320, ra2=360 , dec=-1, dec2= 1 )

    import esutil
    m_dmass, m_true, _ = esutil.htm.HTM(10).match( As_dmass['RA'], As_dmass['DEC'], noblend_clean_cmass_data['RA'], noblend_clean_cmass_data['DEC'], 1./3600, maxmatch = 1)
    true_mask = np.zeros( As_dmass.size, dtype = bool )
    true_mask[m_dmass] = 1
    true = As_dmass[true_mask]
    contaminant = As_dmass[~true_mask]


    # 1) angular clustering signal ----------------------
    
    cmassIndmass, _ = DES_to_SDSS.match( noblend_clean_cmass_data,  As_dmass)
    angularcorr( dmass = As_dmass, subCMASS = cmassIndmass, cmass =noblend_clean_cmass_data, ra = ra, ra2 = ra2, dec=dec, dec2 = dec2, suffix = '_Ashley')
    angularcorr( dmass = contaminant, subCMASS = cmassIndmass, cmass =noblend_clean_cmass_data, ra = ra, ra2 = ra2, dec=dec, dec2 = dec2, suffix = '_Ashley_cont')
    
    # cross correlation
    cross_angular_correlation(data = true, data2 = contaminant, rand = cat_cmass_rand82, rand2= cat_cmass_rand82, suffix = '_Ashley')
    
    
    # 2) Lensing signal ----------------------

    LensingSignal(lense = As_dmass, source = des_im3, rand = cat_cmass_rand82, prefix = 'Ashley_')
    LensingSignal(lense = true, source = des_im3, rand = cat_cmass_rand82, prefix = 'Ashley_true_')
    LensingSignal(lense = contaminant, source = des_im3, rand = cat_cmass_rand82, prefix = 'Ashley_cont_')

    # plotting
    prefix = 'Ashley_true_'
    filename = 'data_txt/'+prefix+'lensing.txt'
    lensingdat = np.loadtxt(filename)
    r_p_bins, LensSignal, LSjkerr, correctedLensSignal, CLSjkerr, BoostFactor, Boostjkerr = lensingdat[:,0], lensingdat[:,1],lensingdat[:,2],lensingdat[:,3],lensingdat[:,4],lensingdat[:,5],lensingdat[:,6]

    fig, ax = plt.subplots(1,1, figsize = (7,7))
    #signals = [[LensSignal, correctedLensSignal], [LSjkerr, CLSjkerr]]
    signals = [[LensSignal], [LSjkerr]]
    labels = ['Lensing', 'Corrected Lensing']
    for i in range(len(signals[0])):
        ax.errorbar(r_p_bins * (1 + 0.1*i), signals[0][i], yerr = signals[1][i], fmt='o', label = labels[i])
    theory = np.loadtxt('data_txt/smd_v_theta_cmass.dat')
    rr_the = theory[:,0]
    delta_sigma_the = theory[:,1]
    error_the = theory[:,2] * np.sqrt(5000/120)
    ax.errorbar(10**rr_the, 10**delta_sigma_the, yerr = 10**error_the, color = 'red', fmt='--o', label = 'theory')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-2,1e3)
    ax.set_xlabel(r'$r$ $(kpc/h)$')
    ax.set_ylabel(r'$\Delta\Sigma$ $(M_{s}h/pc^{2})$')
    ax.set_title('DMASS lensing signal (z_lense = [0.45, 0.55] )'  )
    ax.legend(loc = 'best')

    figname='figure/'+prefix+'lensing.png'
    fig.savefig(figname)
    print 'saving fig to :', figname


    EstimateContaminantFraction( dmass = As_dmass, cmass = noblend_clean_cmass_data )


    #___________________________________________________________
    # systematics ----------------------------------------------

    # 1) systematic maps


    from systematics import GalaxyDensity_Systematics,loadSystematicMaps, chisquare_dof

    properties = ['FWHM', 'AIRMASS', 'SKYSIGMA', 'SKYBRITE', 'NSTARS']
    properties = ['AIRMASS', 'NSTARS']
    filters = ['g','r','i', 'z']
    kind = 'STRIPE82'
    nside = 1024
    njack = 10

    property = 'NSTARS'
    filter = 'i'

    for property in properties:
        fig, ax = plt.subplots(2, 2, figsize = (15, 10))
        ax = ax.ravel()
        for i, filter in enumerate(filters):
            if property == 'NSTARS':
                nside = 512
                filename = 'y1a1_gold_1.0.2_stars_nside1024.fits'
                sysMap_o = loadSystematicMaps( filename = filename, nside = nside )
                
                if kind is 'STRIPE82' :sysMap = sysMap_o[sysMap_o['DEC'] > -5.0]
                if kind is 'SPT' :sysMap = sysMap_o[sysMap_o['DEC'] < -5.0]
                if kind is 'Y1A1':sysMap = sysMap_o
                sysMap82 = sysMap_o.copy() #loadSystematicMaps( filename = filename, nside = nside )
            else :
                sysMap = loadSystematicMaps( property = property, filter = filter, nside = nside , kind = kind)
                sysMap82 = loadSystematicMaps( property = property, filter = filter, nside = nside )
            
            sysMap82 = Cuts.SpatialCuts(sysMap82, ra = 320, ra2=360 , dec=-1 , dec2= 1 )
            
            bins, Cdensity, Cerr = GalaxyDensity_Systematics(noblend_clean_cmass_data, sysMap82, nside = nside, raTag = 'RA', decTag='DEC', property = property)
            bins, Bdensity, Berr = GalaxyDensity_Systematics(As_dmass, sysMap82, nside = nside, raTag = 'RA', decTag='DEC', property = property)
            
            #bins = bins/np.sum(sysMap['SIGNAL']) *len(sysMap)
            #C_jkerr = jksampling(cmass_catalog, sysMap, nside = nside, njack = 10, raTag = 'RA', decTag = 'DEC' )
            #B_jkerr = jksampling(balrog_cmass, sysMap, nside = nside, njack = 10, raTag = 'RA', decTag = 'DEC' )
            
            filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+'.txt'
            DAT = np.column_stack(( bins-(bins[1]-bins[0])*0.1, Cdensity, Cerr, Bdensity, Berr  ))
            np.savetxt(filename, DAT, delimiter = ' ', header = 'bins, Cdensity, Cerr, Bdensity, Berr')
            print "saving data to ", filename
         

    # visualization
    for property in properties:
        
        fig, ax = plt.subplots(2, 2, figsize = (15, 10))
        ax = ax.ravel()
        for i, filter in enumerate(filters):
            
            filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+'.txt'
            data = np.loadtxt(filename)
            bins, Cdensity, Cerr, Bdensity, Berr = [data[:,j] for j in range(5)]
            
            zeromaskC, zeromaskB = ( Cdensity != 0.0 ), (Bdensity != 0.0 )
            Cdensity, Cbins, Cerr = Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
            #C_jkerr = C_jkerr[zeromaskC]
            Bdensity, Bbins, Berr = Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
            #B_jkerr = B_jkerr[zeromaskB]
            
            #fitting
            Cchi, Cchidof = chisquare_dof( Cbins, Cdensity, Cerr )
            Bchi, Bchidof = chisquare_dof( Bbins, Bdensity, Berr )
            
            ax[i].errorbar(Cbins-(bins[1]-bins[0])*0.1, Cdensity, yerr = Cerr, color = 'blue', fmt = '.', label='CMASS, chi2/dof={:>2.2f}'.format(Cchidof))
            ax[i].errorbar(Bbins+(bins[1]-bins[0])*0.1, Bdensity, yerr = Berr, color = 'red', fmt= '.',  label='DMASS, chi2/dof={:>2.2f}'.format(Bchidof))
            ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
            ax[i].set_ylabel('n_gal/n_tot '+str(nside))
            ax[i].set_ylim(0.0, 2)
            #ax[i].set_xlim(8.2, 8.55)
            ax[i].axhline(1.0,linestyle='--',color='grey')
            ax[i].legend(loc = 'best')
            
            #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
            #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
            #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
            #if property == 'NSTARS': ax[i].set_xlim(0.0, 2.5)
        
        fig.suptitle('systematic test (y1a1 DFULL)')
        figname = 'figure/systematic_'+property+'_'+kind+'.png'
        fig.savefig(figname)
        print "saving fig to ", figname





if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        #main(sys.argv)
        main()
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)




