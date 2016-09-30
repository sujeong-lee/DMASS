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
import numpy.lib.recfunctions as rf
from numpy import linalg
#from __future__ import print_function, division
from sklearn.mixture import GMM
import fitsio
#from ..utils import logsumexp, log_multivariate_gaussian, check_random_state
from multiprocessing import Process, Queue
import pdb


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
        #print gmm.weights_
        
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
            
            sys.stdout.write("\r" + 'expected time: {:0.2f} s,  process: {:0.2f} % '.format((t1-t0) * (self.n_iter - i), i * 1./self.n_iter * 100.))
            sys.stdout.flush()
 

            if self.verbose:
                print("%i: log(L) = %.5g" % (i + 1, logL_next))
                print("    (%.2g sec)" % (t1 - t0))
              
            if logL_next < logL + self.tol:
                break
            logL = logL_next

        sys.stdout.write("\r" + 'expected time: {:0.2f} s,  process: {:0.2f} % \n'.format(0, 100))
        
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



def priorCut(data):
    """
    reddeningtag = 'XCORR_SFD98'
    fac = 1.0
    
    if reddening == None :
        fac = 0.0
    elif reddening == 'SLR':
        fac = -1.0
        reddeningtag = 'SLR_SHIFT'
    """

    modelmag_g_des = data['MAG_DETMODEL_G_corrected']# - fac * data[reddeningtag+'_G']
    modelmag_r_des = data['MAG_DETMODEL_R_corrected']# - fac * data[reddeningtag+'_R']
    modelmag_i_des = data['MAG_DETMODEL_I_corrected']# - fac * data[reddeningtag+'_I']
    cmodelmag_i_des = data['MAG_MODEL_I_corrected']# - fac * data[reddeningtag+'_I']
    fib2mag_des = data['MAG_APER_4_I_corrected']# - fac * data[reddeningtag+'_I']
    #dperp = modelmag_r_des - modelmag_i_des - (modelmag_g_des - modelmag_r_des)/8.0

    cut = (#(cmodelmag_i_des > 17) &
           #(cmodelmag_i_des < 22.) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) &
           ((modelmag_r_des - modelmag_i_des ) > 0.0 ) &
           ((modelmag_g_des - modelmag_r_des ) > 0.0 ) &
           ((modelmag_g_des - modelmag_r_des ) < 2.5 ) &
           (fib2mag_des < 22.0 )# &
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
    
    bin = (column < values[0])
    binned_cat.append( cat[bin] )
    binkeep.append(bin)

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



def detMatrixMultiprocessing( T ):

    # split data
    n_process = 12
    T_split = np.array_split( T, n_process, axis=0)
    
    def det_process(q, order, T):
        re = np.array([linalg.det(T[i]) for i in range(T.shape[0])])
        q.put((order, re))
    
    q = Queue()
    Processes = [Process(target = det_process, args=(q, z[0], z[1])) for z in zip(range(n_process), T_split)]
    
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
    Processes = [Process(target = inv_process, args=(q, z[0], z[1])) for z in zip(range(n_process), T_split)]
    
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






def mixing_color(data, suffix = '_corrected', sdss = None, cmass = None ):
    
    filter = ['G', 'R', 'I', 'Z']
    mag = ['MAG_MODEL', 'MAG_DETMODEL', ]
    magtag = [ m+'_'+f+suffix for m in mag for f in filter ]
    del magtag[0], magtag[2]
    err = [ 'MAGERR_MODEL','MAGERR_DETMODEL']
    errtag = [ e+'_'+f for e in err for f in filter ]
    del errtag[0], errtag[2]
    
    X = [ data[mt] for mt in magtag ]
    Xerr = [ data[mt] for mt in errtag ]
    #reddeningtag = 'XCORR_SFD98'

    X = np.vstack(X).T
    Xerr = np.vstack(Xerr).T
    # mixing matrix
    W = np.array([
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],    # i cmag
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],   # r-i
                  [0, 0, 0, 0, 1, -1]])  # i-z

    X = np.dot(X, W.T)

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
    params = [  X_train[:,0], X_train[:,1], X_train[:,2], X_train[:,3] , X_train[:,4]]
    params_t = [  X_test[:,0], X_test[:,1], X_test[:,2], X_test[:,3], X_test[:,4]]

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
    resample = train[ weight > np.median(weight)*50 ]
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
    return resample



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

    filename = 'figure/'+prefix+"diagnostic_histograms3.png"
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



def doVisualization(model_data, real_data, name = ['model', 'real'], labels = None, ranges = None, nbins=100, prefix= 'default'):
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
                axes[i][i].hist(real_data[:,i],bins=xbins,normed=True,label=name[1])
                axes[i][i].set_xlabel(labels[i])
                axes[i][i].hist(model_data[:,i],bins=xbins,normed=True,alpha=0.5,label=name[0])
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

                
                



def _FindOptimalN( N, data, pickleFileName = None):
    from sklearn.mixture import GMM
        
    @pickle_results( pickleFileName )
    def compute_GMM( N, covariance_type='full', n_iter=1000):
        models = [None for n in N]
        for i in range(len(N)):
            sys.stdout.write("\r" + 'Finding optimal number of cluster : {:0.0f} % '.format(i * 1./len(N) * 100.))
            sys.stdout.flush()
            models[i] = GMM(n_components=N[i], n_iter=n_iter,
                            covariance_type=covariance_type)
            models[i].fit(data)
        return models
    models = compute_GMM(N)
    AIC = [m.aic(data) for m in models]
    BIC = [m.bic(data) for m in models]
    i_best = np.argmin(BIC)
    gmm_best = models[i_best]
    sys.stdout.write("\r" + 'Finding optimal number of cluster : {:0.0f} % '.format(100))
    print "\nbest fit converged:", gmm_best.converged_,
    print " n_components =  %i" % N[i_best]
    return N[i_best], AIC, BIC


def XDGMM_model(cmass, lowz, train = None, test = None, mock = None, reverse=None, prefix = ''):

    import esutil
    import numpy.lib.recfunctions as rf
    from multiprocessing import Process, Queue
    
    train, _ = priorCut(train)
    test, _ = priorCut(test)

    # making cmass and lowz mask
    #h = esutil.htm.HTM(10)
    #matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    #m_des, m_sdss, d12 = h.match(train['RA'],train['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    m_des, m_sdss = esutil.numpy_util.match( train['COADD_OBJECTS_ID'], cmass['COADD_OBJECTS_ID'] )
    
    true_cmass = np.zeros( train.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask = true_cmass == 1
    
    print 'num of cmass in train', np.sum(cmass_mask)
    
    """
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
    """
    # stack DES data

    #if reverse is True:
    X_cmass, Xcov_cmass = mixing_color( cmass )
    X_train, Xcov_train = mixing_color( train )
    X_test, Xcov_test = mixing_color( test )
    X, Xcov = np.vstack((X_train,X_test)), np.vstack((Xcov_train, Xcov_test))

    y_train = np.zeros(( train.size, 2 ), dtype=int)
    y_train[:,0][cmass_mask] = 1
    #y_train[:,1][lowz_mask] = 1
    
    
    
    if mock is None : y = np.vstack((y_train, y_test))
    

    print 'train/test', len(X_train), len( X_test )
    
    # train sample XD convolution ----------------------------------
    
    # finding optimal n_cluster

    from sklearn.mixture import GMM
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
    #n_cl_no = 25
    n_cl_all = 25
    # ----------------------------------
  
    #pickleFileName = ['pickle/'+prefix +'XD_all.pkl', 'pickle/'+prefix+'XD_dmass.pkl', 'pickle/'+prefix+'XD_no.pkl']
    pickleFileName = ['pickle/'+prefix2 +'XD_all.pkl', 'pickle/'+prefix1+'XD_dmass.pkl', 'pickle/'+prefix+'XD_no.pkl']
    def XDDFitting( X_train, Xcov_train, init_means=None, init_covars = None, init_weights = None, filename = None, n_cluster = 25 ):
        clf = None
        @pickle_results(filename, verbose = True)
        def compute_XD(n_clusters=n_cluster, n_iter=500, verbose=False):
            clf= XDGMM(n_clusters, n_iter=n_iter, tol=1E-2, verbose=verbose)
            clf.fit(X_train, Xcov_train, init_means=init_means, init_covars = init_covars, init_weights = init_weights)
            return clf
        clf = compute_XD()
        return clf


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
    
    # calculates CMASS fits
    
    #rows = np.random.choice(np.arange(X_train.shape[0]), train.size * 0.33)
    #clf = XDDFitting( X_train[rows,:], Xcov_train[rows,:,:], filename= pickleFileName[0], n_cluster= n_cl_all)
    clf = XDDFitting( X_train, Xcov_train, filename= pickleFileName[0], n_cluster= n_cl_all)
    clf_cmass = XDDFitting( X_cmass, Xcov_cmass, filename=pickleFileName[1], n_cluster=n_cl_cmass)
    
    #clf = XDDFitting( X_train[rows,:], Xcov_train[rows,:,:], init_means=init_means, init_covars = init_covars, init_weights = init_weights, filename= pickleFileName[0], n_cluster= n_cl_all + n_cl_cmass)
    
    #fitsio.write('result_cat/'+prefix+'X_obs.fits' ,X_train[rows,:])
    fitsio.write('result_cat/'+prefix+'X_obs.fits' ,X_train)
    """
    rows = np.random.choice(np.arange(X_train[~cmass_mask].shape[0]), 10 * len(X_train[cmass_mask]))
    init_weightsN, init_meansN, init_covarsN = initialize_deconvolve(X_train[~cmass_mask].data,  Xcov_train[~cmass_mask], n_components = n_cl_no)
    
    clf_nocmass = XDDFitting( X_train[~cmass_mask][rows,:], Xcov_train[~cmass_mask][rows,:,:], init_means=init_meansN, init_covars = init_covarsN, init_weights = init_weightsN, filename= pickleFileName[2], n_cluster=n_cl_no)
    """

    # logprob_a ------------------------------------------
    


    print "calculate loglikelihood gaussian with multiprocessing module"
    #cmass_logprob_a = logsumexp(clf_cmass.logprob_a( X, Xcov, parallel = True ), axis = 1)
    #all_logprob_a = logsumexp(clf.logprob_a( X, Xcov, parallel = True ), axis = 1)
    
    from multiprocessing import Process, Queue
    # split data
    n_process = 12
    
    X_test_split = np.array_split(X_test, n_process, axis=0)
    Xcov_test_split = np.array_split(Xcov_test, n_process, axis=0)
    
    X_split = np.array_split(X, n_process, axis=0)
    Xcov_split = np.array_split(Xcov, n_process, axis=0)
    
    def logprob_process(q,  classname, order,(data, cov)):
        re = classname.logprob_a(data, cov)
        result = logsumexp(re, axis = 1)
        q.put((order, result))

    inputs = [ (X_split[i], Xcov_split[i]) for i in range(n_process) ]
    
    q_cmass = Queue()
    q_all = Queue()
    #q_no = Queue()
    cmass_Processes = [Process(target = logprob_process, args=(q_cmass, clf_cmass, z[0], z[1])) for z in zip(range(n_process), inputs)]
    all_Processes = [Process(target = logprob_process, args=(q_all, clf, z[0], z[1])) for z in zip(range(n_process), inputs)]

    #no_Processes = [Process(target = logprob_process, args=(q_no, clf_nocmass, z[0], z[1])) for z in zip(range(n_process), inputs)]
    


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

    sys.stdout.write("\r" + 'multiprocessing {:0.0f} % \n'.format( 100 ))

    numerator =  np.exp(cmass_logprob_a) * np.sum(y_train[:,0]) * 1.
    denominator = np.exp(all_logprob_a) * len(X_train)
    #denominator = numerator + np.exp(no_logprob_a) * (len(X_train) - np.sum(y_train[:,0]))

    denominator_zero = denominator == 0
    EachProb_CMASS = np.zeros( numerator.shape )
    EachProb_CMASS[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]
    #EachProb_CMASS[EachProb_CMASS > 1.0] = 1.0
    
    #print 'EachProb max', EachProb_CMASS.max()
    
    
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
        q.put((order, re))
    
    inputs = [ (X_sample_all_split[i], X_train, Xcov_train) for i in range(n_process) ]
    
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

    sys.stdout.write("\r" + 'add noise to sample {:0.0f} %'.format( 100 ))
    #result = [q.get() for p in Processes]
    result.sort()
    result = [r[1] for r in result ]

    noisy_X_sample_all = np.vstack( result[:n_process] ) #result[1]
    noisy_X_sample_cmass = add_errors( X_sample_cmass, X_train[cmass_mask], Xcov_train[cmass_mask] )
    
    fitsio.write('result_cat/'+prefix+'noisy_X_sample.fits' ,noisy_X_sample_all)
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


def probability_calibration( des = None, cmass_des = None, prefix = 'test' ):
    from utils import matchCatalogs
    pth_bin, pstep = np.linspace(0.0, 3.0, 50, retstep = True)
    pbin_center = pth_bin[:-1] + pstep/2.
    
    cmass_des, _ = matchCatalogs( des, cmass_des, tag='COADD_OBJECTS_ID' )
    
    N_cmass, _ = np.histogram( cmass_des['EachProb_CMASS'], bins = pth_bin, density=False )
    N_all, _ = np.histogram( des['EachProb_CMASS'], bins = pth_bin, density=False )
    
    prob = N_cmass * 1./N_all
    real = N_cmass * 1./cmass_des.size
    
    for i, j, k in zip(pbin_center, prob, N_cmass): print i, j, k
    
    fig,ax = plt.subplots()
    ax.plot(pbin_center, prob,label='true fraction' )
    ax.plot(pbin_center, real,label='total fraction' )
    ax.set_xlim(0,3)
    ax.set_ylim(0,1)
    ax.legend(loc='best')
    fig.savefig('com_pur_results/'+prefix + '_probability_calibration.png')
    print 'save fig: com_pur_results/'+prefix+'_probability_calibration.png'



def _resampleWithPth( des ):
    
    pth_bin, step = np.linspace(0.01, 1.0, 100, endpoint = True, retstep = True)
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



def resampleWithPth( des, pstart = 0.1, pmax = 0.8 ):
    
    pmin = 0.1
    pmax = pmax # should be adjused with different samples
    p1stbin = 0.1
    
    pth_bin, step = np.linspace(pstart, 1.0, 300, endpoint = True, retstep = True)
    pcenter = pth_bin + step/2.
    
    # ellipse
    pcurve = (pmax-pmin) * np.sqrt( 1 - (pcenter - 1)**2 * 1./(1-p1stbin)**2 ) + pmin
    pcurve[ pcenter < pmin ] = 0.0
    pcurve[ np.argmax(pcurve)+1:] = pmax
    
    # straight
    pcurve2 = np.zeros(pth_bin.size)
    pcurve_front = np.linspace(pstart, 1.0, 300, endpoint = True)
    pcurve2[:pcurve_front.size] = pcurve_front
    pcurve2[pcurve_front.size:] = pmax

    ellipse_dmass = [None for i in range(pth_bin.size)]
    straight_dmass = [None for i in range(pth_bin.size)]
    el_fraction, st_fraction = np.zeros(pth_bin.size),  np.zeros(pth_bin.size)
    
    desInd = np.digitize( des['EachProb_CMASS'], bins= pth_bin )
    
    for i in range(pth_bin.size-1):
        #mask = (des['EachProb_CMASS'] >= pth_bin[i]) & (des['EachProb_CMASS'] < pth_bin[i]+step)
        #cat = des[mask]
        cat = des[ desInd == i+1 ]
        try :
            ellipse_dmass[i] = np.random.choice( cat, size = int(np.around(pcurve[i] * cat.size)) )
            straight_dmass[i] = np.random.choice( cat, size = int(np.around(pcurve2[i] * cat.size)) )
            el_fraction[i] = ellipse_dmass[i].size * 1./cat.size
            st_fraction[i] = straight_dmass[i].size * 1./cat.size
        except ValueError:
            ellipse_dmass[i] = np.random.choice( des, size = 0)
            straight_dmass[i] = np.random.choice( des, size = 0)
            el_fraction[i] = 0
            st_fraction[i] = 0
        #fraction.append( np.sum(mask) * 1./des.size )
        #print pth_bin[i], pth_bin[i]+step, pcurve[i], dmass[i].size * 1./cat.size, dmass[i].size
    
    i = i+1
    mask = (des['EachProb_CMASS'] >= pth_bin[i])
    cat = des[mask]
    ellipse_dmass[i] = np.random.choice( cat, size = int(np.around(pmax * cat.size)) )
    straight_dmass[i] = np.random.choice( cat, size = int(np.around(pmax * cat.size)) )
    #print pth_bin[i], pth_bin[i]+step, 0.75, dmass[i].size * 1./cat.size, dmass[i].size
    
    ellipse_dmass = np.hstack(ellipse_dmass)
    straight_dmass = np.hstack(straight_dmass)
    
    # plotting selection prob----------
    """
    fig, ax = plt.subplots()

    #ax.plot( pcenter, pcurve2, color = 'grey', label = 'Ashley1', linestyle = '--')
    ax.plot( np.insert(pcenter, 0, pcenter[0]), np.insert(pcurve2, 0, 0.0), color = 'grey',linestyle = '--', label='straight')
    ax.plot( np.insert(pcenter, 0, pcenter[0]), np.insert(pcurve, 0, 0.0), color = 'grey' , label = 'ellipse')
    ax.plot( pcenter, el_fraction, color = 'red', linestyle = '--' )
    ax.plot( pcenter, st_fraction, color = 'blue',linestyle = '--'  )
    
    pbin, st = np.linspace(0.0, 1.1, 100, endpoint=False, retstep = True)
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




def EstimateContaminantFraction( dmass = None, cmass = None ):
    #  check difference is smaller than stat error
    
    import esutil
    m_dmass, m_true, _ = esutil.htm.HTM(10).match( dmass['RA'], dmass['DEC'], cmass['RA'], cmass['DEC'], 1./3600, maxmatch = 1)
    true_mask = np.zeros( dmass.size, dtype = bool )
    true_mask[m_dmass] = 1
    
    #cross_data = np.loadtxt('data_txt/acf_cross_noweightcont.txt')
    #cross_data = np.loadtxt('data_txt/acf_cross_weightedcont_prop.txt')
    cross_data = np.loadtxt('data_txt/acf_cross_weightedcont.txt')
    r_tc, w_tc, err_tc = cross_data[:,0], cross_data[:,1], cross_data[:,2]
    
    auto_data = np.loadtxt('data_txt/acf_comparison_photo_dmass.txt')
    true_data = np.loadtxt('data_txt/acf_comparison_photo_true.txt')
    #cont_data = np.loadtxt('data_txt/acf_comparison_noweightcont.txt')
    #cont_data = np.loadtxt('data_txt/acf_comparison_weightedcont_prop.txt')
    cont_data = np.loadtxt('data_txt/acf_comparison_weightedcont.txt')
    r, w, err, w_t, err_t,  w_c, err_c = auto_data[:,0], auto_data[:,1],  auto_data[:,2],  true_data[:,1],  true_data[:,2], cont_data[:,1], cont_data[:,2]
    
    f_c = np.sum(~true_mask) * 1./dmass.size
    print "current f_c = {:>0.2f} %".format( f_c * 100)
    ang_L = 1. + w - (1. -f_c)**2 * (1. + w_t)
    ang_R = 2. * f_c * (1. -f_c)*(1. + w_tc) + f_c**2 * (1+w_c)
    
    keep = (r > 0.001) & (r < 10)
    fractionL = 1. - np.sqrt( (1. + w - ang_L) * 1./(1. + w_t) )
    fractionR = (-(1.+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* ang_R) ) * 1./(w_c-1.-2* w_tc)
    m_fL, m_fR = np.mean(fractionL), np.mean(fractionR) # current max cont fraction
    
    #print "current f_cont (LHS, RHS) : ", m_fL, m_fR
    
    # ideal fraction
    eps82 = np.mean(err[keep]/w[keep])/np.sqrt(np.sum(keep))
    eps = eps82  * np.sqrt(100./1000)
    i_fractionL = np.abs(1. - np.sqrt( (1 + w - eps) * 1./(1 + w_t) ))
    i_fractionR = np.abs((-(1+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* eps) ) * 1./(w_c-1.-2* w_tc))
    
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
    #cont_lensing = np.loadtxt('data_txt/lensing_noweightcont.txt')
    cont_lensing = np.loadtxt('data_txt/lensing_weightedcont_prop.txt')
    #cont_lensing = np.loadtxt('data_txt/lensing_weightedcont.txt')

    
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
    return 0
    
    

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




def RunningXDGMMwithBalrog(des_train, noblend_clean_cmass_data_des):

    import os
    import esutil
    #catpath = '/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/mask/'
    catpath = '/n/des/lee.5922/data/balrog_cat/'
    tables = []
    
    
    for j in range(20):
        keyword = 'EMHUFF.BALROG_Y1A1_SIM_{:05d}'.format(j)
        names = []
        for i in os.listdir(catpath):
            if os.path.isfile(os.path.join(catpath,i)) and keyword in i:
                #print i
                names.append(catpath+i)
        try:
            balrog = io.LoadBalrog(user=keyword, truth=None )
            balrog = AddingReddening(balrog)
            balrog = Cuts.doBasicCuts(balrog, object = 'galaxy')
            result = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des_train, test=balrog, prefix = prefix, mock = True )
            tables.append(result)
        except IOError: pass
    
    tables = np.hstack(tables)
    fitsio.write('result_cat/result_balrog_y1a1.fit', tables2)
    return tables



def main():
    
    """
    Important Catalog
    ------------------
    SSPT_star = fitsio.read('result_cat/SSPT_star.fits')
    full_des_data_st82 = io.getDESY1A1catalogs(keyword = 'STRIPE82', sdssmask=False)
    full_des_data = io.getDESY1A1catalogs(keyword = 'Y1A1', sdssmask=False)   
    sdssmaskedFullDesdata =  io.getDESY1A1catalogs(keyword = 'des_st82', sdssmask=True)
    balrog_st82 = fitsio.read('result_cat/result_balrog_JELENA.fits') # added EachProb_CMASS
    balrog_y1a1 = fitsio.read('result_cat/result_balrog_EMHUFF.fits') # added EachProb_CMASS
    result_photo = fitsio.read('result_cat/result_photo.fits') # fitted with photoCMASS
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    result_smallst82 = fitsio.read('result_cat/result1.fits')
    """
    
    # load dataset

    ra = 320.
    ra2 = 360
    dec = -1
    dec2 = 1

    full_des_data = io.getDESY1A1catalogs(keyword = 'des_st82')
    des_data_clean = keepSDSSgoodregion(Cuts.doBasicCuts(full_des_data, object = 'galaxy'))
    des = Cuts.SpatialCuts(des_data_clean, ra = ra, ra2=ra2, dec= dec, dec2= dec2  )
    
    #cmass_spec= io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    #cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/bosstile-final-collated-boss2-boss38-photoObj.fits.gz')
    
    
    cmass_data_o = io.getSDSScatalogs(file ='/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-combined.fits.gz')
    clean_cmass_data = Cuts.keepGoodRegion(cmass_data_o)
    cmass_data = Cuts.SpatialCuts(clean_cmass_data, ra =ra, ra2=ra2, dec=dec, dec2=dec2 )
    
    noblend_clean_cmass_data, noblend_clean_cmass_data_des = DES_to_SDSS.match( cmass_data, des_data_clean)
    fitsio.write('noblend_clean_cmass_data_test.fits', noblend_clean_cmass_data )
    
    """
    #lowz_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_LOWZ_South-photoObj.fits.gz')
    #lowz_data = Cuts.SpatialCuts(lowz_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    #clean_lowz_data = Cuts.keepGoodRegion(lowz_data)
    """
    
    sdss_prior = fitsio.read('/n/des/lee.5922/data/cmass_cat/cmass_stripe82.fits', upper=True)
    #sdss_data_o = io.getSDSScatalogs(bigSample = True)
    sdss_data_clean = Cuts.doBasicSDSSCuts(sdss_prior)
    fitsio.write('result_cat/cmass_stripe82_photo.fits', sdss_data_clean )
    #sdss = Cuts.SpatialCuts(sdss_prior,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    
    noblend_clean_cmass_data,noblend_clean_cmass_data_des = DES_to_SDSS.match(sdss_data_clean, des)
    fitsio.write('result_cat/noblend_cmass_stripe82_photo.fits', noblend_clean_cmass_data)
    fitsio.write('result_cat/noblend_cmass_stripe82_photo_des.fits',noblend_clean_cmass_data_des)
    
    
    #cmass_sdss = SDSS_cmass_criteria(sdss)

    import esutil
    des_im3_o = esutil.io.read( '/n/des/huff.791/Projects/CMASS/Scripts/main_combined.fits',  columns = ['COADD_OBJECTS_ID', 'DESDM_ZP', 'RA', 'DEC', 'INFO_FLAG' ], upper=True)
    
    des_im3 = des_im3_o[des_im3_o['INFO_FLAG'] == 0]
    
    des_im3_st82_o = io.getDEScatalogs(file = '/n/des/lee.5922/data/im3shape_st82.fits')
    clean_des_im3_st82 = Cuts.keepGoodRegion(des_im3_st82_o)
    des_im3_st82 = Cuts.SpatialCuts(clean_des_im3_st82 ,ra=320, ra2=360 , dec=-1, dec2=1 )
    #des = im3shape_galprof_mask(des_im3, des) # add galaxy profile mode
    
    balrog_o = io.LoadBalrog(user = 'JELENA', truth = None)
    
    balrog_o = io.LoadBalrog(user = 'EMHUFF.BALROG_Y1A1_00000', truth = None)
    #balrog = Cuts.keepGoodRegion( balrog_o)
    """
    print "alphaJ2000, deltaJ2000  -->  ra, dec"
    balrogname = list( balrog.dtype.names)
    alphaInd = balrogname.index('ALPHAWIN_J2000_DET')
    deltaInd = balrogname.index('DELTAWIN_J2000_DET')
    balrogname[alphaInd], balrogname[deltaInd] = 'RA', 'DEC'
    balrog.dtype.names = tuple(balrogname)
    """
    balrog = AddingReddening(balrog)
    balrog = Cuts.doBasicCuts(balrog, object = 'galaxy')
    
    #bcc_st82 = fitsio.read('/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/mask/bcc_stripe82.fit')
    
    # start XD
    prefix = 'small_'
    #prefix = 'photo_'
    
    #devide samples
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(des, des, [0.5,0.5], random_state=0)
    des_train = des[trainInd]
    des_test = des[testInd]


    result = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des_train, test=des, prefix = prefix, mock=True )
    result = addphotoz(result, des_im3)
    
    result_y1a1 = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des_train, test=y1a1, prefix = prefix, mock = True )
    result_y1a1 = addphotoz(result_y1a1, des_im3_o)

    result_balrog = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des_train, test=balrog, prefix = prefix, mock = True )
    
    result_bcc = XDGMM_model(noblend_clean_cmass_data_des, noblend_clean_cmass_data_des, train=des, test=bcc, prefix = prefix, mock = True, bcc=True )
    
    #fitsio.write('result_cat/result_fullst82.fits', result)
    fitsio.write('result_cat/result_photo.fits', result)
    #fitsio.write('result_y1a1.fits', result_y1a1)
    fitsio.write('result_cat/result_balrog.fits', result_balrog)
    #fitsio.write('result_cat/result_balrog_JELENA.fits', result_balrog)
    # Ashley - match distribution and visulaization ----------------------------
    
    #calling catalogs
    #noblend_clean_cmass_data_des = fitsio.read('result_cat/noblend_clean_cmass_data_des.fits')
    #noblend_clean_cmass_data = fitsio.read('result_cat/noblend_clean_cmass_data.fits')
    noblend_clean_cmass_data = fitsio.read('result_cat/noblend_cmass_stripe82_photo.fits')
    noblend_clean_cmass_data_des = fitsio.read('result_cat/noblend_cmass_stripe82_photo_des.fits')

    result = fitsio.read('result_cat/result_photo.fits')
    #result = fitsio.read('result_cat/result_fullst82.fits')
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    
    
    result_balrog = fitsio.read('result_cat/result_balrog.fits')
    
    dmass_stripe82, _ = resampleWithPth(result)
    dmass_y1a1,_ = resampleWithPth(result_y1a1)
    dmass_bcc,_ = resampleWithPth(result_bcc)
    As_dmass,_ = resampleWithPth(result_balrog)
    
    #As_dmass, As_dmass2 = addphotoz( As_dmass, des_im3 ), addphotoz( As_dmass2, des_im3 )
    As_X, _ = mixing_color( resample )
    #As2_X, _ = mixing_color( As_dmass2 )
    Xtrue,_ = mixing_color( noblend_clean_cmass_data_des )
    labels = ['MAG_MODEL_R', 'MAG_MODEL_I', 'g-r', 'r-i', 'i-z']
    ranges =  [[17,22.5], [17,22.5], [0,2], [-.5,1.5], [0.0,.8]]
    doVisualization_1d( Xtrue, As_X, labels = labels, ranges = ranges, nbins=100, prefix='reweight_')
    doVisualization_1d( Xtrue, As2_X, labels = labels, ranges = ranges, nbins=100, prefix='Ashley2_')

    cmassIndmass, _ = DES_to_SDSS.match( noblend_clean_cmass_data, As_dmass)
    angularcorr( dmass = As_dmass, subCMASS = cmassIndmass, cmass =noblend_clean_cmass_data, ra = ra, ra2 = ra2, dec=dec, dec2 = dec2, suffix = '_photo')
    cmassIndmass, _ = DES_to_SDSS.match( noblend_clean_cmass_data,  As_dmass2)
    angularcorr( dmass = As_dmass2, subCMASS = cmassIndmass, cmass =noblend_clean_cmass_data, ra = ra, ra2 = ra2, dec=dec, dec2 = dec2, suffix = '_Ashley2_y1a1')



    weighted_balrog = reweight(train = balrog, test = noblend_clean_cmass_data_des)
    fitsio.write('result_cat/weighted_balrog.fits', weighted_balrog )
    weighted_balrog = fitsio.read('result_cat/weighted_balrog.fits')
    makePlotReweight( train = weighted_balrog, test = noblend_clean_cmass_data_des )
    
    
    
    # corr comparison
    labels = ['DES_SSPT', 'fullcmass_SGC']
    #labels = [ 'photo_cmass', 'photo_dmass', 'photo_true', 'noweightcont', 'weightedcont', 'weightedcont_prop' ]
    #labels = [ 'noweightcont', 'weightedcont', 'weightedcont_prop' ]
    #labels = [  'noweight', 'wnstar_g']
    
    #labels = ['No_mask', 'All_mask', 'mask_AIRMASS', 'mask_SKYBRITE', 'mask_FWHM', 'mask_SKYSIGMA', 'wnstar+mask']
    #labels = ['No_mask', 'All_mask'] + [ 'w+'+l for l in labels[3:6]]
    
    linestyle = ['-']+['--' for i in labels[:-1]]
    fmt = ['.']+['o' for i in labels[:-2]]+['.']
    color = ['red'] + [None for i in labels[:-1]]
    corr_txt = [np.loadtxt('data_txt/acf_comparison_'+s+'.txt') for s in labels]
    
    corr_txt2 = np.loadtxt('data_txt/acf_comparison_cmass_SGC.txt')
    thetaS, wS, Sjkerr = corr_txt2[:,0], corr_txt2[:,1], corr_txt2[:,2]
    
    fig, (ax, ax2) = plt.subplots(2,1, figsize = (10,15))
    ax.errorbar( thetaS, wS, yerr = Sjkerr, label = 'CMASS SGC', color='black', alpha = 0.5)
    ax2.errorbar( thetaS, wS-wS, yerr = 10 * Sjkerr, label = 'CMASS SGC', color='black', alpha = 0.9)
    
    for i in range(len(labels)):
        
        thetaD, wD, Djkerr = corr_txt[i][:,0], corr_txt[i][:,1], corr_txt[i][:,2]
        
        if labels[i] == 'wnstar+mask': markersize = 12
        else : markersize = 5
        
        ax.errorbar( thetaD*(0.95+0.02*i), wD, yerr = Djkerr, fmt = fmt[i], linestyle = linestyle[i] ,label = labels[i], color = color[i], markersize = markersize)
        ax2.errorbar( thetaD*(0.95+0.05*i), 10 * (wD - wS), yerr = 10 *Djkerr, fmt = fmt[i],  linestyle = linestyle[i], label = labels[i],color = color[i], markersize=markersize)
    
    ax.set_xlim(1e-3, 50)
    #ax.set_ylim(-0.02 , 0.5)
    ax.set_ylim(0.001 , 10.0)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')

    ax2.axhline(y=0.0, color = 'black')
    ax2.set_xlim(1e-1, 10)
    ax2.set_ylim(-.2, .5)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\theta(deg)$')
    ax2.set_ylabel(r'$10 \times$ $($ ${w}$ - ${w_{\rm{true}}}$ $)$')
    ax2.legend(loc='best')
    figname = 'figure/acf_comparison.png'
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
    from systematics_module.corr import LensingSignal, cross_angular_correlation, angular_correlation
    
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

    
    # weighting contaminant
    def WeightingCont(result, dmass, true, contaminant):
        
        # cont hist
        pth_bin, step = np.linspace(0.0, 1.01, 50, endpoint = True, retstep = True)
        pcenter = pth_bin + step/2.
        N, edge = np.histogram( contaminant['EachProb_CMASS'], bins = pth_bin, normed=0)
        N_t, edge = np.histogram( true['EachProb_CMASS'], bins = pth_bin,normed=0)
        N_d, edge = np.histogram( dmass['EachProb_CMASS'], bins = pth_bin,normed=0)
        w_cont =   N_t*1./N
        w_cont_prop = pth_bin[:-1]
        N_all, edge = np.histogram( result['EachProb_CMASS'], bins = pth_bin, normed=0)
        
        weight_cont, weight_cont_prop = np.zeros(contaminant.size, dtype=float), np.zeros(contaminant.size, dtype=float)
        for i, p in enumerate(pth_bin[:-1]):
            mask = ( contaminant['EachProb_CMASS'] >= p ) & ( contaminant['EachProb_CMASS'] < p + step)
            print p, p+step, contaminant[mask].size, w_cont[i]
            weight_cont[mask] = w_cont[i]
            weight_cont_prop[mask] = w_cont_prop[i]
        
        """
        fig, (ax,ax2) = plt.subplots(1,2,figsize=(14,7))
        
        ax.plot( pcenter[:-1], N, color = 'red', linestyle = '--', label = 'cont')
        ax.plot( pcenter[:-1], N * w_cont , color = 'green', linestyle = '-', label = 'weighted cont')
        ax.plot( pcenter[:-1], N_t , color = 'blue', linestyle = '--', label = 'true')
        ax.plot( pcenter[:-1], N_d, color = 'grey', linestyle = '--', label = 'all')
        
        
        ax2.plot( pcenter[:-1], N* 1./N_all, color = 'red', linestyle = '--', label = 'cont')
        ax2.plot( pcenter[:-1], N * w_cont* 1./N_all, color = 'green', linestyle = '-', label = 'weighted cont')
        ax2.plot( pcenter[:-1], N_t* 1./N_all , color = 'blue', linestyle = '--', label = 'true')
        ax2.plot( pcenter[:-1], N_d * 1./N_all, color = 'grey', linestyle = '--', label = 'all')
        
        
        ax.set_xlim(0,0.95)
        ax.set_ylim(0,N_d[:10].max() + 1)
        ax.set_ylabel('N')
        ax.set_xlabel('p_threshold')
        ax.legend(loc='best')
        
        ax2.set_xlim(0,1.0)
        ax2.set_ylabel('fraction')
        ax2.set_xlabel('p_threshold')
        ax2.legend(loc='best')
        
        fig.savefig('test')
        print 'figure to ', 'test.png'
        """
        
        
        return weight_cont, weight_cont_prop
    
    w_cont, w_cont_prop = WeightingCont(result, As_dmass, true, contaminant)
    
    angular_correlation(data = noblend_clean_cmass_data, rand = cat_cmass_rand82,  weight = [None, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_cmass')
    angular_correlation(data = true, rand = cat_cmass_rand82,  weight = [None, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_true')
    angular_correlation(data = As_dmass, rand = cat_cmass_rand82,  weight = [None, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_dmass')

    angular_correlation(data = contaminant, rand = cat_cmass_rand82,  weight = [None, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_noweightcont')
    angular_correlation(data = contaminant, rand = cat_cmass_rand82, weight = [ w_cont_prop, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_weightedcont_prop')
    angular_correlation(data = contaminant, rand = cat_cmass_rand82, weight = [ w_cont, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_weightedcont')

    cross_angular_correlation(data = true, data2 = contaminant, rand = cat_cmass_rand82, rand2= cat_cmass_rand82, weight = [None, cat_cmass_rand82['WEIGHT_FKP'], None, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_noweightcont')
    cross_angular_correlation(data = true, data2 = contaminant, rand = cat_cmass_rand82, rand2= cat_cmass_rand82, weight = [None, cat_cmass_rand82['WEIGHT_FKP'], w_cont_prop, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_weightedcont_prop')
    cross_angular_correlation(data = true, data2 = contaminant, rand = cat_cmass_rand82, rand2= cat_cmass_rand82, weight = [None, cat_cmass_rand82['WEIGHT_FKP'], w_cont, cat_cmass_rand82['WEIGHT_FKP']], suffix = '_weightedcont')
    
    

    LensingSignal(lense = contaminant, source = des_im3_st82, rand = cat_cmass_rand82, weight = [None,None,None], suffix = 'noweightcont')
    LensingSignal(lense = contaminant, source = des_im3_st82, rand = cat_cmass_rand82, weight = [w_cont_prop,None,None], suffix = 'weightedcont_prop')
    LensingSignal(lense = contaminant, source = des_im3_st82, rand = cat_cmass_rand82, weight = [w_cont,None,None], suffix = 'weightedcont')
    
    
    # -------------------------
    
    angular_correlation(data = As_dmass, rand = cat_cmass_rand82, suffix = '_photo_dmass')
    angular_correlation(data = noblend_clean_cmass_data, rand = cat_cmass_rand82, suffix = '_photo_true')
    angular_correlation(data = contaminant, rand = cat_cmass_rand82, suffix = '_photo_cont')
    cross_angular_correlation(data = true, data2 = contaminant, rand = cat_cmass_rand82, rand2= cat_cmass_rand82, weight = [None, None, None, None], suffix = '_photo')
    
    
    # 2) Lensing signal ----------------------

    LensingSignal(lense = As_dmass, source = des_im3, rand = cat_cmass_rand82, suffix = 'noweightcont')
    LensingSignal(lense = true, source = des_im3, rand = cat_cmass_rand82, suffix = 'weightedcont_prop')
    LensingSignal(lense = contaminant, source = des_im3, rand = cat_cmass_rand82, suffix = 'weightedcont')

    # plotting
    
    labels = ['noweightcont', 'weightedcont_prop', 'weightedcont']
    
    lensingdat = [np.loadtxt('data_txt/lensing_'+s+'.txt') for s in labels ]
    
    #r_p_bins, LensSignal, LSjkerr, correctedLensSignal, CLSjkerr, BoostFactor, Boostjkerr = lensingdat[:,0], lensingdat[:,1],lensingdat[:,2],lensingdat[:,3],lensingdat[:,4],lensingdat[:,5],lensingdat[:,6]
    fig, ax = plt.subplots(1,1, figsize = (12,7))
    
    for j in range(len(labels)):
        
        r_p_bins, LensSignal, LSjkerr = [lensingdat[j][:,i] for i in range(lensingdat[0][0].size)]
        ax.errorbar(r_p_bins * (1 + 0.1*j), LensSignal, yerr = LSjkerr, fmt='o', label = labels[j])
                                         
    theory = np.loadtxt('data_txt/smd_v_theta_cmass.dat')
    rr_the = theory[:,0]
    delta_sigma_the = theory[:,1]
    error_the = theory[:,2] * np.sqrt(5000/120)
    ax.errorbar(10**rr_the, 10**delta_sigma_the, yerr = 10**error_the, color = 'black', fmt='--o', label = 'theory')
    
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-2,1e3)
    ax.set_xlabel(r'$r$ $(kpc/h)$')
    ax.set_ylabel(r'$\Delta\Sigma$ $(M_{s}h/pc^{2})$')
    ax.set_title('contaminant lensing signal (z_lense = [0.45, 0.55] )'  )
    ax.legend(loc = 'best')

    figname='figure/lensing_weightedcont.png'
    fig.savefig(figname)
    print 'saving fig to :', figname

    EstimateContaminantFraction( dmass = As_dmass, cmass = noblend_clean_cmass_data )


    #___________________________________________________________
    # systematics ----------------------------------------------

    # 1) systematic maps
    
    from systematics import GalaxyDensity_Systematics,loadSystematicMaps, chisquare_dof, MatchHPArea
    
    # random
    balrog_y1a1 = fitsio.read('result_cat/result_balrog_EMHUFF.fits')
    balrog_SSPT = balrog_y1a1[balrog_y1a1['DEC'] < -3]
    #dmass_balrog_SSPT, _ = resampleWithPth(balrog_SSPT)
    rand_clean = Cuts.keepGoodRegion(balrog_SSPT)

    cmass_DR12 = fitsio.read('/n/des/lee.5922/data/cmass_cat/bosstile-final-collated-boss2-boss38-photoObj.fits.gz')
    rand_DR12 = fitsio.read('/n/des/lee.5922/data/random0_DR12v5_CMASS_South.fits.gz')
    #rand_clean_st82 = Cuts.keepGoodRegion(rand_DR11)

    # dmass
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    dmass_y1a1, _ = resampleWithPth(result_y1a1)
    dmass_stripe82 = dmass_y1a1[dmass_y1a1['DEC']>-3]
    dmass_SSPT = MatchHPArea(cat = dmass_y1a1, origin_cat = balrog_SSPT)


    properties = ['FWHM', 'AIRMASS','SKYSIGMA', 'SKYBRITE', 'NSTARS']
    filters = ['g','r','i', 'z']
    kind = 'SPT'
    nside = 1024
    njack = 10

    property = 'FWHM'
    filter = 'g'

    for property in properties:
        for i, filter in enumerate(filters):
            if property == 'NSTARS':
                nside = 512
                filename = 'y1a1_gold_1.0.2_stars_nside1024.fits'
                sysMap_o = loadSystematicMaps( filename = filename, nside = nside )
                
                if kind is 'STRIPE82' :sysMap = sysMap_o[sysMap_o['DEC'] > -5.0]
                elif kind is 'SPT' :sysMap = sysMap_o[sysMap_o['DEC'] < -5.0]
                elif kind is 'Y1A1':sysMap = sysMap_o
                sysMap = MatchHPArea( cat = sysMap, origin_cat=balrog_SSPT)
                sysMap82 = sysMap_o.copy() #loadSystematicMaps( filename = filename, nside = nside )
            else :
                sysMap = loadSystematicMaps( property = property, filter = filter, nside = nside , kind = kind )
                sysMap = MatchHPArea( cat = sysMap, origin_cat=balrog_SSPT)
                #sysMap82 = loadSystematicMaps( property = property, filter = filter, nside = nside )
            
            #sysMap82 = Cuts.SpatialCuts(sysMap82, ra = 320, ra2=360 , dec=-1 , dec2= 1 )
            
            #bins, Cdensity, Cerr, Cf_area = GalaxyDensity_Systematics(dmass_stripe82, sysMap82, rand = rand_clean_st82 ,nside = nside, raTag = 'RA', decTag='DEC', property = property)
            bins, Bdensity, Berr, Bf_area = GalaxyDensity_Systematics(dmass_SSPT, sysMap, rand = rand_clean, nside = nside, raTag = 'RA', decTag='DEC', property = property, filter = filter)
            
            #bins = bins/np.sum(sysMap['SIGNAL']) *len(sysMap)
            #C_jkerr = jksampling(cmass_catalog, sysMap, nside = nside, njack = 10, raTag = 'RA', decTag = 'DEC' )
            #B_jkerr = jksampling(balrog_cmass, sysMap, nside = nside, njack = 10, raTag = 'RA', decTag = 'DEC' )
            
            filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+'.txt'
            DAT = np.column_stack(( bins-(bins[1]-bins[0])*0.1, Bdensity, Berr, Bf_area, Bdensity, Berr, Bf_area  ))
            np.savetxt(filename, DAT, delimiter = ' ', header = 'bins, Cdensity, Cerr, Cfarea, Bdensity, Berr, Bfarea')
            print "saving data to ", filename


    # visualization
    for property in properties:
        
        if property is 'NSTARS' : nside = 512
        
        fig, ax = plt.subplots(2, 2, figsize = (15, 10))
        ax = ax.ravel()
        for i, filter in enumerate(filters):
            filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+'.txt'
            #print filename
            data = np.loadtxt(filename)
            bins, Cdensity, Cerr, Cf_area, Bdensity, Berr, Bf_area = [data[:,j] for j in range(data[0].size)]
            zeromaskC, zeromaskB = ( Cdensity != 0.0 ), (Bdensity != 0.0 )
            Cdensity, Cbins, Cerr = Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
            #C_jkerr = C_jkerr[zeromaskC]
            Bdensity, Bbins, Berr = Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
            #B_jkerr = B_jkerr[zeromaskB]
            
            #fitting
            Cchi, Cchidof = chisquare_dof( Cbins, Cdensity, Cerr )
            Bchi, Bchidof = chisquare_dof( Bbins, Bdensity, Berr )
            
            #ax[i].errorbar(Cbins-(bins[1]-bins[0])*0.1, Cdensity, yerr = Cerr, color = 'blue', fmt = '.', label='CMASS') #, chi2/dof={:>2.2f}'.format(Cchidof))
            ax[i].errorbar(Bbins+(bins[1]-bins[0])*0.1, Bdensity, yerr = Berr, color = 'red', fmt= '.',  label='DMASS')#, chi2/dof={:>2.2f}'.format(Bchidof))
            #ax[i].bar(Cbins+(bins[1]-bins[0])*0.1, Cf_area[zeromaskC],(bins[1]-bins[0]) ,color = 'blue', alpha = 0.3 )
            ax[i].bar(Bbins+(bins[1]-bins[0])*0.1, Bf_area[zeromaskB]+0.5, (bins[1]-bins[0]) ,color = 'red', alpha=0.3 )
            ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
            ax[i].set_ylabel('n_gal/n_tot '+str(nside))
            ax[i].set_ylim(0.5, 1.5)
            #ax[i].set_xlim(8.2, 8.55)
            ax[i].axhline(1.0,linestyle='--',color='grey')
            ax[i].legend(loc = 'best')
            
            #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
            #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
            #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
            if property == 'NSTARS': ax[i].set_xlim(0.0, 2.0)
        
        fig.suptitle('systematic test ({})'.format(kind))
        figname = 'figure/systematic_'+property+'_'+kind+'.png'
        fig.savefig(figname)
        print "saving fig to ", figname



    # correcting systematic errors
    # ** Mask Bad Region
    #
    #

    #properties = ['NSTARS','FWHM', 'AIRMASS','SKYSIGMA', 'SKYBRITE']
    #filters = ['g','r','i', 'z']
    kind = 'SPT'
    nside = 1024
    njack = 10
    
    property = 'NSTARS'
    filter = 'g'

    from systematics import SysMapBadRegionMask, loadSystematicMaps, MatchHPArea
    from systematics_module.corr import angular_correlation

    dmass_stripe82 = dmass_y1a1[dmass_y1a1['DEC'] > -3.0 ]
    dmass_SSPT = MatchHPArea(cat = dmass_y1a1, origin_cat = balrog_SSPT)
    nside = 1024
    kind = 'SPT'

    properties = ['NSTARS','FWHM', 'AIRMASS','SKYSIGMA', 'SKYBRITE']
    #property = [ 'AIRMASS', 'SKYBRITE', 'FWHM', 'SKYSIGMA']
    filter = ['g', 'r', 'i', 'z']

    # Calling maps
    MaskDic = {}
    for i,p in enumerate(property):
        for j,f in enumerate(filter):
            if property == 'NSTARS':
                filename =  'y1a1_gold_1.0.2_stars_nside1024.fits'
                nside = 512
            else : filename = None
            
            sysMap = loadSystematicMaps( filename = filename, property = p, filter = f, nside = nside , kind = kind)
            sysMap = MatchHPArea( cat = sysMap, origin_cat=balrog_SSPT)
            mapname = 'sys_'+p+'_'+f
            MaskDic[mapname] = sysMap




    for p in property:
        correctMask = np.ones(dmass_SSPT.size, dtype=bool)
        
        for i, filter in enumerate(filters):

            bins, Bdensity, Berr, Bf_area = GalaxyDensity_Systematics(dmass_SSPT, sysMap, rand = rand_clean, nside = nside, raTag = 'RA', decTag='DEC', property = p, filter = filter)
            
            filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+'.txt'
            DAT = np.column_stack(( bins-(bins[1]-bins[0])*0.1, Bdensity, Berr, Bf_area, Bdensity, Berr, Bf_area  ))
            np.savetxt(filename, DAT, delimiter = ' ', header = 'bins, Cdensity, Cerr, Cfarea, Bdensity, Berr, Bfarea')
            print "saving data to ", filename

            mapname = 'sys_'+p+'_'+f
            correctMask = SysMapBadRegionMask(dmass_SSPT, MaskDic[mapname], nside = nside, cond = '<=', val = 1.35)
            MaskDic[p+'_mask'] = correctMask
            print p+'_mask', np.sum(correctMask)


    """

    for i,p in enumerate(property):
        
            for j,f in enumerate(filter):
                mapname = 'sys_'+p+'_'+f
                correctMask = correctMask * SysMapBadRegionMask(dmass_SSPT, MaskDic[mapname], nside = nside, cond = '<=', val = value[i][j])
        MaskDic[p+'_mask'] = correctMask
            print p+'_mask', np.sum(correctMask)


    """




    value = [[1.45, 1.45, 1.45, 1.45],  # airmass
             [120, 350, 1150, 2500],  # skybrightness
             [6.5, 5.0, 4.5, 4.0], # fwhm (seeing)
             [5.8, 8.7, 17, 25]]       # skysigma
            #[6., 9.5, 17, 25]]

    for i,p in enumerate(property):
        correctMask = np.ones(dmass_SSPT.size, dtype=bool)
        for j,f in enumerate(filter):
            mapname = 'sys_'+p+'_'+f
            correctMask = correctMask * SysMapBadRegionMask(dmass_SSPT, MaskDic[mapname], nside = nside, cond = '<=', val = value[i][j])
        MaskDic[p+'_mask'] = correctMask
        print p+'_mask', np.sum(correctMask)


    TotalMask = MaskDic['AIRMASS_mask'] * MaskDic['FWHM_mask'] * MaskDic['SKYSIGMA_mask'] * MaskDic['SKYBRITE_mask']

    clean_dmass = dmass_SSPT[TotalMask]
    clean_dmass_AIRMASS = dmass_SSPT[MaskDic['AIRMASS_mask']]
    clean_dmass_SKYBRITE = dmass_SSPT[ MaskDic['SKYBRITE_mask'] ]
    clean_dmass_FWHM = dmass_SSPT[MaskDic['FWHM_mask'] ]
    clean_dmass_SKYSIGMA = dmass_SSPT[  MaskDic['SKYSIGMA_mask'] ]


    # Testing difference
    rand_o = fitsio.read('/n/des/lee.5922/data/random0_DR11v1_CMASS_South.fits.gz')
    rand_clean_st82 = Cuts.keepGoodRegion(rand_o)
    balrog_y1a1 = fitsio.read('result_cat/result_balrog_EMHUFF.fits')
    balrog_SSPT = balrog_y1a1[balrog_y1a1['DEC'] < -3]
    #dmass_balrog_SSPT, _ = resampleWithPth(balrog_SSPT)
    rand_clean = Cuts.keepGoodRegion(balrog_SSPT)

    #angular_correlation(data = dmass_SSPT, rand = rand_clean, suffix = '_No_mask')
    #angular_correlation(data = clean_dmass, rand = rand_clean, suffix = '_All_mask')
    #angular_correlation(data = clean_dmass_AIRMASS, rand = rand_clean, suffix = '_mask_AIRMASS')
    #angular_correlation(data = clean_dmass_SKYBRITE, rand = rand_clean, suffix = '_mask_SKYBRITE')
    #angular_correlation(data = clean_dmass_FWHM, rand = rand_clean, suffix = '_mask_FWHM')
    #angular_correlation(data = clean_dmass_SKYSIGMA, rand = rand_clean, suffix = '_mask_SKYSIGMA')


    # ** Weighting
    #
    #   weighting
    #   NSTAR
    #   FWHM riz bands
    #
    from systematics import ReciprocalWeights

    re_weights = ReciprocalWeights( catalog = clean_dmass, property = 'NSTARS', filter = 'g', nside = 512, kind = kind )
    angular_correlation(data = clean_dmass, rand = rand_clean, weight = [re_weights, None], suffix = '_'+kind+'_w+mask_'+property)


    weight_property = ['FWHM', 'SKYBRITE', 'SKYSIGMA']

    reweights = np.ones(clean_dmass.size, dtype=float)
    for p in weight_property:
        for f in filters:
            reweights = reweights * ReciprocalWeights( catalog = clean_dmass, sysMap = MaskDic['sys_'+p+'_'+f], property = p, filter = f, nside = 1024, kind = kind )
        MaskDic[p+'_weight'] = re_weights
        
        
    result = [ angular_correlation(data = clean_dmass, rand = rand_clean, weight = [MaskDic[p+'_weight'], None], suffix = '_w+mask_'+p) for p in weight_property ]



    # applying weight iteratively -----------------











    # ---------------------------------------------
    # Foreground Stars

    # from Ashley's paper :
    # Take des stars between 17.5 < i_psf < 19.9
    # devide dmass galaxy sample between 18.5 < i_detmag < 19.9 into 5 bins
    # 1) number density as a ftn of degree for 5 different bins around the whole star sample
    # 2) number density for the whole galaxy sample around star between 19 < i_psf < 20.5
    # 3) seeing 6 bins
    # test stripe 82
    # test whole y1a1


    """
    balrog SPT region : result_cat/result_balrog_y1a1.fit'
    """
    
    from systematics import MatchHPArea
    
    # random
    balrog_y1a1 = fitsio.read('result_cat/result_balrog_EMHUFF.fits')
    balrog_SSPT = balrog_y1a1[balrog_y1a1['DEC'] < -3]
    dmass_balrog_SSPT, _ = resampleWithPth(balrog_SSPT)
    rand_clean = Cuts.keepGoodRegion(dmass_balrog_SSPT)
    #random
    #result_barlog = fitsio.read('result_cat/result_balrog_JELENA.fits')
    #cat_cmass_rand82, _ = resampleWithPth(result_balrog)
    #rand_o = fitsio.read('/n/des/lee.5922/data/random0_DR11v1_CMASS_South.fits.gz')
    #rand_clean = Cuts.keepGoodRegion(rand_o)
    
    # dmass
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    dmass_y1a1, _ = resampleWithPth(result_y1a1)
    dmass_SSPT = MatchHPArea(dmass_y1a1, balrog_SSPT)
    
    
    # star
    SSPT_star = fitsio.read('result_cat/SSPT_star.fits')
    #full_des_data = io.getDESY1A1catalogs(keyword = 'des_st82', sdssmask=False)
    #clean_des_star = Cuts.doBasicCuts(full_des_data, object = 'star')
    star_maglimit = ( SSPT_star['MAG_PSF_I'] > 17.5 ) & ( SSPT_star['MAG_PSF_I'] < 19.9 )
    bright_des_star = SSPT_star[star_maglimit]
    

    dmass = dmass_SSPT.copy()
    des_star = SSPT_star.copy()
    bright_des_star = bright_des_star.copy()
    rand = balrog_SSPT.copy()

    from systematics import ForegroundStarCorrelation
    
    bins, step = np.linspace(19.5, 20.5, 5, retstep = True)
    bin_center, binned_dmass, binned_keep = divide_bins( dmass, Tag = 'MAG_MODEL_I', min = bins[0], max = bins[-1], bin_num = bins.size-1 )
    
    bins2, step2 = np.linspace(19.0, 20.5, 5, retstep = True)
    bin_center2, binned_star, binned_keep2 = divide_bins( des_star, Tag = 'MAG_PSF_I', min = bins2[0], max = bins2[-1], bin_num = bins2.size-1 )
    
    labels = [ 'i_mod < {}'.format(bins[0])] + ['{:>0.2f} < i_mod < {:>0.2f}'.format(bins[i], bins[i+1]) for i in range(len(bins)-1) ]
    labels2 = [ 'i_psf < {}'.format(bins2[0])] + ['{:>0.2f} < i_psf < {:>0.2f}'.format(bins2[i], bins2[i+1]) for i in range(len(bins2)-1) ]

    bins3, step = np.linspace(20.25, 21.0, 5, retstep = True)
    bin_center3, binned_dmass3, binned_keep3 = divide_bins( dmass, Tag = 'MAG_APER_4_I', min = bins3[0], max = bins3[-1], bin_num = bins3.size-1 )
    
    labels3 = [ 'i_aper < {}'.format(bins3[0])] + ['{:>0.2f} < i_aper < {:0.2f}'.format(bins3[i], bins3[i+1]) for i in range(len(bins3)-1) ]


    imod_average, ipsf_average, err_imod, err_ipsf = [], [], [], []
    seeing_average, iaper_average, err_seeing, err_iaper = [], [], [], []
    
    for cat, cat2, cat3 in zip(binned_dmass, binned_star, binned_dmass3):
        print cat.size, cat2.size, cat3.size
        theta, n_dmass, err_dmass = ForegroundStarCorrelation(dmass = cat, star = bright_des_star, rand = rand )
        theta, n_star, err_star = ForegroundStarCorrelation(dmass = dmass, star = cat2, rand = rand )
        theta, n_dmass3, err_dmass3 = ForegroundStarCorrelation(dmass = cat3, star = bright_des_star, rand = rand )
        
        imod_average.append(n_dmass)
        ipsf_average.append(n_star)
        iaper_average.append(n_dmass3)
        err_imod.append(err_dmass)
        err_ipsf.append(err_star)
        err_iaper.append(err_dmass3)
    
    data_imod = np.column_stack(( [theta] + imod_average + err_imod ))
    data_ipsf = np.column_stack(( [theta] + ipsf_average + err_ipsf ))
    data_iaper = np.column_stack(( [theta] + iaper_average + err_iaper ))
    np.savetxt('data_txt/foregroundStar_imod.txt', data_imod, delimiter = ' ', header='theta '+str(labels) + ' err')
    np.savetxt('data_txt/foregroundStar_ipsf.txt', data_ipsf, delimiter = ' ', header='theta '+str(labels2)+ ' err')
    np.savetxt('data_txt/foregroundStar_iaper.txt', data_iaper, delimiter = ' ', header='theta '+str(labels3)+ ' err')



    # visualization
    fig, ax = plt.subplots(2,2, figsize = (20,14))
    ax = ax.ravel()

    data_imod = np.loadtxt('data_txt/foregroundStar_imod.txt')
    data_ipsf = np.loadtxt('data_txt/foregroundStar_ipsf.txt')
    data_iaper = np.loadtxt('data_txt/foregroundStar_iaper.txt')
    
    data = [data_imod, data_ipsf, data_iaper]
    label = [labels, labels2, labels3]

    for j,d in enumerate(data):
        theta = d[:,0] * 3600
        for i in range((len(d[0])-1)/2):
            ax[j].errorbar(theta, d[:,i+1], yerr=d[:,i+(len(d[0])-1)/2 + 1], linestyle='-', fmt='.' ,label = label[j][i])
        ax[j].axhline(y=1.0, color='grey', linestyle='--')
        ax[j].legend(loc='best')
        ax[j].set_xlim(0.0, 50)
        #ax[j].set_xscale('log')
        ax[j].set_ylim(0.0, 2.0)
        ax[j].set_ylabel(r'$n$/$n_{avg}$')
        ax[j].set_xlabel(r'$\theta$ $(\rm{arcsec})$')

    fig.savefig('figure/foregroundStar.png')




    # systematic - ideal fraction --------------------------------------


    balrog_y1a1 = fitsio.read('result_cat/result_balrog_EMHUFF.fits')
    balrog_SSPT = balrog_y1a1[balrog_y1a1['DEC'] < -3]
    dmass_balrog_SSPT, _ = resampleWithPth(balrog_SSPT)
    rand_clean = Cuts.keepGoodRegion(dmass_balrog_SSPT)


    #cat_cmass_rand82, _ = resampleWithPth(result_balrog)
    #rand_o = fitsio.read('/n/des/lee.5922/data/random0_DR11v1_CMASS_South.fits.gz')
    #rand_clean = Cuts.keepGoodRegion(rand_o)
    
    # dmass
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    dmass_y1a1, _ = resampleWithPth(result_y1a1)
    dmass_SSPT = MatchHPArea(cat = dmass_y1a1, origin_cat=balrog_SSPT)

    cmass_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_South.fits.gz')
    rand_cmass_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz')
    cmass_NGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_North.fits.gz')
    rand_cmass_NGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_North.fits.gz')

    rand_cmass_SGC =  excludeSt82(rand_cmass_SGC)

    #cmass_photoAll = fitsio.read('/n/des/lee.5922/data/cmass_cat/cmass_photoz_All.fits')

    def excludeSt82(cat):
        mask = (cat['RA'] < 360) & (cat['RA'] > 310) & (cat['DEC'] > -1.5) & (cat['DEC'] < 1.5)
        return cat[~mask]

    cat = [cmass_SGC, rand_cmass_SGC, cmass_NGC, rand_cmass_NGC]
    cmass_SGC, rand_cmass_SGC, cmass_NGC, rand_cmass_NGC = [ excludeSt82(c) for c in cat ]


    from systematics_module.corr import LensingSignal, cross_angular_correlation, angular_correlation
    from systematics import MatchHPArea
    
    # dmass SSPT
    angular_correlation(data = dmass_y1a1, rand = rand_clean,  weight = [None, None], suffix = '_SSPT_y1a1')
    # NGC
    angular_correlation(data = cmass_NGC, rand = rand_cmass_NGC,  weight = [None,None], suffix = '_cmass_NGC')
    angular_correlation(data = cmass_SGC, rand = rand_cmass_SGC,  weight = [None,None], suffix = '_cmass_SGC')


    result = fitsio.read('result_cat/result1.fits')
    As_dmass,_ = resampleWithPth(result)
    noblend_clean_cmass_data = fitsio.read('result_cat/noblend_cmass_stripe82_photo.fits')
    noblend_clean_cmass_data_des = fitsio.read('result_cat/noblend_cmass_stripe82_photo_des.fits')
    EstimateContaminantFraction( dmass = As_dmass, cmass = noblend_clean_cmass_data )

    # calculating contaminant fraction with full correl

    import esutil
    fullcmass_name = "/n/des/lee.5922/data/cmass_cat/cmass_noclean_SGC_NGC.fits"
    fullcmass = esutil.io.read(fullcmass_name, upper=True)
    fullcmass = Cuts.CMASSQaulityCut(fullcmass[fullcmass['FIBER2MAG_I'] < 21.5 ])
    fullcmass = excludeSt82(fullcmass)
    
    """
    NGCmask = (fullcmass['RA'] > 70) & (fullcmass['RA'] < 300)
    fullcmass_NGC, fullcmass_SGC = fullcmass[NGCmask], fullcmass[~NGCmask]
    """
    
    # match HP area
    fullcmass_NGC = MatchHPArea(cat=fullcmass, origin_cat=rand_cmass_NGC, nside=512)
    fullcmass_SGC = MatchHPArea(cat=fullcmass, origin_cat=rand_cmass_SGC, nside=512)

    # SDSS CMASS full correlation (NGC+ SGC - st82)
    angular_correlation(data = fullcmass, rand = rand, suffix = '_SDSS_full')
    # DES CMASS full correlation (half of SPT )
    angular_correlation(data = dmass_SSPT, rand = balrog_SSPT,  weight = [None, None], suffix = '_DES_SSPT')
    angular_correlation(data = cmass_SGC, rand = rand_cmass_SGC,  weight = [None,None], suffix = '_cmass_SGC')
    angular_correlation(data = cmass_NGC, rand = rand_cmass_NGC,  weight = [None,None], suffix = '_cmass_NGC')

    angular_correlation(data = fullcmass_NGC, rand = rand_cmass_NGC,  weight = [None,None], suffix = '_fullcmass_NGC')
    angular_correlation(data = fullcmass_SGC, rand = rand_cmass_SGC,  weight = [None,None], suffix = '_fullcmass_SGC')


if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        #main(sys.argv)
        main()
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)




