import numpy as np
from matplotlib import pyplot as plt
from astroML.decorators import pickle_results
from astroML.density_estimation import XDGMM
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts


def SDSS_cmass_criteria(sdss, prior=None):
    
    modelmag_r = sdss['MODELMAG_R'] - sdss['EXTINCTION_R']
    modelmag_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
    modelmag_g = sdss['MODELMAG_G'] - sdss['EXTINCTION_G']
    cmodelmag_i = sdss['CMODELMAG_I'] - sdss['EXTINCTION_I']
    
    dperp = (modelmag_r - modelmag_i) - (modelmag_g - modelmag_r)/8.0
    fib2mag = sdss['FIBER2MAG_I']
    
    if prior is True:
        priorCut = ((cmodelmag_i > 16.0 ) &
                    (cmodelmag_i < 22.0) &
                    ((modelmag_r - modelmag_i ) < 2.0 )&
                    ((modelmag_g - modelmag_r ) < 3.0 )&
                    (fib2mag < 25.0 ))  #&

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
    
    print "Calculating/applying CMASS object cuts."
    modelmag_g_des = data['MAG_DETMODEL_G'] - data['XCORR_SFD98_G']
    modelmag_r_des = data['MAG_DETMODEL_R'] - data['XCORR_SFD98_R']
    modelmag_i_des = data['MAG_DETMODEL_I'] - data['XCORR_SFD98_I']
    cmodelmag_i_des = data['MAG_MODEL_I'] - data['XCORR_SFD98_I']
    fib2mag_des = data['MAG_APER_4_I']

    cut = ((cmodelmag_i_des > 15) &
           (cmodelmag_i_des < 22.) &
           ((modelmag_r_des - modelmag_i_des ) < 2.0 ) &
           ((modelmag_r_des - modelmag_i_des ) > 0.0 ) &
           ((modelmag_g_des - modelmag_r_des ) > 0.0 ) &
            ((modelmag_g_des - modelmag_r_des ) < 3. )
           # (fib2mag_des < 25.5 )
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


def mixing_color(des):
    
    # des
    des_g = des['MAG_DETMODEL_G'] - des['XCORR_SFD98_G']
    des_r = des['MAG_DETMODEL_R'] - des['XCORR_SFD98_R']
    des_i = des['MAG_DETMODEL_I'] - des['XCORR_SFD98_I']
    #des_z = des['MAG_DETMODEL_Z'] - des['XCORR_SFD98_Z']
    des_ci = des['MAG_MODEL_I'] - des['XCORR_SFD98_I']
    Y = np.vstack([des_ci, des_g, des_r, des_i]).T
    Yerr = np.vstack([ des['MAGERR_MODEL_I'] , des['MAGERR_DETMODEL_G'], des['MAGERR_DETMODEL_R'],des['MAGERR_DETMODEL_I']]).T
    

    
    # mixing matrix
    W = np.array([[1, 0, 0, 0],    # i cmagnitude
                  [0, 1, -1, 0],   # g-r
                  [0, 0, 1, -1]])  # r-i
    Y = np.dot(Y, W.T)
    
    return Y


def mixing_color_sdss(sdss, iter = 0):
    """
    sdss_g = sdss['MODELMAG'][:,1] - sdss['EXTINCTION'][:,1]
    sdss_r = sdss['MODELMAG'][:,2] - sdss['EXTINCTION'][:,2]
    sdss_i = sdss['MODELMAG'][:,3] - sdss['EXTINCTION'][:,3]
    sdss_ci = sdss['CMODELMAG'][:,3] - sdss['EXTINCTION'][:,3]
    """
    sdss_g = sdss['MODELMAG_G'] - sdss['EXTINCTION_G']
    sdss_r = sdss['MODELMAG_R'] - sdss['EXTINCTION_R']
    sdss_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
    sdss_ci = sdss['CMODELMAG_I'] - sdss['EXTINCTION_I']


    X = np.vstack([sdss_ci, sdss_g, sdss_r, sdss_i]).T
    W = np.array([[1, 0, 0, 0],    # i cmagnitude
                  [0, 1, -1, 0],   # g-r
                  [0, 0, 1, -1]])  # r-i

    X = np.dot(X, W.T)
    #Xerr = np.vstack([ sdss['CMODELMAGERR'][:,3] , sdss['MODELMAGERR'][:,1], sdss['MODELMAGERR'][:,2],sdss['MODELMAGERR'][:,3]]).T
    Xerr = np.vstack([ sdss['CMODELMAGERR_I'] , sdss['MODELMAGERR_G'], sdss['MODELMAGERR_R'],sdss['MODELMAGERR_I']]).T
    
    
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    
    # each covariance C = WCW^T
    # best way to do this is with a tensor dot-product
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))
    
    @pickle_results('XD_cmass{}.pkl'.format(iter))
    def compute_XD(n_clusters=12, rseed=0, n_iter=100, verbose=False):
        np.random.seed(rseed)
        clf = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf.fit(X, Xcov)
        return clf
    
    clfX = compute_XD()
    X = clfX.sample(X.shape[0])
    
    return X


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
    #print 'figsave : ../figure/spatialtest.png'



def MachineLearningClassifier( cmass, non, k_neighbors_max = 10 ):

    # Author: Jake VanderPlas
    # License: BSD
    #   The figure produced by this code is published in the textbook
    #   "Statistics, Data Mining, and Machine Learning in Astronomy" (2013)
    #   For more information, see http://astroML.github.com
    #   To report a bug or issue, use the following forum:
    #    https://groups.google.com/forum/#!forum/astroml-general
    

    from astroML.utils import split_samples

    from sklearn.metrics import roc_curve
    from sklearn.naive_bayes import GaussianNB
    from sklearn.lda import LDA
    from sklearn.qda import QDA
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from astroML.classification import GMMBayes
    from astroML.utils import completeness_contamination

    #----------------------------------------------------------------------
    # This function adjusts matplotlib settings for a uniform feel in the textbook.
    # Note that with usetex=True, fonts are rendered with LaTeX.  This may
    # result in an error if LaTeX is not installed on your system.  In that case,
    # you can set usetex to False.
    #from astroML.plotting import setup_text_plots
    #setup_text_plots(fontsize=8, usetex=True)


    #cmass = cmass[::10]
    #non = non[::10]


    # stack colors into matrix X
    
    Ncmass = len(cmass)
    Nnon = len(non)
    X = np.empty((Ncmass + Nnon, 3), dtype=float)

    X[:Ncmass, 0] = cmass['MAG_DETMODEL_G'] - cmass['MAG_DETMODEL_R']
    X[:Ncmass, 1] = cmass['MAG_DETMODEL_R'] - cmass['MAG_DETMODEL_I']
    X[:Ncmass, 2] = cmass['MAG_MODEL_I'] #/np.max(cmass['MAG_MODEL_I'])
    
    X[Ncmass:, 0] = non['MAG_DETMODEL_G'] - non['MAG_DETMODEL_R']
    X[Ncmass:, 1] = non['MAG_DETMODEL_R'] - non['MAG_DETMODEL_I']
    X[Ncmass:, 2] = non['MAG_MODEL_I'] #/np.max(cmass['MAG_MODEL_I'])

    y = np.zeros(Ncmass + Nnon, dtype=int)
    y[:Ncmass] = 1

    # split into training and test sets
    (X_train, X_test), (y_train, y_test) = split_samples(X, y, [0.5, 0.5],
                                                         random_state=0)
    
    #------------------------------------------------------------
    # Compute fits for all the classifiers
    def compute_results(*args):
        names = []
        probs = []
        predictions=[]
        
        for classifier, kwargs in args:
            print classifier.__name__
            model = classifier(**kwargs)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)
            y_pred = model.predict(X_test)
            names.append(classifier.__name__)
            probs.append(y_prob[:, 1])
            predictions.append(y_pred)
        
        return names, probs, predictions

    LRclass_weight = dict([(i, np.sum(y_train == i)) for i in (0, 1)])
    
    names, probs, predictions = compute_results((GaussianNB, {}),
                                   (LDA, {}),
                                   (QDA, {}),
                                   (LogisticRegression,
                                    dict(class_weight=LRclass_weight)),
                                   (KNeighborsClassifier,
                                    dict(n_neighbors=10)),
                                   (DecisionTreeClassifier,
                                    dict(random_state=0, max_depth=12,
                                         criterion='entropy')))
#,
#                                   (GMMBayes, dict(n_components=3, min_covar=1E-5,
#                                                   covariance_type='full')))
#
    

  

    #------------------------------------------------------------
    # Plot results
    fig = plt.figure(figsize=(5, 2.5))
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.9, wspace=0.25)

    # First axis shows the data
    ax1 = fig.add_subplot(121)
    im = ax1.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=4,
                     linewidths=0, edgecolors='none',
                     cmap=plt.cm.binary)
    im.set_clim(-0.5, 1)
    ax1.set_xlim(-0.5, 3.0)
    ax1.set_ylim(-0.3, 1.4)
    ax1.set_xlabel('g - r')
    ax1.set_ylabel('r - i')

    labels = dict(GaussianNB='GNB',
              LinearDiscriminantAnalysis='LDA',
              QuadraticDiscriminantAnalysis='QDA',
              KNeighborsClassifier='KNN',
              DecisionTreeClassifier='DT',
              GMMBayes='GMMB',
              LogisticRegression='LR')


    # Second axis shows the ROC curves
    ax2 = fig.add_subplot(122)
    for name, y_prob in zip(names, probs):
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        
        fpr = np.concatenate([[0], fpr])
        tpr = np.concatenate([[0], tpr])
        
        ax2.plot(fpr, tpr, label=labels[name])


    ax2.legend(loc=4)
    ax2.set_xlabel('false positive rate')
    ax2.set_ylabel('true positive rate')
    ax2.set_xlim(0, 0.15)
    ax2.set_ylim(0.6, 1.01)
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))


    """
    y_pred = predictions[4]
    fig, ax3 = plt.subplots()
    im = ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, s=4,
                     linewidths=0, edgecolors='none',
                     cmap=plt.cm.binary)
    im.set_clim(-0.5, 1)
    ax3.set_xlim(-0.5, 3.0)
    ax3.set_ylim(-0.3, 1.4)
    ax3.set_xlabel('g - r')
    ax3.set_ylabel('r - i')
    from astroML.utils import completeness_contamination
    completeness, contamination = completeness_contamination(y_pred, y_test)
    ax3.set_title('completeness : {}'.format( completeness) )

    """

    plt.show()

    


    def compute_kNN( k_neighbors_max = k_neighbors_max ):
        classifiers = []
        predictions = []
        kvals = np.arange(1,k_neighbors_max)
        purity = []

        for k in kvals:
            classifiers.append([])
            predictions.append([])
        
            clf = KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pur = np.sum(y_pred * y_test) *1./np.sum(y_pred)
            purity.append(pur)
            classifiers[-1].append(clf)
            predictions[-1].append(y_pred)

        completeness, contamination = completeness_contamination(predictions, y_test)
        print '\ncompleteness:', np.array(completeness).ravel()
        #print 'contamination:', np.array(contamination).ravel()
        print 'purity:', np.array(purity).ravel()
        return np.array(completeness).ravel(), np.array(purity).ravel()

    def compute_GMMB():
       
        classifiers = []
        predictions = []
        Ncomp = np.arange(1,10,2)
        purity = []

        for nc in Ncomp:
            classifiers.append([])
            predictions.append([])
            clf = GMMBayes(nc, min_covar=1E-5, covariance_type='full')
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            pur = np.sum(y_pred * y_test) *1./np.sum(y_pred)
            purity.append(pur)
            classifiers[-1].append(clf)
            predictions[-1].append(y_pred)

        completeness, contamination = completeness_contamination(predictions, y_test)
        print '\ncompleteness:', np.array(completeness).ravel()
        print 'purity:', np.array(purity).ravel()
        return np.array(completeness).ravel(), np.array(purity).ravel()

    #completeness, purity = compute_kNN(k_neighbors_max = k_neighbors_max)
    completeness, purity = compute_GMMB()
    return  completeness, purity


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


def XDGMM_model(cmass, lowz, p_threshold = 0.5, train = None, test = None):
    
    
    import esutil
    train, _ = priorCut(train)
    test, _ = priorCut(test)

    # making cmass and lowz mask
    

    h = esutil.htm.HTM(10)
    matchDist = 2/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(train['RA'],train['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass = np.zeros( train.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask = true_cmass == 1

    m_des, m_sdss, d12 = h.match(train['RA'], train['DEC'], lowz['RA'],lowz['DEC'],matchDist,maxmatch=1)
    true_lowz = np.zeros( train.size, dtype = int)
    true_lowz[m_des] = 1
    lowz_mask = true_lowz == 1
    
    
    print 'num of cmass/lowz', np.sum(cmass_mask), np.sum(lowz_mask)


    data = np.hstack(( train, test ))
    # stack DES data
    des_g = data['MAG_DETMODEL_G'] - data['XCORR_SFD98_G']
    des_r = data['MAG_DETMODEL_R'] - data['XCORR_SFD98_R']
    des_i = data['MAG_DETMODEL_I'] - data['XCORR_SFD98_I']
    des_z = data['MAG_DETMODEL_Z'] - data['XCORR_SFD98_Z']
    des_ci = data['MAG_MODEL_I'] - data['XCORR_SFD98_I']
    des_cr = data['MAG_MODEL_R'] - data['XCORR_SFD98_R']


    X = np.vstack([des_cr, des_ci, des_g, des_r, des_i, des_z]).T
    Xerr = np.vstack([ data['MAGERR_MODEL_R'] , data['MAGERR_MODEL_I'], data['MAGERR_DETMODEL_G'], data['MAGERR_DETMODEL_R'],data['MAGERR_DETMODEL_I'],data['MAGERR_DETMODEL_Z']]).T

    
    # mixing matrix
    W = np.array([[1, 0, 0, 0, 0, 0],    # r cmag
                  [0, 1, 0, 0, 0, 0],    # i cmag
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],   # r-i
                  [0, 0, 0, 0, 1, -1]])  # i-z

    X = np.dot(X, W.T)
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    
    # each covariance C = WCW^T
    # best way to do this is with a tensor dot-product
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))

    y_train = np.zeros(( train.size, 2 ), dtype=int)
    # ycov = np.zeros(sdss.size, dtype=int)
    y_train[:,0][cmass_mask] = 1
    y_train[:,1][lowz_mask] = 1

    # mask for train/test sample
    #(trainInd, testInd), (_, _) = split_samples(X, X, [0.9,0.1], random_state=0)
    X_train = X[ :len(train), :]
    X_test = X[-len(test):, :]
    Xcov_train = Xcov[ :len(train), :]
    Xcov_test = Xcov[-len(test):, :]

    print 'train/test', len(X_train), len( X_test ) 
    # train sample XD convolution
  
    """
    @pickle_results("XD_train_all.pkl")
    def compute_XD(n_clusters=30, n_iter=500, verbose=True):
        clf= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        
        sample = np.arange( len(X_train) )
        rows = np.random.choice( sample, size= len(X_train[cmass_mask] * 10 ), replace = False)
        rows_mask = np.zeros( len(X_train), dtype=bool)
        rows_mask[rows] = 1
        
        clf.fit(X_train[rows_mask], Xcov_train[rows_mask])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf
    
    clf = compute_XD()
    X_all = clf.sample(500 * X.shape[0])
    
    """
    
    # cmass extreme deconvolution compute
    @pickle_results("XD_dmass_train_all.pkl")
    def compute_XD(n_clusters=30, n_iter=50, verbose=True):
        clf_cmass= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf_cmass.fit(X_train[cmass_mask], Xcov_train[cmass_mask])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf_cmass

    clf_cmass = compute_XD()
    X_sample_cmass = clf_cmass.sample(100 * X.shape[0])
    
    """
    print "calculate loglikelihood gaussian"
    logprob_a = clf_cmass.logprob_a( X[X_test_ind], Xcov[X_test_ind] )
    sum_logprob_a = logsumexp(logprob_a, axis = 1)
    """

    """
    @pickle_results("XD_dmass_no.pkl")
    def compute_XD(n_clusters=30, n_iter=500, verbose=True):
        clf_nocmass= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        
        sample = np.arange( len(X_train[~cmass_mask]) )
        rows = np.random.choice( sample, size= len(X_train[cmass_mask] * 10 ), replace = False)
        rows_mask = np.zeros( len(X_train[~cmass_mask]), dtype=bool)
        rows_mask[rows] = 1
        
        clf_nocmass.fit(X_train[~cmass_mask][rows_mask], Xcov_train[~cmass_mask][rows_mask])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf_nocmass
    
    clf_nocmass = compute_XD()
    X_sample_no = clf_nocmass.sample(500 * X.shape[0])
    """
    
    @pickle_results("XD_lowz_train20.pkl")
    def compute_XD(n_clusters=20, n_iter=500, verbose=True):
        clf_lowz = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf_lowz.fit(X_train[lowz_mask], Xcov_train[lowz_mask])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf_lowz

    clf_lowz = compute_XD()
    X_sample_lowz = clf_lowz.sample(100 * X.shape[0])


    # 3d number density histogram
    bin0, step0 = np.linspace(X[:,0].min(), X[:,0].max(), 101, retstep=True) # cmodel r
    bin1, step1 = np.linspace(X[:,1].min(), X[:,1].max(), 101, retstep=True) # cmodel i
    bin2, step2 = np.linspace(X[:,2].min(), X[:,2].max(), 201, retstep=True) # gr
    bin3, step3 = np.linspace(X[:,3].min(), X[:,3].max(), 201, retstep=True) # ri
    bin0 = np.append( bin0, bin0[-1]+step0)
    bin1 = np.append( bin1, bin1[-1]+step1)
    bin2 = np.append( bin2, bin2[-1]+step2)
    bin3 = np.append( bin3, bin3[-1]+step3)
    
    
    
    # cmass histogram model probability
    N_XDProb, edges = np.histogramdd(X_sample_cmass[:,1:4], bins = [bin1, bin2, bin3])
    n_CMASS = N_XDProb * 1./np.sum( N_XDProb )

    X_sample_no = X_train[~cmass_mask].copy()
    N_XDProb, edges = np.histogramdd(X_sample_no[:,1:4], bins = [bin1, bin2, bin3])
    n_noCMASS = N_XDProb * 1./np.sum( N_XDProb )

    numerator =  n_CMASS * np.sum(y_train[:,0]) * 1.
    denominator =  (n_CMASS * np.sum(y_train[:,0]) + n_noCMASS * (len(X_train) - np.sum(y_train[:,0]) ))

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
    denominator =  (n_LOWZ * np.sum(y_train[:,1]) + n_noLOWZ * (len(X_train) - np.sum(y_train[:,1]) ))

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

    # Getting galaxies higher than threshold
    GetCMASS_mask = EachProb_CMASS > p_threshold
    X_pred_cmass = X_test[GetCMASS_mask]
    GetLOWZ_mask = EachProb_LOWZ > p_threshold
    X_pred_lowz = X_test[GetLOWZ_mask]
    
    
    
    #Test Histogram -------------------------------
    
    X = X_pred_cmass.copy()
    N_DMASS, edges = np.histogramdd(X[:,1:4], bins = [bin1, bin2, bin3])
    N_DMASS1 = np.sum(np.sum(N_DMASS, axis = 1), axis = 1)
    N_DMASS2 = np.sum(np.sum(N_DMASS, axis = 0), axis = 1)
    N_DMASS3 = np.sum(np.sum(N_DMASS, axis = 0), axis = 0)
    
    X = X_sample_cmass.copy()
    N_CMASS, edges = np.histogramdd(X[:,1:4], bins = [bin1, bin2, bin3])
    N_CMASS1 = np.sum(np.sum(N_CMASS, axis = 1), axis = 1)
    N_CMASS2 = np.sum(np.sum(N_CMASS, axis = 0), axis = 1)
    N_CMASS3 = np.sum(np.sum(N_CMASS, axis = 0), axis = 0)
    
    
    fig, (ax, ax2, ax3) = plt.subplots(1,3, figsize = (15, 5))
    ax.plot( bin1[:-1], N_DMASS1 * 1./np.sum(N_DMASS1), 'b.', label = 'DMASS cmodel i')
    ax.plot( bin1[:-1], N_CMASS1 * 1./np.sum(N_CMASS1), 'g.', label = 'CMASS cmodel i')
    #ax2.plot( bin2[:-1], N_BMASS2 * 1./np.sum(N_BMASS2), 'r.', label = 'BMASS g-r')
    ax2.plot( bin2[:-1], N_DMASS2 * 1./np.sum(N_DMASS2), 'b.', label = 'DMASS g-r')
    ax2.plot( bin2[:-1], N_CMASS2 * 1./np.sum(N_CMASS2), 'g.', label = 'CMASS g-r')
    #ax3.plot( bin3[:-1], N_BMASS3 * 1./np.sum(N_BMASS3), 'r.', label = 'BMASS r-i')
    ax3.plot( bin3[:-1], N_DMASS3 * 1./np.sum(N_DMASS3), 'b.', label = 'DMASS r-i')
    ax3.plot( bin3[:-1], N_CMASS3 * 1./np.sum(N_CMASS3), 'g.', label = 'CMASS r-i')
    ax.legend()
    ax2.legend()
    ax3.legend()
    fig.savefig('AmassHistogram')
    
    
    """
    if testSt82 is True:
        y_test = np.zeros((len(test), 2), dtype = bool)
        m_des, m_sdss, d12 = h.match(test['RA'],test['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
        true_cmass = np.zeros( len(test), dtype = int)
        true_cmass[m_des] = 1
        y_test[:,0] = true_cmass == 1
        
        m_des, m_sdss, d12 = h.match(test['RA'], test['DEC'], lowz['RA'],lowz['DEC'],matchDist,maxmatch=1)
        true_lowz = np.zeros( len(test), dtype = int)
        true_lowz[m_des] = 1
        y_test[:,1] = true_lowz == 1
    
        completeness_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(y_test[:,0])
        purity_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(GetCMASS_mask)
        contaminant_cmass = np.sum( GetCMASS_mask * np.logical_not(y_test[:,0]) )
        
        completeness_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(y_test[:,1])
        purity_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(GetLOWZ_mask)
        contaminant_lowz = np.sum( GetLOWZ_mask * np.logical_not(y_test[:,1]) )
        
        print 'com/purity(cmass)', completeness_cmass, purity_cmass
        print 'com/purity(lowz)', completeness_lowz, purity_lowz
        print 'num of dmass, dlowz in test', np.sum(GetCMASS_mask), np.sum(GetLOWZ_mask)
        print 'comtaminat', contaminant_cmass, contaminant_lowz
    
    """
    return train[cmass_mask], train[lowz_mask], test[GetCMASS_mask], test[GetLOWZ_mask]




    # completeness and purity
    completeness_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(y_test[:,0])
    purity_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(GetCMASS_mask)
    contaminant_cmass = np.sum( GetCMASS_mask * np.logical_not(y_test[:,0]) )

    completeness_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(y_test[:,1])
    purity_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(GetLOWZ_mask)
    contaminant_lowz = np.sum( GetLOWZ_mask * np.logical_not(y_test[:,1]) )

    print 'com/purity(cmass)', completeness_cmass, purity_cmass
    print 'com/purity(lowz)', completeness_lowz, purity_lowz
    print 'num of dmass, dlowz in test', np.sum(GetCMASS_mask), np.sum(GetLOWZ_mask) 
    print 'comtaminat', contaminant_cmass, contaminant_lowz



    # purity vs p_thresh check -----------------------------
    
    
    """
        
    p = np.linspace(0.0, 1.0, 100)
    
    coms = []
    purs = []
    coms2 = []
    purs2 = []
    
    pps = []
    for pp in p:
        
        # Getting galaxies higher than threshold
        GetCMASS_mask = EachProb_CMASS > pp
        X_pred_cmass = X_test[GetCMASS_mask]
        X_cont_cmass = X_test[GetCMASS_mask * np.logical_not(y_test[:,0])]
        GetLOWZ_mask = EachProb_LOWZ > pp
        X_pred_lowz = X_test[GetLOWZ_mask]
        X_cont_lowz = X_test[GetLOWZ_mask * np.logical_not(y_test[:,1])]
        # completeness and purity
        completeness_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(y_test[:,0])
        purity_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(GetCMASS_mask)
        contaminant_cmass = np.sum( GetCMASS_mask * np.logical_not(y_test[:,0]) )
        
        completeness_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(y_test[:,1])
        purity_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(GetLOWZ_mask)
        contaminant_lowz = np.sum( GetLOWZ_mask * np.logical_not(y_test[:,1]) )
        
        coms.append(completeness_cmass)
        purs.append(purity_cmass)
        coms2.append(completeness_lowz)
        purs2.append(purity_lowz)
        
        pps.append(pp)
        print pp, completeness_cmass, purity_cmass


    fig, (ax, ax2) = plt.subplots(1,2, figsize=(14,7))
    ax.plot( pps, coms, 'r.', label = 'completeness')
    ax.plot( pps, purs, 'b.', label = 'purity')
    ax.set_title('CMASS')
    ax.set_xlabel('p_threshold')
    ax2.plot( pps, coms2, 'r.', label = 'completeness')
    ax2.plot( pps, purs2, 'b.', label = 'purity')
    ax2.set_title('LOWZ')
    ax2.set_xlabel('p_threshold')
    ax.legend(loc = 'best')
    ax.set_ylim(0.0, 1.1)
    ax2.legend(loc = 'best')
    ax2.set_ylim(0.0, 1.1)

    fig.savefig('com_pur_check')

    
    """
    
    



def main():
    
    # load dataset

    dec = -1.0
    dec2 = 1.0
    ra = 330.0
    ra2 = 350.


    cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    cmass_data = Cuts.SpatialCuts(cmass_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    clean_cmass_data = Cuts.keepGoodRegion(cmass_data)
    lowz_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_LOWZ_South-photoObj.fits.gz')
    lowz_data = Cuts.SpatialCuts(lowz_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    clean_lowz_data = Cuts.keepGoodRegion(lowz_data)
    
    #sdss_data_o = io.getSDSScatalogs(bigSample = True)
    #sdss = Cuts.SpatialCuts(sdss_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    #sdss = Cuts.doBasicSDSSCuts(sdss)
    
    #full_des_data = io.getDEScatalogs(file = '', bigSample = True )
    full_des_data = io.getDESY1A1catalogs(keyword = 'st82')
    des_data_f = Cuts.SpatialCuts(full_des_data, ra = ra, ra2=ra2, dec= dec, dec2= dec2  )
    des = Cuts.doBasicCuts(des_data_f)
    """
    balrog_o = io.LoadBalrog(user = 'EMHUFF', truth = None)
    balrog = Cuts.SpatialCuts(balrog_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    balrog = Cuts.doBasicCuts(balrog, balrog=True)
    balrogname = list( balrog.dtype.names)
    balrogname[0], balrogname[2] = 'RA', 'DEC'
    balrog.dtype.names = tuple(balrogname)
    balrog = AddingReddening(balrog)
    """
    
    des_data_test = Cuts.SpatialCuts(full_des_data, ra = ra + 10, ra2=ra2+10, dec= dec, dec2= dec2  )
    des_data_test = Cuts.doBasicCuts(des_data_test)

    y1a1 = io.getDESY1A1catalogs(keyword = 'Y1A1_COADD_OBJECTS_000001', size = None)
    des_y1a1 = Cuts.doBasicCuts(y1a1)
    
    
    #split test and train sample
    (trainInd, testInd), (_, _) = split_samples(des, des, [0.8,0.2], random_state=0)
    
    des_train = des[trainInd]
    des_test = des[testInd]
    
    # putting sample in classifier
    trainC, trainL, testC, testL = XDGMM_model(clean_cmass_data, clean_lowz_data, train=des_train, test=des_test )
    
    
    # completeness and purity
    commonC, _ = DES_to_SDSS.match( testC, clean_cmass_data )
    commonL, _ = DES_to_SDSS.match( testL, clean_lowz_data )
    trueC, _ = DES_to_SDSS.match( des_test, clean_cmass_data )
    trueL, _ = DES_to_SDSS.match( des_test, clean_lowz_data )
    
    completeness_cmass = len(commonC) * 1./ len(trueC)
    purity_cmass = len(commonC) * 1./ len(testC)
    completeness_lowz = len(commonL) * 1./ len(trueL)
    purity_lowz = len(commonL) * 1./ len(testL)
    
    print 'com/purity(cmass)', completeness_cmass, purity_cmass
    print 'com/purity(lowz)', completeness_lowz, purity_lowz
    
    

    stop
    # -------------------------------------
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




    # purity vs p_thresh check -----------------------------------
    """
    p = np.linspace(0.0, 0.1, 21)
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (14,7))
    coms = []
    purs = []
    coms2 = []
    purs2 = []

    pps = []
    for pp in p:
    	com, pur, com2, pur2 = XDGMM_model(des, clean_cmass_data, clean_lowz_data, p_threshold = pp )
        coms.append(com)
        purs.append(pur)
        coms2.append(com2)
        purs2.append(pur2)
        
        pps.append(pp)
        print pp, com, pur
    
    ax.plot( pps, coms, 'r.', label = 'completeness')
    ax.plot( pps, purs, 'b.', label = 'purity')
    ax.set_title('CMASS')
    ax.set_xlabel('p_threshold')
    ax2.plot( pps, coms, 'r.', label = 'completeness')
    ax2.plot( pps, purs, 'b.', label = 'purity')
    ax2.set_title('LOWZ')
    ax2.set_xlabel('p_threshold')
    """
    
    
    


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








