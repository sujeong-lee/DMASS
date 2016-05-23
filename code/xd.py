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
    #fig.savefig('../figure/spatialtest')
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


def XDGMM_model(des, cmass, lowz, p_threshold = 0.5, balrog = None, matchSDSS = None):
    import esutil
    des, _ = priorCut(des)   

    # making cmass and lowz mask
    h = esutil.htm.HTM(10)
    matchDist = 2/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(des['RA'],des['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass = np.zeros( des.size, dtype = int)
    true_cmass[m_des] = 1
    cmass_mask = true_cmass == 1

    m_des, m_sdss, d12 = h.match(des['RA'], des['DEC'], lowz['RA'],lowz['DEC'],matchDist,maxmatch=1)
    true_lowz = np.zeros( des.size, dtype = int)
    true_lowz[m_des] = 1
    lowz_mask = true_lowz == 1
    
    print 'num of cmass/lowz', np.sum(cmass_mask), np.sum(lowz_mask)

    # stack DES data
    des_g = des['MAG_DETMODEL_G'] - des['XCORR_SFD98_G']
    des_r = des['MAG_DETMODEL_R'] - des['XCORR_SFD98_R']
    des_i = des['MAG_DETMODEL_I'] - des['XCORR_SFD98_I']
    des_z = des['MAG_DETMODEL_Z'] - des['XCORR_SFD98_Z']
    des_ci = des['MAG_MODEL_I'] - des['XCORR_SFD98_I']
    des_cr = des['MAG_MODEL_R'] - des['XCORR_SFD98_R']

    X = np.vstack([des_cr, des_ci, des_g, des_r, des_i, des_z]).T
    Xerr = np.vstack([ des['MAGERR_MODEL_R'] , des['MAGERR_MODEL_I'], des['MAGERR_DETMODEL_G'], des['MAGERR_DETMODEL_R'],des['MAGERR_DETMODEL_I'],des['MAGERR_DETMODEL_Z']]).T

    
    # mixing matrix
    W = np.array([[1, 0, 0, 0, 0, 0],    # r cmag
                  [0, 1, 0, 0, 0, 0],    # i cmagnitude
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],
                  [0, 0, 0, 0, 1, -1]])

    X = np.dot(X, W.T)
    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    
    # each covariance C = WCW^T
    # best way to do this is with a tensor dot-product
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))

    y = np.zeros( ( true_cmass.size, 2 ), dtype=int)
    # ycov = np.zeros(sdss.size, dtype=int)
    y[:,0][cmass_mask] = 1
    y[:,1][lowz_mask] = 1

    # mask for train/test sample
    (X_train_ind, X_test_ind), (y_train_ind, y_test_ind) = split_samples(X, y, [0.7,0.3],
                                                     random_state=0)
    X_train = X[X_train_ind]
    y_train = y[y_train_ind]
    X_test = X[X_test_ind]
    y_test = y[y_test_ind]


    m_des, m_sdss, d12 = h.match(des[X_train_ind]['RA'],des[X_train_ind]['DEC'], cmass['RA'], cmass['DEC'],matchDist,maxmatch=1)
    true_cmass_train = np.zeros( des[X_train_ind].size, dtype = int)
    true_cmass_train[m_des] = 1
    cmass_mask_train = true_cmass_train == 1

    m_des, m_sdss, d12 = h.match(des[X_train_ind]['RA'], des[X_train_ind]['DEC'], lowz['RA'],lowz['DEC'],matchDist,maxmatch=1)
    true_lowz_train = np.zeros( des[X_train_ind].size, dtype = int)
    true_lowz_train[m_des] = 1
    lowz_mask_train = true_lowz_train == 1


    print 'train/test', len(X_train), len( X_test ) 
    # train sample XD convolution
  
    # cmass extreme deconvolution compute
    @pickle_results("XD_dmass_train_all.kl")
    def compute_XD(n_clusters=20, n_iter=50, verbose=True):
        clf_cmass= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf_cmass.fit(X[X_train_ind][cmass_mask_train], Xcov[X_train_ind][cmass_mask_train])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf_cmass

    clf_cmass = compute_XD()
    X_sample_cmass = clf_cmass.sample(100 * X.shape[0])

    @pickle_results("XD_lowz_train20.pkl")
    def compute_XD(n_clusters=20, n_iter=500, verbose=True):
        clf_lowz = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf_lowz.fit(X[X_train_ind][lowz_mask_train], Xcov[X_train_ind][lowz_mask_train])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf_lowz

    clf_lowz = compute_XD()
    X_sample_lowz = clf_lowz.sample(100 * X.shape[0])


    # 3d number density histogram
    bin0, step0 = np.linspace(X[:,0].min(), X[:,0].max(), 301, retstep=True) # cmodel r
    bin1, step1 = np.linspace(X[:,1].min(), X[:,1].max(), 301, retstep=True) # cmodel i
    bin2, step2 = np.linspace(X[:,2].min(), X[:,2].max(), 501, retstep=True) # gr
    bin3, step3 = np.linspace(X[:,3].min(), X[:,3].max(), 201, retstep=True) # ri
    bin0 = np.append( bin0, bin0[-1]+step0)
    bin1 = np.append( bin1, bin1[-1]+step1)
    bin2 = np.append( bin2, bin2[-1]+step2)
    bin3 = np.append( bin3, bin3[-1]+step3)

    
    # cmass histogram model probability
    N_XDProb, edges = np.histogramdd(X_sample_cmass[:,1:4], bins = [bin1, bin2, bin3])
    n_CMASS = N_XDProb * 1./np.sum( N_XDProb )

    X_sample_no = X[~cmass_mask].copy()
    N_XDProb, edges = np.histogramdd(X_sample_no[:,1:4], bins = [bin1, bin2, bin3])
    n_noCMASS = N_XDProb * 1./np.sum( N_XDProb )

    numerator =  n_CMASS * np.sum(y_train[:,0]) * 1.
    denominator =  (n_CMASS * np.sum(y_train[:,0]) + n_noCMASS * (X_train.size - np.sum(y_train[:,0]) ))

    denominator_zero = denominator == 0
    modelProb_CMASS = np.zeros( numerator.shape )
    modelProb_CMASS[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]  # 3D model probability distriution
    
    # loz histogram model probability
    N_XDProb, edges = np.histogramdd(X_sample_lowz[:,(0,2,3)], bins = [bin0, bin2, bin3])
    n_LOWZ = N_XDProb * 1./np.sum( N_XDProb )

    X_sample_no = X[~lowz_mask].copy()
    N_XDProb, edges = np.histogramdd(X_sample_no[:,(0,2,3)], bins = [bin0, bin2, bin3])
    n_noLOWZ = N_XDProb * 1./np.sum( N_XDProb )    

    numerator =  n_LOWZ * np.sum(y_train[:,1]) * 1.
    denominator =  (n_LOWZ * np.sum(y_train[:,1]) + n_noLOWZ * (X_train.size - np.sum(y_train[:,1]) ))

    denominator_zero = denominator == 0
    modelProb_LOWZ = np.zeros( numerator.shape )
    modelProb_LOWZ[~denominator_zero] = numerator[~denominator_zero]/denominator[~denominator_zero]

    # passing test sample to model probability
    inds0 = np.digitize( X_test[:,0], bin0 )-1
    inds1 = np.digitize( X_test[:,1], bin1 )-1
    inds2 = np.digitize( X_test[:,2], bin2 )-1
    inds3 = np.digitize( X_test[:,3], bin3 )-1
    inds_CMASS = np.vstack([inds1,inds2, inds3]).T
    inds_LOWZ = np.vstack([inds0, inds2, inds3]).T
    inds_CMASS = [ tuple(ind) for ind in inds_CMASS ]    
    inds_LOWZ = [ tuple(ind) for ind in inds_LOWZ ]
    EachProb_CMASS = np.array([modelProb_CMASS[ind] for ind in inds_CMASS ]) # Assign probability to each galaxies
    EachProb_LOWZ = np.array([modelProb_LOWZ[ind] for ind in inds_LOWZ ])

    # Getting galaxies higher than threshold
    GetCMASS_mask = EachProb_CMASS > p_threshold
    X_pred_cmass = X_test[GetCMASS_mask]
    X_cont_cmass = X_test[GetCMASS_mask * np.logical_not(y_test[:,0])]
    GetLOWZ_mask = EachProb_LOWZ > p_threshold
    X_pred_lowz = X_test[GetLOWZ_mask]
    X_cont_lowz = X_test[GetLOWZ_mask * np.logical_not(y_test[:,1])]



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
    p = np.linspace(0.0, 0.05, 100)
    fig, (ax, ax2) = plt.subplots(1,2, figsize=(14,7))
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
    
    ax.plot( pps, coms, 'r.', label = 'completeness')
    ax.plot( pps, purs, 'b.', label = 'purity')
    ax.set_title('CMASS')
    ax.set_xlabel('p_threshold')
    ax2.plot( pps, coms, 'r.', label = 'completeness')
    ax2.plot( pps, purs, 'b.', label = 'purity')
    ax2.set_title('LOWZ')
    ax2.set_xlabel('p_threshold')
    ax.legend(loc = 'best')
    ax.set_ylim(0.5, 1.1)
    ax2.legend(loc = 'best')
    ax2.set_ylim(0.5, 1.1)
    
    fig.savefig('com_pur')
    
    return pps, coms, purs, coms2, purs2
    
    """
    
    
    # for non-matched DES sample --------------------------------------
    if balrog is None:

        inds0 = np.digitize( X[:,0], bin0 )-1
        inds1 = np.digitize( X[:,1], bin1 )-1
        inds2 = np.digitize( X[:,2], bin2 )-1
        inds3 = np.digitize( X[:,3], bin3 )-1

        inds_CMASS = np.vstack([inds1,inds2, inds3]).T
        inds_LOWZ = np.vstack([inds0, inds2, inds3]).T

        inds_CMASS = [ tuple(ind) for ind in inds_CMASS ]
        inds_LOWZ = [ tuple(ind) for ind in inds_LOWZ ]
        EachProb_CMASS = np.array([modelProb_CMASS[ind] for ind in inds_CMASS ]) # Assign probability to each galaxies
        EachProb_LOWZ = np.array([modelProb_LOWZ[ind] for ind in inds_LOWZ ])

        # Getting galaxies higher than threshold
        GetCMASS_mask_total = EachProb_CMASS > p_threshold
        X_pred_cmass_total = X[GetCMASS_mask_total]    
        GetLOWZ_mask_total = EachProb_LOWZ > p_threshold
        X_pred_lowz_total = X[GetLOWZ_mask_total]

        print 'total dmass/dlowz', np.sum(GetCMASS_mask_total), np.sum(GetLOWZ_mask_total)


        return des[X_test_ind][GetCMASS_mask], des[X_test_ind][GetLOWZ_mask], des[GetCMASS_mask_total], des[GetLOWZ_mask_total]



        # Histogram and EXD comparison  ---------------------------------------------
        """
        XX = X_pred_cmass.copy()
        N_DMASS, edges = np.histogramdd(XX[:,1:4], bins = [bin1, bin2, bin3])
        N_DMASS1 = np.sum(np.sum(N_DMASS, axis = 1), axis = 1)
        N_DMASS2 = np.sum(np.sum(N_DMASS, axis = 0), axis = 1)
        N_DMASS3 = np.sum(np.sum(N_DMASS, axis = 0), axis = 0)
        
        XX = X_sample_cmass.copy()
        N_CMASS, edges = np.histogramdd(XX[:,1:4], bins = [bin1, bin2, bin3])
        N_CMASS1 = np.sum(np.sum(N_CMASS, axis = 1), axis = 1)
        N_CMASS2 = np.sum(np.sum(N_CMASS, axis = 0), axis = 1)
        N_CMASS3 = np.sum(np.sum(N_CMASS, axis = 0), axis = 0)
        
        fig, (ax, ax2, ax3) = plt.subplots(1,3)
        #ax.plot( bin1[:-1], N_BMASS1 * 1./np.sum(N_BMASS1), 'r.', label = 'BMASS cmodel i')
        ax.plot( bin1[:-1], N_DMASS1 * 1./np.sum(N_DMASS1), 'b.', label = 'DMASS magmodel i')
        ax.plot( bin1[:-1], N_CMASS1 * 1./np.sum(N_CMASS1), 'g.', label = 'CMASS cmodel i')
        #ax2.plot( bin2[:-1], N_BMASS2 * 1./np.sum(N_BMASS2), 'r.', label = 'BMASS g-r')
        ax2.plot( bin2[:-1], N_DMASS2 * 1./np.sum(N_DMASS2), 'b.', label = 'DMASS g-r')
        ax2.plot( bin2[:-1], N_CMASS2 * 1./np.sum(N_CMASS2), 'g.', label = 'CMASS g-r')
        #ax3.plot( bin3[:-1], N_BMASS3 * 1./np.sum(N_BMASS3), 'r.', label = 'BMASS r-i')
        ax3.plot( bin3[:-1], N_DMASS3 * 1./np.sum(N_DMASS3), 'b.', label = 'DMASS r-i')
        ax3.plot( bin3[:-1], N_CMASS3 * 1./np.sum(N_CMASS3), 'g.', label = 'CMASS r-i')
        
        ax.set_xlabel('mag_i')
        ax2.set_xlabel('g-r')
        ax3.set_xlabel('r-i')
        ax.legend()
        ax2.legend()
        ax3.legend()
        fig.savefig('CD_histogram')
        """
        


        # --------------------------------------
        # testing in histogram
        probable_cmass = X[GetCMASS_mask_total]
        contaminant_cmass = X[~GetCMASS_mask_total] 
        print 'prob/cont', np.sum(GetCMASS_mask_total), np.sum(~GetCMASS_mask_total)
        """
        bin0, step0 = np.linspace(X[:,0].min(), X[:,0].max(), 101, retstep=True) # cmodel r
        bin1, step1 = np.linspace(X[:,1].min(), X[:,1].max(), 101, retstep=True) # cmodel i
        bin2, step2 = np.linspace(X[:,2].min(), X[:,2].max(), 51, retstep=True) # gr
        bin3, step3 = np.linspace(X[:,3].min(), X[:,3].max(), 51, retstep=True) # ri
        N_BMASS, edges = np.histogramdd(X[:,1:4], bins = [bin1, bin2, bin3])

        N_probable, edges = np.histogramdd(probable_cmass[:,1:4], bins = [bin1, bin2, bin3])
        N_cont, edges = np.histogramdd(contaminant_cmass[:,1:4], bins = [bin1, bin2, bin3])
        """
        """
        N_probable1 = np.sum(np.sum(N_probable, axis = 1), axis = 1)
        N_probable2 = np.sum(np.sum(N_probable, axis = 0), axis = 1)
        N_probable3 = np.sum(np.sum(N_probable, axis = 0), axis = 0)
        N_cont1 = np.sum(np.sum(N_cont, axis = 1), axis = 1)
        N_cont2 = np.sum(np.sum(N_cont, axis = 0), axis = 1)
        N_cont3 = np.sum(np.sum(N_cont, axis = 0), axis = 0)
        
        fig, (ax, ax2, ax3) = plt.subplots(1,3)
        ax.plot( bin1[:-1], N_probable1 * 1./np.sum(N_probable1), 'r.', label = 'probable cmodel i')
        ax.plot( bin1[:-1], N_cont1 * 1./np.sum(N_cont1), 'b.', label = 'cont cmodel i')
        ax2.plot( bin2[:-1], N_probable2 * 1./np.sum(N_probable2), 'r.', label = 'probable g-r')
        ax2.plot( bin2[:-1], N_cont2 * 1./np.sum(N_cont2), 'b.', label = 'cont g-r')
        ax3.plot( bin3[:-1], N_probable3 * 1./np.sum(N_probable3), 'r.', label = 'probable r-i')
        ax3.plot( bin3[:-1], N_cont3 * 1./np.sum(N_cont3), 'b.', label = 'cont r-i')
        ax.legend()
        ax2.legend()
        ax3.legend()
        """
        
        """
        fig, ax = plt.subplots(5, 2, figsize = (10, 25))
        ax = ax.ravel()
        for i in range(5):
            ax[2 * i].imshow( np.log(N_probable[55 + 3 * (i+1),:,:]))
            ax[2 * i + 1].imshow( np.log(N_cont[55 + 3 * (i+1),:,:]))

        fig.savefig('hist_ci.png')
        """

        """
        #bin_i = np.linspace(19.0, 20.0, 311)
        #bin_i, step1 = np.linspace(X[:,1].min(), X[:,1].max(), 501, retstep=True) # cmodel i
        fig2, ax2 = plt.subplots(2,5, figsize=(25, 10))
        ax2 = ax2.ravel()
        matrix1, matrix2 = np.mgrid[0:N_XDProb[0,:,0].size ,0:N_XDProb[0,0,:].size]
        
        bin2grid = bin2[matrix1].ravel()
        bin3grid = bin3[matrix2].ravel()
        
        for i in range(10):
            mask = ( X[:,1] > bin1[150 + 10*i]) & (X[:,1] < bin1[150 + 10*i+1])
            #im = ax2[i].imshow( np.log10(modelProb_CMASS[:,250+10*i,:]) )
            probable_cmass_i = X[(GetCMASS_mask_total) & mask]
            cont_cmass_i = X[(~GetCMASS_mask_total) & mask]
            contourplane_i = N_XDProb[150 + 10*i,:,:].ravel()
            #ax2[i].imshow( N_XDProb[:,250 + 20*i,:],  )
            #pcm = ax2[i].pcolormesh(bin2grid, bin3grid, N_XDProb[:,250 + 10*i,:],cmap='RdBu_r')
            pcm = ax2[i].pcolormesh(bin2[matrix1], bin3[matrix2], N_XDProb[150 + 10*i,:,:],cmap='PuBu_r')
            fig2.colorbar(pcm, ax=ax2[i],extend='max')
            #ax2[i].scatter( bin2grid, bin3grid, c=contourplane_i, s=30, alpha = 0.8, edgecolors='none' )
            ax2[i].scatter( cont_cmass_i[:,2], cont_cmass_i[:,3], color = 'green', s = 2, edgecolors='none', label='contaminant')
            ax2[i].scatter( probable_cmass_i[:,2], probable_cmass_i[:,3],  color = 'red', s = 2, edgecolors='none', label='probable')
            ax2[i].legend()
            ax2[i].set_xlim(0,3)
            ax2[i].set_ylim(0,2)
        
        

        
        # sdss dperp ---------------------
        _, cmassInds, _ = h.match(des[GetCMASS_mask_total]['RA'],des[GetCMASS_mask_total]['DEC'], matchSDSS['RA'], matchSDSS['DEC'],matchDist,maxmatch=1)
        _, contInds, _ = h.match(des[~GetCMASS_mask_total]['RA'],des[~GetCMASS_mask_total]['DEC'], matchSDSS['RA'], matchSDSS['DEC'],matchDist,maxmatch=1)


        #cmass_SDSS, _ = DES_to_SDSS.match(matchSDSS, des[GetCMASS_mask_total])
        #cont_SDSS, _ = DES_to_SDSS.match(matchSDSS, des[~GetCMASS_mask_total])
        print 'dmass/dmass_sdss', np.sum(GetCMASS_mask_total), len(cmassInds)
        print 'cont/cont_sdss', np.sum(~GetCMASS_mask_total), len(contInds)


        modelmag_g = matchSDSS['MODELMAG_G'] - matchSDSS['EXTINCTION_G']
        modelmag_r = matchSDSS['MODELMAG_R'] - matchSDSS['EXTINCTION_G']
        modelmag_i = matchSDSS['MODELMAG_I'] - matchSDSS['EXTINCTION_G']
        modelmag_z = matchSDSS['MODELMAG_Z'] - matchSDSS['EXTINCTION_G']
        cmodelmag_i = matchSDSS['CMODELMAG_I'] - matchSDSS['EXTINCTION_I']
        dperp_sdss = (modelmag_r - modelmag_i) - (modelmag_g - modelmag_r)/8.0
        
        dperp_cmass = dperp_sdss[cmassInds]
        dperp_cont = dperp_sdss[contInds]
        ci_cmass = cmodelmag_i[cmassInds]
        ci_cont = cmodelmag_i[contInds]

        dperp_bin = np.linspace(0.45, 1.5, 50)
        #bin_i, _ = np.linspace(18, 20, 91, retstep=True)
        bin_i, _ = np.linspace(18, 20, 51, retstep=True)
        
        N_cmass, edges = np.histogramdd([dperp_cmass, ci_cmass], bins = [dperp_bin, bin_i])
        N_cont, edges = np.histogramdd([dperp_cont, ci_cont], bins = [dperp_bin, bin_i])
        
        return dperp_bin, bin_i, N_cmass, N_cont
        
        dperp_bin, db = np.linspace(0.45, 1.5, 50, retstep = True)

        fig, ax = plt.subplots(5,5, figsize=(25, 25))
        ax = ax.ravel()
        for i in range(20):
            #i_cmass_mask = ( ci_cmass > bin_i[10 + 2*i]) & ( ci_cmass < bin_i[10 + 2*i+1])
            #i_cont_mask = ( ci_cont > bin_i[10 + 2*i]) & ( ci_cont < bin_i[10 + 2*i+1])
            N_dperp_cmass_i = N_cmass[:,10+2*i:10+2*i+1]
            N_dperp_cont_i = N_cont[:,10+2*i:10+2*i+1]
            print N_dperp_cmass_i.shape
            ax[i].bar( dperp_bin[:-1], N_dperp_cont_i, color = 'green', alpha=1.0, width = db, label='contaminant' )
            ax[i].bar( dperp_bin[:-1], N_dperp_cmass_i, color = 'red', alpha=0.5, width = db, label = 'dmass' )
            #ax[i].bar( dperp_bin[:-1], N_dperp_cmass_i * 1./np.sum(N_dperp_cmass_i), color = 'green', alpha=1.0, width = db )
            #ax[i].bar( dperp_bin[:-1], N_dperp_cont_i * 1./np.sum(N_dperp_cont_i), color = 'red', alpha=0.5, width = db )
            ax[i].text(0.95, 0.95, 'cmodel_i={:>0.2f}'.format(bin_i[10+2*i]) ,
                    verticalalignment='top', horizontalalignment='right',
                    transform= ax[i].transAxes,
                    color='black', fontsize=10)
        
            ax[i].axvline(x=0.55, ymin=0, ymax=100, linestyle = '--', color='grey')
        ax[0].legend()
        fig.savefig('hist.png')
        
        
        fig2, ax2 = plt.subplots()
        N_dperp_cont_proj = np.sum( N_cont, axis = 1 )
        N_dperp_cmass_proj = np.sum(N_cmass, axis = 1)
        print N_dperp_cmass_proj.shape
        ax2.bar( dperp_bin[:-1], N_dperp_cont_proj, color = 'green', alpha=1.0, width = db, label='contaminant' )
        ax2.bar( dperp_bin[:-1], N_dperp_cmass_proj, color = 'red', alpha=0.5, width = db, label = 'dmass' )
        ax2.legend()
        ax2.axvline(x=0.55, ymin=0, ymax=100, linestyle = '--', color='grey')
        fig2.savefig('hist_proj.png')



        #fig, (ax, ax2) = plt.subplots(1,2)
        #2dN_probable = N_probable[250, :, :]
        #2dN_cont = N_cont[250, :, :]

            #im = ax.imshow( np.log(2dN_probable) ) 
            #im2 = ax2.imshow( np.log( 2dN_cont ) )
        #fig.colorbar(im, ax = ax)
        #fig.colorbar(im2, ax = ax2 )

        # ax.plot(X_pred_cmass_total[:,2], X_pred_cmass_total[:,3], 'g.', alpha = 0.5 )
        #ax2.plot( X_pred_cmass_total[:,1], X_pred_cmass_total[:,3] - X_pred_cmass_total[:,2]/8.0, 'g.', alpha = 0.3)
         #ax.plot( X[cmass_mask][:,2], X[cmass_mask][:,3], 'r.', alpha = 0.3)
         #ax2.plot( X[cmass_mask][:,1], X[cmass_mask][:,3] - X[cmass_mask][:,2]/8.0, 'r.', alpha = 0.3)
        """
        return des[X_test_ind][GetCMASS_mask], des[X_test_ind][GetLOWZ_mask], des[GetCMASS_mask_total], des[GetLOWZ_mask_total]



    else:
        balrog, _ = priorCut(balrog)
        # stack balrog data
        balrog_g = balrog['MAG_DETMODEL_G'] - balrog['XCORR_SFD98_G']
        balrog_r = balrog['MAG_DETMODEL_R'] - balrog['XCORR_SFD98_R']
        balrog_i = balrog['MAG_DETMODEL_I'] - balrog['XCORR_SFD98_I']
        balrog_z = balrog['MAG_DETMODEL_Z'] - balrog['XCORR_SFD98_Z']
        balrog_ci = balrog['MAG_MODEL_I'] - balrog['XCORR_SFD98_I']
        balrog_cr = balrog['MAG_MODEL_R'] - balrog['XCORR_SFD98_R']

    	X = np.vstack([balrog_cr, balrog_ci, balrog_g, balrog_r, balrog_i, balrog_z]).T

    	# mixing matrix
    	W = np.array([[1, 0, 0, 0, 0, 0],    # r cmag
                  [0, 1, 0, 0, 0, 0],    # i cmagnitude
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],
                  [0, 0, 0, 0, 1, -1]])

        X_balrog = np.dot(X, W.T)
	
    	inds0 = np.digitize( X_balrog[:,0], bin0 )-1
    	inds1 = np.digitize( X_balrog[:,1], bin1 )-1
    	inds2 = np.digitize( X_balrog[:,2], bin2 )-1
        inds3 = np.digitize( X_balrog[:,3], bin3 )-1

        # masking outlier indices
        inds0_mask = inds0 < (bin0.size -1)	
        inds1_mask = inds1 < (bin1.size -1)
        inds2_mask = inds2 < (bin2.size -1)
        inds3_mask = inds3 < (bin3.size -1)
        inds_mask = inds0_mask * inds1_mask * inds2_mask * inds3_mask
        inds0, inds1, inds2, inds3 = inds0[inds_mask], inds1[inds_mask], inds2[inds_mask], inds3[inds_mask]

    	inds_CMASS = np.vstack([inds1,inds2, inds3]).T
    	inds_LOWZ = np.vstack([inds0, inds2, inds3]).T
    	inds_CMASS = [ tuple(ind) for ind in inds_CMASS ]
    	inds_LOWZ = [ tuple(ind) for ind in inds_LOWZ ]
        
    	EachProb_CMASS = np.array([modelProb_CMASS[ind] for ind in inds_CMASS ]) # Assign probability to each galaxies
    	EachProb_LOWZ = np.array([modelProb_LOWZ[ind] for ind in inds_LOWZ ])

        EachProb_CMASS2 = np.zeros( inds_mask.size )
        EachProb_LOWZ2 = np.zeros( inds_mask.size )

        EachProb_CMASS2[inds_mask] = EachProb_CMASS
        EachProb_LOWZ2[inds_mask] = EachProb_LOWZ

    	# Getting galaxies higher than threshold
    	GetCMASS_mask_balrog = EachProb_CMASS2 > p_threshold
    	X_pred_cmass_balrog = X_balrog[GetCMASS_mask_balrog]
    	GetLOWZ_mask_balrog = EachProb_LOWZ2 > p_threshold
    	X_pred_lowz_balrog = X_balrog[GetLOWZ_mask_balrog]


        # Balrog histogram test -----------------------------------------
        """
        X = X_pred_cmass_balrog.copy()
        bin0, step0 = np.linspace(X[:,0].min(), X[:,0].max(), 301, retstep=True) # cmodel r
        bin1, step1 = np.linspace(X[:,1].min(), X[:,1].max(), 301, retstep=True) # cmodel i
        bin2, step2 = np.linspace(X[:,2].min(), X[:,2].max(), 501, retstep=True) # gr
        bin3, step3 = np.linspace(X[:,3].min(), X[:,3].max(), 201, retstep=True) # ri
        N_BMASS, edges = np.histogramdd(X[:,1:4], bins = [bin1, bin2, bin3])
        N_BMASS1 = np.sum(np.sum(N_BMASS, axis = 1), axis = 1)
        N_BMASS2 = np.sum(np.sum(N_BMASS, axis = 0), axis = 1)
        N_BMASS3 = np.sum(np.sum(N_BMASS, axis = 0), axis = 0)

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
        
        
        fig, (ax, ax2, ax3) = plt.subplots(1,3)
        ax.plot( bin1[:-1], N_BMASS1 * 1./np.sum(N_BMASS1), 'r.', label = 'BMASS cmodel i')
        ax.plot( bin1[:-1], N_DMASS1 * 1./np.sum(N_DMASS1), 'b.', label = 'DMASS cmodel i')
        ax.plot( bin1[:-1], N_CMASS1 * 1./np.sum(N_CMASS1), 'g.', label = 'CMASS cmodel i')
        ax2.plot( bin2[:-1], N_BMASS2 * 1./np.sum(N_BMASS2), 'r.', label = 'BMASS g-r')
        ax2.plot( bin2[:-1], N_DMASS2 * 1./np.sum(N_DMASS2), 'b.', label = 'DMASS g-r')
        ax2.plot( bin2[:-1], N_CMASS2 * 1./np.sum(N_CMASS2), 'g.', label = 'CMASS g-r')
        ax3.plot( bin3[:-1], N_BMASS3 * 1./np.sum(N_BMASS3), 'r.', label = 'BMASS r-i')
        ax3.plot( bin3[:-1], N_DMASS3 * 1./np.sum(N_DMASS3), 'b.', label = 'DMASS r-i')
        ax3.plot( bin3[:-1], N_CMASS3 * 1./np.sum(N_CMASS3), 'g.', label = 'CMASS r-i')
        ax.legend()
        ax2.legend()
        ax3.legend()
        """

        
    	print 'balrog dmass/dlowz', np.sum(GetCMASS_mask_balrog), np.sum(GetLOWZ_mask_balrog)
        return balrog[ GetCMASS_mask_balrog], balrog[ GetLOWZ_mask_balrog]

  

def main():
    # load dataset

    dec = -1.0
    dec2 = 1.0
    ra = 320.0
    ra2 = 360.

    #cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj.fits.gz')
    cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    cmass_data = Cuts.SpatialCuts(cmass_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    clean_cmass_data = Cuts.keepGoodRegion(cmass_data)

    # cmass_data = cmass_data[ (cmass_data['Z']>0.43) & (cmass_data['Z']<0.7)]
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

    balrog_o = io.LoadBalrog(user = 'EMHUFF', truth = None)
    balrog = Cuts.SpatialCuts(balrog_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    balrog = Cuts.doBasicCuts(balrog, balrog=True)
    balrogname = list( balrog.dtype.names)
    balrogname[0], balrogname[1] = 'RA', 'DEC' 
    balrog.dtype.names = tuple(balrogname)
    balrog = AddingReddening(balrog)

    #y1a1 = io.getDESY1A1catalogs(keyword = 'Y1A1', size = 1000)
    #des_y1a1 = Cuts.doBasicCuts(y1a1)


    """
    h = esutil.htm.HTM(10)
    matchDist = 2/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(full_des_data['RA'],full_des_data['DEC'], clean_cmass_data['RA'], clean_cmass_data['DEC'],matchDist,maxmatch=1)   
    m_des, m_sdss, d12 = h.match(des_data_f['RA'],des_data_f['DEC'], clean_cmass_data['RA'], clean_cmass_data['DEC'],matchDist,maxmatch=1)
    common_mask = np.zeros(clean_cmass_data.size, dtype=int)
    common_mask[m_sdss] = 1
    excluded_mask= common_mask == 0
    
    excluded_cmass = clean_cmass_data[excluded_mask]
    DAT = np.vstack( [ excluded_cmass['RA'], excluded_cmass['DEC']]  ).T
    np.savetxt('excluded_cmass.txt', DAT,  fmt='%.18e', delimiter=' ', header = '# ra, dec' )
    """


    # extreme deconv classifier
    DMASS,  LOWZ, DMASSALL, LOWZALL = XDGMM_model(des, clean_cmass_data, clean_lowz_data )
    Balrog_DMASS, Balrog_LOWZ = XDGMM_model(des, clean_cmass_data, clean_lowz_data, balrog = balrog)
    DMASS_y1a1, LOWZ_y1a1 = XDGMM_model(des, clean_cmass_data, clean_lowz_data, balrog = des_y1a1)
    # balrog, des histogram comparison
    
    
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





