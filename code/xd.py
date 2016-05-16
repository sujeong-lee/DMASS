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
    
    star = (((sdss['PSFMAG_I'] - sdss['EXPMAG_I']) > (0.2 + 0.2*(20.0 - sdss['EXPMAG_I']))) &
         ((sdss['PSFMAG_Z'] - sdss['EXPMAG_Z']) > (9.125 - 0.46 * sdss['EXPMAG_Z'])))
    
    return sdss[priorCut], priorCut # &star #sdss[cmass]

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


def priorCut(data, sdss=None):
    
    print "Calculating/applying CMASS object cuts."
    
    modelmag_g_des = data['MAG_DETMODEL_G'] - data['XCORR_SFD98_G']
    modelmag_r_des = data['MAG_DETMODEL_R'] - data['XCORR_SFD98_R']
    modelmag_i_des = data['MAG_DETMODEL_I'] - data['XCORR_SFD98_I']
    cmodelmag_i_des = data['MAG_MODEL_I'] - data['XCORR_SFD98_I']
    fib2mag_des = data['MAG_APER_4_I']
    
    """
    quality cut
    16<i<22
    0<g-r<3
    0<r-i<1.5
    i_fib < 21.5
    """
    # dperp_des = ( modelmag_r_des - modelmag_i_des) - (modelmag_g_des - modelmag_r_des)/8.0
    
    cut = ((cmodelmag_i_des > 15) &
           (cmodelmag_i_des < 22.) &
           ((modelmag_r_des - modelmag_i_des ) < 2.0 ) &
           ((modelmag_g_des - modelmag_r_des ) > -2.0 ) &
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


def XDGMM_model(sdss, des, p_treshold = 0.5):

    # divide data train and test
    des, _ = priorCut(des)
    sdss, des = DES_to_SDSS.match(sdss, des)

    # mask should be applied after matching SDSS and DES catalogs (for star-galaxy separation )
    _, cmass_mask = SDSS_cmass_criteria(sdss)
    _, lowz_mask = SDSS_LOWZ_criteria(sdss)   
 
    Ncmass = len(sdss[cmass_mask])
    Nnon = len(sdss[~cmass_mask])
    print 'cmass/non', Ncmass, Nnon
    print 'lowz/non', len(sdss[lowz_mask]), len(sdss[~lowz_mask])


    # sdss
    """
    # stack data
    sdss_g = sdss['MODELMAG_G'] - sdss['EXTINCTION_G']
    sdss_r = sdss['MODELMAG_R'] - sdss['EXTINCTION_R']
    sdss_i = sdss['MODELMAG_I'] - sdss['EXTINCTION_I']
    sdss_z = sdss['MODELMAG_Z'] - sdss['EXTINCTION_Z']
    sdss_ci = sdss['CMODELMAG_I'] - sdss['EXTINCTION_I']

    X = np.vstack([sdss_ci, sdss_g, sdss_r, sdss_i, sdss_z]).T

    Xerr = np.vstack([ sdss['CMODELMAGERR_I'] , sdss['MODELMAGERR_G'], sdss['MODELMAGERR_R'],sdss['MODELMAGERR_I'],sdss['MODELMAGERR_Z']]).T
    """
    

    
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

    y = np.zeros( ( sdss.size, 2 ), dtype=int)
    # ycov = np.zeros(sdss.size, dtype=int)
    y[:,0][cmass_mask] = 1
    y[:,1][lowz_mask] = 1

    # mask for train/test sample
    (X_train_ind, X_test_ind), (y_train_ind, y_test_ind) = split_samples(X, y, [0.3,0.7],
                                                     random_state=0)
    X_train = X[X_train_ind]
    y_train = y[y_train_ind]
    X_test = X[X_test_ind]
    y_test = y[y_test_ind]

    _, cmass_mask_train = SDSS_cmass_criteria(sdss[X_train_ind])
    _, lowz_mask_train = SDSS_LOWZ_criteria(sdss[X_train_ind])

    print 'train/test', len(X_train),len( X_test )  
    # train sample XD convolution
  
    # cmass extreme deconvolution compute
    @pickle_results("XD_dmass_train30_sg.kl")
    def compute_XD(n_clusters=20, n_iter=500, verbose=True):
        clf_cmass= XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf_cmass.fit(X[X_train_ind][cmass_mask_train], Xcov[X_train_ind][cmass_mask_train])
        #clf.fit(X[cmass_mask], Xcov[cmass_mask])
        return clf_cmass

    clf_cmass = compute_XD()
    X_sample_cmass = clf_cmass.sample(100 * X.shape[0])  

    @pickle_results("XD_lowz_train30_sg.kl")
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


    # cmass histogram
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
    
    # loz histogram
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
    GetCMASS_mask = EachProb_CMASS > p_treshold
    X_pred_cmass = X_test[GetCMASS_mask]
    X_cont_cmass = X_test[GetCMASS_mask * np.logical_not(y_test[:,0])]
    GetLOWZ_mask = EachProb_LOWZ > p_treshold
    X_pred_lowz = X_test[GetLOWZ_mask]
    X_cont_lowz = X_test[GetLOWZ_mask * np.logical_not(y_test[:,1])]



    """

    # plotting dmass in color plane
    fig, (ax, ax2) = plt.subplots(1,2)
    ax.plot( X[cmass_mask][:,1], X[cmass_mask][:,2], 'r.', alpha = 0.3)
    ax2.plot( X[cmass_mask][:,0], X[cmass_mask][:,2] - X[cmass_mask][:,1]/8.0, 'r.', alpha = 0.3)
    ax.plot(X_pred[:,1], X_pred[:,2], 'g.', alpha = 0.5 )
    ax2.plot( X_pred[:,0], X_pred[:,2] - X_pred[:,1]/8.0, 'g.', alpha = 0.3)
    ax.plot(X_cont[:,1], X_cont[:,2], 'k.', alpha = 1.0 )
    ax2.plot( X_cont[:,0], X_cont[:,2] - X_cont[:,1]/8.0, 'k.', alpha = 1.0)
    
    ax.set_xlim(-2,3)
    # 1d histogram
    fig2, (ax, ax2, ax3) = plt.subplots(1,3)
    ax.plot( bin1[:-1], np.sum( np.sum( modelProb_CMASS, axis = 1 ), axis = 1), 'r.')
    ax2.plot( bin3[:-1], np.sum( np.sum( modelProb_CMASS, axis = 1 ), axis = 0), 'r.')
    ax3.plot( bin2[:-1], np.sum( np.sum( modelProb_CMASS, axis = 0), axis = 1), 'r.')
    ax.set_xlabel('i_cmod ')
    ax2.set_xlabel('g-r')
    ax3.set_xlabel('r-i')

    # slicing i axis 

    fig3, a =  plt.subplots(4,2, figsize = (10, 20))
    a = a.ravel()
    for i in range(8):
	im = a[i].imshow( np.log10( modelProb_CMASS[:, :, 60 + 15*i])  )
    	fig.colorbar(im, ax = a[i] )
	a[i].text(0.95, 0.01, 'i={:>0.2f}'.format(bin1[i*10]), verticalalignment='bottom', horizontalalignment='right', transform=a[i].transAxes, fontsize=10)

    """

    # completeness and purity
    completeness_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(y_test[:,0])
    purity_cmass = np.sum( GetCMASS_mask * y_test[:,0] )* 1.0/np.sum(GetCMASS_mask)
    contaminant_cmass = np.sum( GetCMASS_mask * np.logical_not(y_test[:,0]) )

    completeness_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(y_test[:,1])
    purity_lowz = np.sum( GetLOWZ_mask * y_test[:,1] )* 1.0/np.sum(GetLOWZ_mask)
    contaminant_lowz = np.sum( GetLOWZ_mask * np.logical_not(y_test[:,1]) )

    print 'com/purity(cmass)', completeness_cmass, purity_cmass
    print 'com/purity(lowz)', completeness_lowz, purity_lowz
    print 'num of dmass,dlowz', np.sum(GetCMASS_mask), np.sum(GetLOWZ_mask) 
    print 'comtaminat', contaminant_cmass, contaminant_lowz
    return des[X_test_ind][GetCMASS_mask], sdss[X_test_ind][GetCMASS_mask], sdss[X_test_ind][GetCMASS_mask * np.logical_not(y_test[:,0])], des[X_test_ind][GetLOWZ_mask], sdss[X_test_ind][GetLOWZ_mask], sdss[X_test_ind][GetLOWZ_mask * np.logical_not(y_test[:,1])]
    #return 0
  

def main():
    # load dataset

    dec = -1.5
    dec2 = 1.5
    ra = 340.0
    ra2 = 360.0

    cmass_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    cmass_data = Cuts.SpatialCuts(cmass_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    # cmass_data = cmass_data[ (cmass_data['Z']>0.43) & (cmass_data['Z']<0.7)]
    lowz_data_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_LOWZ_South-photoObj.fits.gz')
    lowz_data = Cuts.SpatialCuts(lowz_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )

    sdss_data_o = io.getSDSScatalogs(bigSample = True)
    sdss = Cuts.SpatialCuts(sdss_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    sdss = Cuts.doBasicSDSSCuts(sdss)
    
    full_des_data = io.getDEScatalogs(file = '/n/des/huff.791/Projects/CMASS/Data/DES_Y1_S82.fits')
    des_data_f = Cuts.SpatialCuts(full_des_data, ra = ra, ra2=ra2, dec= dec, dec2= dec2  )
    des_data_f = Cuts.doBasicCuts(des_data_f)

    des_data_im3 = io.getDEScatalogs(file = '/n/des/huff.791/Projects/CMASS/Data/im3shape_s82_for_xcorr.fits', bigSample = True)
    des_im3 = Cuts.SpatialCuts(des_data_im3, ra = ra, ra2=ra2, dec= dec, dec2= dec2  )
    des =  im3shape.im3shape_galprof_mask(des_data_im3, des_data_f)
    # des = im3shape.im3shape_photoz( des_im3, des )
    
    # match DES and SDSS catalogs
    MatchedDES,  MatchedSDSS = DES_to_SDSS.match(des, sdss)

    # expmask = MatchedCmass['IM3_GALPROF'] == 1
    # devmask = MatchedCmass['IM3_GALPROF'] == 2
    # noprofmask = MatchedCmass['IM3_GALPROF'] == 0

    # extreme deconv classifier
    DMASS, DMASS_sdss, contaminant_CMASS, LOWZ, LOWZ_sdss, contaminant_LOWZ = XDGMM_model(MatchedSDSS, MatchedDES)



    # purity vs p_thresh check
    p = np.linspace(0.0, 0.1, 21)
    fig, ax = plt.subplots()
    for pp in p:
    	com, pur = XDGMM_model(sdss, des, p_treshold = pp )
   	ax.plot( pp, com, 'r.')
        ax.plot( pp, pur, 'b.')
        print pp, com, pur

    # different region check
    # S1
    sdss1 = Cuts.SpatialCuts(MatchedCmass_sdss,ra =340, ra2=350 , dec= dec, dec2= dec2 )
    des1 = Cuts.SpatialCuts(MatchedCmass,ra =340, ra2=350 , dec= dec, dec2= dec2 )
    sdss2 = Cuts.SpatialCuts(MatchedCmass_sdss,ra =350, ra2=360 , dec= dec, dec2= dec2 )
    des2 = Cuts.SpatialCuts(MatchedCmass,ra =350, ra2=360 , dec= dec, dec2= dec2 )


    com, pur = XDGMM_model(sdss1, des1)
    com, pur = XDGMM_model(sdss2, des2)



   # testing DLOWZ in SDSS photometry -------------------------
    fig, ax = plt.subplots(1,1, figsize = (7, 7))
    dmass_sdss = LOWZ_sdss.copy()
    contaminant = contaminant_LOWZ.copy()

    ax.plot( lowz_data['MODELMAG'][:,1]-lowz_data['MODELMAG'][:,2] - lowz_data['EXTINCTION'][:,1] + lowz_data['EXTINCTION'][:,2], lowz_data['MODELMAG'][:,2] - lowz_data['MODELMAG'][:,3] + lowz_data['EXTINCTION'][:,3] - lowz_data['EXTINCTION'][:,2], 'g.', label = 'DR11 LOWZ')
    ax.plot( dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'],dmass_sdss['MODELMAG_R'] -  dmass_sdss['MODELMAG_I'] + dmass_sdss['EXTINCTION_I'] - dmass_sdss['EXTINCTION_R'], 'r.', label = 'classifier', alpha = 0.5)
    ax.plot( contaminant['MODELMAG_G']-contaminant['MODELMAG_R'] - contaminant['EXTINCTION_G'] + contaminant['EXTINCTION_R'],contaminant['MODELMAG_R'] -  contaminant['MODELMAG_I'] + contaminant['EXTINCTION_I'] - contaminant['EXTINCTION_R'], 'k.', label = 'contaminant')

    ax.plot( dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'], 0.38 + (dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'])/4.0, 'k--')
    ax.plot( dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'], 0.02 + (dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'])/4.0, 'k--')
    ax.set_xlim(0.5,2.5)
    ax.set_ylim(0.2, 1.0)
    ax.set_xlabel('g - r')
    ax.set_ylabel('r - i')



    # testing DMASS in SDSS photometry ------------------------
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (14, 7))

    dmass_sdss = DMASS_sdss.copy()
    dmass = DMASS.copy()
    
    dperp_sdss = dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_R'] + dmass_sdss['EXTINCTION_I']- dmass_sdss['MODELMAG_I'] - (dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'])/8.0
    dperp_des = dmass['MAG_DETMODEL_R'] -  dmass['XCORR_SFD98_R'] + dmass['XCORR_SFD98_I'] - dmass['MAG_DETMODEL_I'] - (dmass['MAG_DETMODEL_G'] - dmass['MAG_DETMODEL_G'] +  dmass['XCORR_SFD98_R'] + dmass['XCORR_SFD98_R'])/8.0

    dperp_cont = contaminant['MODELMAG_R'] - contaminant['EXTINCTION_R'] + contaminant['EXTINCTION_I']- contaminant['MODELMAG_I'] - (contaminant['MODELMAG_G']-contaminant['MODELMAG_R'] - contaminant['EXTINCTION_G'] + contaminant['EXTINCTION_R'])/8.0


    ax.plot( dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'],dmass_sdss['MODELMAG_R'] -  dmass_sdss['MODELMAG_I'] + dmass_sdss['EXTINCTION_I'] - dmass_sdss['EXTINCTION_R'], 'g.', label = 'classifier')
    ax.plot( contaminant['MODELMAG_G']-contaminant['MODELMAG_R'] - contaminant['EXTINCTION_G'] + contaminant['EXTINCTION_R'],contaminant['MODELMAG_R'] -  contaminant['MODELMAG_I'] + contaminant['EXTINCTION_I'] - contaminant['EXTINCTION_R'], 'k.', label = 'contaminant')
    ax.set_xlim(0,3)
    ax.plot( dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'], (dmass_sdss['MODELMAG_G']-dmass_sdss['MODELMAG_R'] - dmass_sdss['EXTINCTION_G'] + dmass_sdss['EXTINCTION_R'])/8.0 + 0.55, 'k--')
    ax.set_ylim(0.5, 1.6) 
    
    ax2.plot( dmass_sdss['CMODELMAG_I'] - dmass_sdss['EXTINCTION_I'], dperp_sdss, 'g.')
    ax2.plot( contaminant['CMODELMAG_I'] - contaminant['EXTINCTION_I'], dperp_cont, 'k.')
    ax2.plot([19.9, 19.9], [0,2], 'k--')
    ax2.plot([15, 22], [0.55, 0.55], 'k--')
    ax2.plot(dmass_sdss['CMODELMAG_I'] - dmass_sdss['EXTINCTION_I'], (dmass_sdss['CMODELMAG_I'] - dmass_sdss['EXTINCTION_I']- 19.86)/1.6 + 0.8, 'k--')
    ax2.set_xlim(17, 20)
    ax2.set_ylim(0.5, 1.6)
    ax.set_xlabel('g - r')
    ax.set_ylabel('r - i')
    ax2.set_xlabel('cmodel_i')
    ax2.set_ylabel('dperp')
    fig.suptitle('DMASS in SDSS photometry')


    # testing dmass with angular correlation function ------------------------------------------------
    
    cmass_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_South.fits.gz')
    rand_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz' )
    cmass_catS = Cuts.SpatialCuts(cmass_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catS = Cuts.SpatialCuts(rand_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    dmass_cat = Cuts.SpatialCuts(dmass, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catD = Cuts.SpatialCuts(rand_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )

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

    # DAT = np.column_stack(( theta_NGC, w_NGC, jkerr_NGC, theta_SGC, w_SGC, jkerr_SGC, thetaS, wS, Sjkerr  ))
    #np.savetxt( 'cmass_acf.txt', DAT, delimiter = ' ', header = ' theta_NGC  w_NGC  jkerr_NGC  theta_SGC  w_SGC   jkerr_SGC   thetaS  wS  Sjkerr ' )

    fig, ax = plt.subplots(1,1, figsize = (7, 7))

    ax.errorbar( theta_SGC*0.95, w_SGC, yerr = jkerr_SGC, fmt = '.', label = 'SGC')
    ax.errorbar( thetaS* 1.05, wS, yerr = Sjkerr, fmt = '.', label = 'Stripe82')
    ax.errorbar( thetaD, wD, yerr = Djkerr, fmt = '.', label = 'DMASS')

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')



    stop
    # -----------------------------------------------------




    """
    quality cut
    16<i<22
    0<g-r<3
    0<r-i<1.5
    """
    






    # extreme deconvolution test
    sdss = cmass_data.copy()
    # stack data
    sdss_g = sdss['MODELMAG'][:,1] - sdss['EXTINCTION'][:,1]
    sdss_r = sdss['MODELMAG'][:,2] - sdss['EXTINCTION'][:,2]
    sdss_i = sdss['MODELMAG'][:,3] - sdss['EXTINCTION'][:,3]
    sdss_z = sdss['MODELMAG'][:,4] - sdss['EXTINCTION'][:,4]
    sdss_ci = sdss['CMODELMAG'][:,3] - sdss['EXTINCTION'][:,3]

    X = np.vstack([sdss_ci, sdss_g, sdss_r, sdss_i, sdss_z]).T
    Xerr = np.vstack([ sdss['CMODELMAGERR'][:,3] , sdss['MODELMAGERR'][:,1], sdss['MODELMAGERR'][:,2],sdss['MODELMAGERR'][:,3],sdss['MODELMAGERR'][:,4]]).T

    # mixing matrix
    W = np.array([[1, 0, 0, 0, 0],    # i cmagnitude
                  [0, 1, -1, 0, 0],   # g-r
                  [0, 0, 1, -1, 0],
                  [0, 0, 0, 1, -1]])


    X = np.dot(X, W.T)

    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    
    # each covariance C = WCW^T
    # best way to do this is with a tensor dot-product
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))


    @pickle_results("XD_cmass.pkl")
    def compute_XD(n_clusters=12, rseed=0, n_iter=100, verbose=True):
        np.random.seed(rseed)
        clf = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf.fit(X, Xcov)
        return clf
    
    clfX = compute_XD()
    X_sample = clfX.sample(X.shape[0])
    X_prob = clfX.logprob_a(X, Xcov)

    
    
    # des
    des_g = des['MAG_DETMODEL_G'] - des['XCORR_SFD98_G']
    des_r = des['MAG_DETMODEL_R'] - des['XCORR_SFD98_R']
    des_i = des['MAG_DETMODEL_I'] - des['XCORR_SFD98_I']
    des_z = des['MAG_DETMODEL_Z'] - des['XCORR_SFD98_Z']
    des_ci = des['MAG_MODEL_I'] - des['XCORR_SFD98_I']

    Y = np.vstack([des_ci, des_g, des_r, des_i, des_z]).T
    Yerr = np.vstack([ des['MAGERR_MODEL_I'] , des['MAGERR_DETMODEL_G'], des['MAGERR_DETMODEL_R'],des['MAGERR_DETMODEL_I'],des['MAGERR_DETMODEL_Z']]).T

    # mixing matrix
    W = np.array([[1, 0, 0, 0],    # i cmagnitude
                  [0, 1, -1, 0],   # g-r
                  [0, 0, 1, -1]])  # r-i

    Y = np.dot(Y, W.T)

    Ycov = np.zeros(Yerr.shape + Yerr.shape[-1:])
    Ycov[:, range(Yerr.shape[1]), range(Yerr.shape[1])] = Yerr**2
    # each covariance C = WCW^T
    # best way to do this is with a tensor dot-product
    Ycov = np.tensordot(np.dot(Ycov, W.T), W, (-2, -1))
    

    # Matched CMASS
    des_g = MatchedCmass['MAG_DETMODEL_G'] - MatchedCmass['XCORR_SFD98_G']
    des_r = MatchedCmass['MAG_DETMODEL_R'] - MatchedCmass['XCORR_SFD98_R']
    des_i = MatchedCmass['MAG_DETMODEL_I'] - MatchedCmass['XCORR_SFD98_I']
    des_z = MatchedCmass['MAG_DETMODEL_Z'] - MatchedCmass['XCORR_SFD98_Z']
    des_ci = MatchedCmass['MAG_MODEL_I'] - MatchedCmass['XCORR_SFD98_I']
   
    Z = np.vstack([des_ci, des_g, des_r, des_i, des_z]).T
    Zerr = np.vstack([ MatchedCmass['MAGERR_MODEL_I'] , MatchedCmass['MAGERR_DETMODEL_G'], MatchedCmass['MAGERR_DETMODEL_R'],MatchedCmass['MAGERR_DETMODEL_I'], MatchedCmass['MAGERR_DETMODEL_Z']]).T
    
    # mixing matrix
    W = np.array([[1, 0, 0, 0, 0],    # i cmagnitude
                  [0, 1, -1, 0, 0],   # g-r
                  [0, 0, 1, -1, 0],
		  [0, 0, 0, 1, -1]])
        
    Z = np.dot(Z, W.T)

    Zcov = np.zeros(Zerr.shape + Zerr.shape[-1:])
    Zcov[:, range(Zerr.shape[1]), range(Zerr.shape[1])] = Zerr**2
    Zcov = np.tensordot(np.dot(Zcov, W.T), W, (-2, -1))




    # save data and fit-------------------
    
    @pickle_results("XD_cmass.pkl")
    def compute_XD(n_clusters=12, rseed=0, n_iter=50, verbose=True):
        np.random.seed(rseed)
        clf = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf.fit(X, Xcov)
        return clf

    clfX = compute_XD()
    X_sample = clfX.sample(X.shape[0])



    @pickle_results("XD_dmass.pkl")
    def compute_XD(n_clusters=12, rseed=0, n_iter=10, verbose=True):
        np.random.seed(rseed)
        clf = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf.fit(Y, Ycov)
        return clf

    clfY = compute_XD()
    Y_sample = clfY.sample(Y.shape[0])



    @pickle_results("XD_MatchedCmass.pkl")
    def compute_XD(n_clusters=12, rseed=0, n_iter=10, verbose=True):
        np.random.seed(rseed)
        clf = XDGMM(n_clusters, n_iter=n_iter, tol=1E-5, verbose=verbose)
        clf.fit(Z, Zcov)
        return clf

    clfZ = compute_XD()
    Z_sample = clfZ.sample(Z.shape[0])





   
# -----------------------------
    
    #des = des[(des['DESDM_ZP'] > 0.38) & (des['DESDM_ZP'] <0.9)]
    
    #sdss, _ = SDSS_cmass_criteria(sdss, prior=True)
    
    #_, dmass = DES_to_SDSS.match(cmass, des_data_f)
    sdss, des = DES_to_SDSS.match(sdss, des)
    
    import numpy.lib.recfunctions as rf
    sdss = rf.append_fields(sdss, 'DESDM_ZP', des['DESDM_ZP'])
    sdss = rf.append_fields(sdss, 'IM3_GALPROF', des['IM3_GALPROF'])
    
    _, prior_mask = priorCut(des)
    
    cmass, cmass_mask = SDSS_cmass_criteria(sdss)
    MachineLearningClassifier( des[cmass_mask], des[~cmass_mask&prior_mask] )
    
    
    
    
    data = des.copy()
    expmask = des['IM3_GALPROF'] == 1
    devmask = des['IM3_GALPROF'] == 2
    
    #exp_des_s = MatchedCmass_sdss[expmask]
    #dev_des_s = MatchedCmass_sdss[devmask]
    exp_des = data[expmask]
    dev_des = data[devmask]
    
    
    Tag = 'DESDM_ZP'
    min = 0.4
    max = 0.8
    bin_num = 6
    
    bin_center, binned_cat, binkeep = divide_bins( data, Tag = Tag, min = min, max = max, bin_num = bin_num)
    #bin_center_s, binned_cat_s, binkeep_s = divide_bins( sdss, Tag = Tag, min = min, max = max, bin_num = bin_num)
    
    """
    bin_center, binned_cat, binkeep = divide_bins( MatchedCmass_sdss, Tag = Tag, min = min, max = max,  bin_num = 6, TagIndex = 3)
    """
    
    
    Ys = []
    Cs = []
    fraction = []
    completeness = []
    
    
    for i, keep in enumerate(binkeep):
        cat_c = data[keep & cmass_mask]
        cat = data[keep & (~cmass_mask)& prior_mask]
        com, f = MachineLearningClassifier( cat_c, cat,k_neighbors_max = 20 )
        #f = np.sum(keep & cmass_mask & prior_mask) * 1./np.sum(keep & prior_mask)
        fraction.append(f)
        #com = np.sum(keep & prior_mask & cmass_mask)* 1./len(sdss[binkeep_s[i] & cmass_mask])
        completeness.append(com)
        #Y = mixing_color_sdss(cat, iter=i)
        C = mixing_color(cat_c)
        Y = mixing_color(cat)
        Ys.append(Y)
        Cs.append(C)
    
    for i, keep in enumerate(binkeep):
        cat_c = data[keep & cmass_mask & devmask]
        cat = data[keep & (~cmass_mask) & devmask& prior_mask]
        com, f = MachineLearningClassifier( cat_c, cat, k_neighbors_max = 10 )
        #f = np.sum(keep & cmass_mask & devmask& prior_mask) * 1./np.sum(keep & devmask& prior_mask)
        fraction.append(f)
        #com = np.sum(keep & cmass_mask & prior_mask)* 1./np.sum(binkeep_s[i] & cmass_mask)
        completeness.append(com)
        #Y = mixing_color_sdss(cat, iter=i)
        C = mixing_color(cat_c)
        Y = mixing_color(cat)
        Ys.append(Y)
        Cs.append(C)

    for i, keep in enumerate(binkeep):
        cat_c = data[keep & cmass_mask & expmask]
        cat = data[keep & (~cmass_mask) & expmask& prior_mask]
        com, f = MachineLearningClassifier( cat_c, cat,k_neighbors_max = 10 )
        #f = np.sum(keep & cmass_mask & expmask& prior_mask) * 1./np.sum(keep & expmask& prior_mask)
        fraction.append(f)
        #com = np.sum(keep & cmass_mask& prior_mask)* 1./np.sum(binkeep_s[i]&cmass_mask)
        completeness.append(com)
        #Y = mixing_color_sdss(cat, iter=i)
        C = mixing_color(cat_c)
        Y = mixing_color(cat)
        Ys.append(Y)
        Cs.append(C)






    fig, ax = plt.subplots(3, len(Ys)/3)
    ax = ax.ravel()

    ibin = np.array([bin_center,bin_center,bin_center]).ravel()
    ibin_edge = np.linspace(min, max, bin_num+1)
    ibin_edge = np.delete(ibin_edge, -1)
    ibin_edge = np.array([ibin_edge,ibin_edge,ibin_edge]).ravel()
    
    x = np.arange(-2, 4)
    for (i, Y), C in zip(enumerate(Ys), Cs):
        ax[i].text(0.05, 0.95, '{}={:>0.2f} \npur={:>0.2f} \ncom={:>0.2f}'.format(Tag, ibin[i], fraction[i], completeness[i]),
                   ha='left', va='top', transform=ax[i].transAxes, fontsize = 8)
        ax[i].scatter(Y[:, 1], Y[:, 2], s=1, lw=0, c='red', alpha=1.0)
        ax[i].scatter(C[:, 1], C[:, 2], s=2, lw=0, c='green', alpha=1.0, label = 'CMASS')
        ax[i].plot( x, 0.35 + x/8., 'k-')
        ax[i].plot( x, -0.4 + x/8. + 0.8 + (ibin_edge[i] - 19.86)/5.6, 'k--' )
        ax[i].set_xlim(0,3)
        ax[i].set_ylim(0.4, 1.4)

    ax[0].set_ylabel('all')
    ax[6].set_ylabel('dev')
    ax[12].set_ylabel('exp')

        #ax[i].set_xlabel('g - r')
        #ax[i].set_ylabel('r - i')


    fig, ax = plt.subplots(3, 6)
    ax = ax.ravel()
    for i, com in enumerate(completeness):
         ax[i].text(0.05, 0.95, '{}={:>0.2f}'.format(Tag, ibin[i]), ha='left', va='top', transform=ax[i].transAxes, fontsize = 8)
         ax[i].plot( np.arange(com.size), com, 'r-', label='completeness')
         ax[i].plot( np.arange(com.size), fraction[i], 'b-', label = 'purity')
         ax[i].set_ylim(0, 1.0)
         ax[i].set_xlabel('n_neighbors')

    ax[0].legend(loc='best')


    stop


    
# -----------------------------

    # extreme_deconvolution
    #from extreme_deconvolution import extreme_deconvolution

    



    # Fit and sample from the underlying distribution
    from astroML.plotting.tools import draw_ellipse
    
    fig,((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2,3, figsize=(14, 14))

    # only plot 1/10 of the stars for clarity
    #ax1 = fig.add_subplot(221)
    #ax1.scatter(X[::10, 1], X[::10, 2], s=9, lw=0, c='k')
    ax1.scatter(X[:, 1], X[:, 2], s=9, lw=0, c='b', alpha = 0.3)
    ax1.set_xlim(-2, 4 )
    ax1.set_ylim(0.2, 2)
    
    #ax2 = fig.add_subplot(222)
    #ax2.scatter(X_sample[::10, 1], X_sample[::10, 2], s=9, lw=0, c='k')
    ax2.scatter(X_sample[:, 2], X_sample[:, 3], s=9, lw=0, c='b', alpha = 0.3)
    ax2.set_xlim(-2, 4 )
    ax2.set_ylim(0.2, 2)

    for i in range(clfX.n_components):
        draw_ellipse(clfX.mu[i], clfX.V[i], scales=[2], ax=ax5,
                     ec='k', fc='gray', alpha=0.2)
    
    ax5.set_xlim(-2, 4 )
    ax5.set_ylim(0.2, 2)

    
    #ax3 = fig.add_subplot(223)
    ax5.scatter(Y[:, 1], Y[:, 2], s=9, lw=0, c='grey', alpha=0.8)
    ax3.scatter(Z[:, 1], Z[:, 2], s=9, lw=0, c='g', alpha = 0.3)
    ax3.set_xlim(-2, 4 )
    ax3.set_ylim(0.2, 2)
    
    #ax4 = fig.add_subplot(224)
    ax4.scatter(Y_sample[:, 1], Y_sample[:, 2], s=9, lw=0, c='grey', alpha=0.8)
    ax4.scatter(Z_sample[:, 1], Z_sample[:, 2], s=9, lw=0, c='g', alpha = 0.3)
    ax4.set_xlim(-2, 4 )
    ax4.set_ylim(0.2, 2)

    for i in range(clfZ.n_components):
        draw_ellipse(clfZ.mu[i], clfZ.V[i], scales=[2], ax=ax6, ec='k', fc='grey', alpha=0.2)

    ax6.set_xlim(-2, 4 )
    ax6.set_ylim(0.2, 2)

    axs = [ax1, ax2, ax3, ax4, ax5, ax6]
    for a in axs:
        a.set_xlim(-2,4)
        a.set_ylim(0.0, 2)
        a.set_xlabel('g - r')
        a.set_ylabel('r - i')


                 
    stop

    fig2 = plt.figure(figsize=(14, 14))
    ax5 = fig2.add_subplot(221)
    #ax5.scatter(X[::10, 2], X[::10, 3], s=9, lw=0, c='k')
    ax5.scatter(X[:, 2], X[:, 3], s=9, lw=0, c='b', alpha = 0.3)
    ax5.set_xlim(-2, 4 )
    ax5.set_ylim(0.2, 2)
    
    ax6 = fig2.add_subplot(222)
    #ax6.scatter(X_sample[::10, 2], X_sample[::10, 3], s=9, lw=0, c='k')
    ax6.scatter(X_sample[:, 2], X_sample[:, 3], s=9, lw=0, c='b', alpha = 0.3)
    ax6.set_xlim(-2, 4 )
    ax6.set_ylim(0.2, 2)

    ax7 = fig2.add_subplot(223)
    #ax5.scatter(X[::10, 2], X[::10, 3], s=9, lw=0, c='k')
    ax7.scatter(Z[:, 1], Z[:, 2], s=9, lw=0, c='g', alpha = 0.3)
    ax7.set_xlim(-2, 4 )
    ax7.set_ylim(0.2, 2)
    
    ax8 = fig2.add_subplot(224)
    #ax6.scatter(X_sample[::10, 2], X_sample[::10, 3], s=9, lw=0, c='k')
    ax8.scatter(Z_sample[:, 1], Z_sample[:, 2], s=9, lw=0, c='g', alpha = 0.3)
    ax8.set_xlim(-2, 4 )
    ax8.set_ylim(0.2, 2)

    ax5.set_xlabel('model_g - model_r')
    ax5.set_ylabel('model_r - model_i')
    ax5.set_title('sdss cmass')

    ax6.set_xlabel('model_g - model_r')
    ax6.set_ylabel('model_r - model_i')
    ax6.set_title('sdss cmass true')

    ax7.set_xlabel('detmodel_g - detmodel_r')
    ax7.set_ylabel('detmodel_r - detmodel_i')
    ax7.set_title('des cmass')

    ax8.set_xlabel('detmodel_g - detmodel_r')
    ax8.set_ylabel('detmodel_r - detmodel_i')
    ax8.set_title('des cmass true')

#-----------------------
#main()


