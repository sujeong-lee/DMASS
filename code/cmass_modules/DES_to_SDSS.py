#import easyaccess as ea
import esutil
import sys
import os
#import healpy as hp
import numpy as np
import pandas as pd
#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
#import seaborn as sns

from ang2stripe import *
import fitsio
from fitsio import FITS, FITSHDR


def dperp_fitting(des, sdss):
    from sklearn import linear_model
    #from sklearn import linear_model
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    
    sdssColor= 'MODELMAG'
    desColor = 'MODELMAG'
    
    dperp_sdss = sdss[sdssColor+'_R'] - sdss[sdssColor+'_I'] - (sdss[sdssColor+'_G'] - sdss[sdssColor+'_R'])/8.
    dperp_des = (des[desColor+'_R_DES'] - des[desColor+'_I_DES']) - (des[desColor+'_G_DES'] - des[desColor+'_R_DES'])/8.

    ColorCorrection = '_SDSS'
    magnitudes = ['MAG_MODEL','MAG_DETMODEL']
    filter = ['G', 'R', 'I', 'Z']

    X = [ des[ mag+'_'+thisfilter+ColorCorrection] for mag, thisfilter in zip(magnitudes, filter) ]
    X.append( dperp_des )
    X = np.array(X).T
    #X = np.array([ des[ mag+'_'+thisfilter+ColorCorrection].data for mag, thisfilter in zip(magnitudes, filter) ]).T
    #X = np.array(dperp_des).reshape((len(dperp_des.data), 1))
    y = dperp_sdss


    keep = ((dperp_des < 1.0) & (dperp_des > -1.0))# |( (dperp_des < 0.3) & (dperp_des > 0.0))

    X_train, X_test, y_train, y_test = train_test_split(X[keep], y[keep], train_size = 0.2, random_state=1)
    
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='auto')
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_lin = SVR(kernel='linear', C=1e3)
    
    y_rbf = svr_rbf.fit(X_train, y_train).predict(X)
    #y_poly = svr_poly.fit(X_train, y_train).predict(X)
    #y_lin = svr_lin.fit(X_train, y_train).predict(X)
    
    
    plt.scatter(dperp_des, y, c='k', label='data', alpha = 0.33)
    plt.hold('on')
    plt.plot(dperp_des , y_rbf,  'g.', label='rbf', alpha = 0.5)
    plt.plot([-10,10], [-10,10], 'r--')
    #plt.plot(dperp_des, y_lin , 'r.', label='Linear model', alpha = 0.2)
    #plt.plot(X, y_poly, c='b', label='Polynomial model')
    
    plt.xlabel('data des')
    plt.ylabel('target sdss')
    plt.xlim(-1., 1.5)
    plt.ylim(-1., 1.5)
    plt.title('Support Vector Regression')
    plt.legend(loc = 'best')
    plt.savefig('../figure/svrfitting')
    plt.clf()
    
    return y_rbf


def DESdperp_to_SDSSdperp(fullsdss, fulldes):
    
    modelmag_g = fullsdss['MODELMAG_G'] - fullsdss['EXTINCTION_G']
    modelmag_r = fullsdss['MODELMAG_R'] - fullsdss['EXTINCTION_R']
    modelmag_i = fullsdss['MODELMAG_I'] - fullsdss['EXTINCTION_I']
    modelmag_z = fullsdss['MODELMAG_Z'] - fullsdss['EXTINCTION_Z']
    dperp_sdss = (modelmag_r - modelmag_i) - (modelmag_g - modelmag_r)/8.0

    des, sdss = match(fulldes, fullsdss)
    
    modelmag_g_des = des['MODELMAG_G_DES'] - des['XCORR_SFD98_G']
    modelmag_r_des = des['MODELMAG_R_DES'] - des['XCORR_SFD98_R']
    modelmag_i_des = des['MODELMAG_I_DES'] - des['XCORR_SFD98_I']
    modelmag_z_des = des['MODELMAG_Z_DES'] - des['XCORR_SFD98_Z']
    dperp_des = (modelmag_r_des - modelmag_i_des) - (modelmag_g_des - modelmag_r_des)/8.0

    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    
    #magcut = ((des['MODELMAG_G_DES'] < 22.0) & (des['MODELMAG_R_DES'] < 22.0) &
    #(des['MODELMAG_I_DES'] < 22.0) & (des['MODELMAG_Z_DES'] < 22.0) )
    #magcut = ((sdss['MODELMAG_R'] < 22.0) & (sdss['MODELMAG_I'] < 22.0)
    #          &(sdss['MODELMAG_G'] < 22.0) & (sdss['MODELMAG_Z'] < 22.0))
    
    use =  (#(18.0 < des['CMODELMAG_I_DES']) &
            #(20.9 > des['CMODELMAG_I_DES']) &
            #(des['MODELMAG_R_DES'] - des['MODELMAG_I_DES'] < 2.) &
            (des['FIBER2MAG_I_DES'] < 21.5 )
            )
            
    #use = use  & magcut
            
    des_exp = des[expcut & use]
    des_dev = des[devcut & use]
    sdss_exp = sdss[expcut & use]
    sdss_dev = sdss[devcut & use]
    
    #des_exp_dperp = dperp_fitting(des_exp, sdss_exp)
    #des_dev_dperp = dperp_fitting(des_dev, sdss_dev)
    
    #SDSSlike_dperp = np.zeros(len(fulldes), dtype=np.float32)
    #SDSSlike_dperp[expcut & use] = des_exp_dperp
    #SDSSlike_dperp[devcut & use] = des_dev_dperp
    
    try :
        #fulldes = rf.append_fields(fulldes, 'DPERP_DES', SDSSlike_dperp)
        fulldes = rf.append_fields(fulldes, 'DPERP', dperp_des)
        fullsdss = rf.append_fields(fullsdss, 'DPERP', dperp_sdss)

    except ValueError:
        #fulldes['DPERP_DES'] = SDSSlike_dperp
        fulldes['DPERP'] = dperp_des
        fullsdss['DPERP'] = dperp_sdss

    return fullsdss, fulldes



def fib2mag_fitting(des, sdss, filter = 'I'):
    from sklearn import linear_model
    from sklearn.cross_validation import train_test_split

    keep = ((sdss['FIBER2MAG_I'] < 30.) & (sdss['FIBER2MAG_I'] > 10.))
    des = des[keep]
    sdss = sdss[keep]

    ColorCorrection = '' #_SDSS'
    magnitudes = ['MAG_APER_3','MAG_APER_4', 'MAG_APER_5', 'MAG_APER_6']
    
    X = np.array([ des[ mag+'_'+filter+ColorCorrection] for mag in magnitudes ]).T
    y = sdss['FIBER2MAG_I']

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    clf = linear_model.LinearRegression()
    clf = linear_model.Lasso(alpha = 0.001)
    #clf = linear_model.ElasticNet(alpha = 0.001, l1_ratio=0.7)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X)

    return y_pred #, sdss, coeff, coeff_list, A
    
    """
    x = [des[ tag+'_'+ filter + correction ] for tag in magnitudes ]
    x = np.vstack((x))
    
    y = sdss['FIBER2MAG_I']
    A = np.vstack([x, np.ones(len(y))]).T
    
    coeff, residuals, __, __ = np.linalg.lstsq(A, y) #, rcond = 1.0)
    des_fib2 = np.dot(A, coeff)
    
    coeff_list = 'coef = ( {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f})'.format(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4])
    
    return des_fib2
    """





def DESfib2mag_to_SDSSfib2mag(sdss, des):
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    
    filter = 'I'
    des_exp_corrected = fib2mag_fitting(des_exp, sdss_exp, filter = filter)
    des_dev_corrected = fib2mag_fitting(des_dev, sdss_dev, filter = filter)
    
    SDSSlike_fibmag = np.zeros(len(des), dtype=np.float32)
    SDSSlike_fibmag[expcut] = des_exp_corrected
    SDSSlike_fibmag[devcut] = des_dev_corrected
    
    #des = rf.append_fields(des, 'FIBER2MAG_'+filter+'_DES', SDSSlike_fibmag)
    
    return SDSSlike_fibmag

def DES_to_SDSS_fitting(sdss, des, filter = 'G', Scolorkind = 'MODELMAG', Dcolorkind = 'MAG_MODEL'):
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    
    colorCorrection = '' #_SDSS'
    
    sdssColor = Scolorkind+'_'+filter
    desColor = Dcolorkind+'_'+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20) & (sdss_exp[sdssColor] > 15) & (des_exp[desColor] > 15)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20) & (sdss_dev[sdssColor] > 15) & (des_dev[desColor] > 15)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    
    alpha, beta = coef_exp
    exp_des_corrected = (des_exp[desColor] - beta)/alpha # exp result
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    
    alpha, beta = coef_dev
    dev_des_corrected = (des_dev[desColor] - beta)/alpha # dev result
    
    SDSSlike_mag = np.zeros(len(des), dtype=np.float32)
    SDSSlike_mag[expcut] = exp_des_corrected
    SDSSlike_mag[devcut] = dev_des_corrected
    
    des = rf.append_fields(des, Scolorkind+'_'+filter+'_DES', SDSSlike_mag)
    return des


def scikitfitting( sdss, des, filter = 'G', Scolorkind = 'MODELMAG'):
    
    import matplotlib.pyplot as plt
    from sklearn import datasets, linear_model
    from sklearn.cross_validation import train_test_split
    from sklearn.svm import SVR
    
    colorCorrection =''# _SDSS'
    sdssColor = Scolorkind+'_'+filter
    #desColor = Dcolorkind+'_'+filter+colorCorrection

    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    #use = (sdss[sdssColor] < 26) & (sdss[sdssColor] > 10)
    #keep = (sdss_exp[sdssColor] < 26) & (sdss_exp[sdssColor] > 10)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]

    tags = ['MAG_DETMODEL', 'MAG_MODEL', 'MAG_PETRO', 'MAG_PSF']
    filters = ['G', 'R', 'I', 'Z']
    magnitudes = [  tag+'_'+thisfilter+colorCorrection for tag, thisfilter in zip(tags, filters)]
    X_exp = np.array([ des_exp[ mag ] for mag in magnitudes ]).T
    y_exp = sdss_exp[sdssColor]
    X_dev = np.array([ des_dev[ mag] for mag in magnitudes ]).T
    y_dev = sdss_dev[sdssColor]


    # scikit-learn svm regression ---------------------------------
    
    
    X_train, X_test, y_train, y_test = train_test_split(X_exp, y_exp, train_size = 0.2, random_state=1)
    
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_lin = SVR(kernel='linear', C=1e3 )
    
    SDSSlike_exp = svr_rbf.fit(X_train, y_train).predict(X_exp)
    #SDSSlike_exp = svr_lin.fit(X_train, y_train).predict(X_exp)
    #SDSSlike_exp = svr_poly.fit(X_train, y_train).predict(X_exp)

    X_train, X_test, y_train, y_test = train_test_split(X_dev, y_dev, train_size = 0.2, random_state=1)
    #SDSSlike_dev = svr_poly.fit(X_train, y_train).predict(X_dev)
    SDSSlike_dev = svr_rbf.fit(X_train, y_train).predict(X_dev)
    #SDSSlike_dev = svr_lin.fit(X_train, y_train).predict(X_dev)
    
    
    # scikit-learn linear regression ---------------------------------
    """
    #exp_x = np.array([ des_exp[ mag+'_'+filter+colorCorrection] for mag in magnitudes ]).T
    #exp_y = sdss_exp[sdssColor]
    keep = (sdss_exp[sdssColor] < 19) & (sdss_exp[sdssColor] > 15)
    X_train, X_test, y_train, y_test = train_test_split(X_exp[keep], y_exp[keep], train_size = 0.5, random_state=1)
    #clf = linear_model.LinearRegression()
    clf = linear_model.Lasso(alpha = 0.001)
    
    clf.fit(X_train, y_train)
    SDSSlike_exp = clf.predict(X_exp)
    
    #dev_x = np.array([des_dev[ mag+'_'+filter+colorCorrection] for mag in magnitudes ]).T
    #dev_y = sdss_dev[sdssColor]
    keep = (sdss_dev[sdssColor] < 19) & (sdss_dev[sdssColor] > 15)
    X_train, X_test, y_train, y_test = train_test_split(X_dev[keep], y_dev[keep], train_size = 0.5, random_state=1)
    clf.fit(X_train, y_train)
    SDSSlike_dev = clf.predict(X_dev)
    """
    

    SDSSlike_mag = np.zeros(len(des), dtype=np.float32)
    SDSSlike_mag[expcut] = SDSSlike_exp
    SDSSlike_mag[devcut] = SDSSlike_dev

    #des = rf.append_fields(des, Scolorkind+'_'+filter+'_DES', SDSSlike_mag)
    
    return SDSSlike_mag #des



def DESmag_to_SDSSmag(sdss_data, des_data):
    
    sdss, des = match(sdss_data, des_data)
    
    sys.stdout.write('DESmag to SDSS mag ')
    
    filters = ['G','R','I','Z']
    Scolorkind = ['MODELMAG', 'CMODELMAG' ]
    import time
    t1 = time.time()
    from multiprocessing import Process, Queue
    
    def multiprocessing_mag(q,sdss, des, thisfilter, Scolor):
        magTag = Scolor+'_'+thisfilter+'_DES'
        q.put(( magTag, scikitfitting(sdss, des, filter = thisfilter, Scolorkind =  Scolor )))
        sys.stdout.write('.')
    
    d_queue = Queue()
    d_processes = []
    for Scolor in Scolorkind:
        for thisfilter in filters:
            p = Process(target=multiprocessing_mag, args=(d_queue, sdss, des, thisfilter, Scolor ))
            d_processes.append(p)

    for p in d_processes:
        p.start()
    
    result = [d_queue.get() for p in d_processes]

    #result.sort()
    desMags = [D[1] for D in result]
    magTaglist = [D[0] for D in result]


    fib2mag = DESfib2mag_to_SDSSfib2mag(sdss, des)
    #dperp3 = DESdperp_to_SDSSdperp(sdss, des)

    """
    for Tag, desMag in zip(magTaglist, desMags):
        des = rf.append_fields(des, Tag, desMag)
        sys.stdout.write('.')

    des = rf.append_fields(des, 'FIBER2MAG_I_DES', fib2mag)
    sys.stdout.write('done')
    """
    # mergin with pandas

    from pandas import DataFrame, concat
    data2 = DataFrame( desMags, index = magTaglist ).T
    fib2mag = DataFrame( fib2mag, columns = ['FIBER2MAG_I_DES'] )
    des = DataFrame(des)

    del des['index']
    des = concat([des, data2, fib2mag], axis=1)

    des = des.to_records()
    

    """
    for thisfilter in filters:
        #print "DESmag to SDSS mag for filter "+thisfilter
        
        #des = DES_to_SDSS_fitting(sdss, des, filter = thisfilter, Scolorkind = 'CMODELMAG', Dcolorkind = 'MAG_MODEL')
        #des = DES_to_SDSS_fitting(sdss, des, filter = thisfilter, Scolorkind = 'MODELMAG', Dcolorkind = 'MAG_DETMODEL')
        
    
        data1 = scikitfitting(sdss, des, filter = thisfilter, Scolorkind = 'MODELMAG')
        data2 = scikitfitting(sdss, des, filter = thisfilter, Scolorkind = 'CMODELMAG')
        
    des = DESfib2mag_to_SDSSfib2mag(sdss, des)
    #des = DESdperp_to_SDSSdperp(sdss, des)
    """
    print '\ntime :', time.time()-t1


    return sdss, des #des #sdss, des2

def transform_DES_to_SDSS(g_des, r_des, i_des, z_des):
    # Transform DES colors to SDSS
    # Transformation equations for stars bluer than (g-r)_sdss = 1.2:
    # Coefficients from Doug Tucker's and Anne Bauer's work, here:
    # https://cdcvs.fnal.gov/redmine/projects/descalibration/wiki
    Fmatrix = np.array([[1. - 0.104,  0.104,0., 0.],
                        [0., 1. - 0.102, 0.102, 0.],
                        [0., 0., 1 - 0.256, 0.256,],
                        [0., 0., -0.086, 1 + 0.086]])
        
    const = np.array( [0.01, 0.02, 0.02, 0.01] )
    
    des_mag =  np.array( [g_des, r_des, i_des, z_des] )
    Finv = np.linalg.inv(Fmatrix)
    
    sdss_mag = np.dot(Finv, (des_mag.T - const).T )
    return sdss_mag

def add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL', independent = None):
    print "Doing des->sdss color transforms for "+magTag_template
    filters = ['G','R','I','Z']
    magTags = []
    
    desMags = np.empty([len(filters),len(data)])
    for i,thisFilter in enumerate(filters):
        magTag = magTag_template+'_'+thisFilter
        desMags[i,:] = data[magTag]
        magTags.append(magTag+'_SDSS')
    sdssMags = transform_DES_to_SDSS(desMags[0,:], desMags[1,:], desMags[2,:], desMags[3,:])

    from pandas import DataFrame, concat
    data = DataFrame( sdssMags, index = magTags).T

    if independent == 'yes':
        fulldata = DataFrame(data)
        data = concat([fulldata, data], axis=1)
        data = data.to_records()
    else : pass

    return data


def ColorTransform(data):
    
    from pandas import DataFrame, concat

    #magTaglist = ['MAG_DETMODEL', 'MAG_MODEL', 'MAG_PETRO', 'MAG_HYBRID', 'MAG_PSF','MAG_AUTO',
    #              'MAG_APER_2', 'MAG_APER_3', 'MAG_APER_4','MAG_APER_5' ] #,'MAG_APER_6','MAG_APER_7','MAG_APER_8','MAG_APER_9', 'MAG_APER_10']
    
    magTaglist = ['MAG_MODEL', 'MAG_AUTO', 'MAG_APER_3', 'MAG_APER_4','MAG_APER_5','MAG_APER_6' ]
    combine = add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL')
    
    for magTag in magTaglist:
    
        strip = add_SDSS_colors(data, magTag_template = magTag)
        combine = concat([combine, strip], axis=1)

    data = DataFrame(data.data)
    data = concat([data, combine], axis=1)
    #del data['index']
    matched_arr = data.to_records()
    return matched_arr



def modestify(data):
    #from Eric's code
    
    modest = np.zeros(len(data), dtype=np.int32)
    
    galcut = (data['FLAGS_I'] <=3) & -( ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'] < 18.0)) | ((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) | ((data['MAG_PSF_I'] > 30.0) & (data['MAG_AUTO_I'] < 21.0)))
    modest[galcut] = 1
    
    starcut = (data['FLAGS_I'] <=3) & ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'] < 18.0) & (data['MAG_PSF_I'] < 30.0) | (((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) & ((data['SPREAD_MODEL_I'] +3*data['SPREADERR_MODEL_I']) > -0.003)))
    modest[starcut] = 3
    
    neither = -(galcut | starcut)
    modest[neither] = 5
    
    data = rf.append_fields(data, 'MODETYPE', modest)
    print np.sum(galcut), np.sum(starcut), np.sum(neither)
    return data

def DESclassifier( des):
    des = modestify(des)
    gal_cut = (des['MODETYPE'] == 1)
    star_cut = (des['MODETYPE'] == 3)
    neither = (des['MODETYPE'] == 5)
    return des[gal_cut], des[star_cut]

def match(sdss, des):
    h = esutil.htm.HTM(10)
    matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(des['RA'], des['DEC'], sdss['RA'],sdss['DEC'],matchDist,maxmatch=1)
    
    print 'matched object ',len(m_des)
    return sdss[m_sdss],des[m_des]

def matchCatalogsWithTag(cat1, cat2 , tag = 'balrog_index'):
    ind1, ind2 = esutil.numpy_util.match(cat1[tag], cat2[tag])
    return cat1[ind1], cat2[ind2]
