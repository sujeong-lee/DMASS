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


def fib2mag_fitting(des, sdss, filter = 'I'):
    
    # x = matrix of 1 and aper 3 ~ 6
    
    keep = ((sdss['FIBER2MAG_I'] < 25.) & (sdss['FIBER2MAG_I'] > 15.))
    des = des[keep]
    sdss = sdss[keep]
    
    correction = '_SDSS'
    taglist = ['MAG_APER_3','MAG_APER_4', 'MAG_APER_5', 'MAG_APER_6']
    # = ['G','R','I','Z']
    
    x = [des[ tag+'_'+ filter + correction ] for tag in taglist ]
    x = np.vstack((x))
    
    y = sdss['FIBER2MAG_I']
    A = np.vstack([x, np.ones(len(y))]).T
    
    coeff, residuals, __, __ = np.linalg.lstsq(A, y) #, rcond = 1.0)
    des_fib2 = np.dot(A, coeff)
    
    coeff_list = 'coef = ( {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f}, {:>.2f})'.format(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4])
    
    return des_fib2, sdss, coeff, coeff_list, A

def DESfib2mag_to_SDSSfib2mag(  sdss, des):
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    
    filter = 'I'
    des_exp,_,_,_,_ = fib2mag_fitting(des_exp, sdss_exp, filter = filter)
    des_dev,_,_,_,_ = fib2mag_fitting(des_dev, sdss_dev, filter = filter)
    
    SDSSlike_fibmag = np.zeros(len(des), dtype=np.float32)
    SDSSlike_fibmag[expcut] = des_exp
    SDSSlike_fibmag[devcut] = des_dev
    
    des = rf.append_fields(des, 'FIBER2MAG_'+filter+'_DES', SDSSlike_fibmag)
    return des

def DES_to_SDSS_fitting(sdss, des, filter = 'G', Scolorkind = 'MODELMAG', Dcolorkind = 'MAG_MODEL'):
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    
    colorCorrection = '_SDSS'
    
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

def DESmag_to_SDSSmag(  sdss_data, des_data):
    
    sdss, des = match(sdss_data, des_data)
    
    filters = ['G','R','I','Z']
    for thisfilter in filters:
        print "DESmag to SDSS mag for filter "+thisfilter
        des = DES_to_SDSS_fitting(sdss, des, filter = thisfilter, Scolorkind = 'CMODELMAG', Dcolorkind = 'MAG_MODEL')
        des = DES_to_SDSS_fitting(sdss, des, filter = thisfilter, Scolorkind = 'MODELMAG', Dcolorkind = 'MAG_DETMODEL')
    
    # fib correction
    des = DESfib2mag_to_SDSSfib2mag(sdss, des)
    return sdss, des

def transform_DES_to_SDSS(g_des, r_des, i_des, z_des):
    # Transform DES colors to SDSSdes
    # Transformation equations for stars bluer than (g-r)_sdss = 1.2:
    # Coefficients from Doug Tucker's and Anne Bauer's work, here:
    # https://cdcvs.fnal.gov/redmine/projects/descalibration/wiki
    Fmatrix = np.array([[1. - 0.104,  0.104,0., 0.],
                        [0., 1. - 0.102, 0.102, 0.],
                        [0., 0., 1 - 0.256, 0.256,],
                        [0., 0., -0.086, 1 + 0.086]])
        
    const = np.array( [0.01, 0.02, 0.02,0.01] )
    
    des_mag =  np.array( [g_des, r_des, i_des, z_des] )
    Finv = np.linalg.inv(Fmatrix)
    
    sdss_mag = np.dot(Finv, (des_mag.T - const).T )
    return sdss_mag

def add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL'):
    print "Doing des->sdss color transforms for "+magTag_template
    filters = ['G','R','I','Z']
    magTags = []
    
    desMags = np.empty([len(filters),len(data)])
    for i,thisFilter in enumerate(filters):
        magTag = magTag_template+'_'+thisFilter
        desMags[i,:] = data[magTag]
    
    sdssMags = transform_DES_to_SDSS(desMags[0,:], desMags[1,:], desMags[2,:], desMags[3,:])
    
    data = rf.append_fields(data, magTag_template+'_G_SDSS', sdssMags[0,:])
    data = rf.append_fields(data, magTag_template+'_R_SDSS', sdssMags[1,:])
    data = rf.append_fields(data, magTag_template+'_I_SDSS', sdssMags[2,:])
    data = rf.append_fields(data, magTag_template+'_Z_SDSS', sdssMags[3,:])
    return data


def ColorTransform(data):
    
    data = add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL')
    data = add_SDSS_colors(data, magTag_template = 'MAG_MODEL')
    data = add_SDSS_colors(data, magTag_template = 'MAG_PSF')
    #data = add_SDSS_colors(data, magTag_template = 'MAG_APER_2')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_3')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_4')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_5')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_6')
    
    return data

def do_CMASS_cuts(data):

    data = add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL')
    data = add_SDSS_colors(data, magTag_template = 'MAG_MODEL')
    data = add_SDSS_colors(data, magTag_template = 'MAG_PSF')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_2')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_3')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_4')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_5')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_6')
    
    # Aperture magnitudes are in apertures of:
    # PHOT_APERTURES  1.85   3.70   5.55   7.41  11.11   14.81   18.52   22.22   25.93   29.63   44.44  66.67
    # (MAG_APER aperture diameter(s) are in pixels)
    # SDSS fiber and fiber2 mags are (sort of) in apertures of 3" and 2", respectively, which correspond to
    # 7.41 pix and 11.11 pix (in DES pixels) respectively, so we'll use mag_aper_4 and mag_aper_5 for the two fiber mags.
    print "Calculating/applying CMASS object cuts."
    
    
    dperp = ( (data['MAG_MODEL_R_SDSS'] - data['MAG_MODEL_I_SDSS']) -
             (data['MAG_MODEL_G_SDSS'] - data['MAG_MODEL_R_SDSS'])/8.0 )
        
    keep_cmass = ( (dperp > 0.55) &
               (data['MAG_DETMODEL_I_SDSS'] < (19.86 + 1.6*(dperp - 0.8) )) &
               (data['MAG_DETMODEL_I_SDSS'] > 17.5 ) & (data['MAG_DETMODEL_I_SDSS'] < 19.9) &
               ((data['MAG_MODEL_R_SDSS'] - data['MAG_MODEL_I_SDSS'] ) < 2 ) &
               (data['MAG_APER_4_I_SDSS'] < 21.5 ) &
               ((data['MAG_PSF_I_SDSS'] - data['MAG_MODEL_I_SDSS']) > (0.2 + 0.2*(20.0 - data['MAG_MODEL_I_SDSS'] ) ) ) &
               ((data['MAG_PSF_Z_SDSS'] - data['MAG_MODEL_Z_SDSS']) > (9.125 - 0.46 * data['MAG_MODEL_Z_SDSS'])) )

    return data[keep_cmass], data

def make_cmass_plots(data, data2):
    fig,((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    
    dperp_exp = ( (data['MODELMAG_R_DES'] - data['MODELMAG_I_DES']) -
                 (data['MODELMAG_G_DES'] - data['MODELMAG_R_DES'])/8.0 )
        
    ax1.plot(data['MODELMAG_G_DES'] - data['MODELMAG_R_DES'],
          data['MODELMAG_R_DES'] - data['MODELMAG_I_DES'],'.')
    ax1.set_xlabel("(g-r)_sdss (model)")
    ax1.set_ylabel("(r-i)_sdss (model)")
    ax1.set_xlim(0,3)
    ax1.set_ylim(0.5,1.6)

    ax2.plot(data['CMODELMAG_I_DES'],dperp_exp,'.',markersize=10)
    ax2.set_xlabel("i_sdss (model)")
    ax2.set_ylabel("dperp")
    ax2.set_xlim(17.0,20.0)
    ax2.set_ylim(0.5,1.6)

    SDSSdperp = ( (data2['MODELMAG'][:,2] - data2['MODELMAG'][:,3]) -
              (data2['MODELMAG'][:,1] - data2['MODELMAG'][:,2])/8.0 )

    ax3.plot(data2['MODELMAG'][:,1] - data2['MODELMAG'][:,2],
          data2['MODELMAG'][:,2] - data2['MODELMAG'][:,3],'.')
    ax3.set_xlabel("(g-r)_sdss (model)")
    ax3.set_ylabel("(r-i)_sdss (model)")
    ax3.set_xlim(0,3)
    ax3.set_ylim(0.5,1.6)

    ax4.plot(data2['CMODELMAG'][:,3],SDSSdperp,'.',markersize=10)
    ax4.set_xlabel("SDSS i_sdss (model)")
    ax4.set_ylabel("dperp")
    ax4.set_xlim(17.0,20.0)
    ax4.set_ylim(0.5,1.6)

    fig.savefig("../figure/cmass_plots")
    print 'fig saved : '+'../figure/cmass_plots'+'.png'

def cmass_criteria(  data):
    
    print "Calculating/applying CMASS object cuts."
    
    modelmag_g = data['MODELMAG_G_DES'] - data['EBV']
    modelmag_r = data['MODELMAG_R_DES'] - data['EBV']
    modelmag_i = data['MODELMAG_I_DES'] - data['EBV']
    
    cmodelmag_g = data['CMODELMAG_G_DES'] - data['EBV']
    cmodelmag_r = data['CMODELMAG_R_DES'] - data['EBV']
    cmodelmag_i = data['CMODELMAG_I_DES'] - data['EBV']
    
    dperp = ( (modelmag_r - modelmag_i) -
             (modelmag_g - modelmag_r)/8.0 )
        
    cmass = ( (dperp > 0.55) &
          (cmodelmag_i < (19.86 + 1.6*(dperp - 0.8) )) &
          (cmodelmag_i > 17.5 ) & (cmodelmag_i< 19.9) &
          ((modelmag_r - modelmag_i ) < 2 ) &
          (data['FIBER2MAG_I_DES'] < 21.5 ) )
            #&((data['MAG_PSF_I_SDSS'] - data['MODELMAG_EXP_I_DES']) > (0.2 + 0.2*(20.0 - data['MODELMAG_EXP_I_DES'] ) ) ) & ((data['MAG_PSF_Z_SDSS'] - data['MODELMAG_EXP_Z_DES']) > (9.125 - 0.46 * data['MODELMAG_EXP_Z_DES'])) )

    print 'cmass galaxies exp ',np.sum(cmass)
    return data[cmass]

def modestify(data):
    #from Eric's code
    
    modest = np.zeros(len(data), dtype=np.int32)
    
    galcut = (data['FLAGS_I'] <=3) & -( ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'] < 18.0)) | ((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) | ((data['MAG_PSF_I'] > 30.0) & (data['MAG_AUTO_I'] < 21.0)))
    modest[galcut] = 1
    
    starcut = (data['FLAGS_I'] <=3) & ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'] < 18.0) & (data['MAG_PSF_I'] < 30.0) | (((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) & ((data['SPREAD_MODEL_I'] +3*data['SPREADERR_MODEL_I']) > -0.003)))
    modest[starcut] = 3
    
    neither = -(galcut | starcut)
    modest[neither] = 5
    
    data = rf.append_fields(data, 'modtype', modest)
    print len(data), np.sum(galcut), np.sum(starcut), np.sum(neither)
    return data

def DESclassifier(  des):
    des = modestify(des)
    gal_cut = (des['modtype'] == 1)
    star_cut = (des['modtype'] == 3)
    neither = (des['modtype'] == 5)
    return des[gal_cut], des[star_cut]

def match(  sdss, des):
    h = esutil.htm.HTM(10)
    matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(des['RA'], des['DEC'], sdss['RA'],sdss['DEC'],matchDist,maxmatch=1)
    
    print 'matched object ',len(m_des)
    return sdss[m_sdss],des[m_des]
