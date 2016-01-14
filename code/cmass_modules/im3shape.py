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

def im3mag_desmag_correction(  im3shape, des ):
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    #neither = - (expcut|devcut)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    #des_star = des[neither]
    
    im3shape_exp, des_exp = match(im3shape, des_exp)
    im3shape_dev, des_dev = match(im3shape, des_dev)
    
    zeropoint = des_exp['MAG_MODEL_I'] - im3shape_exp['MAG_MODEL_I']
    
    for i in range(20):
        print im3shape_exp['DISC_FLUX'][i], des_exp['FLUX_MODEL_I'][i],im3shape_exp['DISC_FLUX'][i]/des_exp['FLUX_MODEL_I'][i]
    
    fig, (ax, ax2) = plt.subplots(nrows =1, ncols =2, figsize = (14, 7))
    ax.plot( im3shape_exp['MAG_MODEL_I'], des_exp['MAG_MODEL_I'], 'r.', label = 'exp_model', alpha = 0.33 )
    ax.plot( im3shape_exp['MAG_MODEL_I'], des_exp['MAG_DETMODEL_I'], 'b.', label = 'exp_detmodel',alpha = 0.33 )
    ax2.plot( im3shape_dev['MAG_MODEL_I'], deos_dev['MAG_MODEL_I'], 'g.', label = 'dev_model', alpha = 0.33 )
    ax2.plot( im3shape_dev['MAG_MODEL_I'], des_dev['MAG_DETMODEL_I'], 'y.', label = 'dev_detmodel',alpha = 0.33 )
    ax.plot([0,30],[0,30],'--',color='red')
    ax.set_xlim(0,30)
    ax.set_xlabel('exp_im3shape_i')
    ax.set_ylabel('des_i')
    ax.set_ylim(0,30)
    ax.legend(loc='best')
    ax2.plot([0,30],[0,30],'--',color='red')
    ax2.set_xlim(0,30)
    ax2.set_xlabel('dev_im3shape_i')
    ax2.set_ylabel('des_i')
    ax2.set_ylim(0,30)
    ax2.legend(loc='best')


    fig.savefig('../figure/fluxcomparison_im3shapeVSdes')
    print 'savefig :', '../figure/fluxcomparison_im3shapeVSdes'

    #return im3sorted, DESsorted



def im3shape_add_radius(  im3shape, fulldes ):
    
    im3radius = np.zeros(len(im3shape), dtype=np.int32)
    
    common_cut = np.in1d(fulldes['COADD_OBJECTS_ID'], im3shape['COADD_OBJECTS_ID'])
    im3radius[common_cut] = im3shape['RADIUS']
    
    data = rf.append_fields(fulldes, 'IM3_RADIUS', im3radius)
    return data



def im3shape_galprof_mask(  im3shape, fulldes ):
    """ add mode to distingush galaxy profiles used in im3shape """
    """ return only full des """
    
    im3galprofile = np.zeros(len(im3shape), dtype=np.int32)
    
    expcut = (im3shape['BULGE_FLUX'] == 0)
    devcut = (im3shape['DISC_FLUX'] == 0)
    neither = - (expcut|devcut)
    
    des_exp = im3shape[expcut]
    des_dev = im3shape[devcut]
    #des_star = im3shape[neither]
    
    expID = des_exp['COADD_OBJECTS_ID']
    devID = des_dev['COADD_OBJECTS_ID']
    fullID = fulldes['COADD_OBJECTS_ID']
    
    expmask = np.in1d(fullID, expID)
    devmask = np.in1d(fullID, devID)
    
    im3galprofile[expmask] = 1
    im3galprofile[devmask] = 2
    
    data = rf.append_fields(fulldes, 'IM3_GALPROF', im3galprofile)
    
    print np.sum(expcut), np.sum(devcut), np.sum(neither)
    return data

def addMagforim3shape(  im3_des):

    zeromag = 25.
    mag = zeromag - 2.5 * np.log10((im3_des['DISC_FLUX'] + im3_des['BULGE_FLUX']) * im3_des['MEAN_FLUX'])
    im3_des = rf.append_fields(im3_des, 'MAG_MODEL_I', mag)
    
    return im3_des