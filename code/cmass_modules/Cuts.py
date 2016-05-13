#import easyaccess as ea
import esutil
import sys
import os
#import healpy as hp
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
#import seaborn as sns

from ang2stripe import *
import fitsio
from fitsio import FITS, FITSHDR

def CmassGal_in_stripe82(data):
    
    list = []
    for i in range(0, len(data)):
        ra, dec = data[i]['RA'], data[i]['DEC']
        stripe_num = ang2stripe(ra,dec)
    
        if stripe_num == 82:
            list.append(data[i])
    
        else : pass
    
    list = np.array(list)
    #selected_data = np.array(list, dtype=data.dtype)
    
    return list


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


def keepGoodRegion(des, balrog=None):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    
    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)
    
    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) )
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    
    if balrog is True:
        
        print "no RA and DEC columns. Use ALPHAWIN_J2000 and DELTAWIN_J2000"
        ra = des['ALPHAWIN_J2000_DET']
        dec = des['DELTAWIN_J2000_DET']
        cut = ra < 0
        ra1 = ra[cut] + 360
        ra[cut] = ra1
        print des['ALPHAWIN_J2000_DET']
        phi = des['ALPHAWIN_J2000_DET'] * np.pi / 180.0
        theta = ( 90.0 - des['DELTAWIN_J2000_DET'] ) * np.pi/180.0

    else:
        
        phi = des['RA'] * np.pi / 180.0
        theta = ( 90.0 - des['DEC'] ) * np.pi/180.0

    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    des = des[keep]
    return des


def doBasicCuts(des, balrog=None, object = 'galaxy'):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    
    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)

    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) )
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    
    if balrog is True:
    
        print "no RA and DEC columns. Use ALPHAWIN_J2000 and DELTAWIN_J2000"
        ra = des['ALPHAWIN_J2000_DET']
        dec = des['DELTAWIN_J2000_DET']
        cut = ra < 0
        ra1 = ra[cut] + 360
        ra[cut] = ra1
        print des['ALPHAWIN_J2000_DET']
        phi = des['ALPHAWIN_J2000_DET'] * np.pi / 180.0
        theta = ( 90.0 - des['DELTAWIN_J2000_DET'] ) * np.pi/180.0
    
    else:
    
        phi = des['RA'] * np.pi / 180.0
        theta = ( 90.0 - des['DEC'] ) * np.pi/180.0
    
    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    des = des[keep]
    
    des = modestify(des)
    
    use = ((des['FLAGS_G'] < 3) &
           (des['FLAGS_R'] < 3) &
           (des['FLAGS_I'] < 3) &
           (des['FLAGS_Z'] < 3))  #& (des['MODETYPE'] == 1)

    if object is 'galaxy': use = use & (des['MODETYPE'] == 1)
    elif object is 'star'  : use = use & (des['MODETYPE'] == 3)
    elif object is None : print 'no object selected. retrieve star + galaxy both'
    
    taglist = ['MAG_MODEL'] #, 'MAG_DETMODEL','MAG_AUTO', 'MAG_PETRO', 'MAG_PSF', 'MAG_HYBRID']
    filters = ['R','I']

    # cut for deleting outlier
    for tag in taglist:
       for thisfilter in filters:
           thistag = tag+'_'+thisfilter
           use = use & ((des[thistag] < 25.0) & (des[thistag] > 10.))

    taglist2 = ['MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5', 'MAG_APER_6']

    for tag in taglist2:
        for thisfilter in filters:
            thistag = tag+'_'+thisfilter
            use = use & ((des[thistag] < 25.0) & (des[thistag] > 15.))


    print 'do Basic Cut', np.sum(use)
    return des[use]


def doBasicSDSSCuts(sdss):
    exclude = 2**1 + 2**11 + 2**5 + 2**19 + 2**5 + 2**19 + 2**7
    blended = 3 # 2**3
    nodeblend = 6 #2**6
    saturated = 18 #2**18
    saturated_center = 2**(32+11)
    use = ( (sdss['CLEAN'] == 1 ) & (sdss['FIBER2MAG_I'] < 25) &
           (sdss['TYPE'] == 3) &
           ( ( sdss['FLAGS'] & exclude) == 0) &
           ( ((sdss['FLAGS'] & saturated) == 0) | (((sdss['FLAGS'] & saturated) >0) & ((sdss['FLAGS'] & saturated_center) == 0)) ) &
           ( ((sdss['FLAGS'] & blended) ==0 ) | ((sdss['FLAGS'] & nodeblend) ==0) ) )
           
    """
    # Covey et al. 2007 Clear SDSS photometry
    deblended_as_moving =2**(32+0)
    primary = 2**0
    edge = 2**2
    psf_flux_interp = 2**(32+15)
    interp_center = 2**(32+12)
    
    clear = ((sdss['FLAGS'] & deblended_as_moving == 0) & (sdss['FLAGS'] & edge == 0) &
             (sdss['FLAGS'] & psf_flux_interp == 0) & (sdss['FLAGS'] & interp_center == 0) &
             (sdss['RESOLVESTATUS'] & primary == 1) )
    
    """
    return sdss[use] # & clear] # & completness95]

def SpatialCuts(  data, ra = 350.0, ra2=355.0 , dec= 0.0 , dec2=1.0 ):
    
    try :
        ascen = data['RA']
        decli = data['DEC']

    except (ValueError, NameError) :
        print "Can't RA and DEC column. Try with 'ALPHAWIN_J2000_DET' column"
        ascen = data['ALPHAWIN_J2000_DET']
        decli = data['DELTAWIN_J2000_DET']
        
        ascen1 = ascen[ascen < 0] + 360
        ascen[ascen<0] = ascen1


    cut =((ascen < ra2) &
      (decli < dec2) &
      (ascen > ra) &
      (decli > dec))

    print 'Spatial Cut ', np.sum(cut)
    return data[cut]


def RedshiftCuts(  data, z_min=0.43, z_max=0.7 ):
    
    cut = ((data['Z'] < z_max ) &
           (data['Z'] > z_min))
    print 'Redshift Cut ', np.sum(cut)
    return data[cut]



def whichGalaxyProfile(sdss):

    exp_L = np.exp(np.array([sdss['LNLEXP_G'],sdss['LNLEXP_R'],sdss['LNLEXP_I'],sdss['LNLEXP_Z']])).T
    dev_L = np.exp(np.array([sdss['LNLDEV_G'],sdss['LNLDEV_R'],sdss['LNLDEV_I'],sdss['LNLDEV_Z']])).T
    star_L = np.exp(np.array([sdss['LNLSTAR_G'],sdss['LNLSTAR_R'],sdss['LNLSTAR_I'],sdss['LNLSTAR_Z']])).T

    expfracL = exp_L /(exp_L + dev_L + star_L)
    devfracL = dev_L /(exp_L + dev_L + star_L)
    
    modelmode = np.zeros((len(sdss), 4), dtype=np.int32)

    expmodel = (expfracL > 0.5)
    modelmode[expmodel] = 0
    devmodel = (devfracL > 0.5)
    modelmode[devmodel] = 1
    neither = - (expmodel | devmodel)
    modelmode[neither] = 2
    
    sdss = rf.append_fields(sdss, 'BESTPROF_G', modelmode[:,0])
    sdss = rf.append_fields(sdss, 'BESTPROF_R', modelmode[:,1])
    sdss = rf.append_fields(sdss, 'BESTPROF_I', modelmode[:,2])
    sdss = rf.append_fields(sdss, 'BESTPROF_Z', modelmode[:,3])
    
    #print ' exp :', np.sum(expmodel),' dev :', np.sum(devmodel), 'neither :', np.sum(neither)
    return sdss


