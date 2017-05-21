#import easyaccess as ea
import esutil
import sys
import os
#import healpy as hp
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
#import seaborn as sns

from ang2stripe import *
import fitsio
from fitsio import FITS, FITSHDR



def priorCut(data):
    modelmag_g_des = data['MAG_DETMODEL_G']
    modelmag_r_des = data['MAG_DETMODEL_R']
    modelmag_i_des = data['MAG_DETMODEL_I']
    cmodelmag_g_des = data['MAG_MODEL_G']
    cmodelmag_r_des = data['MAG_MODEL_R']
    cmodelmag_i_des = data['MAG_MODEL_I']
    magauto_des = data['MAG_AUTO_I']

    cut = (((cmodelmag_r_des > 17) & (cmodelmag_r_des <24)) &
           ((cmodelmag_i_des > 17) & (cmodelmag_i_des <24)) &
           ((cmodelmag_g_des > 17) & (cmodelmag_g_des <24)) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) & # 10122 (95%)
           ((modelmag_r_des - modelmag_i_des ) > 0. ) & # 10120 (95%)
           ((modelmag_g_des - modelmag_r_des ) > 0. ) & # 10118 (95%)
           ((modelmag_g_des - modelmag_r_des ) < 2.5 ) & # 10122 (95%)
           (magauto_des < 21. ) #&  10124 (95%)
        )
    return cut




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


def modestify(data, suffix='_corrected'):
    #from Eric's code
    
    modest = np.zeros(len(data), dtype=np.int32)
    
    galcut = (data['FLAGS_I'] <=3) & -( ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'+suffix] < 18.0)) | ((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) | ((data['MAG_PSF_I'] > 30.0) & (data['MAG_AUTO_I'+suffix] < 21.0)))
    modest[galcut] = 1
    
    starcut = (data['FLAGS_I'] <=3) & ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'+suffix] < 18.0) & (data['MAG_PSF_I'] < 30.0) | (((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) & ((data['SPREAD_MODEL_I'] +3*data['SPREADERR_MODEL_I']) > -0.003)))
    modest[starcut] = 3
    
    neither = -(galcut | starcut)
    modest[neither] = 5
    
    #data = rf.append_fields(data, 'MODETYPE', modest)
    print np.sum(galcut), np.sum(starcut), np.sum(neither)
    return modest


def _keepGoodRegion(des, hpInd = False, balrog=None):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    
    path = '/n/des/lee.5922/data/balrog_cat/'

    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    fraction = hp.read_map(path+'Y1A1_WIDE_frac_combined_griz_o.4096_t.32768_EQU.fits')
    
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)

    ind_good_ring = np.where(( gdmask >= 1)
                             & ((bdmask.astype('int64') & (64+32+8)) == 0)
                             & (fraction > 0.8)
                             )
    
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    phi = des['RA'] * np.pi / 180.0
    theta = ( 90.0 - des['DEC'] ) * np.pi/180.0

    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    des = des[keep]
    if hpInd is True:
        return ind_good_ring
    else:
        return des


    
    

def keepGoodRegion(des, hpInd = False, balrog=None):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    
    path = '/n/des/lee.5922/data/systematic_maps/'
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_il22_seeil4.0_nside4096ring_redlimcut.fits')
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_redlimcut_il22_seeil4.0_4096ring.fits')
    #Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits
    frac_cut = LSSGoldmask['FRAC'] > 0.8
    ind_good_ring = LSSGoldmask['PIXEL'][frac_cut]
    
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    phi = des['RA'] * np.pi / 180.0
    theta = ( 90.0 - des['DEC'] ) * np.pi/180.0

    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd, ind_good_ring)
    des = des[keep]
    if hpInd is True:
        return ind_good_ring
    else:
        return des


def limitingDepth(catalog, nside = 4096):

    print "Don't use this function"
    import healpy as hp
    Npix = hp.nside2npix(nside)
    path = '/n/des/lee.5922/data/balrog_cat/'
    mask = np.ones(Npix, dtype = bool)
    for f in ['g','r','i','z']:
        maglim = hp.read_map(path+'Y1A1_SPT_and_S82_IMAGE_SRC_band_'+f+'_nside4096_oversamp4_maglimit__.fits.gz')
        mask = mask * (maglim > 22 )

    ind_good_ring = np.where(mask)
    nside=4096
    phi = catalog['RA'] * np.pi / 180.0
    theta = ( 90.0 - catalog['DEC'] ) * np.pi/180.0
    
    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    return catalog[keep]



def doGoldBasicCuts(des, raTag = 'RA', decTag = 'DEC', object = 'galaxy'):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    print "Don't use this function"
    """
    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    fraction = hp.read_map(path+'Y1A1_WIDE_frac_combined_griz_o.4096_t.32768_EQU.fits')
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)
    
    mask = np.ones(fraction.size, dtype = bool)
    for f in ['g','r','i','z']:
        maglim = hp.read_map(path+'Y1A1_SPT_and_S82_IMAGE_SRC_band_'+f+'_nside4096_oversamp4_maglimit__.fits.gz')
        mask = mask * (maglim > 22 )

    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) & (fraction > 0.8) & mask )
    """
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    
    phi = des[raTag] * np.pi / 180.0
    theta = ( 90.0 - des[decTag] ) * np.pi/180.0

    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    des = des[keep]
    
    #modtype = modestify(des)
    modtype = des['MODEST_CLASS']
    use = np.ones(des.size, dtype=bool)

    if object is 'galaxy': use = use & (modtype == 1)
    elif object is 'star'  : use = use & (modtype == 2)
    elif object is None : print 'no object selected. retrieve star + galaxy both'
    
    taglist = ['MAG_MODEL'] #, 'MAG_DETMODEL','MAG_AUTO', 'MAG_PETRO', 'MAG_PSF', 'MAG_HYBRID']
    filters = ['R','I']
    
    # cut for deleting outlier
    for tag in taglist:
        for thisfilter in filters:
            thistag = tag+'_'+thisfilter
            use = use & ((des[thistag] < 25.0) & (des[thistag] > 10.))

    print 'do Basic Cut', np.sum(use)
    return des[use]





def _doBasicCuts(des, raTag = 'RA', decTag = 'DEC', balrog=None, object = 'galaxy', suffix='_corrected'):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
     
    use = ((des['FLAGS_G'] < 3) &
           (des['FLAGS_R'] < 3) &
           (des['FLAGS_I'] < 3) &
           (des['FLAGS_Z'] < 3))
    
    """
    taglist = ['MAG_MODEL'] 
    filters = ['R','I']

    # cut for deleting outlier
    for tag in taglist:
       for thisfilter in filters:
           thistag = tag+'_'+thisfilter+suffix
           use = use & ((des[thistag] < 25.0) & (des[thistag] > 10.))
    """
    
    taglist2 = ['MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5', 'MAG_APER_6']

    filters = ['G', 'R','I', 'Z']
    for tag in taglist2:
        for thisfilter in filters:
            thistag = tag+'_'+thisfilter+suffix
            use = use & ((des[thistag] < 30.0) & (des[thistag] > 15.))


    print 'do Basic Cut', np.sum(use)
    return des[use]





def doBasicCuts(des, raTag = 'RA', decTag = 'DEC', balrog=None, object = 'galaxy', suffix=''):
    import healpy as hp
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    
    path = '/n/des/lee.5922/data/balrog_cat/'
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_il22_seeil4.0_nside4096ring_redlimcut.fits')
    ind_good_ring = LSSGoldmask['PIXEL']
    
    
    
    #maglim = hp.read_map(path+'Y1A1_SPT_and_S82_IMAGE_SRC_band_i_nside4096_oversamp4_maglimit__.fits.gz')
    #mask = maglim > 21.
    
    """
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    fraction = hp.read_map(path+'Y1A1_WIDE_frac_combined_griz_o.4096_t.32768_EQU.fits')
    LSSmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    
    mask = np.ones(fraction.size, dtype = bool)
    
    for f in ['g','r','i','z']:
        maglim = hp.read_map(path+'Y1A1_SPT_and_S82_IMAGE_SRC_band_'+f+'_nside4096_oversamp4_maglimit__.fits.gz')
        mask = mask * (maglim > 22 )
    
    
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)

    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) & (fraction > 0.8) & mask)
    """
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    

    phi = des[raTag] * np.pi / 180.0
    theta = ( 90.0 - des[decTag] ) * np.pi/180.0
    
    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    des = des[keep]
    
    use = ((des['FLAGS_G'] <= 3) &
           (des['FLAGS_R'] <= 3) &
           (des['FLAGS_I'] <= 3) &
           (des['FLAGS_Z'] <= 3))

    
    if object is 'galaxy': 
        #modtype = modestify(des, suffix=suffix)
        modtype = des['MODEST_CLASS']
        use = use & (modtype == 1)
    elif object is 'star' :
        #modtype = modestify(des, suffix=suffix)
        modtype = des['MODEST_CLASS']
        use = use & (modtype == 3)
    elif object is None : print 'no object selected. retrieve star + galaxy both'
    
    taglist = ['MAG_MODEL'] #, 'MAG_DETMODEL','MAG_AUTO', 'MAG_PETRO', 'MAG_PSF', 'MAG_HYBRID']
    filters = ['R','I']

    # cut for deleting outlier
    for tag in taglist:
        for thisfilter in filters:
            thistag = tag+'_'+thisfilter+suffix
            use = use & ((des[thistag] < 25.0) & (des[thistag] > 10.))
           
    taglist2 = ['MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5', 'MAG_APER_6']

    filters = ['G', 'R','I', 'Z']
    for tag in taglist2:
        for thisfilter in filters:
            thistag = tag+'_'+thisfilter+suffix
            use = use & ((des[thistag] < 25) & (des[thistag] > 15.))
            #use = use & (des[thistag] != 99)

    print 'do Basic Cut', np.sum(use)
    return des[use]


def doBasicSDSSCuts(sdss):
    # (Reid 2016 Section 2.2)
    # photometric quality flags
    import healpy as hp
    # bad region mask (DES footprint)
    path = '/n/des/lee.5922/data/balrog_cat/'
    LSSGoldmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    ind_good_ring = LSSGoldmask['PIXEL']
    
    """
    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)

    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) )
    """
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    phi = sdss['RA'] * np.pi / 180.0
    theta = ( 90.0 - sdss['DEC'] ) * np.pi/180.0


    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd,ind_good_ring)
    sdss  = sdss[keep]


    # quality cut ( Reid et al. 2016 Section 2.2 )
    exclude = 2**1 + 2**5 + 2**7 + 2**11 + 2**19 # BRIGHT, PEAK CENTER, NO PROFILE, DEBLENDED_TOO_MANY_PEAKS, NOT_CHECKED
    # blended object
    blended = 2**3
    nodeblend = 2**6
    # obejct not to be saturated
    saturated = 2**18
    saturated_center = 2**(32+11)
    
    use =  ( 
            (sdss['CLEAN'] == 1 ) &
            #(sdss['FIBER2MAG_I'] < 22.5) &
            (sdss['TYPE'] == 3) &
           ( ( sdss['FLAGS'] & exclude) == 0) &
           ( ((sdss['FLAGS'] & saturated) == 0) | (((sdss['FLAGS'] & saturated) > 0) & ((sdss['FLAGS'] & saturated_center) == 0) ) )&
           ( ((sdss['FLAGS'] & blended) == 0 ) | ((sdss['FLAGS'] & nodeblend) ==0) ) )
    
    """
    
    # Cuts Ashley used
    binned = 1879048192
    blending = 8
    bright =2
    edge = 4
    saturated = 2**18
    
    use = (
              #  ((sdss['FLAGS'] & binned) > 0) |
           ((sdss['FLAGS'] & blending) < 8) &
           ((sdss['FLAGS'] & bright) == 0) &
           ((sdss['FLAGS'] & edge) == 0) &
           ((sdss['FLAGS'] & saturated) == 0)
              )
    """
    return sdss[use] # & clear] # & completness95]

def CMASSQaulityCut( sdss ):

    # quality cut ( Reid et al. 2016 Section 2.2 )
    exclude = 2**1 + 2**5 + 2**7 + 2**11 + 2**19 # BRIGHT, PEAK CENTER, NO PROFILE, DEBLENDED_TOO_MANY_PEAKS, NOT_CHECKED
    # blended object
    blended = 2**3
    nodeblend = 2**6
    edge = 2**2
    # obejct not to be saturated
    saturated = 2**18
    saturated_center = 2**(32+11)
    
    # Ashley
    use1 = (
           (sdss['CLEAN'] == 1 ) &
           #(sdss['TYPE'] == 3) &
           ((sdss['FLAGS'] & edge ) == 0) &
           (~((sdss['FLAGS'] & 88 ) == 8)) &
           ((sdss['FLAGS'] & 2 ) == 0) &
           ((sdss['FLAGS'] & edge ) == 0) &
           ((sdss['FLAGS'] & saturated ) == 0) &
           (~((sdss['FLAGS'] & 1879048192 ) == 0))
           )
    # Reid
    use =  (
            (sdss['CLEAN'] == 1 ) &
            #(sdss['FIBER2MAG_I'] < 21.5) &
            (sdss['TYPE'] == 3) & ((sdss['FLAGS'] & edge ) == 0) &
            ( ( sdss['FLAGS'] & exclude) == 0) &
            ( ((sdss['FLAGS'] & saturated) == 0) | (((sdss['FLAGS'] & saturated) > 0) & ((sdss['FLAGS'] & saturated_center) == 0) ) )&
            ( ((sdss['FLAGS'] & blended) == 0 ) | ((sdss['FLAGS'] & nodeblend) ==0) ) )

    print np.sum(~use), " objects excluded"
    return sdss[use]


def SpatialCuts(  data, ra = 350.0, ra2=355.0 , dec= 0.0 , dec2=1.0 ):
    
    try :
        ascen = data['RA']
        decli = data['DEC']

    except (ValueError, NameError) :
        print "Can't RA and DEC column. Try with 'ALPHAWIN_J2000_DET' and 'DELTAWIN_J2000_DET' column"
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


