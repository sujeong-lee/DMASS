#import easyaccess as ea
import esutil
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy.lib.recfunctions as rf
#import seaborn as sns
import fitsio
from fitsio import FITS, FITSHDR


def SearchFitsByName(path = None, keyword = None, no_keyword=None, columns = None):
    import os, sys
    
    print '\n--------------------------------\n Existing catalog\n--------------------------------'
    
    if no_keyword is None : 
        tables = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and keyword in i:
                tables.append(path+i)
                print i
                sys.stdout.flush()

    else : 
        tables = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and keyword in i:
                if no_keyword not in i:
                    tables.append(path+i)
                    print i
                    sys.stdout.flush()

                else: pass

    #data = esutil.io.read(tables, columns = columns, combine = True)
    return tables


def SearchAndCallFits(path = None, keyword = None, no_keyword=None, columns = None):
    import os, sys
    print no_keyword
    stop
    print '\n--------------------------------\n calling catalog\n--------------------------------'
    
    if no_keyword is None : 
        tables = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and keyword in i:
                #if no_keyword not in i:
                tables.append(path+i)
                print i
                sys.stdout.flush()

    elif no_ieyword is not None : 
        print "not none"
        tables = []
        for i in os.listdir(path):
            if os.path.isfile(os.path.join(path,i)) and keyword in i:
                if no_keyword in i: 
                    pass
                elif no_keyword not in i:
                    tables.append(path+i)
                    print i
                    sys.stdout.flush()

               #else: pass

    data = esutil.io.read(tables, columns = columns, combine = True)
    return data

def getSGCCMASSphotoObjcat():
    
    print '\n--------------------------------\n calling BOSS SGC CMASS catalog\n--------------------------------'
    
    import esutil
    import numpy as np
    
    path = '/n/des/lee.5922/data/cmass_cat/'
    """
    cmass_photo = esutil.io.read(path+'bosstile-final-collated-boss2-boss38.fits.gz', upper=True)
    cmass_specprimary = esutil.io.read(path+'bosstile-final-collated-boss2-boss38-specObj.fits.gz', upper=True)
    cmass_zwarning_noqso = esutil.io.read(path+'bosstile-final-collated-boss2-boss38-photoObj-specObj.fits.gz')
    
    use2 = (( (cmass_photo['BOSS_TARGET1'] & 2) != 0 )
           #(cmass_specprimary['SPECPRIMARY'] == 1) &
           #(cmass_zwarning_noqso['ZWARNING_NOQSO'] == 0 )
           )
    """
    cmass = esutil.io.read(path+'boss_target_selection.fits', upper=True)
    use = (( cmass['BOSS_TARGET1'] & 2 != 0 ) &
           ((cmass['CHUNK'] != 'boss1' ) & (cmass['CHUNK'] != 'boss2')) &
           ((cmass['FIBER2MAG_I'] - cmass['EXTINCTION_I']) < 21.5 )
            )
           
    cmass = cmass[use]
    
    _, ind = np.unique(cmass['OBJID'], return_index=True, return_inverse=False, return_counts=False)
    no_duplicate_mask = np.zeros(cmass.size, dtype=bool)
    no_duplicate_mask[ind] = 1

    cmass = cmass[no_duplicate_mask]

    print "Applying Healpix BOSS SGC footprint mask"
    print "Change healpix mask to spatial cut later..... Don't forget!!! "
    HPboss = esutil.io.read('/n/des/lee.5922/data/cmass_cat/healpix_boss_footprint_SGC_1024.fits')
    from systematics import hpRaDecToHEALPixel
    HealInds = hpRaDecToHEALPixel( cmass['RA'],cmass['DEC'], nside= 1024, nest= False)
    BOSSHealInds = np.in1d( HealInds, HPboss )    
    return cmass[BOSSHealInds]



def getSDSScatalogs(  file = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit', bigSample = False):
    
    
    if bigSample is True:
        filepath = '/n/des/lee.5922/data/'
        #file = filepath+'sdss_ra330_10.fit'
        
        sdss_files = [filepath+'sdss_ra340_345_dec1_0_sjlee_0.fit',
                      filepath+'sdss_ra340_345_decm1_0_sjlee_0.fit',
                      filepath+'sdss_ra345_350_decm1_0_sjlee_0.fit',
                      filepath+'sdss_ra345_350_dec1_0_sjlee_0.fit',
                      filepath+'sdss_ra355_360_decm1_0_sjlee.fit',
                      filepath+'sdss_ra355_360_dec1_0_sjlee.fit',
                      #filepath +'st82_355_360_SujeongLee_0.fit',
                      '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit',
                      '/n/des/huff.791/Projects/CMASS/Data/S82_SDSS_0_10.fit'
                      ]
        sdss_data = esutil.io.read(sdss_files,combine=True)
        
        #esutil.io.write('sdss_ra330_10.fit',sdss_data)
        #sdss_data = fitsio.read(file)
    
    else:
    
        #file1 = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit'
        #file1 = '/Users/SJ/Dropbox/repositories/CMASS/data/test_emhuff.fit'
        #file3 = '../data/sdss_clean_galaxy_350_360_m05_0.fits'
        #file4 = '../data/sdss_clean_galaxy_350_351_m05_0.fits'
        sdss_data = fitsio.read(file)
        #data = esutil.io.read_header(file1,ext=1)
        
    sdss_data.dtype.names = tuple([ sdss_data.dtype.names[i].upper() for i in range(len(sdss_data.dtype.names))])
    
    return sdss_data


def getCatalogsWithKeys(keyword = None, path = None):
    
    import os, esutil, sys
    
    tables = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and keyword in i:
            tables.append(path+i)
            print i
            sys.stdout.flush()
    des_data = esutil.io.read( tables, combine=True)
    return des_data
 
    
    
def getDESY1A1catalogs(keyword = 'Y1A1', gold = False, size = None, sdssmask=True, im3shape=None):
    
    import time
    import os, sys
    
    #colortags = ['FLUX_MODEL', 'FLUX_DETMODEL', 'FLUXERR_MODEL', 'FLUXERR_DETMODEL', 'FLAGS', 'MAG_MODEL', 'MAG_DETMODEL', 'MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5','MAG_APER_6', 'XCORR_SFD98', 'MAGERR_DETMODEL', 'MAG_AUTO', 'MAG_PETRO', 'MAG_PSF' ]
    colortags = ['FLAGS', 'MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5','MAG_APER_6']
    
    filters = ['G', 'R', 'I', 'Z']
    colortags = [ colortag + '_'+filter for colortag in colortags for filter in filters ]

    sdssmasktags = ['bad_field_mask', 'unphot_mask', 'bright_star_mask', 'rykoff_bright_star_mask','collision_mask', 'centerpost_mask']

    if sdssmask is False : sdssmasktags=[]
    
    #tags = ['RA', 'DEC', 'COADD_OBJECTS_ID', 'SPREAD_MODEL_I', 'SPREADERR_MODEL_I' ,'CLASS_STAR_I', 'MAGERR_MODEL_I', 'MAGERR_MODEL_R'] + colortags + sdssmasktags
    tags = ['RA', 'DEC', 'COADD_OBJECTS_ID'] + colortags + sdssmasktags
    #tags = ['COADD_OBJECTS_ID'] + colortags + sdssmasktags
    path = '/n/des/lee.5922/data/y1a1_coadd/'
    if gold is True : path = '/n/des/lee.5922/data/gold_cat/'
    #path = '/n/des/huff.791/Projects/CMASS/Data/Stripe82/' # path for sdss veto masked cat
    tables = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and keyword in i:
            tables.append(path+i)
            print i
            sys.stdout.flush()

    rows = None
    if size is not None:
        sample = np.arange(152160)
        rows = np.random.choice( sample, size=size , replace = False)

     
    if gold is True: 
        #tags = None
        colortags = ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_AUTO', 'MAGERR_MODEL', 'MAGERR_DETMODEL']
        tags = ['RA', 'DEC', 'COADD_OBJECTS_ID', 'FLAGS_GOLD', 'MODEST_CLASS', 'DESDM_ZP', 'HPIX' ]\
        + [ c+'_'+f for f in filters for c in colortags ]

    des_data = esutil.io.read( tables, combine=True, columns = tags, rows = rows)
    
    return des_data



def LoadBalrog(user = 'JELENA', truth = None):

    import time
    import os

    colortags = ['FLUX_MODEL', 'FLAGS', 'MAG_MODEL', 'MAG_DETMODEL', 'MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5','MAG_APER_6', 'MAGERR_MODEL', 'MAGERR_DETMODEL']
    filters = ['G', 'R', 'I', 'Z']
    colortags = [ colortag + '_'+filter for colortag in colortags for filter in filters ]

    if truth is True:
        kind = 'TRUTH'
        tags = ['RA', 'DEC', 'ID', 'Z', ]

    elif truth is None:
        kind = 'SIM'
        tags = ['ALPHAWIN_J2000_DET', 'DELTAWIN_J2000_DET', 'SPREAD_MODEL_I', 'SPREADERR_MODEL_I', 'MAG_AUTO_I','CLASS_STAR_I','MAG_PSF_I'] + colortags

    path = '/n/des/lee.5922/data/balrog_cat/'
    tables = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and user.upper() in i and kind in i:
            print i
            tables.append(path+i)

    balrog_data = esutil.io.read( tables, combine=True, columns = tags)

    print "no RA and DEC columns. Use ALPHAWIN_J2000 and DELTAWIN_J2000"
    ra = balrog_data['ALPHAWIN_J2000_DET']
    dec = balrog_data['DELTAWIN_J2000_DET']
    cut = ra < 0
    ra1 = ra[cut] + 360
    ra[cut] = ra1

    print "alphaJ2000, deltaJ2000  -->  ra, dec"
    balrogname = list( balrog_data.dtype.names)
    alphaInd = balrogname.index('ALPHAWIN_J2000_DET')
    deltaInd = balrogname.index('DELTAWIN_J2000_DET')
    balrogname[alphaInd], balrogname[deltaInd] = 'RA', 'DEC'
    balrog_data.dtype.names = tuple(balrogname)

    """
    print "alphaJ2000, deltaJ2000  -->  ra, dec"
    balrogname = list( balrog_data.dtype.names)
    alphaInd = balrogname.index('ALPHAWIN_J2000_DET')
    deltaInd = balrogname.index('DELTAWIN_J2000_DET')
    balrogname[alphaInd], balrogname[deltaInd] = 'RA', 'DEC'
    balrog_data.dtype.names = tuple(balrogname)
    """
    return balrog_data

