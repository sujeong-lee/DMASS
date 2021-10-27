import os, sys, scipy
import esutil
import healpy as hp
import numpy as np
%matplotlib inline

# call required functions from modules 
sys.path.append('code_py3/')
from cmass_modules import io
from utils import matchCatalogsbyPosition, hpHEALPixelToRaDec, HealPixifyCatalogs, spatialcheck
from xd import mixing_color, XD_fitting_X, assignCMASSProb, doVisualization_1d
from run_DMASS_Y3 import priorCut_test

# Calling Y1 GOLD v2.0 (training)
# All catalogs are in the 'input_path' directory 
# The 'SearchAndCallFits' function below loads all 
# catalogs in the directory including 'input_keyword' in its name
input_path = '/n/des/lee.5922/data/gold_cat/GOLD_STRIPE_82/'
# call only first 9 catalogs for a fast run.
# to call all catalogs in the directory, use 'Y3_GOLD' as input_keyword 
# but that will consume huge memory
input_keyword = 'Y1A1_GOLD_STRIPE82_v2'  
# Columns to call
columns =  ['RA', 'DEC', 
            'HPIX_4096',     # Healpix in ring order, nside=4096
            'COADD_OBJECT_ID', 
            'SOF_CM_MAG_CORRECTED_G', # mag_[griz]
            'SOF_CM_MAG_CORRECTED_R',
            'SOF_CM_MAG_CORRECTED_I',
            'SOF_CM_MAG_CORRECTED_Z',
            'SOF_CM_MAG_ERR_G',       # mag error_[griz]
            'SOF_CM_MAG_ERR_R',
            'SOF_CM_MAG_ERR_I',
            'SOF_CM_MAG_ERR_Z']
gold_st82 = io.SearchAndCallFits(path = input_path, keyword = input_keyword, columns=columns)

# Color/Magnitude cuts to exclude extremely high or low mag/color sources.
# 16 < mag_riz < 24, 0 < (r-i) < 1.5, 0 < (g-r) < 2.5
# These galaxies are less likeliy to be a CMASS, therefore unnecessary. 
# We apply these cuts to reduce the sample size to speed up the codes
mask_magcut = priorCut_test(gold_st82)
gold_st82 = gold_st82[mask_magcut]

# cmass catalog is stored in cosmos machine
cmass_filename = '/n/des/lee.5922/data/cmass_cat/cmass-dr12v4-S-Reid-full.dat.fits'
cmass = esutil.io.read(cmass_filename)

# Add healpix index (nside=4096) to the CMASS catalog
healConfig = {'map_inside':4096,
              'out_nside':4096,
              'nest':False}
cmass = HealPixifyCatalogs(catalog=cmass, healConfig=healConfig, ratag='RA', dectag = 'DEC')

# this function returns indices of common galaxies (CMASS) in each catalog. 
# mg1: indices of common galaxies in cmass catalog
# mg2: indicies of common galaxies in Y3 GOLD catalog 
mg1, mg2, _ = esutil.htm.HTM(10).match(cmass['RA'], cmass['DEC'], gold_st82['RA'], \
                                     gold_st82['DEC'],2./3600, maxmatch=1)
# Apply indices to each catalog to select common galaxies. 
# Selected galaxies are then ordered in the same way. 
# (i.e clean_cmass_data_sdss[0] and clean_cmass_data_des[0] are the same galaxy) 
clean_cmass_data_sdss = cmass[mg1]           # common galaxies in cmass catalog
clean_cmass_data_des = gold_st82[mg2] # common galaxies in Y1 Gold

# We also need non-CMASS galaxies in the DES Y3 gold catalog
cmass_mask = np.zeros(gold_st82.size, dtype=bool)
cmass_mask[mg2] = 1
nocmass = gold_st82[~cmass_mask]

print(('num of cmass in des side', clean_cmass_data_des.size, '({:0.0f}%)'.format(clean_cmass_data_des.size*1./cmass.size*100.)))
-------------------------------

#matching catalogs (train_st82):
'''
#Su's flags and color cut:
mask_all=cut
f'=f[cut]
train_sample=fitsfile
#print('total num of train', len(train_sample))
'''
#reading in needed values:
cmass_ra=hdu[1].data['RA     ']
cmass_dec=hdu[1].data['DEC    ']

mg1, mg2,_= esutil.htm.HTM(10).match(cmass_ra, cmass_dec, ra, dec, 2./3600, maxmatch=1)

cmass_mask=np.zeros(ra.size, dtype=bool)
cmass_mask[mg2]=1
clean_cmass_data_des_ra, nocmass_ra=ra[cmass_mask], ra[~cmass_mask]
clean_cmass_data_des_dec, nocmass_dec=dec[cmass_mask], dec[~cmass_mask]

print(('num of cmass in des side', clean_cmass_data_des_ra.size, '({:0.0f}%)'.format(clean_cmass_data_des_ra.size*1./cmass_ra.size*100.)))

---------------------

import esutil
import numpy as np
    
mask_all = priorCut_test(gold_st82)
gold_st82 = gold_st82[mask_all]
    
# calling BOSS cmass and applying dmass goodregion mask ----------------------------
#cmass = io.getSGCCMASSphotoObjcat()
train_sample = esutil.io.read(train_sample_filename)

print('total num of train', train_sample.size)

print('\n--------------------------------\n applying DES veto mask to CMASS\n--------------------------------')   
    
train_sample = Cuts.keepGoodRegion(train_sample)
fitsio.write( output_dir+'/cmass_in_st82.fits', train_sample)

print('num of train_sample after des veto', train_sample.size)

print('\n--------------------------------\n matching catalogues\n--------------------------------')

# find cmass in des_gold side --------------------
mg1, mg2, _ = esutil.htm.HTM(10).match(cmass_ra, cmass_dec, ra, \
                                         dec,2./3600, maxmatch=1)
cmass_mask = np.zeros(ra.size, dtype=bool)
cmass_mask[mg2] = 1
clean_cmass_data_des_ra, nocmass_ra = ra[cmass_mask], ra[~cmass_mask]
clean_cmass_data_des_dec, nocmass_dec = dec[cmass_mask], dec[~cmass_mask]


print('num of cmass in des side',clean_cmass_data_des_ra.size,'({:0.0f}%)'.format(clean_cmass_data_des_ra.size*1./cmass_ra.size*100))