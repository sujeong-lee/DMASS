import sys
sys.path.append('../')
from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
from systematics_module import *
import os
from numpy.lib.recfunctions import append_fields
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
from run_systematics import sys_iteration, weightmultiply, fitting_allSP, calling_sysMap
from sys_functions import *

#debugging:
#import ipdb
#ipdb.set_trace()

# -----------------------
run_name = "no_weights" #"final_linear"

#variables to set:

SPT = True
SPmap = True
custom = True

sys_weights = False

# -----------------------

# DMASS in NEST by default
if SPT == True:
    print("SPT Region...")
    dmass_spt = calling_catalog('/fs/scratch/PCON0008/warner785/bwarner/dmass_spt.fits')
    random_spt = uniform_random_on_sphere(dmass_spt, size = 10*int(np.sum(dmass_spt['CMASS_PROB']))) 
    random_spt = keepGoodRegion(random_spt)
    #random_spt = appendColumn(random_spt, value=np.ones(random_spt.size), name='WEIGHT')
    index_mask = np.argsort(dmass_spt)
    dmass_chron = dmass_spt[index_mask] # ordered by hpix values
    
else:
    print("Validation Region...")
    dmass_val = calling_catalog('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')
    #dmass_val = esutil.io.read('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')   
    random_val = uniform_random_on_sphere(dmass_val, size = 10*int(np.sum(dmass_val['CMASS_PROB'])))#larger size of randoms
    random_val = keepGoodRegion(random_val)
    #random_val = appendColumn(random_val, value=np.ones(random_val.size), name='WEIGHT')
    index_mask = np.argsort(dmass_val)
    dmass_chron = dmass_val[index_mask]
    
dmass_chron['HPIX_4096'] = hp.nest2ring(4096, dmass_chron['HPIX_4096'])
print(dmass_chron.shape)

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')

if SPT ==True:
    phi = random_spt['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_spt['DEC'] ) * np.pi/180.0

else:    
    phi = random_val['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_val['DEC'] ) * np.pi/180.0
    
random_pix = hp.ang2pix(4096, theta, phi)

frac = np.zeros(hp.nside2npix(4096))
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
frac[fracDet['PIXEL']] = fracDet['SIGNAL']
fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']

frac_obj = frac[random_pix]

u = np.random.rand(len(random_pix))
#select random points with the condition u < frac_obj
if SPT == True:
    random_spt_fracselected = random_spt[u < frac_obj]
else:
    random_val_fracselected = random_val[u < frac_obj]

dmass_chron_weights =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/iterativeALL.fits')
print(dmass_chron_weights.shape)


# mocks load in: 

mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/'
n_pca = 34
cov = []
cov_template = 'cov{0}'
for i_pca in range(n_pca): #n_pca
    cov_input= cov_template.format(i_pca)
    cov.append(cov_input)
#print(cov)

cov_template = 'covarSP34_{0}'
for i_pca in range(n_pca): #n_pca
    cov_keyword = cov_template.format(i_pca)
    #print(cov_keyword)
    with open(mock_outdir + cov_keyword + '.txt') as mocks:
        array1 = [x.split() for x in mocks]
        array2 = np.array(array1)
        print(i_pca)
        cov[i_pca] = array2.astype(float)
    mocks.close()

# -------------------------------------------------------------------------------------

#input_path = '/fs/scratch/PCON0008/warner785/bwarner/band_i/' 
input_path = '/fs/scratch/PCON0008/warner785/bwarner/'

mock_path = '/fs/scratch/PCON0008/warner785/bwarner/mocks/cats/'
mock_template = 'y3_cmass_mocks_full_catalog{0}.'

#test weighted pca-dmass on the sp checks

band = []
band_template = 'band_{0}'
fil = ('g','r','i','z')
for x in range(4):
    band_input= band_template.format(fil[x])
    band.append(band_input)
print(band) 

input_path = '/fs/scratch/PCON0008/warner785/bwarner/PCA/'
y1 = 7
y_all = 10
chi2_dmass = []

AIRMASS =('y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz')

SKYBRITE = ('y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz')
        
SIGMA_MAG_ZERO = ('y3a2_g_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz','y3a2_r_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz','y3a2_i_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz','y3a2_z_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz')
        
FGCM_GRY = ('y3a2_g_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz')
        
SKYVAR_UNCERT = ('y3a2_g_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz','y3a2_r_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz','y3a2_z_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz')
        
T_EFF_EXPTIME = ('y3a2_g_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz')
        
FWHM_FLUXRAD = ('y3a2_g_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz')

ex_dir = '/fs/scratch/PCON0008/warner785/bwarner/PCA/extinction/'
SFD98 = ("ebv_sfd98_fullres_nside_4096_nest_equatorial_des.fits.gz")

sof_dir = '/fs/scratch/PCON0008/warner785/bwarner/PCA/sof_depth/'
SOF_DEPTH = ("y3a2_gold_2_2_1_sof_nside4096_nest_g_depth.fits.gz", "y3a2_gold_2_2_1_sof_nside4096_nest_r_depth.fits.gz", "y3a2_gold_2_2_1_sof_nside4096_nest_i_depth.fits.gz", "y3a2_gold_2_2_1_sof_nside4096_nest_z_depth.fits.gz")

stars_dir = '/fs/scratch/PCON0008/warner785/bwarner/PCA/stars/'
STELLAR_DENS = ("stars_extmashsof0_16_20_zeros_footprint_nside_4096_nest.fits.gz")

maps = np.array([AIRMASS, SKYBRITE, SIGMA_MAG_ZERO, FGCM_GRY, SKYVAR_UNCERT, T_EFF_EXPTIME, FWHM_FLUXRAD, SFD98, STELLAR_DENS, SOF_DEPTH])
map_name = ['AIRMASS','SKYBRITE', 'SIGMA_MAG_ZERO', 'FGCM_GRY', 'SKYVAR_UNCERT', 'T_EFF_EXPTIME', 'FWHM_FLUXRAD', 'SFD98', 'STELLAR_DENS', 'SOF_DEPTH']
     
i_pca = -1
for y in range(y_all):
    if y<y1:
        for x in range(4):
            i_pca+=1
            full_path = input_path + band[x]+'/'
            print("path: ", full_path)
            current_map = map_name[y]+fil[x]
            current = maps[y]
            print("current map: ", current_map)
            input_keyword = current[x]
            
            pcenter, norm_number_density, error_cov, fracerr_norm = go_through_SP(full_path, input_keyword, current_map, fracHp, cov, i_pca, dmass_chron, dmass_chron_weights, sys_weights = sys_weights)
            plot_figure(input_keyword, current_map, run_name, pcenter, norm_number_density, error_cov, fracerr_norm, sys_weights = sys_weights)
                
    else:
        if y == 7:
            print("SECOND SET")
            amount = 1
            full_path = ex_dir
            print("path: ", full_path)
            current_map = map_name[y]
            print("current map: ", current_map)

        if y == 8:
            amount = 1 
            full_path = stars_dir
            print("path: ", full_path)
            current_map = map_name[y]  
            print("current map: ", current_map)
        if y == 9:
            amount = 4
        for x in range(amount):
            i_pca+=1
            if y == 9:
                full_path = sof_dir
                print("path: ", full_path)
                current_map = map_name[y]+fil[x]
                print("current map: ", current_map)
                
            current = maps[y]
            input_keyword = current[x]
                                
            pcenter, norm_number_density, error_cov, fracerr_norm = go_through_SP(full_path, input_keyword, current_map, fracHp, cov, i_pca, dmass_chron, dmass_chron_weights, sys_weights = sys_weights)
            plot_figure(input_keyword, current_map, run_name, pcenter, norm_number_density, error_cov, fracerr_norm, sys_weights = sys_weights)
                  
            chi2_, chi2_reduced = chi2(norm_number_density, np.ones(12), error_cov, 0, None, SPT = False)
            chi2_dmass.append(chi2_reduced)