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
run_name = "vali_newweighttrue" #newweight newweight
sys_weight = "vali19"#'final27'

#variables to set:

SPT = False
SPmap = True
custom = True

sys_weights = True #True
equal_area = True
maglim = False
DMASS_area = True

# -----------------------
#https://zenodo.org/records/8207175/files/csfd_ebv.fits?download=1
# DMASS in NEST by default, maglim as well
if SPT == True:
    print("SPT Region...")
    if maglim !=True:
        #PROB CUT:
        dmass_spt = calling_catalog('/fs/scratch/PCON0008/warner785/bwarner/dmass_spt.fits')
        random_chron = esutil.io.read('/users/PCON0003/warner785/DMASSY3/code_py3/random_spt_chron.fits')
        
        #NO PROB CUT:
        #dmass_spt = esutil.io.read('/fs/scratch/PCON0008/warner785/bwarner/dmass_spt.fits')
# CHECKING MAGLIM ----------------------------------------------------------------------------------
    else:
        dmass_spt = esutil.io.read('/fs/project/PCON0008/des_y3/maglim/mag_lim_lens_sample_combined_jointmask_sample.fits.gz')
    # for maglim only:
        dmass_spt = appendColumn(dmass_spt, value=np.ones(dmass_spt.size), name='CMASS_PROB')
        nside = 4096
        phi = dmass_spt['RA'] * np.pi / 180.0
        theta = ( 90.0 - dmass_spt['DEC'] ) * np.pi/180.0
        hpInd = hp.ang2pix(nside,theta,phi,nest=True)
        dmass_spt = appendColumn(dmass_spt, value=hpInd, name='HPIX_4096')
        dmass_spt = cutPCA(dmass_spt, SPT, SPmap = False, maglim = True, cat = True)
# --------------------------------------------------------------------------------------------------    
    index_mask = np.argsort(dmass_spt)
    dmass_chron = dmass_spt[index_mask] # ordered by hpix values
    
else:
    print("Validation Region...")
    dmass_val = calling_catalog('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')
    random_chron = esutil.io.read('/fs/scratch/PCON0008/warner785/bwarner/june23_validation/random_val_chron.fits')
    #dmass_val = esutil.io.read('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')   
    index_mask = np.argsort(dmass_val)
    dmass_chron = dmass_val[index_mask]
    
dmass_chron['HPIX_4096'] = hp.nest2ring(4096, dmass_chron['HPIX_4096'])
print(dmass_chron.shape)

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')

if maglim != True or DMASS_area == True:
    frac = np.zeros(hp.nside2npix(4096))
    fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
    frac[fracDet['PIXEL']] = fracDet['SIGNAL']
    fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
    fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']    
else:
    path = '/fs/project/PCON0008/des_y3/maglim/mask/'
    fracDet = fitsio.read(path+'y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits.gz')
    frac = np.zeros(hp.nside2npix(4096))
    #fracDet["HPIX"] = hp.nest2ring(4096, fracDet['HPIX'])
    frac[fracDet['HPIX']] = fracDet['FRACGOOD']
    fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
    fracHp[fracDet['HPIX']] = fracDet['FRACGOOD']

if maglim != True:
    dmass_chron_weights =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_validation/'+sys_weight+'.fits')
    ##june23_tests/'+sys_weight+'.fits')
    #dmass_chron_weights = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/Oct23_tests/SPweight12.fits')
else:
    weights =fitsio.read('/fs/project/PCON0008/des_y3/maglim/weight_maps/w_map_bin2_nside4096_nbins1d_10_2sig_v2.0.fits.gz')
    y = np.full(hp.nside2npix(4096), hp.UNSEEN)
    y[weights['HPIX']] = weights['VALUE']
    dmass_chron_weights = y[dmass_chron['HPIX_4096']].copy()
print(dmass_chron_weights.shape)

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
#print(band) 

input_path = '/fs/scratch/PCON0008/warner785/bwarner/PCA/'
y1 = 7
y_all = 10
chi2_dmass = []
chi2_weight = []

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

stars_dir = '/fs/scratch/PCON0008/warner785/bwarner/'
STELLAR_DENS = ("y3_stellar_density_4096_ring_jointmask_v2.2.fits.gz")

maps = np.array([AIRMASS, SKYBRITE, SIGMA_MAG_ZERO, FGCM_GRY, SKYVAR_UNCERT, T_EFF_EXPTIME, FWHM_FLUXRAD, SFD98, STELLAR_DENS, SOF_DEPTH])
map_name = ['AIRMASS','SKYBRITE', 'SIGMA_MAG_ZERO', 'FGCM_GRY', 'SKYVAR_UNCERT', 'T_EFF_EXPTIME', 'FWHM_FLUXRAD', 'SFD98', 'STELLAR_DENS', 'SOF_DEPTH']
     
i_pca = -1
p = -1
for y in range(y_all):
    if y<y1 and y < 7:
        for x in range(4):
            i_pca+=1
            if y == 3:
                p+=1
                cov = np.loadtxt('/fs/scratch/PCON0008/warner785/bwarner/'+'covSP_fgcm_FULL'+str(p)+'.txt')
            else:
                cov = np.loadtxt('/fs/scratch/PCON0008/warner785/bwarner/'+'covSP_area'+str(i_pca)+'.txt')
            diag_cov = np.diagonal(cov)
            error_cov = np.sqrt(diag_cov)
            cov_matrix = diag_cov
            vec_c = np.array(cov_matrix)
            diag_matrix = np.diag(vec_c)
            cov_chi2 = diag_matrix
            
            full_path = input_path + band[x]+'/'
            print("path: ", full_path)
            current_map = map_name[y]+fil[x]
            current = maps[y]
            print("current map: ", current_map)
            input_keyword = current[x]
            
            pcenter, norm_number_density, norm_number_density_ran, error_cov, fracerr_norm, fracerr_ran_norm, sysval_gal = go_through_SP(full_path, input_keyword, current_map, y, fracHp, cov, i_pca, dmass_chron, dmass_chron_weights, random_chron, name = run_name, sys_weights = sys_weights, maglim = maglim)
          
            plot_figure(input_keyword, current_map, y, run_name, pcenter, norm_number_density, error_cov, fracerr_norm, norm_number_density_ran, fracerr_ran_norm, sys_weights = sys_weights)
            #if current_map != 'STELLAR_DENS' and current_map != 'SFD98':
            chi2_, chi2_reduced = chi2(norm_number_density, np.ones(10), fracerr_norm, 0, cov_chi2, SPT = SPT)
            chi2_dmass.append(chi2_reduced)
            print("chi2 dmass", chi2_reduced)
            #assign_weight(pcenter, norm_number_density, y, fracerr_norm, sysval_gal, cov_matrix, run_name, current_map, chi2_weight, cov_chi2, error_cov, i_pca)
            #else:
                #chi2_dmass.append(0)
                #chi2_weight.append(0)
                
    #else:   
    #if y>7 and y<9:
    if y == 10:
        if y == 7:
            print("SECOND SET")
            amount = 1
            csfd = True
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
            
            if y == 8:
                cov = np.loadtxt('/fs/scratch/PCON0008/warner785/bwarner/'+'covSP_star0'+'.txt')
            else:
                cov = np.loadtxt('/fs/scratch/PCON0008/warner785/bwarner/'+'covSP_area'+str(i_pca)+'.txt')
            diag_cov = np.diagonal(cov)
            error_cov = np.sqrt(diag_cov)
            cov_matrix = diag_cov
            vec_c = np.array(cov_matrix)
            diag_matrix = np.diag(vec_c)
            cov_chi2 = diag_matrix
                                
            pcenter, norm_number_density, norm_number_density_ran, error_cov, fracerr_norm, fracerr_ran_norm, sysval_gal = go_through_SP(full_path, input_keyword, current_map, y, fracHp, cov, i_pca, dmass_chron, dmass_chron_weights, random_chron, name = run_name, sys_weights = sys_weights, maglim = maglim, csfd = csfd)

            plot_figure(input_keyword, current_map, y, run_name, pcenter, norm_number_density, error_cov, fracerr_norm, norm_number_density_ran, fracerr_ran_norm, sys_weights = sys_weights)
            #if current_map != 'STELLAR_DENS' and current_map != 'SFD98':      
            chi2_, chi2_reduced = chi2(norm_number_density, np.ones(10), fracerr_norm, 0, cov_chi2, SPT = SPT)
            chi2_dmass.append(chi2_reduced)
            print("chi2 dmass", chi2_reduced)
            #assign_weight(pcenter, norm_number_density, y, fracerr_norm, sysval_gal, cov_matrix, run_name, current_map, chi2_weight, cov_chi2, error_cov, i_pca)
            #else:
                #chi2_dmass.append(0)
                #chi2_weight.append(0)
            
# save chi2 for later comparison
np.savetxt(run_name+'_chi2_dmass_SP.txt', chi2_dmass)
np.savetxt(run_name+'chi2_trend_SP.txt', chi2_weight)

#no_weight
#final_linear