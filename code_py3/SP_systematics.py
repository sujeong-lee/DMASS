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

#variables to set:

run_name = "iterativeALL"

# SP map plots (unweighted, 50, 107, chi2>2 for both) + quadratic run values

SPT = True
SPmap = True
frac_weight = None
custom = True

sys_weights = True
SP_run1 = True
SP_run2 = False # other sp maps in band folders

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

#lin50weights.fits //
#linALLweights.fits //
#quad50weights.fits 
#quadALLweights.fits 

#linchi2_50.fits //
#linchi2_ALL.fits //
#quadchi2_50.fits 
#quadchi2_ALL.fits 

#quad_update1weights.fits //
#quad_updateALLweights.fits //
#quadchi_update1weights.fits //
#quadchi_updateALLweights.fits


# mocks load in: 

mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/'
n_pca = 16
cov = []
cov_template = 'cov{0}'
for i_pca in range(n_pca): #n_pca
    cov_input= cov_template.format(i_pca)
    cov.append(cov_input)
#print(cov)

cov_template = 'covarSP_{0}'
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

#test weighted pca-dmass on the sp checks

chi2_dmass = []

band = []
band_template = 'band_{0}'
fil = ('g','r','i','z')
for x in range(4):
    band_input= band_template.format(fil[x])
    band.append(band_input)
print(band)

if SP_run1 or SP_run2 == True:
    if SP_run1 == True:
        input_path = '/fs/scratch/PCON0008/warner785/bwarner/'
        y1 = 4
        covariance = None

        AIRMASS =('y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz')

        EXPTIME = ('y3a2_g_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz','y3a2_r_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz','y3a2_i_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz','y3a2_z_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz')

        FWHM = ('y3a2_g_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz')

        SKYBRITE = ('y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz')

        maps = np.array([AIRMASS, EXPTIME, FWHM, SKYBRITE])
        map_name = ['AIRMASS', 'EXPTIME', 'FWHM', 'SKYBRITE']
    
    else: 
        input_path = '/fs/scratch/PCON0008/warner785/bwarner/PCA/'
        y1 = 21
        covariance = None
        
        AIRMASS_MAX = ('y3a2_g_o.4096_t.32768_AIRMASS.MAX_EQU.fits.gz','y3a2_r_o.4096_t.32768_AIRMASS.MAX_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_AIRMASS.MAX_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_AIRMASS.MAX_EQU.fits.gz')
        
        MAGLIM = ('y3a2_g_o.4096_t.32768_maglim_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_maglim_EQU.fits.gz','y3a2_i_o.4096_t.32768_maglim_EQU.fits.gz','y3a2_z_o.4096_t.32768_maglim_EQU.fits.gz')
        
        AIRMASS_MIN = ('y3a2_g_o.4096_t.32768_AIRMASS.MIN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_AIRMASS.MIN_EQU.fits.gz','y3a2_i_o.4096_t.32768_AIRMASS.MIN_EQU.fits.gz','y3a2_z_o.4096_t.32768_AIRMASS.MIN_EQU.fits.gz')
        
        SIGMA_MAG = ('y3a2_g_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz','y3a2_r_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz','y3a2_i_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz','y3a2_z_o.4096_t.32768_SIGMA_MAG_ZERO.QSUM_EQU.fits.gz')
        
#        SKYVAR_MAX = ('y3a2_g_o.4096_t.32768_SKYVAR.MAX_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_SKYVAR.MAX_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYVAR.MAX_EQU.fits.gz','y3a2_z_o.4096_t.32768_SKYVAR.MAX_EQU.fits.gz')
        
        FGCM_MIN = ('y3a2_g_o.4096_t.32768_FGCM_GRY.MIN_EQU.fits.gz','y3a2_r_o.4096_t.32768_FGCM_GRY.MIN_EQU.fits.gz','y3a2_i_o.4096_t.32768_FGCM_GRY.MIN_EQU.fits.gz','y3a2_z_o.4096_t.32768_FGCM_GRY.MIN_EQU.fits.gz')
#        SKYVAR_MIN = ('y3a2_g_o.4096_t.32768_SKYVAR.MIN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_SKYVAR.MIN_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYVAR.MIN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_SKYVAR.MIN_EQU.fits.gz')
        
        FGCM = ('y3a2_g_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_FGCM_GRY.WMEAN_EQU.fits.gz')
        
#        SKYVAR_SQRT = ('y3a2_g_o.4096_t.32768_SKYVAR_SQRT_WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_SKYVAR_SQRT_WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_SKYVAR_SQRT_WMEAN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_SKYVAR_SQRT_WMEAN_EQU.fits.gz')
        
        FRAC = ('y3a2_g_o.4096_t.32768_frac_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_frac_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_frac_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_frac_EQU.fits.gz')
        
#        SKYVAR_UNCER = ('y3a2_g_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz','y3a2_r_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz','y3a2_z_o.4096_t.32768_SKYVAR.UNCERTAINTY_EQU.fits.gz')
        
#        FWHM_FLUXRAD_MAX = ('y3a2_g_o.4096_t.32768_FWHM_FLUXRAD.MAX_EQU.fits.gz','y3a2_r_o.4096_t.32768_FWHM_FLUXRAD.MAX_EQU.fits.gz','y3a2_i_o.4096_t.32768_FWHM_FLUXRAD.MAX_EQU.fits.gz','y3a2_z_o.4096_t.32768_FWHM_FLUXRAD.MAX_EQU.fits.gz')
        
        SKYVAR = ('y3a2_g_o.4096_t.32768_SKYVAR_WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_SKYVAR_WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_SKYVAR_WMEAN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_SKYVAR_WMEAN_EQU.fits.gz')
        
        
#        FWHM_FLUXRAD_MIN = ('y3a2_g_o.4096_t.32768_FWHM_FLUXRAD.MIN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FWHM_FLUXRAD.MIN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FWHM_FLUXRAD.MIN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_FWHM_FLUXRAD.MIN_EQU.fits.gz')
        
        
        T_EFF_EXPTIME = ('y3a2_g_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_T_EFF_EXPTIME.SUM_EQU.fits.gz')
        
        FWHM_FLUXRAD = ('y3a2_g_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_FWHM_FLUXRAD.WMEAN_EQU.fits.gz')
        
        T_EFF_MAX = ('y3a2_g_o.4096_t.32768_T_EFF.MAX_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_T_EFF.MAX_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_T_EFF.MAX_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_T_EFF.MAX_EQU.fits.gz')
        
        FWHM_MAX = ('y3a2_g_o.4096_t.32768_FWHM.MAX_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FWHM.MAX_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FWHM.MAX_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_FWHM.MAX_EQU.fits.gz')
        
        T_EFF_MIN = ('y3a2_g_o.4096_t.32768_T_EFF.MIN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_T_EFF.MIN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_T_EFF.MIN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_T_EFF.MIN_EQU.fits.gz')
        
        FWHM_MIN = ('y3a2_g_o.4096_t.32768_FWHM.MIN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_FWHM.MIN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_FWHM.MIN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_FWHM.MIN_EQU.fits.gz')
        
        T_EFF = ('y3a2_g_o.4096_t.32768_T_EFF.WMEAN_EQU.fits.gz', 'y3a2_r_o.4096_t.32768_T_EFF.WMEAN_EQU.fits.gz', 'y3a2_i_o.4096_t.32768_T_EFF.WMEAN_EQU.fits.gz', 'y3a2_z_o.4096_t.32768_T_EFF.WMEAN_EQU.fits.gz')
        

        maps = np.array([AIRMASS_MAX, MAGLIM, AIRMASS_MIN, SIGMA_MAG, SKYVAR_MAX, FGCM_MIN, SKYVAR_MIN, FGCM, SKYVAR_SQRT, FRAC, SKYVAR_UNCER, FWHM_FLUXRAD_MAX, SKYVAR, FWHM_FLUXRAD_MIN, T_EFF_EXPTIME, FWHM_FLUXRAD, T_EFF_MAX, FWHM_MAX, T_EFF_MIN, FWHM_MIN, T_EFF])
        map_name = ['AIRMASS_MAX', 'MAGLIM', 'AIRMASS_MIN', 'SIGMA_MAG', 'SKYVAR_MAX', 'FGCM_MIN', 'SKYVAR_MIN', 'FGCM', 'SKYVAR_SQRT', 'FRAC', 'SKYVAR_UNCER', 'FWHM_FLUXRAD_MAX', 'SKYVAR', 'FWHM_FLUXRAD_MIN', 'T_EFF_EXPTIME', 'FWHM_FLUXRAD', 'T_EFF_MAX', 'FWHM_MAX', 'T_EFF_MIN', 'FWHM_MIN', 'T_EFF']
        
    i_mock = -1
    for y in range(y1):
        for x in range(4):
            i_mock+=1
            full_path = input_path + band[x]+'/'
            print("path: ", full_path)
            current_map = map_name[y]+fil[x]
            current = maps[y]
#current_map = 'AIRMASSi'
            print("current map: ", current_map)
    
#input_keyword = 'y3a2_i_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz'
            input_keyword = current[x]
            print("fits file", input_keyword)
        # use fitsio.read in separate file:
            sysMap = io.SearchAndCallFits(path = full_path, keyword = input_keyword)

            path = '/fs/scratch/PCON0008/warner785/bwarner/'
#sysMap['PIXEL'] = hp.nest2ring(4096, sysMap['PIXEL'])
            sysMap = cutPCA(sysMap,SPT=SPT,SPmap=SPmap)
    
            covariance_i = cov[i_mock]
            covariance = np.copy(covariance_i)
            covariance[0][-1]=0
            covariance[-1][0]=0
#diag_cov = np.diagonal(covariance)
            error_cov = np.sqrt(diag_cov)


            print(dmass_chron_weights)

            h, sysval_gal = number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = sys_weights)
            area = area_pixels(sysMap, None, fracDet, SPmap=SPmap, custom = custom)
            pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)

#plotting:

            fig, ax = plt.subplots()
            if SPT == True: # change based on used weights ---------------------------------------
                ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass spt, weights "+ run_name)
            #ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms spt")
            else:
                ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass val")
            #ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms val")
            plt.legend()
            xlabel = input_keyword
            plt.xlabel(current_map)
            plt.ylabel("n_gal/n_tot 4096")
            plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
            if sys_weights == True:
                if SPT == True:
                    plt.title(current_map+' SPT region with '+run_name+'weights applied')
                    if custom == True:
                        fig.savefig('../SPmap_custom/'+run_name+current_map+' spt_check.pdf')
                    else:
                        fig.savefig('../SPmap_official/'+run_name+current_map+' spt_check.pdf')
                else:
                    plt.title(current_map+' VAL region with weights applied')
                    fig.savefig('../SPmap_check/'+current_map+' val.pdf')
            else:
                if SPT == True:
                    plt.title('systematics check, no weights: '+current_map+' in spt')
                    if custom == True:
                        fig.savefig('../SPmap_custom/'+current_map+'spt.pdf')
                    else:
                        fig.savefig('../SPmap_official/'+current_map+'spt.pdf')
                else:
                    plt.title('systematics check, no weights: '+current_map+' in val')
                    fig.savefig('../SPmap_check/'+current_map+'val.pdf')
                
            chi2_, chi2_reduced = chi2(norm_number_density, np.ones(12), error_cov, 0, None, SPT = False)
            chi2_dmass.append(chi2_reduced)
    

else:  #third run:
    
    ex_dir = '/fs/scratch/PCON0008/warner785/bwarner/PCA/extinction'
    EXTINCTION = ("ebv_sfd98_fullres_nside_4096_nest_equatorial_des.fits.gz")
    sof_dir = '/fs/scratch/PCON0008/warner785/bwarner/PCA/sof_depth'
    SOF_DEPTH = ("y3a2_gold_2_2_1_sof_nside4096_nest_g_depth.fits.gz", "y3a2_gold_2_2_1_sof_nside4096_nest_r_depth.fits.gz", "y3a2_gold_2_2_1_sof_nside4096_nest_i_depth.fits.gz", "y3a2_gold_2_2_1_sof_nside4096_nest_z_depth.fits.gz")
    stars_dir = '/fs/scratch/PCON0008/warner785/bwarner/PCA/stars'
    STARS = ("stars_extmashsof0_16_20_zeros_footprint_nside_4096_nest.fits.gz")
    
    
    maps = np.array([EXTINCTION, STARS, SOF_DEPTH])
    map_name = ['EXTINCTION', 'STARS','SOF_DEPTH']

    for y in range(3):
        if y == 0:
            amount = 1
            full_path = ex_dir
            print("path: ", full_path)
            current_map = map_name[y] 

        if y == 1:
            amount = 1 
            full_path = stars_dir
            print("path: ", full_path)
            current_map = map_name[y]
            
            
        if y == 2:
            amount = 4
        for x in range(amount):
            if y == 2:
                full_path = input_path + band[x]+'/'
                print("path: ", full_path)
                current_map = map_name[y]+fil[x]
                
            current = maps[y]
            input_keyword = current[x]
                
#current_map = 'AIRMASSi'
            print("current map: ", current_map)
            print("fits file", input_keyword)
        # use fitsio.read in separate file:
            sysMap = io.SearchAndCallFits(path = full_path, keyword = input_keyword)

            path = '/fs/scratch/PCON0008/warner785/bwarner/'
#sysMap['PIXEL'] = hp.nest2ring(4096, sysMap['PIXEL'])
            sysMap = cutPCA(sysMap,SPT=SPT,SPmap=SPmap)


            print(dmass_chron_weights)

            h, sysval_gal = number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = sys_weights)
            area = area_pixels(sysMap, None, fracDet, SPmap=SPmap, custom = custom)
            pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)

#plotting:

            fig, ax = plt.subplots()
            if SPT == True: # change based on used weights ---------------------------------------
                ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass spt, weights "+ run_name)
            #ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms spt")
            else:
                ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass val")
            #ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms val")
            plt.legend()
            xlabel = input_keyword
            plt.xlabel(current_map)
            plt.ylabel("n_gal/n_tot 4096")
            plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
            if sys_weights == True:
                if SPT == True:
                    plt.title(current_map+' SPT region with '+run_name+'weights applied')
                    if custom == True:
                        fig.savefig('../SPmap_custom/'+run_name+current_map+' spt_check.pdf')
                    else:
                        fig.savefig('../SPmap_official/'+run_name+current_map+' spt_check.pdf')
                else:
                    plt.title(current_map+' VAL region with weights applied')
                    fig.savefig('../SPmap_check/'+current_map+' val.pdf')
            else:
                if SPT == True:
                    plt.title('systematics check, no weights: '+current_map+' in spt')
                    if custom == True:
                        fig.savefig('../SPmap_custom/'+current_map+'spt.pdf')
                    else:
                        fig.savefig('../SPmap_official/'+current_map+'spt.pdf')
                else:
                    plt.title('systematics check, no weights: '+current_map+' in val')
                    fig.savefig('../SPmap_check/'+current_map+'val.pdf')
                
            chi2_, chi2_reduced = chi2(norm_number_density, np.ones(12), fracerr_norm, 0, None, SPT = False)
            chi2_dmass.append(chi2_reduced)
    
# save chi2 for later comparison
np.savetxt(run_name+'_chi2_dmass_SP.txt', chi2_dmass)