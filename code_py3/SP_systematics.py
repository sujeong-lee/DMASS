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

run_name = "quad50"

# SP map plots (unweighted, 50, 107, chi2>2 for both) + quadratic run values

SPT = True
SPmap = True
linear = True
quadratic = False
frac_weight = None
custom = True

sys_weights = True

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

dmass_chron_weights =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/quad50weights.fits')
print(dmass_chron_weights.shape)

#lin50weights.fits 
#linALLweights.fits 
#quad50weights.fits
#quadALLweights.fits

#linchi2_50.fits
#linchi2_ALL.fits
#quadchi2_50.fits
#quadchi2_ALL.fits


'''
mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/'
n_pca = 1
cov = []
cov_template = 'cov{0}'
for i_pca in range(n_pca): #n_pca
    cov_input= cov_template.format(i_pca)
    cov.append(cov_input)
#print(cov)

cov_template = 'covariance{0}'
for i_pca in range(n_pca): #n_pca
    cov_keyword = cov_template.format(i_pca)
    #print(cov_keyword)
    with open(mock_outdir + cov_keyword + '.txt') as mocks:
        array1 = [x.split() for x in mocks]
        array2 = np.array(array1)
        print(i_pca)
        cov[i_pca] = array2.astype(float)
    mocks.close()
    
covariance = cov[0]
#diag_cov = np.diagonal(covariance)
error_cov = np.sqrt(diag_cov)
'''

# -------------------------------------------------------------------------------------
    
#input_path = '/fs/scratch/PCON0008/warner785/bwarner/band_i/' 
input_path = '/fs/scratch/PCON0008/warner785/bwarner/'

chi2_dmass = []
covariance = None

#test weighted pca-dmass on the sp checks

#y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz
#y3a2_g_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz 
#y3a2_g_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz 
#y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz

AIRMASS =('y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz')

EXPTIME = ('y3a2_g_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz','y3a2_r_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz','y3a2_i_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz','y3a2_z_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz')

FWHM = ('y3a2_g_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz')

SKYBRITE = ('y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_r_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_i_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz','y3a2_z_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz')

maps = np.array([AIRMASS, EXPTIME, FWHM, SKYBRITE])
map_name = ['AIRMASS', 'EXPTIME', 'FWHM', 'SKYBRITE']

band = []
band_template = 'band_{0}'
fil = ('g','r','i','z')
for x in range(4):
    band_input= band_template.format(fil[x])
    band.append(band_input)
print(band)    

for y in range(4):
    for x in range(4):
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
        #print(sysMap.dtype.names)
#frac_weight = fracHp[sysMap['PIXEL']]
#sysMap = sysMap[frac_weight != hp.UNSEEN]
#print(sum(frac_weight == hp.UNSEEN))

        #h_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/h_ran_'+current_map+'.fits')
        #norm_number_density_ran =   fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/norm_ran_'+current_map+'.fits')
        #fracerr_ran_norm = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/fracerr_ran_'+current_map+'.fits')

        print(dmass_chron_weights)
#sysMap = cutPCA(sysMap)
#fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
#fracDet['PIXEL'] = hp.nest2ring(4096, fracDet['PIXEL'])

#    randoms4096 = random_pixel(random_val_fracselected)
#    index_ran_mask = np.argsort(randoms4096)
#    random_chron = randoms4096[index_ran_mask]
#    h_ran,_= number_gal(sysMap, random_chron, sys_weights = False)
#    area = area_pixels(sysMap, fracDet)
#    pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)
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