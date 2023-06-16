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
    #dmass_val = calling_catalog('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')
    dmass_val = esutil.io.read('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')   
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

dmass_chron_weights =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/dmass_sys_weight_spt_custom_ALL.fits')
print(dmass_chron_weights.shape)

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
'''
# -------------------------------------------------------------------------------------
    
input_path = '/fs/scratch/PCON0008/warner785/bwarner/band_i/' 
#/band_g,r,i,z/:
#band = ['g', 'r', 'i', 'z']

#for x in range(4)
    #full_path = input_path + band[x] + '/'
#test weighted pca-dmass on the sp checks

#y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz
#y3a2_g_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz 
#y3a2_g_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz 
#y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz
#redmine
#nersc pca location for sp

'''covariance = cov[0]
#diag_cov = np.diagonal(covariance)
error_cov = np.sqrt(diag_cov)'''

current_map = 'AIRMASSi'

input_keyword = 'y3a2_i_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz'
sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword)

path = '/fs/scratch/PCON0008/warner785/bwarner/'
#sysMap['PIXEL'] = hp.nest2ring(4096, sysMap['PIXEL'])
sysMap = cutPCA(sysMap,SPT=SPT,SPmap=SPmap)
print(sysMap.dtype.names)
#frac_weight = fracHp[sysMap['PIXEL']]
#sysMap = sysMap[frac_weight != hp.UNSEEN]
#print(sum(frac_weight == hp.UNSEEN))

h_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/h_ran_'+current_map+'.fits')
norm_number_density_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/norm_ran_'+current_map+'.fits')
fracerr_ran_norm = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/fracerr_ran_'+current_map+'.fits')

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
area = area_pixels(sysMap, None, fracDet, SPmap=SPmap)
pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)

#plotting:

fig, ax = plt.subplots()
if SPT == True:
    ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in spt")
    ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms in spt")
else:
    ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in validation")
    ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms in validation")
plt.legend()
xlabel = input_keyword
plt.xlabel(current_map)
plt.ylabel("n_gal/n_tot 4096")
#plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
#plt.ylim(bottom=0.85)
plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
if sys_weights == True:
    if SPT == True:
        plt.title(current_map+' SPT region systematic check')
        if custom == True:
            fig.savefig('../SPmap_custom/'+current_map+' spt_check.pdf')
        else:
            fig.savefig('../SPmap_official/'+current_map+' spt_check.pdf')
    else:
        plt.title(current_map+' VAL region systematic check')
        fig.savefig('../SPmap_check/'+current_map+' val.pdf')
else:
    if SPT == True:
        plt.title('systematics check: '+current_map+' in spt')
        if custom == True:
            fig.savefig('../SPmap_custom/'+current_map+'spt.pdf')
        else:
            fig.savefig('../SPmap_official/'+current_map+'spt.pdf')
    else:
        plt.title('systematics check: '+current_map+' in val')
        fig.savefig('../SPmap_check/'+current_map+'val.pdf')  