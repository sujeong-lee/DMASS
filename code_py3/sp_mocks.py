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

n_mock = 292
n_pca = 16

SPT = True
SPmap = True
frac_weight = None
custom = True
mocks = True

sys_weights = False

# -----------------------

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])

# -------------------------------------------------------------------------------------

#input_path = '/fs/scratch/PCON0008/warner785/bwarner/band_i/' 
input_path = '/fs/scratch/PCON0008/warner785/bwarner/'

mock_path = '/fs/scratch/PCON0008/warner785/bwarner/mocks/cats/'
mock_template = 'y3_cmass_mocks_full_catalog{0}.'

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

ndens_array = np.zeros((n_pca,n_mock,12)) #<N PCA maps>,<N mocks>,<N PCA bins>
for mock_i in range(n_mock):
    
    i_pca=-1
    mock_keyword = mock_template.format(mock_i)
    print("using ", mock_keyword, "...")
    dmass_spt = io.SearchAndCallFits(path = mock_path, keyword = mock_keyword)
    dmass_chron = dmass_spt
    
    dmass_weight = np.full(dmass_chron.size, 1)
       # hpix from ra, dec ==> create new column
        
    phi = dmass_chron['ra'] * np.pi / 180.0
    theta = ( 90.0 - dmass_chron['dec'] ) * np.pi/180.0 
    dmass_hpix = hp.ang2pix(4096, theta, phi)

    for y in range(4):
        for x in range(4):
            i_pca+=1
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
#print(sum(frac_weight == hp.UNSEEN) 

            h, sysval_gal = number_gal(sysMap, dmass_hpix, dmass_weight, sys_weights = sys_weights, mocks = mocks)
            area = area_pixels(sysMap, None, fracDet, SPmap=SPmap, custom = custom)
            pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)

            ndens = norm_number_density
            print("pca: ",i_pca, "mock: ",mock_i)
            ndens_array[i_pca,mock_i] = ndens
        
    mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/mocks/'    
    for i_pca in range(n_pca):
        np.savetxt(mock_outdir+"mocks_SP"+str(i_pca)+".txt", ndens_array[i_pca])
