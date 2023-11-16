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
custom = True
mocks = False

sys_weights = False

# -----------------------

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])

frac = np.zeros(hp.nside2npix(4096))
frac[fracDet['PIXEL']] = fracDet['SIGNAL']
fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']

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
covariance = None
n_mock = 1
n_pca = 34

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
     

#ndens_array = np.zeros((n_pca,n_mock,12)) #<N PCA maps>,<N mocks>,<N PCA bins>
for mock_i in range(n_mock):
    i_pca=-1
    mock_keyword = mock_template.format(mock_i)
    print("using ", mock_keyword, "...")
    dmass_spt = io.SearchAndCallFits(path = mock_path, keyword = mock_keyword)
    dmass_chron = dmass_spt
    
    dmass_weight = np.full(dmass_chron.size, 1)    
    phi = dmass_chron['ra'] * np.pi / 180.0
    theta = ( 90.0 - dmass_chron['dec'] ) * np.pi/180.0 
    dmass_hpix = hp.ang2pix(4096, theta, phi)
                      
    for y in range(y_all):
        if y == 7 : # get specific maps
            if y<y1:
                for x in range(4):
                    i_pca+=1
                    full_path = input_path + band[x]+'/'
                    print("path: ", full_path)
                    current_map = map_name[y]+fil[x]
                    current = maps[y]
                    print("current map: ", current_map)
                    input_keyword = current[x]
                
                    print("pca: ",i_pca, "mock: ",mock_i)
                    go_through_maps(full_path, input_keyword, current_map, fracHp,  dmass_hpix, dmass_weight)
                
            else:
                if y == 7:
                    print("SECOND SET")
                    amount = 1
                    full_path = ex_dir
                    print("path: ", full_path)
                    current_map = map_name[y]
                    print("current map: ", current_map)
                    star_map = True # usually False, except csfd
                    flatten = False
                if y == 8:
                    amount = 1 
                    full_path = stars_dir
                    print("path: ", full_path)
                    current_map = map_name[y]  
                    star_map = True
                    flatten = True
                    print("current map: ", current_map)
                if y == 9:
                    amount = 4
                    star_map = True
                    flatten = True
                for x in range(amount):
                    i_pca+=1
                    if y == 9:
                        full_path = sof_dir
                        print("path: ", full_path)
                        current_map = map_name[y]+fil[x]
                        print("current map: ", current_map)
                
                    current = maps[y]
                    if y!=8:
                        input_keyword = current[x]
                    else:
                        input_keyword = 'y3_stellar_density_4096_ring_jointmask_v2.2.fits.gz'
                
                    print("pca: ",i_pca, "mock: ",mock_i)                
                    go_through_maps(full_path, input_keyword, current_map, fracHp, dmass_hpix, dmass_weight, star_map = star_map, flatten = flatten)
                  
    #mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/mocks/'    
    #for i_pca in range(n_pca):
        #np.savetxt(mock_outdir+"mock_SP_34maps"+str(i_pca)+".txt", ndens_array[i_pca])
            