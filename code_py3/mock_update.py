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

run_name = "equal_area" #"finalized_linear_vFINAL2"
SPT = True
SPmap = False
linear_run = True
sys_weights = False
STOP = False
custom = True
iterative = False # -- reading in weights from other runs
first_set = True # pca0-pca49 = first set; pca 50-106 = second set
equal_area = True

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
frac = np.zeros(hp.nside2npix(4096))
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
frac[fracDet['PIXEL']] = fracDet['SIGNAL']
fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']

mock_path = '/fs/scratch/PCON0008/warner785/bwarner/mocks/cats/'
mock_template = 'y3_cmass_mocks_full_catalog{0}.'

# ('bin', 'ra', 'dec')
# plotting ra, dec as check

n_mock = 700 #700
n_pca = 50 #50

mock_plus = -1
import time
Start = time.time()
ndens_array = np.zeros((n_pca,4,10)) #<N PCA maps>,<N mocks>,<N PCA bins>
for mock_i in range(n_mock):
    if mock_i > 695:
        mock_plus +=1
        mock_keyword = mock_template.format(mock_i)
        print("using ", mock_keyword, "...")
        dmass_spt = io.SearchAndCallFits(path = mock_path, keyword = mock_keyword)

        input_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_SP107_SPT_v2_cformat/'
        #y3/band_z/
        keyword_template = 'pc{0}_'
       
        for i_pca in range(n_pca): #50
            sys_weights = False
        
            input_keyword = keyword_template.format(i_pca)
            print(input_keyword)
            sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword)
            frac_weight = fracHp[sysMap['PIXEL']]
            sysMap = sysMap[frac_weight != hp.UNSEEN]
            path = '/fs/scratch/PCON0008/warner785/bwarner/'
    
# ---------------------  

            dmass_chron = dmass_spt
        #read_in = time.time()
        #files_read = read_in - Start
        
            dmass_weight = np.full(dmass_chron.size, 1)
       # hpix from ra, dec ==> create new column
        
            phi = dmass_chron['ra'] * np.pi / 180.0
            theta = ( 90.0 - dmass_chron['dec'] ) * np.pi/180.0 
            dmass_hpix = hp.ang2pix(4096, theta, phi)
        
#        fig, ax = plt.subplots()
#        ax.plot( dmass_chron['ra'], dmass_chron['dec'], "b,")
#        fig.savefig('mock_footprint.pdf')
        #area_start = time.time()
            area = area_pixels(sysMap, frac_weight, equal_area = equal_area)
        #area_finish = time.time()
        #area_time = area_finish - area_start
        #print("found area in ",area_time, "seconds")
#        pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)
        
            h, sysval_gal = number_gal(sysMap, dmass_hpix, dmass_weight, sys_weights = False, mocks = True, equal_area = equal_area)
        
            pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area, equal_area = equal_area)
    
            ndens = norm_number_density
            print("pca: ",i_pca, "mock: ",mock_i)
            ndens_array[i_pca,mock_plus] = ndens
#        frac_nor.append(fracerr_norm)
        
    #plotting:

##        fig, ax = plt.subplots()
##        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "mock dmass")
#        ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms")
##        plt.legend()
##        xlabel = input_keyword
##        plt.xlabel(xlabel)
##        plt.ylabel("n_gal/n_tot 4096")
    #plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
    #plt.ylim(bottom=0.85)
##        plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
##        plt.title(xlabel+mock_keyword+' systematics mock check')
##        fig.savefig(xlabel+mock_keyword+'sys_check_mock.pdf')        
    
    # save data for each mock -- for all pcas used:
        End = time.time()
        mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/mocks/'    
        for i_pca in range(n_pca):
            np.savetxt(mock_outdir+"mocks_area_pca_plus"+str(i_pca)+".txt", ndens_array[i_pca])
    
        Time = End - Start
        print("time taken for mock: ",Time, "seconds")
        Start = time.time()
#    mockfiles.append(num_den)
    
#print(mockfiles)
#np.savetxt('mocks_density.txt', mockfiles)
#    np.savetxt('mocks_frac'+mock_keyword+'.txt', frac_nor)
        

# define arrays and fill as comp. 
# 50 files 
# test with 3 mocks

# first set of mocks:
# np.savetxt(mock_outdir+"mocks_set20_pca"+str(i_pca)+".txt", ndens_array[i_pca])