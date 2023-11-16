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

run_name = "sp_chi2_fgcm" #"finalized_linear_vFINAL2"
SPT = True
SPmap = True
linear_run = True
sys_weights = False
STOP = False
custom = True
iterative = False # -- reading in weights from other runs
first_set = True # pca0-pca49 = first set; pca 50-106 = second set
equal_area = True

#path = '/fs/scratch/PCON0008/warner785/bwarner/'
#fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
#frac = np.zeros(hp.nside2npix(4096))
#fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
#frac[fracDet['PIXEL']] = fracDet['SIGNAL']
#fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
#fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']

#mock_path = '/fs/scratch/PCON0008/warner785/bwarner/mocks/cats/'
#mock_template = 'y3_cmass_mocks_full_catalog{0}.'

# ('bin', 'ra', 'dec')
# plotting ra, dec as check

import numpy as np
m_pca = 4

number_density_array = []
pca_template = 'pca{0}'
for i_pca in range(m_pca):
    pca_input= pca_template.format(i_pca)
    number_density_array.append(pca_input)
    
mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/mocks/'
keyword_template = 'mock_SP_fgcm{0}'

#keyword2_template = "mocks_second_set_pca{0}"

for i_pca in range(m_pca):
    input_keyword = keyword_template.format(i_pca)
    #print(input_keyword)
    with open(mock_outdir + input_keyword + '.txt') as mocks:
        array1 = [x.split() for x in mocks]
        array2 = np.array(array1)
        number_density_array[i_pca] = array2.astype(float)
    mocks.close()

#pcenter_array = np.loadtxt('/users/PCON0003/warner785/DMASSY3/code_py3/'+'FGCM_GRY'+ num +'pcenter.txt')
#pcenter_array = np.array(pcenter_array)

n_mock = 508 #700
n_pca = 4 #50

num = ['g', 'r', 'i', 'z']

import time
Start = time.time()
lin_chi2_array = np.zeros((n_pca,n_mock)) #<N PCA maps>,<N mocks>,<N PCA bins>
null_chi2_array =  np.zeros((n_pca,n_mock))
for mock_i in range(n_mock):
    if mock_i > -1: #can get to particular mocks
        keyword_template = 'pc{0}_'
        for i_pca in range(n_pca): #50
            if i_pca != -1:
                covariance = np.loadtxt('/fs/scratch/PCON0008/warner785/bwarner/'+'covSP_fgcm_FULL'+str(i_pca)+'.txt')
                diag_cov = np.diagonal(covariance)
                cov_matrix = diag_cov
                vec_c = np.array(cov_matrix)
                diag_matrix = np.diag(vec_c)
                cov_chi2 = diag_matrix
                pcenter_array = np.loadtxt('/users/PCON0003/warner785/DMASSY3/code_py3/'+'FGCM_GRY'+ num[i_pca]+'pcenter.txt')
                pcenter = np.array(pcenter_array)
                norm_number_density = number_density_array[i_pca][mock_i]
            
                chi2_, chi2_reduced = chi2(norm_number_density, np.ones(10), None, 0, cov_chi2, SPT = True)
                print("pca: ",i_pca, "mock: ",mock_i)
                null_chi2_array[i_pca,mock_i] = chi2_reduced
            
                #FOR LINEAR CHI2--------------------------------
                params = scipy.optimize.curve_fit(lin, pcenter, norm_number_density, sigma = cov_matrix)
                [m, b1] = params[0]
                trend_chi2, trend_chi2_reduced = chi2(norm_number_density, lin(pcenter, m, b1), None, 2, cov_chi2, SPT = True)
                chi2_lin = trend_chi2_reduced
            # ----------------------------------------------
                lin_chi2_array[i_pca,mock_i] = chi2_lin #chi2_i
            else: 
                null_chi2_array[i_pca,mock_i] = 0
                lin_chi2_array[i_pca,mock_i] = 0

    # save data for each mock -- for all pcas used:
        End = time.time()
        mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/mocks/'    
        for i_pca in range(n_pca):
            np.savetxt(mock_outdir+"lin_sp_chi2_fgcmFULL_"+str(i_pca)+".txt", lin_chi2_array[i_pca])
            np.savetxt(mock_outdir+"null_sp_chi2_fgcmFULL_"+str(i_pca)+".txt", null_chi2_array[i_pca])
    
        Time = End - Start
        print("time taken for mock: ",Time, "seconds")
        Start = time.time()

# pca:
# null mocks: chi2_dist2_
# linear mocks: lin_chi2_dist_