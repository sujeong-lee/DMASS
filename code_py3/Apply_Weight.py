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

file_path = '/users/PCON0003/warner785/DMASSY3/code_py3/'

name = 'finalized_linear_v2'
# linear_run1
# linear_run2
# quad_updated1
# quad_updated2

multiply = False # multiplying existing weights == True 

with open(file_path + name +'chi2_dmassi_spt.txt') as dmass:
#    w  = [float(x) for x in next(dmass).split()]
    chi2_dmass = [float(x) for x in dmass]
dmass.close()

keyword_template = 'pc{0}_'

if multiply != True:
    #dmass_chron =   fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+name+'pc0_dmass_weight_spt.fits')
    dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+name+"pc0_"+'dmass_weight_spt.fits')
    norm_weight1 = 1
#norm_weight1 = np.mean(dmass_chron['SYS_WEIGHT'][dmass_chron['SYS_WEIGHT']!=0])
    dmass_chron_i = dmass_chron/norm_weight1
#print(dmass_chron_i)
#y = [50]
#dmass_i = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/dmass_sys_weight_pca.fits')
#dmass_chron_i = dmass_i['SYS_WEIGHT']
#print(dmass_chron_i.size)
    for x in range(50): #50, 106
        #if x>50: #x>50
        if chi2_dmass[x]>2 and x>0: #x-50
#        print(y[x])
            input_keyword = keyword_template.format(x) #y[x]
            print(input_keyword)
            dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+name+input_keyword+'dmass_weight_spt.fits')
            norm_weight = 1
        #norm_weight = np.mean(dmass_chron['SYS_WEIGHT'][dmass_chron['SYS_WEIGHT']!=0])
            norm_dmass = (dmass_chron/norm_weight)
            full_dmass_sysweights = np.multiply(norm_dmass,dmass_chron_i)
            dmass_chron_i = full_dmass_sysweights
        #print(dmass_chron_i)
#    y_new = y[x]+1
#    y.append(y_new)

if multiply == True:
    dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'quadchi_update1weights.fits')
    norm_weight1 = 1
    dmass_chron_i = dmass_chron/norm_weight1
 
    dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'quadchi_update2weights.fits')
    norm_weight = 1
    norm_dmass = (dmass_chron/norm_weight)
    full_dmass_sysweights = np.multiply(norm_dmass,dmass_chron_i)
    dmass_chron_i = full_dmass_sysweights
    

print(full_dmass_sysweights)

#fig, ax = plt.subplots()
#ax.hist(full_dmass_sysweights[full_dmass_sysweights!=0])

#dmass_chron['SYS_WEIGHT'] = full_dmass_sysweights
outdir = '/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'
os.makedirs(outdir, exist_ok=True)
esutil.io.write( outdir+'final_linear.fits', full_dmass_sysweights, overwrite=True)

print(np.mean(full_dmass_sysweights))


