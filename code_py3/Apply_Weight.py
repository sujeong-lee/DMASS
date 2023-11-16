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

name = 'vali'

multiply = False # multiplying existing weights == True 
iterative = True
SP_addition = False

with open(file_path + name +'_chi2_dmass_SP.txt') as dmass: # chi2_dmassi_spt.txt
#    w  = [float(x) for x in next(dmass).split()]
    chi2_null = [float(x) for x in dmass]
dmass.close()


with open(file_path+'sp_chi2_threshold.txt') as dm5: # chi2_threshold.txt
#    w  = [float(x) for x in next(randoms).split()]
    thres_chi2 = [float(x) for x in dm5]
dm5.close()

with open(file_path+'stars_chi2_threshold.txt') as dm5: # chi2_threshold.txt
#    w  = [float(x) for x in next(randoms).split()]
    stars_thres_chi2 = [float(x) for x in dm5]
dm5.close()

with open(file_path+'fgcm_FULL_chi2_threshold.txt') as dm5: # chi2_threshold.txt
#    w  = [float(x) for x in next(randoms).split()]
    fgcm_thres_chi2 = [float(x) for x in dm5]
dm5.close()

with open(file_path+ name+'chi2_trend_SP.txt') as dm4: # chi2_trend_spt.txt
#    w  = [float(x) for x in next(randoms).split()]
    chi2_trend = [float(x) for x in dm4]
dm4.close()
 
diff = []
for i in range(34): #50 
    diff.append(chi2_null[i]*10 - chi2_trend[i]*8)

threshold = []
i = 0
while i<34: #50 
    if i == 12:
        for x in range(4):
            threshold.append(diff[i]/fgcm_thres_chi2[x])
            i+=1
    if i == 29:
        threshold.append(diff[i]/stars_thres_chi2[0])
    if i!=12 and i!=29:
        threshold.append(diff[i]/thres_chi2[i]) 
    i+=1
threshold = np.array(threshold)

keyword_template = 'pc{0}_'

if multiply != True and iterative != True and SP_addition != True:
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
        if (threshold[x])>2 and x>0: #x-50
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

# FOR ONE PCA WEIGHTS:
    full_dmass_sysweights = dmass_chron_i

if multiply == True:
    dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'quad_v5.fits')
    norm_weight1 = 1
    dmass_chron_i = dmass_chron/norm_weight1
 
    dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'quad_v6.fits')
    norm_weight = 1
    norm_dmass = (dmass_chron/norm_weight)
    full_dmass_sysweights = np.multiply(norm_dmass,dmass_chron_i)
    dmass_chron_i = full_dmass_sysweights
    
if iterative == True:
    for x in range(34): #50
        if (threshold[x])>2 and threshold[x] == threshold.max(): #and x!=28 and x!=29:
            print(x, threshold.max())
            #dmass_chron_i = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+name+"pc"+str(x)+"_"+'dmass_weight_spt.fits')
            dmass_chron_i = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/oct23_tests/'+name+'sp'+str(x)+'weight.fits')
            #for first run only:
            full_dmass_sysweights = dmass_chron_i
            
            #for any run other than first run:
            # pca maps:
            #dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'final27.fits')
            # sp maps:
            #dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/Oct23_tests/'+'SPweight0.fits')
            #full_dmass_sysweights = np.multiply(dmass_chron,dmass_chron_i)
            
if SP_addition == True: # want to correct for sp map 14, 28
    diff = []
    print(chi2_trend)
    diff.append(chi2_null[14]*10 - chi2_trend[0]*8)
    diff.append(chi2_null[28]*10 - chi2_trend[1]*8)
    diff = np.array(diff)
    i = 0
    for x in range(2):
        if diff[x] == diff.max():
            print(x, diff.max())
            if i == 0:
                dmass_chron_i = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/Sept23_tests/'+name+'FGCM_GRYiweight.fits') 
            
            if i == 1:
                dmass_chron_i = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/Sept23_tests/'+name+'SFD98weight.fits')
        i+=1
    #dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'SPlin1.fits')
    #full_dmass_sysweights = np.multiply(dmass_chron,dmass_chron_i)
    full_dmass_sysweights = dmass_chron_i
                
print(full_dmass_sysweights)

#fig, ax = plt.subplots()
#ax.hist(full_dmass_sysweights[full_dmass_sysweights!=0])

#dmass_chron['SYS_WEIGHT'] = full_dmass_sysweights
outdir = '/fs/scratch/PCON0008/warner785/bwarner/oct23_tests/'
os.makedirs(outdir, exist_ok=True)
esutil.io.write( outdir+'SPfinal0.fits', full_dmass_sysweights, overwrite=True)

print(np.mean(full_dmass_sysweights))

#final_linear_v2.fits

#final_weight_sept_v2.fits