from xd import *
import esutil
import matplotlib.pyplot as plt
import numpy as np

file_path = '/users/PCON0003/warner785/DMASSY3/code_py3/'
#name = 'final27'

first_run = True

name = 'vali'
new_name = 'vali0'

with open(file_path + name +'chi2_dmassi_spt.txt') as dmass:
#    w  = [float(x) for x in next(dmass).split()]
    chi2_null = [float(x) for x in dmass]
dmass.close()
chi2_null = np.array(chi2_null)

with open(file_path+'chi2_threshold.txt') as dm5:
#    w  = [float(x) for x in next(randoms).split()]
    thres_chi2 = [float(x) for x in dm5]
dm5.close()
thres_chi2 = np.array(thres_chi2)

with open(file_path+ name+'chi2_trend_spt.txt') as dm4:
#    w  = [float(x) for x in next(randoms).split()]
    chi2_trend = [float(x) for x in dm4]
dm4.close()

diff = []
print(len(chi2_null), len(chi2_trend))
for i in range(50):
    diff.append(chi2_null[i] - chi2_trend[i])

threshold = []
for i in range(50):
    threshold.append(diff[i]/thres_chi2[i])
threshold = np.array(threshold)

keyword_template = 'pc{0}_'
    
if threshold.max()<2:
    print("----------------EVERYTHING WEIGHTED-----------------")
    contin = False
else:
    contin = True
    for x in range(50):
        if (threshold[x])>2 and threshold[x] == threshold.max():
            print(x, threshold.max())
            dmass_chron_i = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_validation/'+name+"pc"+str(x)+"_"+'dmass_weight_spt.fits')
            
            #for first run only:
            if first_run == True:
                full_dmass_sysweights = dmass_chron_i
            
            #for any run other than first run:
            else:
                dmass_chron = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_validation/'+name+'.fits')
                full_dmass_sysweights = np.multiply(dmass_chron,dmass_chron_i)

    print(full_dmass_sysweights)

#fig, ax = plt.subplots()
#ax.hist(full_dmass_sysweights[full_dmass_sysweights!=0])

#dmass_chron['SYS_WEIGHT'] = full_dmass_sysweights
    outdir = '/fs/scratch/PCON0008/warner785/bwarner/june23_validation/'
    os.makedirs(outdir, exist_ok=True)
    esutil.io.write( outdir+new_name+'.fits', full_dmass_sysweights, overwrite=True)

    print(np.mean(full_dmass_sysweights))