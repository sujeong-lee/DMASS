import os, sys
import esutil
import healpy as hp
import numpy as np
import fitsio
sys.path.append('../')
from xd import *
from run_DMASS_Y3 import *
from utils import *
import healpy as hp
from systematics import *
from cmass_modules import io
os.chdir('../../DMASS_XDGMM/code_py3/')
print(os.getcwd())
from xdgmm import XDGMM as XDGMM_Holoien
os.chdir('../../DMASSY3/code_py3/')
print(os.getcwd())

#gold_spt = fitsio.read('../output/test/train_cat/y3/gold_spt.fits')
gold_spt = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+'cardinal_masked.fits')

def GenerateRegions(jarrs, jras, jdecs, jfile, njack, jtype):

    import kmeans_radec
    
    if jtype=='generate':
        rdi = np.zeros( (len(jarrs),2) )
        rdi[:,0] = jras# jarrs[gindex][jras[gindex]]
        rdi[:,1] = jdecs #jarrs[gindex][jdecs[gindex]]

        if jfile == None:
            jfile = 'JK-{0}.txt'.format(njack)
        km = kmeans_radec.kmeans_sample(rdi, njack, maxiter=200, tol=1.0e-5)

        if not km.converged:
            raise RuntimeError("k means did not converge")
        np.savetxt(jfile, km.centers)

    elif jtype=='read':
        print('read stored jfile :', jfile)
        centers = np.loadtxt(jfile)
        km = kmeans_radec.KMeans(centers)
        njack = len(centers)

    return [km, jfile]


def AssignIndex(jarrs, jras, jdecs, km):
    
    ind = []
    for i in range(len(jarrs)):
        rdi = np.zeros( (len(jarrs[i]),2) )
        rdi[:,0] = jras[i]
        rdi[:,1] = jdecs[i]
        index = km.find_nearest(rdi)
        ind.append(index[0])
    return np.array(ind)


def construct_jk_catalog( cat, njack = 10, root='./', jtype = 'generate', jfile = 'jkregion.txt', suffix = '' , retind = False):

    import os
    os.system('mkdir '+root)
    km, jfile = GenerateRegions(cat, cat['RA'], cat['DEC'], root+jfile, njack, jtype)
    ind = AssignIndex(cat, cat['RA'], cat['DEC'], km)
    #ind_rand = AssignIndex(rand, rand['RA'], rand['DEC'], km)
    
    if retind : 
        if 'JKindex' in cat.dtype.names : cat['JKindex'] = ind
        else : cat = appendColumn(cat, name = 'JKindex', value = ind, dtypes=None)
        return cat

    else : 
        catlist = []
        for i in range(njack):
            mask = (ind == i)
            catlist.append(cat[mask])
            #mask_rand = (ind_rand == i)
            #_constructing_input_file(cat[mask], rand[mask_rand], root = root, suffix = suffix)
            
        #os.remove(jkfile)  

        return catlist
        
split_spt = construct_jk_catalog(gold_spt)

# save as fits file:
 # change to scratch space
 # read and write to scratch space -- purged if unused in ~2months

outdir = '/fs/scratch/PCON0008/warner785/bwarner/'
os.makedirs(outdir, exist_ok=True)

#plt.scatter(ra,dec,marker='.',c=ind)

for i in range(len(split_spt)):
    fitsio.write( outdir+'split_cardinal_' + str(i) + '.fits', split_spt[i], overwrite=True)
    
