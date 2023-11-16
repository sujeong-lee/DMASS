import esutil, yaml, sys, os, argparse
import healpy as hp
#from systematics import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import astropy.io.fits as fits
import h5py

#opening DES Y6 Cardinal simulations catalog:
filename='/fs/scratch/PCON0008/warner785/bwarner/Cardinal-3_v2.0_Y6a_gold.h5'
f=h5py.File(filename,'r')

print(list(f.keys()))
dset = f['catalog/']
print(list(dset.keys()))
mask = f['masks/gold/']

golddset = f['catalog/gold/']
print(list(golddset.keys()))

print(list(mask.keys())) 

#read in values:
sof_g = f['catalog/gold/mag_g'][:]
err_g = f['catalog/gold/mag_err_g'][:]

sof_r = f['catalog/gold/mag_r'][:]
err_r = f['catalog/gold/mag_err_r'][:]

sof_i = f['catalog/gold/mag_i'][:]
err_i = f['catalog/gold/mag_err_i'][:]

ra = f['catalog/gold/ra'][:]
dec = f['catalog/gold/dec'][:]
coadd_object_id = f['catalog/gold/coadd_object_id'][:]

#create fits file:
cardinal = np.zeros( len(sof_g), dtype=[('COADD_OBJECT_ID','int'), ('SOF_CM_MAG_CORRECTED_G','float'), ('SOF_CM_MAG_ERR_G','float'), ('SOF_CM_MAG_CORRECTED_R','float'), ('SOF_CM_MAG_ERR_R','float'), ('SOF_CM_MAG_CORRECTED_I','float'), ('SOF_CM_MAG_ERR_I','float')])
cardinal['COADD_OBJECT_ID'] = coadd_object_id
#cardinal['RA'] = ra
#cardinal['DEC'] = dec
cardinal['SOF_CM_MAG_CORRECTED_G'] = sof_g
cardinal['SOF_CM_MAG_ERR_G'] = err_g
cardinal['SOF_CM_MAG_CORRECTED_R'] = sof_r
cardinal['SOF_CM_MAG_ERR_R'] = err_r
cardinal['SOF_CM_MAG_CORRECTED_I'] = sof_i
cardinal['SOF_CM_MAG_ERR_I'] = err_i

#save fits file:
outdir = '/fs/scratch/PCON0008/warner785/bwarner/'
os.makedirs(outdir, exist_ok=True)
esutil.io.write(outdir+'cardinal.fits', cardinal, overwrite=True)

f.close()