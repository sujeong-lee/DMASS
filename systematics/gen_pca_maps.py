"""
This script is designed to read in the DES Y3 Survey property maps (SP maps)
and run a PCA on them

We will provide options to create PCAS for validation region, or science region
"""

import numpy as np
from sklearn.decomposition import PCA
import os 
import fitsio as fio
import healpy as hp 

#label used when saving PCA maps
label = 'SP106_validationregion'
output_order = 'ring'

test = True

if test == True:
    outdir = './pca_{0}_test/'.format(label)
else:
    outdir = './pca_{0}/'.format(label)
if os.path.exists(outdir) == False: #if the output directory does not exist
    os.mkdir(outdir) #make it

#path to the observational SP maps on NERSC
sp_dir1 = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/band_g/'
sp_dir2 = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/band_r/'
sp_dir3 = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/band_i/'
sp_dir4 = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/band_z/'
sp_dir5 = '/global/cfs/cdirs/des/monroy/systematics_analysis/spmaps/sof_depth/'
sp_files =  [sp_dir1 + filename for filename in os.listdir(sp_dir1)] + \
            [sp_dir2 + filename for filename in os.listdir(sp_dir2)] + \
            [sp_dir3 + filename for filename in os.listdir(sp_dir3)] + \
            [sp_dir4 + filename for filename in os.listdir(sp_dir4)] + \
            [sp_dir5 + filename for filename in os.listdir(sp_dir5)]

#paths to the astrophysical SP maps on NERSC
starfile1 = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/stars/stars_extmashsof0_16_20_zeros_footprint_nside_4096_nest.fits.gz'
sp_files.append(starfile1)

#This one has UNSEEN values and might be redundant?
#will remove for now
#starfile2 = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/stars/y3_stellar_density_4096_ring_jointmask_v2.2.fits.gz'
#sp_files.append(starfile2)

extfile = '/global/cfs/projectdirs/des/monroy/systematics_analysis/spmaps/extinction/ebv_sfd98_fullres_nside_4096_nest_equatorial_des.fits.gz'
sp_files.append(extfile)

if test == True:
    sp_files = sp_files[:3]

nmaps = len(sp_files)

#load the LSS mask (slightly different format to usual)
lssmask_file = '/global/cfs/cdirs/des/jelvinpo/cats/y3/bao/v2p2/MASK_Y3LSSBAOSOF_22_3_v2p2_y3_format.fits.gz'
lssmask = fio.read(lssmask_file)
mask = np.zeros(hp.nside2npix(4096))
mask[hp.ring2nest(4096, lssmask['HPIX'])] = 1.
mask = mask.astype('bool')

#validation region mask
ra_pix,dec_pix = hp.pix2ang(4096,np.arange(hp.nside2npix(4096)),nest=True,lonlat=True) #ra and dec of all pixels
mask4 = (ra_pix>18)&(ra_pix<43)
mask4 = mask4 & (dec_pix>-10) & (dec_pix<10)

totalmask = mask & mask4

index_r2n = hp.nest2ring(4096,np.arange(hp.nside2npix(4096))) #for converting hp arrays 
index_n2r = hp.ring2nest(4096,np.arange(hp.nside2npix(4096))) #remember the order is counter intuative

spmaps = []
for ifile, filename in enumerate(sp_files):
    print("LOADING SP {0}: {1}".format(ifile, filename))
    
    #load the SP maps
    
    #Three maps are saved differently so need to be loaded like this
    if 'sof_depth' in filename or 'stars_extmashsof0_16_20' in filename or 'y3_stellar_density' in filename:
        sp_hp = fio.read(filename)['I'].flatten()
        
        if 'ring' in filename:
            index_r2n = hp.nest2ring(4096,np.arange(hp.nside2npix(4096)))
            sp_hp = sp_hp[index_r2n]
    else:
        sp_data = fio.read(filename)
    
        #make a healpix array
        sp_hp = np.ones(hp.nside2npix(4096))*hp.UNSEEN

        if 'ring' in filename:
            sp_hp[hp.ring2nest(4096,sp_data['PIXEL'])] = sp_data['SIGNAL']
        else:
            sp_hp[sp_data['PIXEL']] = sp_data['SIGNAL']
    
    sp_patch = sp_hp[totalmask]

    if (sp_patch == hp.UNSEEN).any():
        raise RuntimeError('{0}/{1} pixels in the patch are unseen for map {2}'.format( sum(sp_patch == hp.UNSEEN),len(sp_patch),filename ))

    spmaps.append(sp_patch)
    
#Fit the PCA
print("FITTING PCA")
pca = PCA(n_components=len(spmaps))
data = np.array(spmaps)
pca.fit(data.T)

np.savetxt(outdir + 'components.txt', pca.components_)
f = open(outdir + 'sp_input.txt','w')
f.write('\n'.join(sp_files))
f.close()

#construct the PCA maps from the coefficients (components)
pcamaps = []
for imap in range(len(spmaps)):
    print("SAVING PC{0}".format(imap))
    
    pci = np.ones(hp.nside2npix(4096))*hp.UNSEEN
    #pci[totalmask] = np.sum([pca.components_[imap][i]*data[i] for i in range(len(spmaps))],axis=0)
    pci[totalmask] = np.sum(pca.components_[imap]*data.T,axis=1)
    pcamaps.append(pci)
    
    pca_filename = 'pc{0}_{1}_4096{2}.fits.gz'.format(imap, label, output_order)
    
    if output_order == 'nest':
        outarray = pci
    elif output_order == 'ring':
        outarray = pci[index_n2r]
    fio.write(outdir + pca_filename, outarray)
    
    
