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
label = 'SP107_validationregion'
output_order = 'ring'

test = False
resume=True

if test == True:
    outdir = './pca_{0}_test/'.format(label)
else:
    outdir = './pca_{0}/'.format(label)
if os.path.exists(outdir) == False: #if the output directory does not exist
    os.mkdir(outdir) #make it

#path to the observational SP maps on NERSC
sp_dir1 = '/fs/scratch/PCON0008/warner785/bwarner/PCA/band_g/'
sp_dir2 = '/fs/scratch/PCON0008/warner785/bwarner/PCA/band_r/'
sp_dir3 = '/fs/scratch/PCON0008/warner785/bwarner/PCA/band_i/'
sp_dir4 = '/fs/scratch/PCON0008/warner785/bwarner/PCA/band_z/'
sp_dir5 = '/fs/scratch/PCON0008/warner785/bwarner/PCA/sof_depth/'
sp_files =  [sp_dir1 + filename for filename in os.listdir(sp_dir1)] + \
            [sp_dir2 + filename for filename in os.listdir(sp_dir2)] + \
            [sp_dir3 + filename for filename in os.listdir(sp_dir3)] + \
            [sp_dir4 + filename for filename in os.listdir(sp_dir4)] + \
            [sp_dir5 + filename for filename in os.listdir(sp_dir5)]

#paths to the astrophysical SP maps on NERSC
starfile1 = '/fs/scratch/PCON0008/warner785/bwarner/PCA/stars/stars_extmashsof0_16_20_zeros_footprint_nside_4096_nest.fits.gz'
sp_files.append(starfile1)

#This one has UNSEEN values and might be redundant?
#will remove for now

#starfile2 = '/global/cfs/cdirs/des/jelvinpo/sysmaps/y3/stars/y3_stellar_density_4096_ringbaosample_v2p2.fits'
#sp_files.append(starfile2)

extfile = '/fs/scratch/PCON0008/warner785/bwarner/PCA/extinction/ebv_sfd98_fullres_nside_4096_nest_equatorial_des.fits.gz'
sp_files.append(extfile)

if test == True:
    sp_files = sp_files[-4:]

nmaps = len(sp_files)

#load the LSS mask (slightly different format to usual)
lssmask_file = '/fs/scratch/PCON0008/warner785/bwarner/MASK_Y3LSSBAOSOF_22_3_v2p2.fits'
lssmask = fio.read(lssmask_file)
mask = np.zeros(hp.nside2npix(4096))
mask[hp.ring2nest(4096, lssmask['PIXEL'])] = 1.
mask = mask.astype('bool')

ra_pix,dec_pix = hp.pix2ang(4096,np.arange(hp.nside2npix(4096)),nest=True,lonlat=True) #ra and dec of all pixels

# validation region mask
mask4 = (ra_pix>18)&(ra_pix<43)
mask4 = mask4 & (dec_pix>-10) & (dec_pix<10)

# SPT region mask
mask_train = (ra_pix>310) & (ra_pix<360)|(ra_pix['RA']<7)
mask_train = mask_train & (dec_pix>-10) & (dec_pix<10)
mask_spt = not mask_4 & not mask_train

# SPT *including* validation mask
mask_all = not mask_train  # not sliver

totalmask = mask & mask4 # change depending on region

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
    
    #standardize the SP map (mean=0, std=1)
    sp_patch = (sp_patch-np.mean(sp_patch))/np.std(sp_patch)

    if (sp_patch == hp.UNSEEN).any():
        raise RuntimeError('{0}/{1} pixels in the patch are unseen for map {2}'.format( sum(sp_patch == hp.UNSEEN),len(sp_patch),filename ))

    spmaps.append(sp_patch)
    
#Fit the PCA
print("FITTING PCA")
pca = PCA(n_components=len(spmaps))
data = np.array(spmaps)
pca.fit(data.T)

#save the PCA
np.savetxt(outdir + 'components.txt', pca.components_)
np.savetxt(outdir + 'explained_variance.txt', pca.explained_variance_)
f = open(outdir + 'sp_input.txt','w')
f.write('\n'.join(sp_files))
f.close()
np.save(outdir + 'pca_object.npy', pca)

#construct the PCA maps from the coefficients (components)

#since the mask of all the PCs are the same
#we can use a single array for the maps
#this might help to conserve memory
pci = np.ones(hp.nside2npix(4096))*hp.UNSEEN #since the mask of all the PCs are the smae 

for imap in range(len(spmaps)):
    print("SAVING PC{0}".format(imap))

    pca_filename = 'pc{0}_{1}_4096{2}.fits.gz'.format(imap, label, output_order)
    if os.path.exists(outdir + pca_filename)==True and resume==True:
        print("SKIPPING PC{0}, file already found".format(imap))
        continue
    else:
        print("SAVING PC{0}".format(imap))
        
    
    pci[totalmask] = np.sum(pca.components_[imap]*data.T,axis=1)
    
    if output_order == 'nest':
        fio.write(outdir + pca_filename, pci)
    elif output_order == 'ring':
        fio.write(outdir + pca_filename, pci[index_n2r])

    
