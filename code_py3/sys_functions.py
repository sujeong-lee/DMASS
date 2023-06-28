# These functions are used in PCAmap weight creation for systematics, as well as
# in checking the applied weights with the SPmaps:

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

def calling_catalog(catname=None):

    catdir = ''.join([ c+'/' for c in catname.split('/')[:-1]])
    os.system('mkdir '+catdir)
    dmass = esutil.io.read(catname)
    #w_dmass = dmass['CMASS_PROB']
    #print ('Calculating DMASS systematic weights...')
    #dmass = appendColumn(dmass, name='WEIGHT', value= w_dmass )
    dmass = dmass[ dmass['CMASS_PROB'] > 0.01 ]   # for low probability galaxies

    print ('Resulting catalog size')
    print ('DMASS=', np.sum(dmass['CMASS_PROB']) )
    #print ('randoms=', randoms.size)
    return dmass #, randoms

def keepGoodRegion(des, hpInd = False, balrog=None):
    import healpy as hp
    import fitsio
    
    path = '/fs/scratch/PCON0008/warner785/bwarner/'
    LSSGoldmask = fitsio.read(path+'MASK_Y3LSSBAOSOF_22_3_v2p2.fits')
    ringhp = hp.nest2ring(4096, [LSSGoldmask['PIXEL']])
    ind_good_ring = ringhp
    
    # healpixify the catalog.
    nside=4096
    # Convert silly ra/dec to silly HP angular coordinates.
    phi = des['RA'] * np.pi / 180.0
    theta = ( 90.0 - des['DEC'] ) * np.pi/180.0

    hpInd = hp.ang2pix(nside,theta,phi,nest=False)
    keep = np.in1d(hpInd, ind_good_ring)
    des = des[keep]
    if hpInd is True:
        return ind_good_ring
    else:
        return des
    
def ra_dec_to_xyz(ra, dec):
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return (cos_ra * sin_dec,
            sin_ra * sin_dec,
            cos_dec)

def uniform_sphere(RAlim, DEClim, size=1):
    zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
    DEC = (180. / np.pi) * np.arcsin(z)
    RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)
    
    return RA, DEC

def uniform_random_on_sphere(data, size = None ):
    ra = data['RA']
    dec = data['DEC']
    
    n_features = ra.size
    ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                 (min(dec), max(dec)),
                                 size)
    #random redshift distribution
    
    data_R = np.zeros((ra_R.size,), dtype=[('RA', 'float'), ('DEC', 'float')])
    data_R['RA'] = ra_R
    data_R['DEC'] = dec_R
                              
    return data_R

def random_pixel(random_val_fracselected):
    phi = random_val_fracselected['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_val_fracselected['DEC'] ) * np.pi/180.0
    nside= 4096

    HPIX_4096 = hp.ang2pix(4096, theta, phi)

    random_val = append_fields(random_val_fracselected, 'HPIX_4096', HPIX_4096, usemask=False)
    
    return random_val

def cutPCA(sysMap, SPT, SPmap):
    
    #for systematic maps in NEST (SPmaps)
    if SPmap == True:
        sysMap['PIXEL'] = hp.nest2ring(4096, sysMap['PIXEL'])
    RA, DEC = hp.pix2ang(4096, sysMap['PIXEL'], lonlat=True)
    sysMap = append_fields(sysMap, 'RA', RA, usemask=False)
    sysMap = append_fields(sysMap, 'DEC', DEC, usemask=False)
    #print(sysMap.dtype.names)

    sysMap = keepGoodRegion(sysMap)
    
# for spt region
    if SPT == True:
        mask_spt = (sysMap['RA']>295)&(sysMap['RA']<360)|(sysMap['RA']<105)
        mask_spt = mask_spt & (sysMap['DEC']>-68) & (sysMap['DEC']<-10)
        sysMap = sysMap[mask_spt]
    
# for validation region
    else:
        mask4 =(sysMap['RA']>18)&(sysMap['RA']<43)
        mask4 = mask4 & (sysMap['DEC']>-10) & (sysMap['DEC']<10)
        sysMap = sysMap[mask4]
    
 # for training region
    #mask = (sysMap['RA']>310) & (sysMap['RA']<360)|(sysMap['RA']<7)
    #mask = mask & (sysMap['DEC']>-10) & (sysMap['DEC']<10)
    #sysMap = sysMap[mask]
    
    return sysMap

def number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = False, mocks = False): # apply systematic weights here
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2

    x = np.full(hp.nside2npix(4096), hp.UNSEEN)
    x[sysMap['PIXEL']] = sysMap['SIGNAL']
    if mocks != True:
        sysval_gal = x[dmass_chron['HPIX_4096']].copy()
    else: 
        sysval_gal = x[dmass_chron].copy()
    
    if mocks != True:
        if sys_weights == True:
            print("weights being applied...")
            h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["CMASS_PROB"][sysval_gal != hp.UNSEEN]*dmass_chron_weights[sysval_gal != hp.UNSEEN])
        else:
            h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["CMASS_PROB"][sysval_gal != hp.UNSEEN]) # -- density of dmass sample, not gold sample
    else:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron_weights[sysval_gal != hp.UNSEEN])
    
    return h, sysval_gal

def area_pixels(sysMap, frac_weight, fracDet, SPmap, custom):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2
    
# number of galaxies in each pixel:
    if SPmap != True:
        if custom == True:
            sys_signal = sysMap['SIGNAL']
            area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = frac_weight)
        else:   
            sys_signal = sysMap['SIGNAL']
            area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = sysMap['FRACDET'][sys_signal != hp.UNSEEN])

# number of galaxies in each pixel:
    else:
        sys_signal = sysMap['SIGNAL']

        sys = sysMap
        mask = np.full(hp.nside2npix(4096), hp.UNSEEN)
        frac_mask = np.in1d(fracDet["PIXEL"], sys["PIXEL"], assume_unique=False, invert=False)

        mask[sys["PIXEL"]] = sys["SIGNAL"]
        frac_sys = mask[fracDet["PIXEL"][frac_mask]]
        area,_ = np.histogram(frac_sys[frac_sys != hp.UNSEEN] , bins=pbin , weights = fracDet["SIGNAL"][frac_mask][frac_sys != hp.UNSEEN])

    
    return area

def number_density(sysMap, h, area):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2

    number_density = []
    for x in range(len(h)):
        den = h[x]/area[x]
        number_density.append(den)

    total_area = 0
    #Normalize based on total number density of used footprint:
    for x in range(len(area)):
        total_area += area[x]

    # total galaxies:
    total_h = 0
    for x in range(len(h)):
        total_h += h[x]

    #normalization: 
    total_num_density = total_h/total_area
    
    # apply normalization: 
    #print(number_density)
    norm_number_density = number_density/total_num_density
    #print(norm_number_density_ran)

    fracerr = np.sqrt(h) #1 / sqrt(number of randoms cmass galaxies in each bin)
    fracerr_norm = (fracerr/area)/total_num_density
    
    return pcenter, norm_number_density, fracerr_norm

def chi2(norm_number_density, x2_value, fracerr_norm, n, covariance, SPT):
    #chi**2 values for qualitative analysis:
    # difference of (randoms-horizontal line)**2/err_ran**2
    x1 = norm_number_density
    x2 = x2_value
    X = x1-x2
    cov = covariance
    #import pdb
    #pdb.set_trace()
    if SPT != True:
        err = fracerr_norm
        chi2 = (X)**2 / err**2 
        sum_chi2 = chi2
        chi2_reduced = sum(chi2)/(chi2.size-n) # n = 2 for linear fit, 3 for quad.
    else:
        inv_cov = np.linalg.inv(cov)
        Matrix = np.matrix(X)
        X_T = np.transpose(Matrix)
        chi2 = (Matrix)*inv_cov*X_T
        sum_chi2=float(chi2)
        chi2_reduced = sum_chi2/(len(norm_number_density)-n)
    
    return sum_chi2, chi2_reduced

def lin(pcenter,m,b1):
    return m*pcenter+b1

def quad(pcenter,a,b2,c):
    return a*pcenter**2+b2*pcenter+c

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)
