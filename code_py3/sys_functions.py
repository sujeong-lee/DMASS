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

def number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = False, mocks = False, iterative = False): # apply systematic weights here
    
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
            if iterative != True:
                h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["CMASS_PROB"][sysval_gal != hp.UNSEEN]) # -- density of dmass sample, not gold sample
        if iterative == True:
            print("weights being applied...")
            h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["CMASS_PROB"][sysval_gal != hp.UNSEEN]*dmass_chron_weights[sysval_gal != hp.UNSEEN])
    else:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron_weights[sysval_gal != hp.UNSEEN])
    
    return h, sysval_gal

def area_pixels(sysMap, frac_weight, custom):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2
    
# number of galaxies in each pixel:
    #if debug != True:
    if custom == True:
        sys_signal = sysMap['SIGNAL']
        area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = frac_weight)
    else:   
        sys_signal = sysMap['SIGNAL']
        area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = sysMap['FRACDET'][sys_signal != hp.UNSEEN])
            
    #else:  (for SP if other method isn't working)
# number of galaxies in each pixel:
        #debug this to be like the PCA set
        #sys_signal = sysMap['SIGNAL']

        #sys = sysMap
        #mask = np.full(hp.nside2npix(4096), hp.UNSEEN)
        #frac_mask = np.in1d(fracDet["PIXEL"], sys["PIXEL"], assume_unique=False, invert=False)

        #mask[sys["PIXEL"]] = sys["SIGNAL"]
        #frac_sys = mask[fracDet["PIXEL"][frac_mask]]
        #area,_ = np.histogram(frac_sys[frac_sys != hp.UNSEEN] , bins=pbin , weights = fracDet["SIGNAL"][frac_mask][frac_sys != hp.UNSEEN])

    
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

def go_through_mock(full_path, input_keyword, current_map, fracHp, ndens_array, dmass_hpix, dmass_weight):
        
    map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_cut/'
    print("fits file", input_keyword)
    sysMap = io.SearchAndCallFits(path = map_path, keyword = current_map+'.fits')
    
    frac_weight = fracHp[sysMap['PIXEL']]
    sysMap = sysMap[frac_weight != hp.UNSEEN]
    
    h, sysval_gal = number_gal(sysMap, dmass_hpix, dmass_weight, sys_weights = False, mocks = True)
    area = area_pixels(sysMap, frac_weight, custom = True)
    pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)

    ndens = norm_number_density
    
    return ndens

def go_through_maps(full_path, input_keyword, current_map, fracHp, dmass_hpix, dmass_weight, star_map = False, flatten = False):
        
        print("fits file", input_keyword)
        sysMap = io.SearchAndCallFits(path = full_path, keyword = input_keyword)
        if star_map == True:
            if flatten == True:
                flat = sysMap['I'].flatten()
                pixels = np.zeros(flat.size)
            else:
                pixels = np.zeros(sysMap.size)
            for x in range(pixels.size):
                if x>0:
                    pixels[x]=pixels[x-1]+1
            sysMap = np.zeros( len(pixels), dtype=[('PIXEL','int'), ('SIGNAL','float')])
            if flatten == True:
                sysMap['SIGNAL'] = flat
            else:
                signal = sysMap['I']
                sysMap['SIGNAL'] = signal
            sysMap['PIXEL'] = pixels
    
#    path = '/fs/scratch/PCON0008/warner785/bwarner/'
        sysMap = cutPCA(sysMap,SPT=True,SPmap=True)
        
        outdir = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_cut/'
        os.makedirs(outdir, exist_ok=True)
        esutil.io.write( outdir+current_map+'.fits', sysMap, overwrite=True)


def go_through_SP(full_path, input_keyword, current_map, fracHp, cov, i_pca, dmass_chron, dmass_chron_weights, sys_weights = False):

    map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_cut/'
    print("fits file", input_keyword)
    sysMap = io.SearchAndCallFits(path = map_path, keyword = current_map+'.fits')

    frac_weight = fracHp[sysMap['PIXEL']]
    sysMap = sysMap[frac_weight != hp.UNSEEN]
    
    covariance_i = cov[i_pca]
    covariance = np.copy(covariance_i)
    covariance[0][-1]=0
    covariance[-1][0]=0
    if is_pos_def(covariance) == True:
        cov_matrix = covariance
    else:
        print("NOT POSITIVE DEFINITE") # try not zeroing out, or taking anout zeroing out //
        covariance[1][-1] = 0
        covariance[-1][1] = 0
        covariance[0][-2] = 0
        covariance[-2][0] = 0
        if is_pos_def(covariance) == True:
            cov_matrix = covariance
        else: 
            print("STILL NOT POSITIVE DEFINITE")
    
    diag_cov = np.diagonal(covariance)
    error_cov = np.sqrt(diag_cov)

    h, sysval_gal = number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = sys_weights)
    area = area_pixels(sysMap, frac_weight, custom = True)
    pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)
    
    return pcenter, norm_number_density, error_cov, fracerr_norm


def plot_figure(input_keyword, current_map, run_name, pcenter, norm_number_density, error_cov, fracerr_norm, sys_weights = False, SPT = True, custom = True):
    
    fig, ax = plt.subplots()
    if SPT == True: # change based on used weights ---------------------------------------
        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass spt, weights "+ run_name)
            #ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms spt")
    else:
        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass val")
            #ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms val")
    plt.legend()
    xlabel = input_keyword
    plt.xlabel(current_map)
    plt.ylabel("n_gal/n_tot 4096")
    plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
    if sys_weights == True:
        if SPT == True:
            plt.title(current_map+' SPT region with '+run_name+'weights applied')
            if custom == True:
                fig.savefig('../SPmap_custom/'+run_name+current_map+' spt_check.pdf')
            else:
                fig.savefig('../SPmap_official/'+run_name+current_map+' spt_check.pdf')
        else:
            plt.title(current_map+' VAL region with weights applied')
            fig.savefig('../SPmap_check/'+current_map+' val.pdf')
    else:
        if SPT == True:
            plt.title('systematics check, no weights: '+current_map+' in spt')
            if custom == True:
                fig.savefig('../SPmap_custom/'+current_map+'spt.pdf')
            else:
                fig.savefig('../SPmap_official/'+current_map+'spt.pdf')
        else:
            plt.title('systematics check, no weights: '+current_map+' in val')
            fig.savefig('../SPmap_check/'+current_map+'val.pdf')


'''          
extra bit:

        print("fits file", input_keyword)
        # use fitsio.read in separate file:
    sysMap = io.SearchAndCallFits(path = full_path, keyword = input_keyword)

    path = '/fs/scratch/PCON0008/warner785/bwarner/'
#sysMap['PIXEL'] = hp.nest2ring(4096, sysMap['PIXEL'])
    sysMap = cutPCA(sysMap,SPT=True,SPmap=True)
    if star_map == True:
        if flatten == True:
            flat = sysMap['I'].flatten()
            pixels = np.zeros(flat.size)
        else:
            pixels = np.zeros(sysMap.size)
        for x in range(pixels.size):
            if x>0:
                pixels[x]=pixels[x-1]+1
        sysMap = np.zeros( len(sysMap), dtype=[('PIXEL','int'), ('SIGNAL','float')])
        sysMap['PIXEL'] = pixels
        if flatten == True:
            sysMap['SIGNAL'] = flat
        else:
            sysMap['SIGNAL'] = sysMap['I']
            
'''  