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

def cutPCA(sysMap, SPT, SPmap, maglim = False, cat = False, stars = False):
    
    #for systematic maps in NEST (SPmaps)
    if cat != True:
        if SPmap == True and stars == False:
            sysMap['PIXEL'] = hp.nest2ring(4096, sysMap['PIXEL'])
        RA, DEC = hp.pix2ang(4096, sysMap['PIXEL'], lonlat=True)
        sysMap = append_fields(sysMap, 'RA', RA, usemask=False)
        sysMap = append_fields(sysMap, 'DEC', DEC, usemask=False)
    #print(sysMap.dtype.names)
    
    if maglim != True:
        sysMap = keepGoodRegion(sysMap)
    else:
        sysMap = keepMaglim(sysMap)
    # different combinations -- maglim spmap area, spt spmap area etc. 
    
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

def number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = False, mocks = False, iterative = False, equal_area = True): # apply systematic weights here
    
    if equal_area != True:
        minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
        maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
        pbin, pstep = np.linspace( minimum, maximum, 11, retstep=True)
        pcenter = pbin[:-1] + pstep/2
        
        #data = sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN]
        #fig, ax = plt.subplots()
        #ax.hist(data, bins = pbin)
        #fig.savefig('../Final_Checks/''fgcm.pdf')
        
    else:    
        import numpy 
        # create binning in area option: 10 bins
        percentile  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Calculate the bin edges
        bin_edges = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], percentile)
        pbin = bin_edges 
        data = sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN]
        digitized = numpy.digitize(data, pbin)
        pmean = [data[digitized == i].mean() for i in range(1, len(pbin))]

    x = np.full(hp.nside2npix(4096), hp.UNSEEN)
    x[sysMap['PIXEL']] = sysMap['SIGNAL']
    #print("sysmap: ", sysMap['SIGNAL'])
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


def area_pixels(sysMap, frac_weight, custom = True, equal_area = True):
    
    if equal_area != True:
        minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
        maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
        #print(maximum)
        pbin, pstep = np.linspace( minimum, maximum, 11, retstep=True)
        pcenter = pbin[:-1] + pstep/2
        
    else:    
        import numpy 
        # create binning in area option:
        percentile  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Calculate the bin edges
        bin_edges = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], percentile)
        pbin = bin_edges 
        data = sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN]
        digitized = numpy.digitize(data, pbin)
        pmean = [data[digitized == i].mean() for i in range(1, len(pbin))]
    
# number of galaxies in each pixel:
    #if debug != True:
    
    if custom == True:
        sys_signal = sysMap['SIGNAL']
        print(frac_weight[sys_signal != hp.UNSEEN].shape)
        print(sys_signal[sys_signal != hp.UNSEEN].shape)
        area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = frac_weight[sys_signal != hp.UNSEEN])
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

def number_density(sysMap, h, area, equal_area = True):
    
    if equal_area != True:
        minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
        maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
        pbin, pstep = np.linspace( minimum, maximum, 11, retstep=True)
        pcenter = pbin[:-1] + pstep/2
        
    else:    
        import numpy 
        # create binning in area option:
        percentile  = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        # Calculate the bin edges
        bin_edges = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], percentile)
        pbin = bin_edges 
        data = sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN]
        digitized = numpy.digitize(data, pbin)
        pcenter = [data[digitized == i].mean() for i in range(1, len(pbin))]
        # actually a pmean, but keeping the same for variable sake

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
    #print("total_num_density: ", total_num_density, "here")
    
    # apply normalization: 
    #print(number_density)
    norm_number_density = number_density/total_num_density
    #print(norm_number_density_ran)

    fracerr = np.sqrt(h) #1 / sqrt(number of randoms cmass galaxies in each bin)
    fracerr_norm = (fracerr/area)/total_num_density
    
    return pcenter, norm_number_density, fracerr_norm

def chi2(norm_number_density, x2_value, fracerr_norm, n, covariance, SPT = False): ##########
    #chi**2 values for qualitative analysis:
    # difference of (randoms-horizontal line)**2/err_ran**2
    x1 = norm_number_density
    #print("number_density: ", x1)
    x2 = x2_value
    #print("x2: ", x2)
    X = x1-x2
    #print("difference: ", X)
    cov = covariance
    #print(cov)
    #import pdb
    #pdb.set_trace()
    if SPT != True:
        err = fracerr_norm
        chi2 = (X)**2 / err**2 
        sum_chi2 = sum(chi2)
        chi2_reduced = sum(chi2)/(chi2.size-n) # n = 2 for linear fit, 3 for quad.
    else:
        inv_cov = np.linalg.inv(cov)
        #if n == 0:
            #np.savetxt('/fs/scratch/PCON0008/warner785/bwarner/covariance.txt', cov)
            #np.savetxt('/fs/scratch/PCON0008/warner785/bwarner/inv_covariance.txt', inv_cov)
        Matrix = np.matrix(X)
        X_T = np.transpose(Matrix)
        chi2 = (Matrix)*inv_cov*X_T
        sum_chi2=float(chi2)
        chi2_reduced = sum_chi2/(len(norm_number_density)-n)
    
    return sum_chi2, chi2_reduced

def lin(pcenter,m,b1):
    import numpy as np
    p = np.array(pcenter)
    return m*p+b1

def quad(pcenter,a,b2,c):
    p = np.array(pcenter)
    return a*p**2+b2*p+c

def cubic(pcenter, a2, b3, c2, d):
    p = np.array(pcenter)
    return a2*p**3+b3*p**2+c2*p+d

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) >= 0)

def go_through_mock(full_path, input_keyword, current_map, y, fracHp, ndens_array, dmass_hpix, dmass_weight):
    
    if y == 8:
        map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_star/'
    else:
        map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_cut/'
    print("fits file", input_keyword)
    sysMap = io.SearchAndCallFits(path = map_path, keyword = current_map+'.fits')
    
    frac_weight = fracHp[sysMap['PIXEL']]
    sysMap = sysMap[frac_weight != hp.UNSEEN]
    
    if y == 3:
        equal_area = False
    else:
        equal_area = True
    
    h, sysval_gal = number_gal(sysMap, dmass_hpix, dmass_weight, sys_weights = False, mocks = True, equal_area = equal_area)
    area = area_pixels(sysMap, frac_weight, custom = True, equal_area = equal_area)
    pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area, equal_area = equal_area)

    ndens = norm_number_density
    
    return ndens

def go_through_maps(full_path, input_keyword, current_map, fracHp, dmass_hpix, dmass_weight, star_map = False, flatten = False, skip = True):
        
        print("fits file", input_keyword)
        #sysMap = io.SearchAndCallFits(path = full_path, keyword = input_keyword)
        sysMap = esutil.io.read('/fs/scratch/PCON0008/warner785/bwarner/csfd_formated.fits')
        if skip == False:
            if star_map == True:
                if flatten == True:
                    flat = sysMap['I'].flatten()
                    pixels = np.zeros(flat.size)
                else:
                    pixels = np.zeros(sysMap.size)
                for x in range(pixels.size):
                    if x>0:
                        pixels[x]=pixels[x-1]+1
                signal = np.array(sysMap)
                sysMap = np.zeros( len(pixels), dtype=[('PIXEL','int'), ('SIGNAL','float')])
                if flatten == True:
                    sysMap['SIGNAL'] = flat
                else: 
                    sysMap['SIGNAL'] = signal
                sysMap['PIXEL'] = pixels
    
#    path = '/fs/scratch/PCON0008/warner785/bwarner/'
        print(star_map)
        sysMap = cutPCA(sysMap,SPT=True,SPmap=True, maglim=False, stars = star_map)
        
        outdir = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_csfd/'
        os.makedirs(outdir, exist_ok=True)
        esutil.io.write( outdir+current_map+'.fits', sysMap, overwrite=True)


def go_through_SP(full_path, input_keyword, current_map, y, fracHp, cov, i_pca, dmass_chron, dmass_chron_weights, random_chron, name, sys_weights = False, maglim = False, csfd = False, SPT = False):  ##############
    
    if maglim == True:
        print("maglim...")
        map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_maglim2/'
    if SPT == True:
        if y == 8:
            map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_star/'
        else:
            map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_cut/'
    if SPT == False:
        map_path = '/fs/scratch/PCON0008/warner785/bwarner/SPmaps_val/'
        
    if csfd == True:
        map_path = '/fs/scratch/PCON0008/warner785/bwarner/'
        sysMap = io.SearchAndCallFits(path = map_path, keyword = 'csfd_spt.fits')
    #print("fits file", input_keyword)
    
    else:
        sysMap = io.SearchAndCallFits(path = map_path, keyword = current_map+'.fits')

    frac_weight = fracHp[sysMap['PIXEL']]
    sysMap = sysMap[frac_weight != hp.UNSEEN]
    
    #covariance = cov[i_pca]
#    covariance = np.copy(covariance_i)
    #covariance[0][-1]=0
    #covariance[-1][0]=0
    diag_cov = np.diagonal(cov)
    error_cov = np.sqrt(diag_cov)
    
#    if is_pos_def(covariance) == True:
#        cov_matrix = covariance
#    else:
#        print("NOT POSITIVE DEFINITE") # try not zeroing out, or taking anout zeroing out //
#        covariance[1][-1] = 0
#        covariance[-1][1] = 0
#        covariance[0][-2] = 0
#        covariance[-2][0] = 0
#        if is_pos_def(covariance) == True:
#            cov_matrix = covariance
#        else:
#            print("STILL NOT POSITIVE DEFINITE")
#            covariance[0][-3] = 0
#            covariance[-3][0] = 0
#            covariance[1][-2] = 0
#            covariance[-2][1] = 0
#            covariance[2][-1] = 0
#            covariance[-1][2] = 0
#            if is_pos_def(covariance) == True:
#                cov_matrix = covariance
#            else:
#                print("STILL NOT POSITIVE DEFINITE x2")
#                cov_matrix = error_cov
    if y == 3:
        equal_area = False
    else:
        equal_area = True
    h, sysval_gal = number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = sys_weights, equal_area = equal_area)
    h_ran,_= number_gal(sysMap, random_chron, None, sys_weights = False, equal_area = equal_area)
    area = area_pixels(sysMap, frac_weight, custom = True, equal_area = equal_area)
    pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area, equal_area = equal_area)
    pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area, equal_area = equal_area)
    
    #for checking figure:
    #np.savetxt(current_map+"pcenter.txt", pcenter)
    #np.savetxt(current_map+name+"number_density.txt", norm_number_density)
    #np.savetxt(current_map+name+"error.txt", error_cov) #error_cov, #fracerr_norm
    
    return pcenter, norm_number_density, norm_number_density_ran, error_cov, fracerr_norm, fracerr_ran_norm, sysval_gal


def plot_figure(input_keyword, current_map, y, run_name, pcenter, norm_number_density, error_cov, fracerr_norm, norm_number_density_ran, fracerr_ran_norm, sys_weights = False, SPT = False, custom = True): #############
        
    fig, ax = plt.subplots()
    
    if SPT == True: # change based on used weights ---------------------------------------
        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass spt, weights "+ run_name)
        ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms spt")
    else:
        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass val")
        ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms val")
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
                fig.savefig('../Final_Checks/'+run_name+current_map+' spt_check.pdf')
            else:
                fig.savefig('../SPmap_official/'+run_name+current_map+' spt_check.pdf')
        else:
            plt.title(current_map+' VAL region with weights applied')
            fig.savefig('../SPmap_check/'+current_map+' val.pdf')
    else:
        if SPT == True:
            plt.title('systematics check, no weights: '+current_map+' in spt')
            if custom == True:
                fig.savefig('../Final_Checks/'+current_map+'spt.pdf')
            else:
                fig.savefig('../SPmap_official/'+current_map+'spt.pdf')
        else:
            plt.title('systematics check, no weights: '+current_map+' in val')
            fig.savefig('../SPmap_check/'+current_map+'val.pdf')
    plt.close(fig)
    
    #if SPmap == True:
        #np.savetxt(current_map+run_name+"number_density.txt", norm_number_density)
        #np.savetxt(current_map+run_name+"error.txt", error_cov)


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

def reconstruct(matrix):
    from numpy import linalg
    
    # w = eigenvalue, v = eigenvector
    w, v = linalg.eig(matrix)
    #print(w)
    w_new = []
    for x in range(w.size):
        if w[x]<0:
            print(w[x])
            new = np.absolute(w[x])
        else:
            new = w[x]
        w_new.append(new)
    #print(w_new)
        
    # reconstruct matrix with new values:
    inv_v = np.linalg.inv(v)
    mat_inv = np.matrix(inv_v)
    vec_w = np.array(w_new)
    #I = np.identity(12, dtype = float)
    Lambda = np.diag(vec_w)
    mat_v = np.matrix(v)
    cov = mat_v*Lambda*mat_inv
    
    #check if this worked:
    check = np.all(np.linalg.eigvals(cov) >= 0)
    if check == True:
        print("working properly")
    else: 
        print("need to fix")

    return cov

def invertible(matrix):
    import numpy
    from numpy import linalg
    invertible = False
    #np.savetxt('covariance_example.txt', matrix)
    #print(type(matrix[0][0]))
    det = np.linalg.det(matrix)
    if det != 0:
        invertible = True
    else:
        invertible = False
        
    return invertible
 
def is_sym(mat):
    import numpy as np
    N = 10
    sym = True
    tr = np.matrix.transpose(mat)
    for i in range(N):
        for j in range(N):
            if (mat[i][j] != tr[i][j]):
                print("not symmetric")
                sym = False
    return sym

            
# python3 program of the above approach
N = 10
 
# Function to convert the given matrix
# into a symmetric matrix by replacing
# transpose elements with their mean
def symmetric(mat):
 
    # Loop to traverse lower triangular
    # elements of the given matrix
    for i in range(0, N):
        for j in range(0, N):
            if (j < i):
                mat[i][j] = mat[j][i] = (mat[i][j] +
                                         mat[j][i]) // 2
    return mat

def assign_weight(pcenter, norm_number_density, y, fracerr_norm, sysval_gal, cov_matrix, run_name, current_map, chi2_weight, cov_chi2, error_cov, i_pca):
    
    SPT = True
    linear = True
    q = False
    cub = False
    
    if linear == True:
        fig,ax = plt.subplots(1,1)
        params = scipy.optimize.curve_fit(lin, pcenter, norm_number_density, sigma = cov_matrix)
        [m, b1] = params[0]
        print("linear fit variables: ",m, b1)
        
        ax.plot(pcenter,lin(pcenter,m, b1),"r--")
        plt.title(current_map+' systematic linear trendline')
        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
        fig.savefig('/users/PCON0003/warner785/DMASSY3/Oct23/' +current_map+'SP_WEIGHT.pdf')
        trend_chi2, trend_chi2_reduced = chi2(norm_number_density, lin(pcenter, m, b1), fracerr_norm, 2, cov_chi2, SPT = SPT)
        print('linear trend_chi2: ', trend_chi2_reduced)
        chi2_weight.append(trend_chi2_reduced)
        weight_object = (1/lin(sysval_gal, m,b1))
        weight_object[sysval_gal == hp.UNSEEN] = 0 # check into this-- nothing should be zero
        avg = np.average(weight_object[weight_object!=0])
        weight_object = weight_object/avg
        
    if q == True:
        fig,ax = plt.subplots(1,1)
        params = scipy.optimize.curve_fit(quad, pcenter, norm_number_density, sigma = cov_matrix)             
        [a, b2, c] = params[0]
        print("quadratic fit variables: ",a, b2, c)

        ax.plot(pcenter,quad(pcenter,a,b2,c),"r--")
        plt.title(current_map+' systematic quadatric trendline')
        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
        fig.savefig('/users/PCON0003/warner785/DMASSY3/Sept23/' +current_map+'SP_WEIGHT.pdf')
        trend2_chi2, trend2_chi2_reduced = chi2(norm_number_density, quad(pcenter, a, b2, c), fracerr_norm, 3, cov_chi2, SPT = SPT)
        chi2_weight.append(trend2_chi2_reduced)
        weight_object = (1/quad(sysval_gal, a,b2,c))
        weight_object[sysval_gal == hp.UNSEEN] = 0
        avg = np.average(weight_object[weight_object!=0])
        weight_object = weight_object/avg
        
    #testing higher order--
    if cub == True:
        fig,ax = plt.subplots(1,1)
        params = scipy.optimize.curve_fit(cubic, pcenter, norm_number_density, sigma = cov_matrix)             
        [a2, b3, c2, d] = params[0]
        print("cubic fit variables: ",a2, b3, c2, d)

        ax.plot(pcenter,cubic(pcenter,a2,b3,c2,d),"r--")
        plt.title(current_map+' systematic cubic trendline')
        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
        fig.savefig('/users/PCON0003/warner785/DMASSY3/Oct23/' +current_map+'SP_WEIGHT.pdf')
        trend3_chi2, trend3_chi2_reduced = chi2(norm_number_density, cubic(pcenter,a2,b3,c2,d), fracerr_norm, 4, cov_chi2, SPT = SPT)
        chi2_weight.append(trend3_chi2_reduced)
        weight_object = (1/cubic(sysval_gal, a2,b3,c2,d))
        weight_object[sysval_gal == hp.UNSEEN] = 0
        avg = np.average(weight_object[weight_object!=0])
        weight_object = weight_object/avg
        
    dmass_chron_sys_weight = weight_object
    print("weights being applied: ",dmass_chron_sys_weight, "weights that are zero: ", dmass_chron_sys_weight[dmass_chron_sys_weight==0].size)
    plt.close()
    
    outdir = '/fs/scratch/PCON0008/warner785/bwarner/oct23_tests/'
    os.makedirs(outdir, exist_ok=True)
    print("saving files...")
     # ONLY SAVE WEIGHTS COLUMN FOR FASTER RUN #
    esutil.io.write( outdir+run_name+'sp'+str(i_pca)+'weight.fits', dmass_chron_sys_weight, overwrite=True)
    
def keepMaglim(des, hpInd = False, balrog=None):
    
    import healpy as hp
    import fitsio
    
    path = '/fs/project/PCON0008/des_y3/maglim/mask/'
    LSSGoldmask = fitsio.read(path+'y3_gold_2.2.1_RING_joint_redmagic_v0.5.1_wide_maglim_v2.2_mask.fits.gz')
    #ringhp = hp.nest2ring(4096, [LSSGoldmask['HPIX']])
    ringhp = LSSGoldmask['HPIX']
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
    

# OPENING COVARIANCE MATRICES: -----------------------
'''
if first_set == True:
    cov = []
    m_pca = 50 #50
    cov_template = 'cov{0}'
    for i_pca in range(m_pca): #n_pca
        cov_input= cov_template.format(i_pca)
        cov.append(cov_input)

    cov_template = 'cov_fixed_{0}' #'cov_initial_{0}' #'cov_all_{0}'
    for i_pca in range(m_pca): #n_pca
        cov_keyword = cov_template.format(i_pca)
        #print(cov_keyword)
        with open(mock_outdir + cov_keyword + '.txt') as mocks:
            array1 = [x.split() for x in mocks]
            array2 = np.array(array1)
            cov[i_pca] = array2.astype(float)
        mocks.close()

else:
    cov = []
    m_pca = 56 #56
    cov_template = 'cov{0}'
    for i_pca in range(m_pca): #n_pca
        cov_input= cov_template.format(i_pca)
        cov.append(cov_input)

    cov_template = 'covar107_{0}'
    for i_pca in range(m_pca): #n_pca
        cov_keyword = cov_template.format(i_pca+50)
        #print(cov_keyword)
        with open(mock_outdir + cov_keyword + '.txt') as mocks:
            array1 = [x.split() for x in mocks]
            array2 = np.array(array1)
            cov[i_pca] = array2.astype(float)
        mocks.close()
'''
# ----------------------------------------------------

        #covariance_i = cov[i_pca]
        #covariance = np.copy(covariance_i)
        #covariance[0][-1]=0
        #covariance[-1][0]=0
        
        #covariance[1][-1] = 0
        #covariance[-1][1] = 0
        #covariance[0][-2] = 0
        #covariance[-2][0] = 0

'''
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
                covariance[0][-3] = 0
                covariance[-3][0] = 0
                covariance[1][-2] = 0
                covariance[-2][1] = 0
                covariance[2][-1] = 0
                covariance[-1][2] = 0
                if is_pos_def(covariance) == True:
                    cov_matrix = covariance
                else:
                    print("STILL NOT POSITIVE DEFINITE x2")
                    if invertible(covariance) == True:
                        cov_matrix = covariance
                    else:
                        print("NOT INVERTIBLE")
                        cov_new = reconstruct(covariance)
                        if invertible(cov_new) == True:
                            cov_matrix = cov_new
                        else:
                            print("STILL NOT INVERTIBLE")
                            cov_matrix = error_cov
#'''

'''    
        else:
            print("FIXING")
            #np.savetxt(run_name+'covariance.txt', covariance)
            cov_new = reconstruct(cov_sym)
            if is_pos_def(cov_new) == True:
                cov_matrix = cov_new
                cov_chi2 = cov_new
            else:
                print("NOT WORKING")
                cov_matrix = error_cov
                vec_c = np.array(cov_matrix)
                diag_matrix = np.diag(vec_c)
                cov_chi2 = diag_matrix
            #np.savetxt(run_name+'covariance.txt', covariance)

#'''   