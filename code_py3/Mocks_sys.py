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

#import time
# Start = time.time()
# testing how long numpy to add column

#debugging:
#import ipdb
#ipdb.set_trace()

#read in fracDet once
path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
fracDet['PIXEL'] = hp.nest2ring(4096, fracDet['PIXEL'])


## --------------------------------------------

def number_gal(sysMap, dmass_hpix, dmass_weight, sys_weights = False): # apply systematic weights here
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    #minimum = np.min(sysMap['SIGNAL'][dim_mask]) #FWHM signal (for g filter)
    #maximum = np.percentile(sysMap['SIGNAL'][dim_mask], 99)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
    #maximum = np.max(sysMap['SIGNAL'][dim_mask])
    #print("min: ", minimum)
    #print("max: ", maximum)

    #pbin = np.linspace(-.03, .04, 1000)
    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2

    #x = np.zeros(hp.nside2npix(512))
    x = np.full(hp.nside2npix(4096), hp.UNSEEN)
    #print(x, sum(x))
    #x[sysMap['PIXEL'][dim_mask]] = sysMap['SIGNAL'][dim_mask]
#    x[sysMap['HPIX_512']] = sysMap['SIGNAL']
    x[sysMap['PIXEL']] = sysMap['SIGNAL']

    #print(hp.visufunc.mollview(x)) # this is fine
    #print(hp.UNSEEN)

    #systematic value at galaxy location:

#    sysval_gal = x[dmass_chron['HPIX_512']].copy()
    sysval_gal = x[dmass_hpix].copy()

    #which healpixels have values in the sysMap signal

    #print(sum(sysval_gal[sysval_gal != hp.UNSEEN]))
    #print(hp.UNSEEN)

    #print(x.size, sysval_gal.size, dmass_chron.size)
    #print(maximum, minimum)
    #print((sysval_gal != 0.0).any())
    
    if sys_weights == True:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["WEIGHT"][sysval_gal != hp.UNSEEN]*dmass_chron["SYS_WEIGHT"][sysval_gal != hp.UNSEEN])
    else:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_weight[sysval_gal != hp.UNSEEN]) # -- density of dmass sample, not gold sample
    #print(h)
    
    return h, sysval_gal

def area_pixels(sysMap, fracDet):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2
# number of galaxies in each pixel:
    sys_signal = sysMap['SIGNAL']

    #n,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin )

    sys = sysMap
    mask = np.full(hp.nside2npix(4096), hp.UNSEEN)

    #Only look at pixels where fracDet has value
    ##frac_mask = np.in1d(fracDet["PIXEL"], sys["PIXEL"], assume_unique=False, invert=False)

    #make an array with signals corresponding to pixel values 
    #mask[sys["PIXEL"]] = sys["SIGNAL"]

    #array only including fracDet/sys seen pixels sys signal values 
    ##frac_sys = mask[fracDet["PIXEL"][frac_mask]]
    
    fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
    fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']
    
    frac_weight = fracHp[sysMap['PIXEL']]
    assert (frac_weight != hp.UNSEEN).all() #.any()

    #weights of fracDet in the overlap applied for accurate areas
    area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = frac_weight)

    # area = units of healpixels

    return area


def number_density(sysMap, h, area):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    #minimum = np.min(sysMap['SIGNAL'][dim_mask]) #FWHM signal (for g filter)
    #maximum = np.percentile(sysMap['SIGNAL'][dim_mask], 99)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
    #print(minimum)
    #print(maximum)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2
    # change to number density: divide by area

    #fig, ax = plt.subplots()
    #ax.errorbar( pcenter, h_ran)
    #ax.legend(chi2_reduced)
    #plt.title('number of random galaxies per bin')
    #fig.savefig('ran_gal_bin.pdf')

    #print(hp.visufunc.mollview(sysMap['SIGNAL'][dim_mask]))

    # h_ran = number of galaxies
    #print("number of random galaxies: ", h_ran)

    # number density in bins: h/area

    number_density = []
    for x in range(len(h)):
        den = h[x]/area[x]
        number_density.append(den)
    
    #print("randoms number density: ", number_density_ran)


    total_area = 0
    #Normalize based on total number density of used footprint:
    for x in range(len(area)):
        total_area += area[x]

    #print("total_area: ", total_area)

    # total galaxies:
    total_h = 0
    for x in range(len(h)):
        total_h += h[x]

    #print("total galaxies: ", total_h)

    #normalization: 
    total_num_density = total_h/total_area

    #print("total number density: ", total_num_density_ran)
    
    # apply normalization: 
    #print(number_density)
    norm_number_density = number_density/total_num_density
    #print(norm_number_density_ran)

    fracerr = np.sqrt(h) #1 / sqrt(number of cmass galaxies in each bin)
    fracerr_norm = (fracerr/area)/total_num_density
    #print("normalized error: ", fracerr_ran_norm)
    
    return pcenter, norm_number_density, fracerr_norm


def chi2(norm_number_density, x2_value, fracerr_norm, n):
    #chi**2 values for qualitative analysis:
    # difference of (randoms-horizontal line)**2/err_ran**2
    x1 = norm_number_density
    x2 = x2_value
    err = fracerr_norm
    chi2 = (x1-x2)**2 / err **2 
    chi2_reduced = sum(chi2)/(chi2.size-n)  # n = 2 for linear fit, 3 for quad.
    #print("chi2: ",chi2_reduced)
    
    return chi2, chi2_reduced
    
#--------------------------different loaded files:------------------------------------------------#

#plt.rcParams.update({
#  "text.usetex": False,
#  "font.family": "Helvetica"
#})


mock_path = '/fs/scratch/PCON0008/warner785/bwarner/mocks/set2/cats/'
mock_template = 'y3_cmass_mocks_full_2_catalog_catalog{0}.'

# ('bin', 'ra', 'dec')
# plotting ra, dec as check

n_mock = 700
n_pca = 50

import time
Start = time.time()
ndens_array = np.zeros((n_pca,n_mock,12)) #<N PCA maps>,<N mocks>,<N PCA bins>
for mock_i in range(n_mock):
    
    mock_keyword = mock_template.format(mock_i)
    print("using ", mock_keyword, "...")
    dmass_spt = io.SearchAndCallFits(path = mock_path, keyword = mock_keyword)

    input_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_SP107_SPT_v2_cformat/'
    #y3/band_z/
    keyword_template = 'pc{0}_'
       
    for i_pca in range(n_pca): #50
        
        sys_weights = False
        
        input_keyword = keyword_template.format(i_pca)
        print(input_keyword)
        sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword)
        path = '/fs/scratch/PCON0008/warner785/bwarner/'
    
# ---------------------  

        dmass_chron = dmass_spt
        read_in = time.time()
        files_read = read_in - Start
        
        #import pdb
        #pdb.set_trace()
        
       # column for weights  == 1
        dmass_weight = np.full(dmass_chron.size, 1)
       # hpix from ra, dec ==> create new column
        
        phi = dmass_chron['ra'] * np.pi / 180.0
        theta = ( 90.0 - dmass_chron['dec'] ) * np.pi/180.0 
        dmass_hpix = hp.ang2pix(4096, theta, phi)
        
#        fig, ax = plt.subplots()
#        ax.plot( dmass_chron['ra'], dmass_chron['dec'], "b,")
#        fig.savefig('mock_footprint.pdf')
        area_start = time.time()
        area = area_pixels(sysMap, fracDet)
        area_finish = time.time()
        area_time = area_finish - area_start
        print("found area in ",area_time, "seconds")
#        pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)
        
        h, sysval_gal = number_gal(sysMap, dmass_hpix, dmass_weight, sys_weights = False) # change this to true if sys weights run
        
        pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)
    
        ndens = norm_number_density
        print("pca: ",i_pca, "mock: ",mock_i)
        ndens_array[i_pca,mock_i] = ndens
#        frac_nor.append(fracerr_norm)
        
    #plotting:

##        fig, ax = plt.subplots()
##        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "mock dmass")
#        ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms")
##        plt.legend()
##        xlabel = input_keyword
##        plt.xlabel(xlabel)
##        plt.ylabel("n_gal/n_tot 4096")
    #plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
    #plt.ylim(bottom=0.85)
##        plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
##        plt.title(xlabel+mock_keyword+' systematics mock check')
##        fig.savefig(xlabel+mock_keyword+'sys_check_mock.pdf')        
    
    # save data for each mock -- for all pcas used:
    End = time.time()
    mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/mocks/'    
    for i_pca in range(n_pca):
        np.savetxt(mock_outdir+"mocks_second_set_pca"+str(i_pca)+".txt", ndens_array[i_pca])
    
    Time = End - Start
    print("time taken for mock: ",Time, "seconds")
    Start = time.time()
#    mockfiles.append(num_den)
    
#print(mockfiles)
#np.savetxt('mocks_density.txt', mockfiles)
#    np.savetxt('mocks_frac'+mock_keyword+'.txt', frac_nor)
        

# define arrays and fill as comp. 
# 50 files 
# test with 3 mocks

# first set of mocks:
# np.savetxt(mock_outdir+"mocks_set20_pca"+str(i_pca)+".txt", ndens_array[i_pca])