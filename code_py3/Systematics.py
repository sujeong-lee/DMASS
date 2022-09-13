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

#debugging:
#import ipdb
#ipdb.set_trace()

# calling map 
path = '/fs/scratch/PCON0008/warner785/bwarner/'
LSSGoldmask = fitsio.read(path+'MASK_Y3LSSBAOSOF_22_3_v2p2.fits') #BAO is different measurement, may be different from this analysis
# 'Y1LSSmask_v2_redlimcut_il22_seeil4.0_4096ring.fits'
#GoldMask = callingEliGoldMask()
LSSMask = LSSGoldmask
GoldMask = LSSGoldmask

pixra, pixdec = hp.pix2ang(nside=4096,ipix=GoldMask['PIXEL'],nest=True,lonlat=True)

LSSMask = LSSMask[pixdec >-3.0 ]
#GoldMask_st82 = Cuts.SpatialCuts(GoldMask, ra=320, ra2=360, dec=-2, dec2=2)
GoldMask_st82 = GoldMask[ pixdec > -3.0 ]
GoldMask_spt = GoldMask[ pixdec < -3.0 ]
#GoldMask_spt = Cuts.SpatialCuts(GoldMask_spt, ra=0, ra2 = 100, dec=-52, dec2 = -48)

pixarea = hp.nside2pixarea( 4096, degrees = True)
sptnpix = GoldMask_spt['PIXEL'].size #hp.get_map_size( GoldMask_spt['PIXEL'] )
st82npix =  GoldMask_st82['PIXEL'].size # hp.get_map_size( GoldMask_st82 )
SPTMaparea = pixarea * sptnpix
ST82Maparea = pixarea * st82npix

def calling_lens_catalog(catname=None):

    catdir = ''.join([ c+'/' for c in catname.split('/')[:-1]])
    os.system('mkdir '+catdir)
    dmass = esutil.io.read(catname)
    w_dmass = dmass['CMASS_PROB']
    print ('Calculating DMASS systematic weights...')
    dmass = appendColumn(dmass, name='WEIGHT', value= w_dmass )
#   dmass = dmass[ dmass['CMASS_PROB'] > 0.01 ]   # for low probability galaxies
    esutil.io.write(catname, dmass)
    #randoms = esutil.io.read('/n/des/lee.5922/data/dmass_cat/random_x50_dmass_spt_masked.fits')
    
    randoms = esutil.io.read('/fs/scratch/PCON0008/warner785/bwarner/random_dmass_y1_public_v1.fits')
#    randoms = esutil.io.read('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_st82_DET200_randoms.fits')
    
#    catdir = ''.join([ c+'/' for c in catname.split('/')[:-1]])
#    os.system('mkdir '+catdir)
#    dmass = esutil.io.read('/n/des/lee.5922/data/dmass_cat/dmass_spt_sys_v3.fits')
    #w_dmass = dmass['CMASS_PROB'] *dmass['WEIGHT0_fwhm_r']*dmass['WEIGHT1_airmass_z']
    #print ('Calculatig DMASS systematic weights...')
    #dmass = appendColumn(dmass, name='WEIGHT', value= w_dmass )
    #dmass = dmass[ dmass['CMASS_PROB'] > 0.01 ]
    #esutil.io.write(catname, dmass)
    #randoms = esutil.io.read('/n/des/lee.5922/data/dmass_cat/random_x50_dmass_spt_masked.fits')

#    randoms = esutil.io.read('/n/des/lee.5922/data/dmass_cat/random_x50_dmass_spt_masked.fits')

    print ('Resulting catalog size')
    print ('DMASS=', np.sum(dmass['WEIGHT']) )
    print ('randoms=', randoms.size)
    return dmass, randoms

def keepGoodRegion(des, hpInd = False, balrog=None):
    import healpy as hp
    import fitsio
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    
    path = '/fs/scratch/PCON0008/warner785/bwarner/'
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_il22_seeil4.0_nside4096ring_redlimcut.fits')
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    LSSGoldmask = fitsio.read(path+'MASK_Y3LSSBAOSOF_22_3_v2p2.fits')
    ringhp = hp.nest2ring(4096, [LSSGoldmask['PIXEL']])
    #Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits
    #frac_cut = LSSGoldmask['FRAC'] > 0.8
    #ind_good_ring = LSSGoldmask['PIXEL'][frac_cut]
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
    """Convert ra & dec to Euclidean points
    Parameters
    ----------
    ra, dec : ndarrays
    Returns
    x, y, z : ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return (cos_ra * sin_dec,
            sin_ra * sin_dec,
            cos_dec)

def uniform_sphere(RAlim, DEClim, size=1):
    """Draw a uniform sample on a sphere
    Parameters
    ----------
    RAlim : tuple
        select Right Ascension between RAlim[0] and RAlim[1]
        units are degrees
    DEClim : tuple
        select Declination between DEClim[0] and DEClim[1]
    size : int (optional)
        the size of the random arrays to return (default = 1)
    Returns
    -------
    RA, DEC : ndarray
        the random sample on the sphere within the given limits.
        arrays have shape equal to size.
    """
    zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
    DEC = (180. / np.pi) * np.arcsin(z)
    RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)
    
    return RA, DEC

def uniform_random_on_sphere(data, size = None ):
    ra = data['RA']
    dec = data['DEC']
    
    n_features = ra.size
    #size = 100 * data.size
    
    # draw a random sample with N points
    ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                 (min(dec), max(dec)),
                                 size)
    #data = np.asarray(ra_dec_to_xyz(ra, dec), order='F').T
    #data_R = np.asarray(ra_dec_to_xyz(ra_R, dec_R), order='F').T
    
    #random redshift distribution
    
    data_R = np.zeros((ra_R.size,), dtype=[('RA', 'float'), ('DEC', 'float')])
    data_R['RA'] = ra_R
    data_R['DEC'] = dec_R
                              
    return data_R

## ---------------------------------------------

# import DMASS in validation region
#lens, randoms = calling_lens_catalog('/fs/scratch/PCON0003/warner785/bwarner/dmass_y1_public_v1.fits')
dmass_val, randoms = calling_lens_catalog('/fs/scratch/PCON0008/warner785/bwarner/dmass_st82_DET200.fits')

random_val = uniform_random_on_sphere(dmass_val, size = 10*int(np.sum(dmass_val['WEIGHT']))) #larger size of randoms
# applying LSS mask 
random_val = keepGoodRegion(random_val)

plt.rcParams.update({
  "text.usetex": False,
  "font.family": "Helvetica"
})

random_val = appendColumn(random_val, value=np.ones(random_val.size), name='WEIGHT')

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')

phi = random_val['RA'] * np.pi / 180.0
theta = ( 90.0 - random_val['DEC'] ) * np.pi/180.0
random_pix = hp.ang2pix(4096, theta, phi)
print(random_pix.size)

frac = np.zeros(hp.nside2npix(4096))
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
#sysHp[sysMap['PIXEL'][dim_mask]] = sysMap['SIGNAL'][dim_mask]
frac[fracDet['PIXEL']] = fracDet['SIGNAL']

frac_obj = frac[random_pix]

u = np.random.rand(len(random_pix))
#select random points with the condition u < frac_obj
random_val_fracselected = random_val[u < frac_obj]

def cutPCA(sysMap):
    RA, DEC = hp.pix2ang(4096, sysMap['PIXEL'], lonlat=True)
    sysMap = append_fields(sysMap, 'RA', RA, usemask=False)
    sysMap = append_fields(sysMap, 'DEC', DEC, usemask=False)
    #print(sysMap.dtype.names)

    sysMap = keepGoodRegion(sysMap)

    mask4 =(sysMap['RA']>18)&(sysMap['RA']<43)
    mask4 = mask4 & (sysMap['DEC']>-10) & (sysMap['DEC']<10)
    sysMap = sysMap[mask4]
    
    return sysMap


def cut_and_downgradePCA(sysMap):
    #print(sysMap.dtype.names)

    RA, DEC = hp.pix2ang(4096, sysMap['PIXEL'], lonlat=True)
    sysMap = append_fields(sysMap, 'RA', RA, usemask=False)
    sysMap = append_fields(sysMap, 'DEC', DEC, usemask=False)
    #print(sysMap.dtype.names)

    sysMap = keepGoodRegion(sysMap)

    mask4 =(sysMap['RA']>18)&(sysMap['RA']<43)
    mask4 = mask4 & (sysMap['DEC']>-10) & (sysMap['DEC']<10)
    sysMap = sysMap[mask4]
    
    sysHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
    #sysHp[sysMap['PIXEL'][dim_mask]] = sysMap['SIGNAL'][dim_mask]
    sysHp[sysMap['PIXEL']] = sysMap['SIGNAL']
    #print(sysHp.size)
    #print(sysHp[35369:35469])

    #print(sysMap['PIXEL'][dim_mask][0])

    nside_in = hp.pixelfunc.get_nside(sysHp)
    #print(nside_in)
    downgrade = hp.pixelfunc.ud_grade(sysHp, 512, pess=False, order_in='RING', order_out=None, power=None, dtype=None)
    #downgrade128 = hp.pixelfunc.ud_grade(sysHp, 128, pess=False, order_in='RING', order_out=None, power=None, dtype=None)
    #downgrade256 = hp.pixelfunc.ud_grade(sysHp, 256, pess=False, order_in='RING', order_out=None, power=None, dtype=None)

    #default order_out = ring, change to nest? -- check
    nside_out = hp.pixelfunc.get_nside(downgrade)
    #print(nside_out)

    n_good_pixels_at_512 = hp.nside2npix(nside_out)
    #np.zeros(n_good_pixels_at_512, dtype=[('PIXEL','int'), ('SIGNAL','float')] )
    pixels = np.zeros(n_good_pixels_at_512)
    #print(pixels.size)

    #print(hp.visufunc.mollview(sysHp))
    

    #print(sysMap.size)
    
    for x in range(pixels.size):
        if x>0:
            pixels[x]=pixels[x-1]+1
    #print(pixels)

    sysMap = np.zeros( len(pixels), dtype=[('HPIX_512','int'), ('SIGNAL','float'),('RA','float'),('DEC','float')])
    sysMap['HPIX_512'] = pixels
    sysMap['SIGNAL'] = downgrade

    #print(sysMap.size)
    #sysMap = keepGoodRegion(sysMap)
    #print(sysMap.size)

    #restrict to validation region area:
    RA, DEC = hp.pix2ang(512, sysMap['HPIX_512'], lonlat=True)
    #theta, phi = hp.pix2ang(512, sysMap['PIXEL'])
    ##sys_area = np.zeros( len(phi), dtype=[('RA','float'), ('DEC','float')] )

    # Convert silly ra/dec to silly HP angular coordinates.
    ##phi = des['RA'] * np.pi / 180.0
    #RA = phi*(180.0/np.pi)
    #DEC = -(180.0/np.pi)*theta+90.0
    ##theta = ( 90.0 - des['DEC'] ) * np.pi/180.0

    sysMap['RA']= RA
    sysMap['DEC'] = DEC
    
    return sysMap

def downgrade_fracDet(fracDet):
    frac = np.zeros(hp.nside2npix(4096))
    fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
    #sysHp[sysMap['PIXEL'][dim_mask]] = sysMap['SIGNAL'][dim_mask]
    frac[fracDet['PIXEL']] = fracDet['SIGNAL']
    downgrade_frac = hp.pixelfunc.ud_grade(frac, 512, pess=False, order_in='RING', order_out=None, power=None, dtype=None)
    
    n_good_pixels_at_512 = hp.nside2npix(512)
    #np.zeros(n_good_pixels_at_512, dtype=[('PIXEL','int'), ('SIGNAL','float')] 
 
    pixels = np.arange( n_good_pixels_at_512)


    fracDet_512 = np.zeros( len(pixels), dtype=[('HPIX_512','int'), ('SIGNAL','float')])
    fracDet_512['HPIX_512'] = pixels
    fracDet_512['SIGNAL'] = downgrade_frac    
    
    return fracDet_512


def downgrade_dmass(dmass_val):
    index_mask = np.argsort(dmass_val['HPIX_4096'])
    dmass_chron = dmass_val[index_mask] # ordered by hpix values
    theta, phi = hp.pix2ang(4096, dmass_chron['HPIX_4096'], nest = True)
    HPIX_512 = hp.ang2pix(512, theta, phi)

    dmass_chron = append_fields(dmass_chron, 'HPIX_512', HPIX_512, usemask=False)
    
    return dmass_chron

def number_gal(sysMap, dmass_chron, sys_weights = False): # apply systematic weights here
    
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
    sysval_gal = x[dmass_chron['HPIX_4096']].copy()

    #which healpixels have values in the sysMap signal

    #print(sum(sysval_gal[sysval_gal != hp.UNSEEN]))
    #print(hp.UNSEEN)

    #print(x.size, sysval_gal.size, dmass_chron.size)
    #print(maximum, minimum)
    #print((sysval_gal != 0.0).any())
    
    if sys_weights == True:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["WEIGHT"][sysval_gal != hp.UNSEEN]*dmass_chron["SYS_WEIGHT"][sysval_gal != hp.UNSEEN])
    else:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["WEIGHT"][sysval_gal != hp.UNSEEN]) # -- density of dmass sample, not gold sample
    #print(h)
    
    return h, sysval_gal

def area_pixels(sysMap, fracDet):
    
    #minimum = np.percentile(sysMap['SIGNAL'][dim_mask], 1)
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    #minimum = np.min(sysMap['SIGNAL'][dim_mask]) #FWHM signal (for g filter)
    #maximum = np.percentile(sysMap['SIGNAL'][dim_mask], 99)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
    #print(minimum)
    #print(maximum)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2
# number of galaxies in each pixel:

    sys_signal = sysMap['SIGNAL']

    #print(sys_signal[sys_signal != hp.UNSEEN])
    #print(sys_signal[sys_signal != hp.UNSEEN].size)

    n,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin )
    print('area without weights:')
    print(n)

    #corrected fracDet nside 512 //

    #matched_sys2 = sysMap[dim_mask]
    sys = sysMap
#    mask = np.full(hp.nside2npix(512), hp.UNSEEN)
    mask = np.full(hp.nside2npix(4096), hp.UNSEEN)

#    print(fracDet_512["HPIX_512"])
    print(fracDet["PIXEL"])

    #Only look at pixels where fracDet has value
#    frac_mask = np.in1d(fracDet_512["HPIX_512"], sys["HPIX_512"], assume_unique=False, invert=False)
    frac_mask = np.in1d(fracDet["PIXEL"], sys["PIXEL"], assume_unique=False, invert=False)

    #make an array with signals corresponding to pixel values 
#    mask[sys["HPIX_512"]] = sys["SIGNAL"]
    mask[sys["PIXEL"]] = sys["SIGNAL"]

    #array only including fracDet/sys seen pixels sys signal values 
    #print(mask[mask != hp.UNSEEN])
#    frac_sys = mask[fracDet_512["HPIX_512"][frac_mask]]
    frac_sys = mask[fracDet["PIXEL"][frac_mask]]

    #print(frac_sys[frac_sys != hp.UNSEEN])
    #print(frac_sys[frac_sys != hp.UNSEEN].size)

    #print(frac_sys[frac_sys != hp.UNSEEN])


    #print("sum: ", sum(fracDet_512["SIGNAL"]))

    #weights of fracDet in the overlap applied for accurate areas
    area,_ = np.histogram(frac_sys[frac_sys != hp.UNSEEN] , bins=pbin , weights = fracDet["SIGNAL"][frac_mask][frac_sys != hp.UNSEEN])
    print('area with weights:')
    print(area)
    # area = units of healpixels

    return area

def random_pixel(random_val_fracselected):
    phi = random_val_fracselected['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_val_fracselected['DEC'] ) * np.pi/180.0
    nside= 4096

    HPIX_4096 = hp.ang2pix(4096, theta, phi)

    random_val = append_fields(random_val_fracselected, 'HPIX_4096', HPIX_4096, usemask=False)
    #print(random_val.dtype.names)
    
    return random_val

def downgrade_ran(random_val_fracselected):
    # convert nside for randoms:
    phi = random_val_fracselected['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_val_fracselected['DEC'] ) * np.pi/180.0
    nside= 4096

    HPIX_512 = hp.ang2pix(512, theta, phi)

    random_val = append_fields(random_val_fracselected, 'HPIX_512', HPIX_512, usemask=False)
    #print(random_val.dtype.names)

    index_ran_mask = np.argsort(random_val['HPIX_512'])
    random_chron = random_val[index_ran_mask] # ordered by hpix values
    
    return random_chron


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

    fracerr = np.sqrt(h) #1 / sqrt(number of randoms cmass galaxies in each bin)
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


input_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_maps_jointmask_no_stars1623/'
#y3/band_z/
keyword_template = 'pca{0}_'
for i_pca in range(50): #50
    input_keyword = keyword_template.format(i_pca)
    print(input_keyword)
    sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword)

    path = '/fs/scratch/PCON0008/warner785/bwarner/'
    
    sys_weights = False
    
    linear = False
    quadratic = False
    
#    sysMap = cut_and_downgradePCA(sysMap)
    sysMap = cutPCA(sysMap)
    fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
    fracDet['PIXEL'] = hp.nest2ring(4096, fracDet['PIXEL'])
#    fracDet_512 = downgrade_fracDet(fracDet)
#    dmass_chron = downgrade_dmass(dmass_val)
#    random_chron = downgrade_ran(random_val_fracselected)
    if sys_weights == True:
        dmass_chron = fitsio.read('../output/test/train_cat/y3/'+input_keyword+'dmass_sys_weight.fits')
        random_chron = fitsio.read('../output/test/train_cat/y3/'+input_keyword+'randoms.fits')
    else:
        index_mask = np.argsort(dmass_val)
        dmass_chron = dmass_val[index_mask] # ordered by hpix values
        dmass_chron['HPIX_4096'] = hp.nest2ring(4096, dmass_chron['HPIX_4096']) 
        randoms4096 = random_pixel(random_val_fracselected)
        index_ran_mask = np.argsort(randoms4096)
        random_chron = randoms4096[index_ran_mask]
        
    h, sysval_gal = number_gal(sysMap, dmass_chron, sys_weights = False)
    area = area_pixels(sysMap, fracDet)
    h_ran,_= number_gal(sysMap, random_chron, sys_weights = False)

    pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)
    pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)
    

    #plotting:

    fig, ax = plt.subplots()
    ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in validation")
    ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms in validation")
    plt.legend()
    xlabel = input_keyword
    plt.xlabel(xlabel)
    plt.ylabel("n_gal/n_tot 4096")
    #plt.ylim(top=1.2)  # adjust the top leaving bottom unchanged
    #plt.ylim(bottom=0.85)
    plt.axhline(y=1, color='grey', linestyle='--')
#    plt.title(xlabel+' systematic check')
    if sys_weights == True:
        plt.title(xlabel+' sys weights applied')
        fig.savefig(xlabel+'sys_applied.pdf')
    else:
        plt.title(xlabel+' systematics check')
        fig.savefig(xlabel+'sys_check.pdf')        

    ran_chi2, ran_chi2_reduced = chi2(norm_number_density_ran, 1, fracerr_ran_norm, 0)
    print('ran_chi2: ', ran_chi2_reduced)
    
    if sys_weights == True:
        trend_chi2, trend_chi2_reduced = chi2(norm_number_density, 1, fracerr_norm, 0)
        print('applied_sys_chi2: ', trend_chi2_reduced)
    
    if sys_weights == False:    
        dmass_chi2, dmass_chi2_reduced = chi2(norm_number_density, 1, fracerr_norm, 0)
        print('checking chi2 before correction: ', dmass_chi2_reduced)
        #trendline:
        # fit to trend:
        fig,ax = plt.subplots(1,1)
        #linear trends first -- chi2 for higher order study --- check for threshold value (afterward)
        z = np.polyfit(pcenter, norm_number_density, 1)
        p = np.poly1d(z)

        print(p)

        ax.plot(pcenter,p(pcenter),"r--")
        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in validation")
        plt.title(xlabel+' systematic linear trendline')
        fig.savefig(xlabel+'linear.pdf')

        trend_chi2, trend_chi2_reduced = chi2(norm_number_density, p(pcenter), fracerr_norm, 2)

        print('linear trend_chi2: ', trend_chi2_reduced)
    
# difference between sum(chi2) between models (free parameters-- 1 new, want more than 1 better in sum(chi2))

        # second trendline:
        # fit to trend:
        fig,ax = plt.subplots(1,1)
        #linear trends first -- chi2 for higher order study --- check for threshold value (afterward)
        z2 = np.polyfit(pcenter, norm_number_density, 2)
        p2 = np.poly1d(z2)

        print(p2)

        ax.plot(pcenter,p2(pcenter),"r--")
        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in validation")
        plt.title(xlabel+' systematic quadratic trendline')
        fig.savefig(xlabel+'quadratic.pdf')

        trend2_chi2, trend2_chi2_reduced = chi2(norm_number_density, p2(pcenter), fracerr_norm, 3)
        diff_chi2 = abs(sum(trend_chi2)-sum(trend2_chi2))
        print('quadratic trend_chi2: ', trend2_chi2_reduced)

        print("difference of chi2 between models: ", diff_chi2)
        if diff_chi2 > 1:
            quadratic=True
            print("Quadratic is better fit for ", xlabel)
        else:
            linear=True
            print("Linear fit is suitable for ", xlabel)
        
        # work on applying the weights to dmass:

        #linear:
        #weight_pixel = (1/p(sysMap["PIXEL"]))
        if linear==True:
            
            #check chi2 first
            chi2_ = np.linspace(0,30,100)
            y = np.abs((100*(1.-scipy.stats.chi2(12).cdf(chi2_))-5.))  #for 5% p-value threshold
            index = np.where(y == y.min())[0][0]
            threshold = chi2_[index]/12
            print(threshold)
            for x in range(len(trend_chi2)):
                if trend_chi2[x]>threshold:
                    print(xlabel, " needs to be flagged", x)
            
            #make sure object density stays the same
            weight_object = (1/p(sysval_gal))
            weight_object[sysval_gal == hp.UNSEEN] = 0
            avg = np.average(weight_object)
            print(avg)  # should be aprox. 1
            # normalize density
            weight_object = weight_object/avg
        
        # quadratic:
        #weight_pixel = (1/p2(sysMap["PIXEL"]))
        if quadratic==True:
            
            #check chi2 first
            chi2_ = np.linspace(0,30,100)
            y = np.abs((100*(1.-scipy.stats.chi2(12).cdf(chi2_))-5.))  #for 5% p-value threshold
            index = np.where(y == y.min())[0][0]
            threshold = chi2_[index]/12
            print(threshold)
            for x in range(len(trend2_chi2)):
                if trend2_chi2[x]>threshold:
                    print(xlabel, " needs to be flagged", x)
                    
            #make sure object density stays the same
            weight_object = (1/p2(sysval_gal))
            weight_object[sysval_gal == hp.UNSEEN] = 0
            avg = np.average(weight_object)
            print(avg)  # should be aprox. 1
            #normalize density
            weight_object = weight_object/avg
        
        dmass_chron = append_fields(dmass_chron, 'SYS_WEIGHT', weight_object, usemask=False)
        print(dmass_chron["SYS_WEIGHT"], dmass_chron["SYS_WEIGHT"].size)
    
        outdir = '../output/test/train_cat/y3/'
        os.makedirs(outdir, exist_ok=True)
        esutil.io.write( outdir+xlabel+'dmass_sys_weight.fits', dmass_chron, overwrite=True)
        esutil.io.write( outdir+xlabel+'randoms.fits', random_chron, overwrite=True)
        