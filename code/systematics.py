#import easyaccess as ea
import esutil, sys, os, fitsio
import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from suchyta_utils import jk
from utils import *
#import seaborn as sns

#from ang2stripe import *
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts


def brelchisqr(xi, xi2, invcov, brelsqr):
    DiffVector = xi2 - brelsqr*xi
    chi2 = np.dot( np.dot( DiffVector , invcov), DiffVector )
    return chi2

def find_nearest(array, value, brell):
    
    ind_min = array.argmin()
    array1 = np.asarray(array[:ind_min])
    idx1 = (np.abs(array1 - value)).argmin()
    
    array2 = np.asarray(array[ind_min:])
    idx2 = (np.abs(array2 - value)).argmin()
    
    return brell[:ind_min][idx1], brell[ind_min], brell[ind_min:][idx2]


def brel_chisqr_fitting(xi1, xi2, Fisher, verbose=False):
    
    brelsqr = np.linspace(0.8, 1.2, 5000)   
    brelarr = np.sqrt(brelsqr) 
    chisqr_result = np.zeros(brelsqr.size)
    
    i=0
    for b in brelsqr:
        chisqr_result[i] = brelchisqr(xi1, xi2, Fisher, b)
        i+=1   
        
    minarg = chisqr_result.argmin()
    chisqr_min = chisqr_result.min()
    brel = np.sqrt(brelsqr[minarg])
    
    b_cmass = 2.0
    db = b_cmass * (1. - 1./brel)
    
       
    b_cmass = 2.0

    br1, brmin, br2 = find_nearest( chisqr_result, chisqr_min + 1,brelarr )
    err_brel = np.abs(br1-br2)/2.
    err_db = err_brel *b_cmass
    if verbose : print 'err db, db / br1, brmin, br2 :', err_db, db, br1, brmin, br2  
    #return err_db, db, chisqr_result
    return err_brel, brmin, chisqr_result
    
    

def excludeBadRegions(des, balrogObs, balrogTruthMatched, balrogTruth, band=None):
    path = '/n/des/lee.5922/data/balrog_cat/'
    eliMap = hp.read_map(path +'y1a1_gold_1.0.2_wide_badmask_4096.fit', nest=True)
    nside = hp.npix2nside(eliMap.size)
    maskIndices = np.arange(eliMap.size)
    badIndices = maskIndices[eliMap == 1]
    
    if band is not None:
        raTag = 'RA_'+band
        decTag = 'DEC_'+band
    else:
        raTag = 'RA'
        decTag = 'DEC'
    obsKeepIndices = getGoodRegionIndices(catalog=balrogObs, badHPInds=badIndices, nside=nside, raTag = raTag, decTag = decTag)
    truthKeepIndices = getGoodRegionIndices(catalog=balrogTruth, badHPInds=badIndices, nside=nside, raTag = raTag, decTag = decTag)
    desKeepIndices = getGoodRegionIndices(catalog=des, badHPInds=badIndices, nside=nside, raTag = raTag, decTag = decTag)

    balrogObs = balrogObs[obsKeepIndices]
    balrogTruthMatched = balrogTruthMatched[obsKeepIndices]
    balrogTruth = balrogTruth[truthKeepIndices]
    des = des[desKeepIndices]
    
    return des, balrogObs, balrogTruthMatched, balrogTruth


def pixfracMask( catalog, nside = 4096 ):

    import healpy as hp
    import numpy as np
    from systematics import hpHEALPixelToRaDec
    path = '/n/des/lee.5922/data/balrog_cat/'
    # Note that the masks here in in equatorial, ring format.
    fraction = hp.read_map(path+'Y1A1_WIDE_frac_combined_griz_o.4096_t.32768_EQU.fits')

    if nside != 4096:
        fraction = hp.ud_grade(fraction, nside_out = nside )

    ind_good_ring = np.where(fraction > 0.8)[0]

    hpInd = hpRaDecToHEALPixel(catalog['RA'], catalog['DEC'], nside=nside, nest= False)
    keep = np.in1d(hpInd,ind_good_ring)

    return catalog[keep]



def callingEliGoldMask( nside = 4096 ):
    
    # calling gold mask ----------------------------------------------------------------
    import healpy as hp
    import numpy as np
    from systematics import hpHEALPixelToRaDec
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    path = '/n/des/lee.5922/data/systematic_maps/'
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_il22_seeil4.0_nside4096ring_redlimcut.fits')
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_redlimcut_il22_seeil4.0_4096ring.fits')
    frac_cut = (LSSGoldmask['FRAC'] >= 0.8)
    LSSGoldmask = LSSGoldmask[frac_cut]

    #path = '/n/des/lee.5922/data/balrog_cat/'
    #LSSGoldmask =  fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    #ind_good_ring2 = LSSGoldmask['PIXEL']
    
    
    """
    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)
    fraction = hp.read_map(path+'Y1A1_WIDE_frac_combined_griz_o.4096_t.32768_EQU.fits')

    
    if nside != 4096:
        print "if resolution is degraded, pixel fraction column would not work properly "
        gdmask = hp.ud_grade(gdmask, nside_out = nside )
        bdmask = hp.ud_grade(bdmask, nside_out = nside )
        fraction = hp.ud_grade(fraction, nside_out = nside )
    
    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) & (fraction > 0.8))[0]
    # healpixify the catalog.

    GoldMask = np.zeros((ind_good_ring.size, ), dtype=[('PIXEL', 'i4'), ('RA', 'f8'), ('DEC', 'f8')])
    GoldMask['PIXEL'] = ind_good_ring
    #GoldMask['FRAC'] = fraction[ind_good_ring]
    sys_ra, sys_dec = hpHEALPixelToRaDec(ind_good_ring, nside = nside)
    GoldMask['RA'] = sys_ra
    GoldMask['DEC'] = sys_dec
    """
    """ should consider cut-off ?? """
    # ----------------------------------------------------------------
    return LSSGoldmask
    #return GoldMask



def callingY1GoldMask( nside = 4096 ):
    
    # calling gold mask ----------------------------------------------------------------
    import healpy as hp
    import numpy as np
    from systematics import hpHEALPixelToRaDec
    # 25 is the faintest object detected by DES
    # objects larger than 25 considered as Noise
    #path = '/n/des/lee.5922/data/systematic_maps/'
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_il22_seeil4.0_nside4096ring_redlimcut.fits')
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    #LSSGoldmask = fitsio.read(path+'Y1LSSmask_v2_redlimcut_il22_seeil4.0_4096ring.fits')
    #frac_cut = (LSSGoldmask['FRAC'] >= 0.8)
    #LSSGoldmask = LSSGoldmask[frac_cut]

    #path = '/n/des/lee.5922/data/balrog_cat/'
    #LSSGoldmask =  fitsio.read(path+'Y1LSSmask_v1_il22seeil4.04096ring_redlimcut.fits')
    #ind_good_ring2 = LSSGoldmask['PIXEL']
    
    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)
    #fraction = hp.read_map(path+'Y1A1_WIDE_frac_combined_griz_o.4096_t.32768_EQU.fits')

    
    if nside != 4096:
        print "if resolution is degraded, pixel fraction column would not work properly "
        gdmask = hp.ud_grade(gdmask, nside_out = nside )
        bdmask = hp.ud_grade(bdmask, nside_out = nside )
        print 'DOWNGRADE = ', nside
        #fraction = hp.ud_grade(fraction, nside_out = nside )
    
    #ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) & (fraction > 0.8))[0]
    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0))[0]
    # healpixify the catalog.

    GoldMask = np.zeros((ind_good_ring.size, ), dtype=[('PIXEL', 'i4'), ('RA', 'f8'), ('DEC', 'f8')])
    GoldMask['PIXEL'] = ind_good_ring
    #GoldMask['FRAC'] = fraction[ind_good_ring]
    sys_ra, sys_dec = hpHEALPixelToRaDec(ind_good_ring, nside = nside)
    GoldMask['RA'] = sys_ra
    GoldMask['DEC'] = sys_dec
    
    """ should consider cut-off ?? """
    # ----------------------------------------------------------------
    #return LSSGoldmask
    return GoldMask

def loadSystematicMaps(  property ='AIRMASS', filter='g', nside = 1024, filename = None, kind = 'STRIPE82', path = None):

    import healpy as hp

    if path is None : 
        if kind is 'SPT' : 
            path = '/n/des/lee.5922/data/systematic_maps/Y1A1NEW_COADD_SPT/nside4096_oversamp4/'
            if property is 'FWHM': path = '/n/des/lee.5922/data/systematic_maps/seeing_i_spt/'
        if kind is 'Y1A1': path = '/n/des/lee.5922/data/systematic_maps/Y1A1_SPT_and_S82_IMAGE_SRC/nside4096_oversamp4/'
        if kind is 'STRIPE82' : path ='/n/des/lee.5922/data/systematic_maps/'


    if filename is not None:
        try:
            print '\nPATH = ',path
            print filename
            sysMap_hp = hp.read_map( path+filename, nest=False)
            if nside != 4096 : 
                sysMap_hp = hp.ud_grade(sysMap_hp, nside_out = nside )
                print 'DOWNGRADE = ', nside
            goodmask = hp.mask_good(sysMap_hp)
            maskIndices = np.arange(sysMap_hp.size)
            goodIndices = maskIndices[goodmask]
            clean_map = sysMap_hp[goodmask]

        except IOError:
            print "Input map is not fits format. Not masked"
            clean_map = np.loadtxt(path+filename)
            goodIndices = np.arange(clean_map.size)
            

    else :       

        if property is 'GE': 

            nside = 512
            
            reddening_nest = esutil.io.read('/n/des/lee.5922/data/2mass_cat/lambda_sfd_ebv.fits', ensure_native=True)
            reddening_ring = hp.reorder(reddening_nest['TEMPERATURE'], inp='NEST', out='RING')
            clean_map = rotate_hp_map(reddening_ring, coord = ['C', 'G'])
            goodmask = hp.mask_good(clean_map)
            maskIndices = np.arange(clean_map.size)
            goodIndices = maskIndices[goodmask]

        else : 
            if property is 'FWHM':keyword = '_MEAN_coaddweights3' 
            #(FWHM_MEAN, FWHM_PIXELFREE, FWHM_FROMFLUXRADIUS_MEAN) should be used instead of the simple FWHM (see Eli's emails!)
            elif property is not 'FWHM' : keyword = '_coaddweights3_mean'
            
            for i in os.listdir(path):
                if os.path.isfile(os.path.join(path,i)) and property.upper()+keyword in i and 'band_'+filter.lower() in i:
                    print '\nPATH = ',path
                    print i
                    
                    sysMap_hp = hp.read_map( path+i, nest=False)
                    if nside != 4096 : 
                        sysMap_hp = hp.ud_grade(sysMap_hp, nside_out = nside )
                        print 'DOWNGRADE = ', nside
                    goodmask = hp.mask_good(sysMap_hp)
                    
                    maskIndices = np.arange(sysMap_hp.size)
                    goodIndices = maskIndices[goodmask]
                    clean_map = sysMap_hp[goodmask]

    sysMap = np.zeros((clean_map.size, ), dtype=[('PIXEL', 'i4'), ('SIGNAL', 'f8'), ('RA', 'f8'), ('DEC', 'f8')])
    sysMap['PIXEL'] = goodIndices
    sysMap['SIGNAL'] = clean_map
    
    sys_ra, sys_dec = hpHEALPixelToRaDec(goodIndices, nside = nside)
    sysMap['RA'] = sys_ra
    sysMap['DEC'] = sys_dec
    
    return sysMap


def _GalaxyDensity_Systematics( catalog, sysMap, rand = None, weight = None, nside = 4096, raTag = 'RA', decTag='DEC', property = 'NSTARS', filter='g'):
    #property ='AIRMASS', filter='g', nside = 128, raTag = 'RA', decTag='DEC'):
    import healpy as hp
    
    if weight is None: weight = 1.0 * np.ones(catalog.size)
    
    # gal number density vs survey property
    catHpInd = hpRaDecToHEALPixel(catalog[raTag], catalog[decTag], nside=nside, nest= False)
    HpIdxInSys_mask = np.in1d(catHpInd, sysMap['PIXEL'])
    HpIdxInSys = catHpInd[HpIdxInSys_mask]
    
    #N_validPixel = len(set(HpIdxInSys)) #validPixel.size #len(sysMap) #len(set(hpInd[validPixelInd])) * 1.0
    N_validPixel = len(sysMap)
    FullArea = hp.nside2pixarea(nside, degrees = True) * N_validPixel

    # total number of galaxies in the input catalog
    Ngal_total = np.sum(weight[HpIdxInSys_mask]) # HpIdxInSys.size

    P = 1e+4
    n_bar = 1e-4
    
    bin_num = 15
    min = np.min(sysMap['SIGNAL'])
    max = np.max(sysMap['SIGNAL'])
    
    
    if property == 'NSTARS': bin_num = bin_num * 2
    if property == 'AIRMASS':
        if filter == 'z' : min = 1.0
    if property == 'FWHM' :
        #bin_num = bin_num * 2
        if filter == 'i' : max = 5.0
        if filter == 'z' : max = 5.0
    if property == 'SKYBRITE' :
        #bin_num = bin_num * 2
        if filter == 'g' : max = 150
        if filter == 'r' : max = 380
        if filter == 'i' : max = 1200
    if property == 'SKYSIGMA' :
        #bin_num = bin_num * 2
        if filter == 'g' : max = 6.5
        if filter == 'r' : max = 9.5
        if filter == 'i' : max = 18
        if filter == 'z' : max = 26

    print 'max', max

    bin_center, binned_cat, keeps = divide_bins( sysMap, Tag = 'SIGNAL', min = min, max = max, bin_num = bin_num )

    if rand is not None:
        randHpInd = hpRaDecToHEALPixel(rand[raTag], rand[decTag], nside=nside, nest= False)
        randHpIdxInSys_mask = np.in1d(randHpInd, sysMap['PIXEL'])
        randHpIdxInSys = randHpInd[randHpIdxInSys_mask]
        Nrand_total = randHpIdxInSys.size
    
    """
    z_bin, step = np.linspace(0.0, 1.0, 20, retstep = True)
    #z_bin_center = z_bin[:-1] + step/2.
    N_z, _ = np.histogram( catalog['Z'], bins = z_bin )
    n_z = N_z / FullArea #((360 - 317) * 3.)
    n_bar = np.sum(n_z)/len(z_bin)
    """
    
    w_FKP = 1./( 1 + n_bar * P )
    galaxy_density_list = []
    err = []
    f_area = []
    for i, sys_i in enumerate(binned_cat):
        
        HpIdxInsys_i_mask = np.in1d(catHpInd, sys_i['PIXEL'])
        HpIdxInsys_i = catHpInd[HpIdxInsys_i_mask]
        #Npix = len(set(HpIdxInsys_i))
        Npix = len(sys_i)
        Ngal = np.sum(weight[HpIdxInsys_i_mask])  #HpIdxInsys_i.size
        area_i = hp.nside2pixarea(nside, degrees = True) * Npix
        f_area.append(area_i * 1./FullArea)
        
        
        if rand is not None:
            randHpIdxInsys_i_mask = np.in1d(randHpInd, sys_i['PIXEL'])
            randHpIdxInsys_i = randHpInd[randHpIdxInsys_i_mask]
            #Npix = len(set(HpIdxInsys_i))
            Npix = len(sys_i)
            Nrand = randHpIdxInsys_i.size
    
        try:
            if rand is not None : norm_galaxy_density = (Ngal * 1./Nrand) * ( Nrand_total/Ngal_total )
            else : norm_galaxy_density = (Ngal * 1./Npix) * ( N_validPixel * 1./Ngal_total )
            
            
            #print 'N_gal/Npix, N_tot/Npix,tot :', Ngal *1./Npix, Ngal_total*1./N_validPixel, norm_galaxy_density
        
        except ZeroDivisionError : norm_galaxy_density = float('nan')

        #print 'N_gal/Npix, N_tot/Npix,tot :', Ngal *1./Npix, Ngal_total*1./N_validPixel, norm_galaxy_density
        
        if np.isnan(norm_galaxy_density) == True:
            galaxy_density_list.append(0.0)
            err.append( 0.0 )
        else:
            err.append( 1./np.sqrt(Ngal * w_FKP) * norm_galaxy_density)
            galaxy_density_list.append(norm_galaxy_density)

        #print 'bin :', bin_center[i], ' Ngal :', Ngal, ' Ngal_tot :', Ngal_total, ' ngal : ',norm_galaxy_density
    return np.array(bin_center), np.array(galaxy_density_list), np.array(err), np.array(f_area)




def GalaxyDensity_Systematics( catalog, sysMap, rand = None, nside = 4096, 
			       raTag = 'RA', decTag='DEC', property = 'NSTARS', filter='g', 
			       FullArea = None, nbins=20, reweight = None, pixelmask = None):
    #property ='AIRMASS', filter='g', nside = 128, raTag = 'RA', decTag='DEC'):
    """
    pixelmask : should be 1d, same size with sysMap one band
    """
    import healpy as hp


    #if pixelmask is not None : sysMap = sysMap[pixelmask]

    # gal number density vs survey property
    catHpInd = hpRaDecToHEALPixel(catalog[raTag], catalog[decTag], nside=nside, nest= False)
    randHpInd = hpRaDecToHEALPixel(rand[raTag], rand[decTag], nside=nside, nest= False)
    
    P = 1e+4
    n_bar = 3.0e-4
    log = False
    
    bin_num = nbins
    min = np.min(sysMap['SIGNAL'])
    max = np.max(sysMap['SIGNAL'])
    
    
    if property == 'NSTARS_allband': max = 2.0
    if property == 'AIRMASS':
        if filter == 'z' : min = 1.0
    if property == 'FWHM' : pass
        #bin_num = bin_num * 2
        #if filter == 'g' : max = 6.0
        #if filter == 'r' : max = 5.5
        #if filter == 'i' : max = 4.5
        #if filter == 'z' : max = 4.5
    if property == 'SKYBRITE' : pass
        #bin_num = bin_num * 2
        #if filter == 'g' : max = 150
        #if filter == 'r' : max = 450
        #if filter == 'i' : max = 1200
        #if filter == 'z' : max = 3000
    if property == 'SKYSIGMA' : pass
        #bin_num = bin_num * 2
        #if filter == 'g' : max = 6.5
        #if filter == 'r' : max = 11
        #if filter == 'i' : max = 18
        #if filter == 'z' : max = 26
    if property == 'EXPTIME': 
        #pass
        bin_num = bin_num*2
        #max = 600    
    #if property == 'DEPTH':
    if property == 'GE' : 
        log = True
        max = 0.2

    if property == 'SLR': 
        if filter == 'i' : 
            bin_num = bin_num*2
            min = -0.5
            max = 0.5
        elif filter == 'z' : 
            bin_num = bin_num*2
            min = -0.5
            max = 0.5
        else : pass
        #if filter == 'r' : max = 5.5
        #if filter == 'i' : max = 4.5
        #if filter == 'z' : max = 4.5    
    
    if property == 'SLRSFDRES': 
        if filter == 'g' : 
            #bin_num = bin_num*2
            min = -0.2
            max = 0.2
        elif filter == 'r' : 
            #bin_num = bin_num*2
            min = -0.04
            max = 0.04
        elif filter == 'i' : 
            #bin_num = bin_num*2
            min = -0.01
            max = 0.01
        elif filter == 'z' : 
            bin_num = bin_num*10
            min = -0.04
            #min = -0.12
            max = 0.8
        else : pass

    bin_center, binned_cat, keeps = divide_bins( sysMap, Tag = 'SIGNAL', min = min, max = max, bin_num = bin_num, log=log )
    
    w_FKP = 0.32607782  #1./( 1 + n_bar * P ), cmass sgc mean(wfkp)
    galaxy_density_list = []
    f_area = []
    #Ngal = np.zeros(len(binned_cat))
    #Nrand = np.zeros(len(binned_cat))
    Ngal = []
    Nrand = []
    
    
    if reweight is None : reweight = np.ones(catalog.size)
     
    for i, sys_i in enumerate(binned_cat):
        
        HpIdxInsys_i_mask = np.in1d(catHpInd, sys_i['PIXEL'])
        Npix = len(np.unique(sys_i['PIXEL']))
        ngal = np.sum(reweight[HpIdxInsys_i_mask]) 
        Ngal.append( ngal )
        area_i = hp.nside2pixarea(nside, degrees = True) * Npix

        #if ngal == 0 : area_i = 0
        f_area.append(area_i * 1./FullArea)

             
        #if rand is not None:
        randHpIdxInsys_i_mask = np.in1d(randHpInd, sys_i['PIXEL'])
        Nrand.append(np.sum(randHpIdxInsys_i_mask))# * np.sum(w_re_rand[randHpIdxInsys_i_mask])  
        #print Ngal[i], Nrand[i]
     
    Ngal = np.array(Ngal)
    Nrand = np.array(Nrand)
    f_area = Ngal * 1./Ngal.max()

    HpIdxInsys_mask = np.in1d(catHpInd, sysMap['PIXEL'])
    randHpIdxInsys_mask = np.in1d(randHpInd, sysMap['PIXEL'])
    
    #Ngal_total = np.sum(Ngal)
    #Nrand_total = np.sum(Nrand)
    Ngal_total = np.sum(reweight[HpIdxInsys_mask])
    Nrand_total = np.sum(randHpIdxInsys_mask)

    #f_area = Ngal * 1./Ngal_total

    #try:
    Ngal_avg = Ngal *1./Nrand
    ratio = Ngal_total * 1./Nrand_total
    norm_galaxy_density = Ngal_avg *1./ratio # (Ngal * 1./Nrand ) * ( Nrand_total * 1./ (Ngal_total) )   
    norm_galaxy_density[Nrand == 0] = 0.0
    err = 1./np.sqrt(Ngal * w_FKP) * norm_galaxy_density
    #nanmask = np.ma.masked_invalid(err).mask
    err[Ngal == 0] = 0.0

    return np.array(bin_center), np.array(norm_galaxy_density), np.array(err), np.array(f_area)






#################################
# Fitting Systematic Properties
#################################


#data_mask = Cbins < 5.0
#def linear_fitting( filename )
def fitting_linear( x_predict, xdata, ydata, yerr):
    import scipy
    #powerlaw = lambda x, amp, index: amp * (x**index)
    #logx = np.log10(xdata)
    #logy = np.log10(ydata)
    #logyerr = yerr / ydata

    # define our (line) fitting function
    fitfunc = lambda p, x: p[0] + p[1] * x
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    pinit = [1.0, -1.0]
    out = scipy.optimize.leastsq(errfunc, pinit,
                           args=(xdata, ydata, yerr), full_output=1)

    pfinal = out[0]
    covar = out[1]

    #print pfinal
    #print covar


    #index = pfinal[1]
    #amp = 10.0**pfinal[0]
    
    #print index, amp
    
    #indexErr = np.sqrt( covar[0][0] )
    #ampErr = np.sqrt( covar[1][1] ) * amp
    
    #y_predict = powerlaw(x_predict, amp, index)
    y_predict = fitfunc(pfinal, x_predict)
    return x_predict, y_predict

#data_mask = Cbins < 5.0
#def linear_fitting( filename )
def fitting_errftn( x_predict, xdata, ydata, yerr):
    import scipy
    #powerlaw = lambda x, amp, index: amp * (x**index)
    #logx = np.log10(xdata)
    #logy = np.log10(ydata)
    #logyerr = yerr / ydata

    # define our (line) fitting function
    #fitfunc = lambda p, x: p[0] + p[1] * x
    fitfunc = lambda p, x : p[0] * ( 1. - scipy.special.erf( (x-p[1])*1./p[2]  ))
    errfunc = lambda p, x, y, err: (y - fitfunc(p, x)) / err

    #pinit = [1.0, -1.0]
    
    xmid = (xdata.max() - xdata.min()) /2.
    pinit = [1, xmid, xmid ]
    out = scipy.optimize.leastsq(errfunc, pinit,
                           args=(xdata, ydata, yerr), full_output=1)

    pfinal = out[0]
    covar = out[1]
    
    y_predict = fitfunc( pfinal, x_predict )
    return x_predict, y_predict


# call file

def fitting_SP( property = None, filter=None, kind = None, suffix='', plot=False, function = None,
                path = '../data_txt/systematics/'):

   
    import scipy
    
    for p in property : 
        if plot : 
            fig, ax = plt.subplots(2,2,figsize = (15,10))
            ax = ax.ravel()
            i = 0
        for f in filter : 
            filename = path+'systematic_'+p+'_'+f+'_'+kind+'_'+suffix+'.txt'
            #print filename
            data = np.loadtxt(filename)
            bins, Cdensity, Cerr, Cf_area, Rdensity, Rerr, Rf_area = [data[:,j] for j in range(data[0].size)]
            zeromaskC, zeromaskR = ( Cdensity != 0.0 ), (Rdensity != 0.0 )
            Cbins, Cdensity, Cerr = bins[zeromaskC], Cdensity[zeromaskC], Cerr[zeromaskC]
            Rbins, Rdensity, Rerr = bins[zeromaskR], Rdensity[zeromaskR], Rerr[zeromaskR]                                                  
            
            C_bin_predict = np.zeros(bins.size)
            R_bin_predict = np.zeros(bins.size)
            #predict[~zeromaskC] = np.nan

            _, R_predict = fitting_linear( bins, Rbins, Rdensity, Rerr)
            if function == 'linear':
                _, C_predict = fitting_linear( bins, Cbins, Cdensity, Cerr)
                #_, R_predict = fitting_linear( bins, Rbins, Rdensity, Rerr)
                
            elif function == 'sqrt' : 
                _, C_predict = fitting_linear( np.sqrt(bins), np.sqrt(Cbins), Cdensity, Cerr)
            elif function == 'square' : 
                _, C_predict = fitting_linear( bins**2, Cbins**2, Cdensity, Cerr)    
            elif function == 'log' : 
                _, C_predict = fitting_linear( np.log10(bins), np.log10(Cbins), Cdensity, Cerr)  
                
            elif function == 'errftn' : 
                _, C_predict = fitting_errftn( bins, Cbins, Cdensity, Cerr)
                
                #_, R_predict = fitting_errftn( bins, Rbins, Rdensity, Rerr)                 
                
            else : print 'Please enter the kind of fitting function'
                
            Cchi2_model = np.sum((Cdensity - C_predict[zeromaskC])**2 *1./Cerr**2)
            Rchi2_model = np.sum((Rdensity - R_predict[zeromaskR])**2 *1./Rerr**2)
 
            Cchi2_null = np.sum((Cdensity - 1.0)**2 *1./Cerr**2)
            Rchi2_null = np.sum((Rdensity - 1.0)**2 *1./Rerr**2)
        
            Del_Cchi2 = Cchi2_null - Cchi2_model 
            Del_Rchi2 = Rchi2_null - Rchi2_model
            
            print 'chi2_null = ', Cchi2_null
            print 'chi2_mod. = ', Cchi2_model
            print 'Delta chi2 (sample) =', Cchi2_null - Cchi2_model 
            

            
            DAT = np.column_stack((bins, C_predict, R_predict))
            header = 'Delta chi2 (sample) = '+ str(Del_Cchi2)+ '\nDelta chi2 (random) = '+str(Del_Rchi2) +\
            '\nbins, Sample, Random'
            
            np.savetxt(filename+'.model', DAT, header =header)
            print 'output save to ', filename+'.model\n'
            
            if plot : 
                ax[i].errorbar(Cbins, Cdensity, yerr=Cerr, fmt='b-', label='dmass' + ' $\Delta \chi^2={:0.2f}$'.format(Del_Cchi2))
                ax[i].errorbar(Rbins, Rdensity, yerr=Rerr, fmt='r-', label='random')
                ax[i].plot(bins, C_predict, 'b--', alpha=1.0)
                #ax[i].plot(Rbins*1.005, R_predict, 'r-', alpha = 0.3)
                ax[i].plot(bins*1.005, R_predict, 'r--', alpha = 1.0)
                ax[i].axhline(y=1.0, color='grey', ls = '--')
                ax[i].set_xlabel(p+'_'+f)
                ax[i].set_ylabel('galaxy num density')
                ax[i].set_ylim(0.7, 1.3)
                ax[i].legend(loc=1)
                if function == 'log' : ax[i].set_xscale('log')
                #if p == 'EXPTIME' : 
                ax[i].set_xlim(None, Rbins.max() * 1.01)
                i += 1
                
        if plot : 
            figname = path+'systematic_fitting_'+p+'_'+f+'_'+kind+'_'+suffix+'.png'
            fig.savefig(figname)
            print 'saving fig to ', figname



def calculate_weight( property = None, filter=None, kind = None, suffix='', plot=False, function = None,
                path = '../data_txt/systematics/', catalog = None, sysMap= None, 
                weight=False, raTag ='RA', decTag='DEC', nside=4096):

   
    import scipy
        
    #for p in property :
    p = property
    f = filter

    if p == 'GE' : nside = 512
    elif p == 'NSTARS_allband' : nside = 1024
    else : nside = 4096
        
    if plot:
        #fig, ax = plt.subplots(2,2,figsize = (15,10))
        fig, ax = plt.subplots()
        #ax = ax.ravel()
        #i = 0
    
    #for f in filter : 
    filename = path+'systematic_'+p+'_'+f+'_'+kind+'_'+suffix+'.txt'
    #print filename
    data = np.loadtxt(filename)
    bins, Cdensity, Cerr, Cf_area, Rdensity, Rerr, Rf_area = [data[:,j] for j in range(data[0].size)]
    zeromaskC, zeromaskR = ( Cdensity != 0.0 ), (Rdensity != 0.0 )
    Cbins, Cdensity, Cerr = bins[zeromaskC], Cdensity[zeromaskC], Cerr[zeromaskC]
    Rbins, Rdensity, Rerr = bins[zeromaskR], Rdensity[zeromaskR], Rerr[zeromaskR]                                                  
    
    #C_bin_predict = np.zeros(bins.size)
    #R_bin_predict = np.zeros(bins.size)
    #predict[~zeromaskC] = np.nan
    
    mapname = 'sys_'+p+'_'+f+'_'+kind #+'_masked'
    catHpInd = hpRaDecToHEALPixel(catalog[raTag], catalog[decTag], nside=nside, nest= False)
    #signal = sysMap[mapname]['SIGNAL'][catHpInd]

    min = np.min(sysMap[mapname]['SIGNAL'])
    max = np.max(sysMap[mapname]['SIGNAL'])
    bin_num = 100

    if function == 'log' : log = True
    else : log = False

    bin_center, binned_cat, keeps = divide_bins( sysMap[mapname], Tag = 'SIGNAL', \
                                                min = min, max = max, bin_num = bin_num, log=log )


    _, R_predict = fitting_linear( bin_center, Rbins, Rdensity, Rerr )
    
    if function == 'linear':
        _, C_predict = fitting_linear( bin_center, Cbins, Cdensity, Cerr)

    elif function == 'sqrt' : 
        _, C_predict = fitting_linear( np.sqrt(bin_center), np.sqrt(Cbins), Cdensity, Cerr)

    elif function == 'square' : 
        _, C_predict = fitting_linear( bin_center**2, Cbins**2, Cdensity, Cerr)

    elif function == 'log' : 
        _, C_predict = fitting_linear( np.log10(bin_center), np.log10(Cbins), Cdensity, Cerr)  

    elif function == 'errftn' : 
        _, C_predict = fitting_errftn( bin_center, Cbins, Cdensity, Cerr)                

    else : 
        print 'Please enter the kind of fitting function'
        return 0


    if plot : 
        ax.errorbar(Cbins, Cdensity, yerr=Cerr, fmt='b-', label='dmass')
        ax.errorbar(Rbins, Rdensity, yerr=Rerr, fmt='r-', label='random')
        ax.plot(bin_center, C_predict, 'b--', alpha=1.0)
        #ax[i].plot(Rbins*1.005, R_predict, 'r-', alpha = 0.3)
        ax.plot(bin_center*1.005, R_predict, 'r--', alpha = 1.0)
        ax.axhline(y=1.0, color='grey', ls = '--')
        ax.set_xlabel(p+'_'+f)
        ax.set_ylabel('galaxy num density')
        ax.set_ylim(0.7, 1.3)
        ax.legend(loc=1)
        ax.set_xlim(Rbins.min() * 0.95, Rbins.max()*1.05 )
        if function == 'log' : ax.set_xscale('log')
	plt.show() 

    wg = np.zeros( catalog.size, dtype=float)
    for i, sysMap_i in enumerate(binned_cat):
        HpIdxInSys_mask = np.in1d(catHpInd, sysMap_i['PIXEL'])
        wg[HpIdxInSys_mask] = 1./C_predict[i]
        #print np.sum(HpIdxInSys_mask), 1./C_predict[i], sysMap_i['PIXEL'].size
                        
    return wg


def plotting_significance( property = None, filter=None, kind = None, suffix='', 
                path = '../data_txt/systematics/', deltachi2 = False):

   
    import scipy
    
    key = 'significance'
    if deltachi2 : key = 'Delta chi2 (sample)'
    label = []
    siglist = []
    
    for p in property : 
        if p in ['NSTARS_allband', 'GE', 'FWHM_pca', 'SKYBRITEpca0', 'SKYBRITEpca1', 'SKYBRITEpca2', 'SKYBRITEpca3','SKYBRITEpca1'] : filt_effec = filter[0]
        else : filt_effec = filter
        for f in filt_effec : 
            filename = path+'systematic_'+p+'_'+f+'_'+kind+'_'+suffix+'.txt.model'
            file = open(filename, 'r')
            data = file.readlines()
            for s in data : 
                if key in s.split('#')[1] : 
                    sig = s.split('=')[1]
                    siglist.append( float(sig) )
                    label.append( p+'_'+f )
                    #print  p+'_'+f , sig
                    break
       
    arg = np.argsort(np.array(siglist), kind='quicksort')[::-1]
    #print arg
    #siglist = siglist[arg]
    #label = label[arg]
    siglist = [siglist[a] for a in arg]
    label = [label[a] for a in arg]
    print siglist
    fig, ax = plt.subplots(figsize = (10,5))
    ax.plot( np.arange( len(siglist) ), siglist,   'o'  )
    #ax.axvline(x = 2, ls = '--', color='k')
    #ax.axvline(x = 3, ls = '--', color='k')
    ax.set_ylim(0.01,1000)
    ax.set_xlabel(key)
    ax.set_xticks( np.arange( len(siglist) ) )
    ax.set_xticklabels(label, rotation = 90)
    ax.set_yscale('log')
    plt.show()
    fig.savefig(path+'chisquare_'+p+'_'+f+'_'+kind+'_'+suffix+'.png')
    return label, siglist
         


def sys_ngal(cat1 = None, cat2=None, rand1 = None, rand2 = None, sysmap = None, nside=4096, FullArea = None, 
             nbins = 10, properties = None, kind='SPT', suffix='', 
             outdir='../data_txt/systemtaics/', pixelmask = None, reweight=None):
    
    from systematics import ReciprocalWeights, jksampling, GalaxyDensity_Systematics
    for p in properties:
        if p is 'NSTARS_allband':
            nside = 1024
            filter = ['g']
        elif p is 'GE':
            nside = 512
            filter = ['g']
        elif p is 'FWHM_pca':
            nside = 4096
            filter = ['g']
        elif p in ['SKYBRITEpca0','SKYBRITEpca1', 'SKYBRITEpca2', 'SKYBRITEpca3'] :
            nside = 4096
            filter = ['g']
        elif p is 'SLR':
            nside = 512
            filter = ['g', 'r', 'i', 'z']
        else :
            nside = nside
            filter = ['g', 'r', 'i', 'z']
        for j,f in enumerate(filter):

            mapname = 'sys_'+p+'_'+f+'_'+kind #+'_masked'
            bins, Bdensity, Berr, Bf_area = GalaxyDensity_Systematics(cat1, sysmap[mapname], rand = rand1, nside = nside,\
                                                                        property = p, filter = f, nbins=nbins, FullArea = FullArea,\
                                                                        pixelmask = pixelmask, reweight=reweight)
            bins, Rdensity, Rerr, Rf_area = GalaxyDensity_Systematics(cat2, sysmap[mapname], rand = rand2, nside = nside,\
                                                                        property = p, filter = f, nbins=nbins, FullArea = FullArea,\
                                                                        pixelmask = pixelmask)
            #print Rdensity
            #bins = bins/np.sum(sysMap['SIGNAL']) *len(sysMap)
            #B_jkerr = jksampling(clean_dmass, MaskDic[mapname], property = p, nside = nside, njack = 30, raTag = 'RA', decTag = 'DEC' )
            os.system('mkdir '+outdir)
            filename = outdir+'systematic_'+p+'_'+f+'_'+kind+'_'+suffix+'.txt'
            #DAT = np.column_stack(( bins-(bins[1]-bins[0])*0.1, Bdensity, Berr, Bf_area, Bdensity, Berr, Bf_area  ))
            DAT = np.column_stack(( bins, Bdensity, Berr, Bf_area, Rdensity, Rerr, Rf_area  ))
            np.savetxt(filename, DAT, delimiter = ' ', header = 'bins, Ddensity, Derr, Dfarea, Rdensity, Rerr, Rfarea, R_jkerr')
            print "saving data to ", filename




    





def GetCov(full_j, it_j, njack):
    cov_j = []
    norm = 1. * (njack-1)/njack

    csize = len(full_j)
    matrix1, matrix2 = np.mgrid[0:csize, 0:csize]

    cov = 0
    for k in range(len(it_j)):
        print cov
        cov +=  (it_j[k][matrix1] - full_j[matrix1]) * (it_j[k][matrix2]- full_j[matrix2] )

    cov = norm * cov
    return cov


def jksampling(catalog, sysMap, property = 'NSTARS', nside = 256, njack = 10, raTag = 'RA', decTag = 'DEC' ):
    import os
    
    # jk error
    jkfile = './jkregion.txt'
    njack = njack
    jk.GenerateJKRegions( catalog[raTag], catalog[decTag], njack, jkfile)
    jktest = jk.SphericalJK( target = GalaxyDensity_Systematics, jkargs=catalog, jkargsby=[raTag, decTag], jkargspos=None, nojkargs=[sysMap], nojkargspos=1, nojkkwargs={ 'nside':nside, 'raTag':raTag, 'decTag':decTag, 'property':property }, regions = jkfile)
    jktest.DoJK( regions = jkfile)
    jkresults = jktest.GetResults(jk=True, full = True)

    full_j = jkresults['full'][1]
    it_j = np.array( [jkresults['jk'][i][1] for i in range(njack)] )

    print full_j.size
    print len(it_j)

    njack = len(it_j) #redefine njack
    norm = 1. * (njack-1)/njack
    
    cov = 0
    for k in range(len(it_j)):
        cov +=  (it_j[k] - full_j)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    
    cov = norm * cov
    #os.remove(jkfile)

    return np.sqrt(cov)



def w_subsample(subsample, gal_ra, gal_dec, sys_ra, sys_dec, delta_g, delta_sys,nside = 4096 ):
    sysMap = subsample.copy()
    
    sys_i = sysMap['SIGNAL']
    sys_mean = np.sum(sys_i)/len(sys_i)
    delta_sys = sys_i/sys_mean - 1.  # systematic overdensty 1d array corresponding to sysMap['PIXEL']
    sys_theta, sys_phi = hp.pix2ang(nside, sysMap['PIXEL'])
    sys_ra, sys_dec = GetRaDec(sys_phi,sys_theta)
    matrix1, matrix2 = np.mgrid[0:len(delta_g), 0:len(delta_sys)]
    gal_ra_m = gal_ra[matrix1]
    gal_dec_m = gal_dec[matrix1]
    sys_ra_m = sys_ra[matrix2]
    sys_dec_m = sys_dec[matrix2]
    sep = separation(gal_ra_m, gal_dec_m, sys_ra_m, sys_dec_m )
    #sep = hp.rotator.angdist([gal_ra_m, gal_dec_m],[sys_ra_m, sys_dec_m], lonlat=True)
    cross = delta_sys[matrix2] * delta_g[matrix1]
    sep_ravel = np.array(sep).ravel()
    cross_ravel = np.array(cross).ravel()
    
    # putting theta bins and count pairs
    sep_bins = np.logspace(0.1, 5.0, 10, base = 10)
    digitized_count = np.digitize(sep_ravel, sep_bins)
    pair_count = [ len(sep_ravel[digitized_count == i]) for i in range(len(sep_bins))]
    angularw = [np.sum(cross_ravel[digitized_count == i]) for i in range(len(sep_bins))]
    angular_theta = np.array(angularw) / np.array(pair_count)
    return sep_bins, angular_theta



def CrossCorrel(catalog, property = 'AIRMASS', filter = 'g', raTag = 'RA', decTag='DEC'):
    import healpy as hp

    path = '/n/des/lee.5922/data/systematic_maps/'
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and property.upper()+'__mean' in i and 'band_'+filter.lower() in i:
            print i
            sysMap = fitsio.read(path+i)

    nside = 4096

    # galaxy
    hpInd = hpRaDecToHEALPixel(catalog[raTag], catalog[decTag], nside=nside, nest= False)
    healCat = rf.append_fields(catalog,'HEALIndex', hpInd, dtypes=hpInd.dtype)
    N_validPixel = len(set(hpInd))
    FullArea = hp.nside2pixarea(nside, degrees = True) * N_validPixel
    N_mean = len(catalog)/FullArea
    Number_g = [len(catalog[healCat['HEALIndex'] == pixel]) for pixel in set(hpInd)]
    hpIndices = [pixel for pixel in set(hpInd)]
    A_pix = hp.nside2pixarea(nside, degrees = True)
    N_g = Number_g/A_pix
    delta_g = N_g/N_mean - 1.  # galaxy overdensity 1d array corresponding to hpInd
    gal_theta, gal_phi = hp.pix2ang(nside, hpIndices)
    gal_ra, gal_dec = GetRaDec(gal_phi,gal_theta)


    # split sys samples with mask
    n_sample = 10000
    mask = np.zeros(sysMap.size, dtype = bool)
    masklist = []
    for i in range(sysMap.size/n_sample):
        mask[i * n_sample:(i+1)*n_sample-1] = 1
        print mask
        masklist.append(mask)
        mask = np.zeros(sysMap.size, dtype = bool)

    mask[(sysMap.size/n_sample -1) * n_sample:] = 1
    masklist.append(mask)
    sys_samples = [sysMap[mask] for mask in masklist]



    angular_theta_list = []
    for sample in sys_samples[0:10]:
        bins, sample_w = w_subsample(sample, gal_ra, gal_dec, sys_ra, sys_dec, delta_g, delta_sys,nside = 4096 )
        angular_theta_list.append(sample_w)

    N_samples = len(sys_samples)
    averaged_angular_theta = np.sum( np.array(angular_theta_list ), axis = 0) / N_samples

    # error
    var = [np.sum((averaged_angular_theta - w_i)**2 )/N_samples**2 for w_i in angular_theta_list]
    error = np.sqrt(var)


    # auto correlation








    # gallist, syslist
    """
    matrix1, matrix2 = np.mgrid[0:len(delta_g), 0:len(delta_sys)]
    gal_ra = gal_ra[matrix1]
    gal_dec = gal_dec[matrix1]
    sys_ra = sys_ra[matrix2]
    sys_dec = sys_dec[matrix2]
    sep = hp.rotator.angdist([gal_ra, gal_dec],[sys_ra, sys_dec], lonlat=True)
    """


    
    sep_list = []
    angularw = []
    ones = np.ones(sys_ra.size)
    for j in range(len(delta_g)):
        sep = separation(gal_ra[j] * ones, gal_dec[j] * ones, sys_ra, sys_dec )
        w = sep * delta_sys * delta_g[j] * ones
        sep_list.append(sep)
        angularw.append(w)

    count = count.ravel()
    angularw = angularw.ravel()
    digitized_count = np.digitize(count, sep_bins)
    pair_count = [ len(count[digitized_count == i])/2. for i in range(len(sep_bins))]
    angularw = [len(angularw[digitized_count == i])/2. for i in range(len(sep_bins))]

    cross_correl = angularw/pair_count




    # TreeCorr --------------------------------
    """
    sys_i = sysMap['SIGNAL']
    sys_mean = np.sum(sys_i)/len(sys_i)
    delta_sys = sys_i/sys_mean - 1.  # systematic overdensty 1d array corresponding to sysMap['PIXEL']
    sys_theta, sys_phi = hp.pix2ang(nside, sysMap['PIXEL'])
    sys_ra, sys_dec = GetRaDec(sys_phi,sys_theta)
    
    
    import treecorr
    cat_gal = treecorr.Catalog(ra=gal_ra, dec=gal_dec, w= delta_g,ra_units='deg', dec_units='deg')
    cat_sys = treecorr.Catalog(ra=sys_ra, dec=sys_dec, w= delta_sys, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=cmass_rand['RA'], dec=cmass_rand['DEC'], is_rand=True, ra_units='deg', dec_units='deg')
    cat_rand2 = treecorr.Catalog(ra=sys_ra, dec=sys_dec, is_rand=True, ra_units='deg', dec_units='deg')

    nbins = 30
    bin_size = 0.2
    min_sep = 0.05
    dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
    dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
    rd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
    rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
    dd.process_cross(cat_gal, cat_sys)
    dr.process(cat_gal, cat_rand)
    rd.process(cat_sys, cat_rand2)
    rr.process(cat_rand)

    xi, varxi = dd.calculateXi(rr,dr,rd)
    return dd.meanr, xi, varxi
    """

    galaxy_density_list = []
    for k in keeps:
        keephpInd = sysMap[k]['PIXEL']
        keep = np.in1d(hpInd, keephpInd)
        Npix = len(set(hpInd[keep]))
        PixelArea = hp.nside2pixarea(nside, degrees = True) * Npix
        galaxy_density = np.sum(keep)/PixelArea /N_total
        galaxy_density_list.append(galaxy_density)


"""
def _acf(data, rand, weight = None):
    # cmass and balrog : all systematic correction except obscuration should be applied before passing here
    
    import treecorr
    
    weight_data = None
    weight_rand = None

    if weight is not None:
        weight_data = weight[0]
        weight_rand = weight[1]

    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand, ra_units='deg', dec_units='deg')

    nbins = 20
    bin_size = 0.5
    min_sep = .01
    sep_units = 'degree'
    dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dd.process(cat)
    dr.process(cat,cat_rand)
    rr.process(cat_rand)

    xi, varxi = dd.calculateXi(rr,dr)

    return dd.meanr, xi, varxi

def angular_correlation(data = None, rand = None, weight = None, suffix = ''):
    # jk sampling
    import os
    from suchyta_utils import jk
    print 'calculate angular correlation function'
    #r, xi, xierr = _acf( data, rand, weight = weight )
    
    jkfile = './jkregion.txt'
    njack = 30
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _acf, jkargs=[ data, rand ], jkargsby=[[raTag, decTag],[raTag, decTag]], regions = jkfile, nojkkwargs = {'weight':weight})
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    r, xi, varxi = jkresults['full']
    
    # getting jk err
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    norm = 1. * (njack-1)/njack
    xi_cov = 0
    for k in range(njack):
        xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)

    filename = 'data_txt/acf_comparison'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename
    #return r, xi, jkerr

def _cross_acf(data, data2, rand, rand2, weight = None):
    
    import treecorr
    
    weight_data1 = None
    weight_data2 = None
    weight_rand1 = None
    weight_rand2 = None
    
    if weight is not None:
        weight_data1 = weight[0]
        weight_rand1 = weight[1]
        weight_data2 = weight[2]
        weight_rand2 = weight[3]

    
    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data1, ra_units='deg', dec_units='deg')
    cat2 = treecorr.Catalog(ra=data2['RA'], dec=data2['DEC'], w= weight_data2, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand1, ra_units='deg', dec_units='deg')
    cat_rand2 = treecorr.Catalog(ra=rand2['RA'], dec=rand2['DEC'], is_rand=True, w = weight_rand2, ra_units='deg', dec_units='deg')

    #nbins = 30
    #bin_size = 0.2
    #min_sep = 0.1
    #sep_units = 'arcmin'
    
    nbins = 20
    bin_size = 0.5
    min_sep = .001
    sep_units = 'degree'
    
    dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    rd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    
    dd.process(cat, cat2)
    dr.process(cat, cat_rand)
    rd.process(cat2, cat_rand2)
    rr.process(cat_rand, cat_rand2)
    
    xi, varxi = dd.calculateXi(rr,dr,rd)
    return dd.meanr, xi, varxi


def cross_angular_correlation(data = None, data2 = None, rand = None, rand2= None, weight = None, suffix = ''):
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
    njack = 10
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cross_acf, jkargs=[data, data2, rand, rand2], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag],[raTag, decTag]], nojkkwargs = {'weight':weight}, jkargspos=None, regions = jkfile)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    r, xi, varxi = _cross_acf(data, data2, rand, rand2, weight = weight )
    
    # getting jk err
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    norm = 1. * (njack-1)/njack
    xi_cov = 0
    for k in range(njack):
        xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)

    filename = 'data_txt/acf_cross'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename
    #return r, xi, jkerr

"""
def CMASSclassifier(sdss_data, des_data, train_sample = None ):

    from cmass_stripe82 import cmass_criteria, classifier3
    
    classifier_tags = ['flux_model'] #,'modelmag', 'cmodelmag']
    
    classifier_tags_g = [tag+'_g' for tag in classifier_tags]
    classifier_tags_r = [tag+'_r' for tag in classifier_tags]
    classifier_tags_i = [tag+'_i' for tag in classifier_tags]
    classifier_tags_z = [tag+'_z' for tag in classifier_tags]
    
    des_tags = classifier_tags_g + classifier_tags_r + classifier_tags_i
    des_tags = [ tag.upper() for tag in des_tags ]


    sdss_classifier_tags = ['modelMag','modelMagErr','fiber2Mag','cModelMag','cModelMagErr',
                        'fracDeV','extinction','skyIvar','psffwhm',
                        'expMag','expMagErr','deVMag','deVMagErr','psfMag']
    
    sdss_tags_g = [tag +'_g' for tag in sdss_classifier_tags]
    sdss_tags_r = [tag +'_r' for tag in sdss_classifier_tags]
    sdss_tags_i = [tag +'_i' for tag in sdss_classifier_tags]
    sdss_tags_z = [tag +'_z' for tag in sdss_classifier_tags]
    sdss_tags = sdss_tags_g + sdss_tags_r + sdss_tags_i + sdss_tags_z + ['type','clean']
    
    sdss_tags = [ tag.upper() for tag in sdss_tags ]
    
    des_prior_cut = cmass_criteria(des_data)
    des_data = des_data[des_prior_cut]
    
    train_size = len(des_data)*1/5
    
    if train_sample is not None:
        train_sample = train_sample[ cmass_criteria(train_sample)]
        train_size = len(train_sample)*1/5

    cl, predict_mask, good_mask = classifier3(sdss_data, des_data, des_tags = des_tags, sdss_tags = sdss_tags, train_size = train_size, train = train_sample)
    print predict_mask.size, des_data.size

    return predict_mask

def jk_error(cmass_catalog, njack = 30 , target = None, jkargs=[], jkargsby=[], raTag = 'RA', decTag = 'DEC'):

    import os
    from suchyta_utils import jk
    
    # jk error
    jkfile = './jkregion.txt'
    jk.GenerateJKRegions( cmass_catalog[raTag], cmass_catalog[decTag], njack, jkfile)
    jktest = jk.SphericalJK( target = target, jkargs=jkargs, jkargsby=jkargsby, jkargspos=None, nojkargs=None, nojkargspos=None, nojkkwargs={}, regions = jkfile)
    jktest.DoJK( regions = jkfile)
    jkresults = jktest.GetResults(jk=True, full = True)
    
    full_j = jkresults['full'][1]
    it_j = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    njack = len(it_j) #redefine njack
    norm = 1. * (njack-1)/njack
    
    cov = 0
    for k in range(len(it_j)):cov +=  (it_j[k] - full_j)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    
    cov = norm * cov
    os.remove(jkfile)
    jkerr = np.sqrt(cov)

    return full_j, jkerr



def make2SphereRandoms( ra_min = 0, ra_max = 360, dec_min = -90, dec_max= 90, size = 10000 , plot = False ):
    
    theta_min, phi_min = convertRaDecToThetaPhi(ra_min, dec_min)
    theta_max, phi_max = convertRaDecToThetaPhi(ra_max, dec_max)
    v_min, v_max = (np.cos(theta_min) + 1)/2., (np.cos(theta_max) + 1)/2.
    u_min, u_max = phi_min/(np.pi * 2), phi_max/(np.pi * 2)
    
    u = np.random.uniform(u_min, u_max, size)
    v = np.random.uniform(v_min, v_max, size)

    R = 1.0

    theta = np.arccos(2 * v - 1)
    phi = 2*np.pi * u
    
    ra, dec = convertThetaPhiToRaDec(theta, phi)
    
    random = np.zeros((size,), dtype=[('RA', 'f8'), ('DEC', 'f8')])
    random['RA'] = ra
    random['DEC'] = dec
    
    if plot == True:
    
        xx = R * np.sin(theta) * np.cos(phi)
        yy = R * np.sin(theta) * np.sin(phi)
        zz = R * np.cos(theta)
        
        
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.set_aspect("equal")

        # top: elev=90, side: elev=0
        ax.view_init(elev=0, azim=0)

        u = np.linspace(0, 2 * np.pi, 120)
        v = np.linspace(0, np.pi, 60)

        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha = 0.3, linewidth = 0)
        ax.scatter(xx,yy,zz,color="k",s=1)
        plt.title('Side View')
        #plt.axis('off')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.set_aspect("equal")
        ax2.view_init(elev=90, azim=0)
        ax2.plot_surface(x, y, z,  rstride=1, cstride=1, color='c', alpha = 0.3, linewidth = 0)
        ax2.scatter(xx,yy,zz,color="k",s=1)
        
        plt.title('Top View')
        #plt.axis('off')
        plt.show()
        fig.savefig('../figure/randomSphere_3d')

        fig2, ax = plt.subplots(1,1)
        ax.plot(ra, dec, 'b.')
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        ax.set_title('Random sample in RA and DEC, size = 5000')
        fig2.savefig('../figure/randomSphere_radec')
        print 'fig saved :','../figure/randomSphere_3d.png',' ../figure/randomSphere_radec.png'

    return random


def SystematicMask( catalog, nside = 1024, raTag = 'RA', decTag = 'DEC' ):
    
    path = '/n/des/lee.5922/data/systematic_maps/'

    sysMap_name = ['Y1A1NEW_COADD_STRIPE82_band_g_nside4096_oversamp4_FWHM_coaddweights3_mean.fits.gz',
                   'Y1A1NEW_COADD_STRIPE82_band_g_nside4096_oversamp4_SKYBRITE_coaddweights3_mean.fits.gz',
                   'Y1A1NEW_COADD_STRIPE82_band_r_nside4096_oversamp4_SKYBRITE_coaddweights3_mean.fits.gz',
                   'Y1A1NEW_COADD_STRIPE82_band_g_nside4096_oversamp4_SKYSIGMA_coaddweights3_mean.fits.gz',
                   'Y1A1NEW_COADD_STRIPE82_band_r_nside4096_oversamp4_SKYSIGMA_coaddweights3_mean.fits.gz',
                   'Y1A1NEW_COADD_STRIPE82_band_i_nside4096_oversamp4_SKYSIGMA_coaddweights3_mean.fits.gz']
                   
    sysMap_hp = [ hp.read_map( path+sysMap, nest=False ) for sysMap in sysMap_name ]
    sysMap_ud = [ hp.ud_grade(sysMap, nside_out = nside ) for sysMap in sysMap_hp ]
    
    default = hp.mask_good(sysMap_ud[0])
    goodmask =  (
                 #(sysMap_ud[0] > 4.7 ) &
                 (sysMap_ud[1] < 170 ) &
                 #(sysMap_ud[2] < 390 ) &
                 #(sysMap_ud[3] < 6.2 ) &
                 (sysMap_ud[4] < 9.8 ) #&
                 #(sysMap_ud[5] < 16 )
                 )

    clean = default & goodmask

    maskIndices = np.arange(sysMap_ud[0].size)
    goodIndices = maskIndices[clean]
    
    hpInd = hpRaDecToHEALPixel(catalog[raTag], catalog[decTag], nside=nside, nest= False)
    
    CleanCatalogMask = np.in1d( hpInd, goodIndices )
    #CleanCatalog = catalog[CleanCatalogMask]
    return CleanCatalogMask


"""
def _LS(lense, source, rand, weight = None, Boost = False):


    #Calculate Lensing Signal deltaSigma, Boost Factor, Corrected deltaSigma
    
    #parameter
    #---------
    #lense : lense catalog
    #source : source catalog (shear catalog that contains e1, e2)
    #rand : random catalog
    
        
    
    
    import treecorr
    
    weight_lense, weight_source, weight_rand = weight
    
    z_s_min = 0.7
    z_s_max = 1.0
    
    num_lense_bin = 3
    num_source_bin = 3
    z_l_bins = np.linspace(0.45, 0.55, num_lense_bin)
    z_s_bins = np.linspace(z_s_min, z_s_max, num_source_bin)
    #matrix1, matrix2 = np.mgrid[0:z_l_bins.size, 0:z_s_bins.size]
    source = source[ (source['DESDM_ZP'] > z_s_min ) & (source['DESDM_ZP'] < z_s_max )]
    lense = lense[ (lense['DESDM_ZP'] > 0.45) & (lense['DESDM_ZP'] < 0.55) ]
    rand = rand[ (rand['Z'] > 0.45) & (rand['Z'] < 0.55) ]

    z_l_bincenter, lense_binned_cat,_ = divide_bins( lense, Tag = 'DESDM_ZP', min = 0.45, max = 0.55, bin_num = num_lense_bin)
    z_r_bincenter, rand_binned_cat,_ = divide_bins( rand, Tag = 'Z', min = 0.45, max = 0.55, bin_num = num_lense_bin)
    z_s_bincenter, source_binned_cat,_ = divide_bins( source, Tag = 'DESDM_ZP', min = z_s_min, max = z_s_max, bin_num = num_source_bin)
    
    # angular diameter
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70,Om0=0.274)
    
    from astropy import constants as const
    from astropy import units as u
    
    h = 0.7
    c = const.c.to(u.megaparsec / u.s)
    G = const.G.to(u.megaparsec**3 / (u.M_sun * u.s**2))
    
    dA_l = cosmo.angular_diameter_distance(z_l_bins) * h
    dA_s = cosmo.angular_diameter_distance(z_s_bins) * h
    matrix1, matrix2 = np.mgrid[0:z_l_bins.size, 0:z_s_bins.size]
    dA_ls = cosmo.angular_diameter_distance_z1z2(z_l_bins[matrix1],z_s_bins[matrix2]) * h
    dA_l_matrix = dA_l[matrix1]
    dA_s_matrix = dA_s[matrix2]
    Sigma_critical = c**2 / (4 * np.pi * G) * dA_s_matrix/(dA_l_matrix * dA_ls)
    
    
  
  

    min_sep_1 = 0.001
    bin_size = 0.4
    nbins = 30
    
    theta = [min_sep_1 * np.exp(bin_size * i) for i in range(nbins) ]
    r_p_bins = theta * dA_l[0]


    if Boost is True:
        n_SL = []
        n_SR = []
        N_SL = []
        N_SR = []
        
        weight = 1./(Sigma_critical)**2
        weight = np.ones(Sigma_critical.shape)
        
        print " **  To do : add Boost codes Weight "
        
        for i, S in enumerate(source_binned_cat):
            source_cat = treecorr.Catalog(ra=S['RA'], dec=S['DEC'], ra_units='deg', dec_units='deg')
            for (j, dA), L, R in zip( enumerate(dA_l), lense_binned_cat, rand_binned_cat ):
                min_sep = theta[0] * dA_l[0]/dA
                lense_cat = treecorr.Catalog(ra=L['RA'], dec=L['DEC'], ra_units='deg', dec_units='deg')
                rand_cat = treecorr.Catalog(ra=R['RA'], dec=R['DEC'], ra_units='deg', dec_units='deg')
                SL = treecorr.NNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins = nbins, sep_units='degree')
                SR = treecorr.NNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins = nbins, sep_units='degree')
                SL.process(lense_cat, source_cat)
                SR.process(rand_cat, source_cat)
                w = 1./weight[j,i]**2
                n_SL.append(w * SL.npairs/np.sum(SL.npairs))
                n_SR.append(w * SR.npairs/np.sum(SR.npairs))
                N_SL.append(SL.npairs)
                N_SR.append(SR.npairs)
        
            N_SL_list = np.sum( N_SL, axis = 0)
            N_SR_list = np.sum( N_SR, axis = 0)
            n_SL_list = np.sum( n_SL, axis = 0)
            n_SR_list = np.sum( n_SR, axis = 0)

            BoostFactor = n_SL_list/n_SR_list

            nbar = 1e-4
            P = 1e+4
            w_FKP = 1./(nbar*P + 1)
            err = 1./ np.sqrt(w_FKP) / np.sqrt(N_SL_list) * BoostFactor
            #err = 1./ np.sqrt(n_SL) * BoostFactor




    # Test Boost
"""
"""
    rand2 = cmass_rand[ (cmass_rand['Z'] > z_s_min) & (cmass_rand['Z'] < z_s_max) ]
    rand2_cat = treecorr.Catalog(ra=rand2['RA'], dec=rand2['DEC'], ra_units='deg', dec_units='deg',is_rand =True )
    #z_r_bincenter, rand2_binned_cat,_ = divide_bins( rand2, Tag = 'Z', min = 0.7, max = 1.0, bin_num = num_source_bin)
    cross_ang = []
    err_varxi = []
    #for i, S in enumerate(source_binned_cat):
    source_cat = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], ra_units='deg', dec_units='deg')
    #rand2_cat = treecorr.Catalog(ra=rand2['RA'], dec=R2['DEC'], ra_units='deg', dec_units='deg',is_rand =True )
    for dA, L, R in zip( dA_l, lense_binned_cat, rand_binned_cat ):
        min_sep = theta[0] * dA_l[0]/dA
        lense_cat = treecorr.Catalog(ra=L['RA'], dec=L['DEC'], ra_units='deg', dec_units='deg')
        rand_cat = treecorr.Catalog(ra=R['RA'], dec=R['DEC'], ra_units='deg', dec_units='deg',is_rand =True )
        SL = treecorr.NNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins = nbins, sep_units='degree')
        SR = treecorr.NNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins = nbins, sep_units='degree')
        LR = treecorr.NNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins = nbins, sep_units='degree')
        RR = treecorr.NNCorrelation(min_sep=min_sep, bin_size=bin_size, nbins = nbins, sep_units='degree')
        SL.process(lense_cat, source_cat)
        SR.process(rand2_cat, source_cat)
        LR.process(lense_cat, rand_cat)
        RR.process(rand_cat, rand2_cat)
        xi, varxi = SL.calculateXi(RR,SR,LR)
        cross_ang.append(xi)
        err_varxi.append(varxi)
    cross_rp = np.sum(cross_ang, axis = 0)
    err_varxi = np.sum(err_varxi, axis = 0)
"""
"""


    # lensing Signal
    
    lense_cat_tot = treecorr.Catalog(ra=lense['RA'], dec=lense['DEC'], w = weight_lense, ra_units='deg', dec_units='deg')
    source_cat_tot = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], w = weight_source, g1=source['E1'], g2 = source['E2'], ra_units='deg', dec_units='deg')
    
    gamma_matrix = []
    varxi_matrix = []

    for dA in dA_l:
        min_sep = theta[0] * dA_l[0]/dA
        ng = treecorr.NGCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
        ng.process(lense_cat_tot, source_cat_tot)
        gamma_matrix.append(ng.xi)
        varxi_matrix.append(ng.varxi)

    varxi = np.sum(varxi_matrix, axis= 0)/len(dA_l)
    gamma_matrix = np.array(gamma_matrix)
    gamma_avg = np.sum(np.array(gamma_matrix), axis = 0)/len(z_l_bins)


    matrix1, matrix2 = np.mgrid[0:dA_l.size, 0:dA_s.size]
    delta_sigma=[]
    delta_sigma_1bin = []
    summed_sigma_crit = []
    for i in range(len(r_p_bins)):
        g = gamma_matrix[:,i]
        g_matrix = g[matrix1]
        ds = np.sum(g_matrix * Sigma_critical).to(u.Msun/u.parsec**2)
        ds_1bin = np.sum(g_matrix * Sigma_critical, axis = 1).to(u.Msun/u.parsec**2)
        sc = np.sum(Sigma_critical).to(u.Msun/u.parsec**2)
        delta_sigma.append(ds.value)
        delta_sigma_1bin.append(ds_1bin.value)
        summed_sigma_crit.append(sc.value)

    delta_sigma = np.array(delta_sigma)/(len(z_l_bins) * len(z_s_bins))

    if Boost is True :
        Corrected_delta_sigma = np.array(BoostFactor) * delta_sigma
        return r_p_bins, np.array(BoostFactor), delta_sigma, Corrected_delta_sigma

    else :
        #delta_sigma_1bin = np.array(delta_sigma_1bin)/len(z_s_bins)
        #summed_sigma_crit = np.array(summed_sigma_crit)/(len(z_l_bins) * len(z_s_bins))
        #error_tot = np.sqrt(summed_sigma_crit**2 * varxi)
    
        return r_p_bins, delta_sigma


def LensingSignal(lense = None, source = None, rand = None, weight = None, suffix = ''):
    
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
    njack = 10
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( lense[raTag], lense[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _LS, jkargs=[lense, source, rand], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag]], nojkkwargs={'weight':weight}, regions = jkfile)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    r_p_bins, LensSignal = jkresults['full']
    
    # getting jk err
    LensSignal_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    #correctedLensSignal_i = np.array( [jkresults['jk'][i][2] for i in range(njack)] )
    #Boost_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    norm = 1. * (njack-1)/njack
    #Boost_cov, LensSignal_cov, correctedLensSignal_cov = 0, 0, 0
    LensSignal_cov = 0
    for k in range(njack):
        #Boost_cov +=  (Boost_i[k] - BoostFactor)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
        LensSignal_cov +=  (LensSignal_i[k] - LensSignal)**2
        #correctedLensSignal_cov +=  (correctedLensSignal_i[k] - correctedLensSignal)**2

    #Boostjkerr = np.sqrt(norm * Boost_cov)
    LSjkerr = np.sqrt(norm * LensSignal_cov)
    #CLSjkerr = np.sqrt(norm * correctedLensSignal_cov)

    filename = 'data_txt/lensing_'+suffix+'.txt'
    header = 'r_p_bins, LensSignal, LSjkerr'
    DAT = np.column_stack((r_p_bins, LensSignal, LSjkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename
"""



def ReciprocalWeights( catalog = None, sysMap = None, property = None, filter = None, nside = 4096, kind = None ):

    if sysMap is None:
        if property == 'NSTARS':
            nside = 1024
            filename = 'y1a1_gold_1.0.2_stars_nside1024.fits'
            sysMap_o = loadSystematicMaps( filename = filename, nside = nside )
                
            if kind is 'STRIPE82' :sysMap = sysMap_o[sysMap_o['DEC'] > -3.0]
            if kind is 'SPT' :sysMap = sysMap_o[sysMap_o['DEC'] < -3.0]
            if kind is 'Y1A1':sysMap = sysMap_o
        
        else :
            sysMap = loadSystematicMaps( property = property, filter = filter, nside = nside , kind = kind)

    else : pass

    filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+'_masked.txt'
    data = np.loadtxt(filename)
    bins, density, error = data[:,0], data[:,4], data[:,5]
    weights = np.ma.fix_invalid(1./density, fill_value = 0.0).data

    catHpInd = hpRaDecToHEALPixel(catalog['RA'], catalog['DEC'], nside=nside, nest = False)
    
    bin_num = bins.size
    step = (bins[2] - bins[1])
    bin_center, binned_cat, keeps = divide_bins( sysMap, Tag = 'SIGNAL', min = bins.min() - step/2., max = bins.max() + step/2., bin_num = bin_num )
    
    wg = np.zeros( catalog.size, dtype=float)
    for i, sysMap_i in enumerate(binned_cat):
        HpIdxInSys_mask = np.in1d(catHpInd, sysMap_i['PIXEL'])
        wg[HpIdxInSys_mask] = weights[i]
        #print i, bin_center[i], weights[i], np.sum(HpIdxInSys_mask)

    return wg



def SysMapBadRegionMask(catalog, sysMap, nside = 512, cond = None, val = None):
    """
    get badregion pixel num from sysMap
    get matched HpInx from catalog
    make mask
    return mask
    """
    
    if cond=='=':
        use = (sysMap['SIGNAL']==val)
    elif cond=='<':
        use = (sysMap['SIGNAL'] < val)
    elif cond=='<=':
        use = (sysMap['SIGNAL'] <= val)
    elif cond=='>':
        use = (sysMap['SIGNAL'] > val)
    elif cond=='>=':
        use = (sysMap['SIGNAL'] >= val)

    MaskedSysMap = sysMap[use]

    catHpInd = hpRaDecToHEALPixel(catalog['RA'], catalog['DEC'], nside=nside, nest= False)
    HpIdxInSys_mask = np.in1d(catHpInd, MaskedSysMap['PIXEL'])
    #HpIdxInSys = catHpInd[HpIdxInSys_mask]
    
    return MaskedSysMap, HpIdxInSys_mask



def ForegroundStarCorrelation( dmass = None, star = None, rand = None, weight = [None, None, None]):
    
    """
    Counts number of galaxies around bright stars and calculates the normalized number density as a function of theta(arcseconds).
    """
    
    import treecorr
    
    w_dmass, w_star, w_rand = weight
    
    dmasscat = treecorr.Catalog(ra=dmass['RA'], dec=dmass['DEC'], w=w_dmass, ra_units='deg', dec_units='deg')
    starcat = treecorr.Catalog(ra=star['RA'], dec=star['DEC'], w=w_star, ra_units='deg', dec_units='deg')
    randcat = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], w=w_rand, is_rand=True, ra_units='deg', dec_units='deg')
    
    bin_size = 0.1
    min_sep = 0.0002
    max_sep = 0.05
    sep_units = 'degrees'
    
    ds = treecorr.NNCorrelation( bin_size = bin_size, min_sep= min_sep, max_sep = max_sep, sep_units=sep_units, verbose=1)
    rs = treecorr.NNCorrelation( bin_size = bin_size, min_sep= min_sep, max_sep = max_sep, sep_units=sep_units, verbose=1)
    ds.process(dmasscat, starcat)
    rs.process(randcat, starcat)
    
    import suchyta_utils
    from suchyta_utils import hp
    
    A = suchyta_utils.hp.GetArea(cat=dmass, ra='RA', dec='DEC', nside=4096)
    A_rand = suchyta_utils.hp.GetArea(cat=rand, ra='RA', dec='DEC', nside=4096)
    
    theta = ds.meanr
    
    n = ds.npairs * 1./ds.tot # A
    n_rand = rs.npairs * 1./rs.tot #A_rand
    
    averaged_n = n/n_rand
    
    
    P = 1e+4
    n_bar = 1e-4
    w_FKP = 1./( 1 + n_bar * P )
    err =  1./np.sqrt(ds.npairs * w_FKP) * averaged_n
    
    
    return theta, averaged_n, err



def BCCDESfootprintMasking():

    import os
    import esutil
    import healpy as hp
    # for loop
    # load small portions of bcc map
    # remain bcc map pixels in sysMap
    # save map


    path = '/n/des/lee.5922/data/balrog_cat/'
    goodmask = path+'y1a1_gold_1.0.2_wide_footprint_4096.fit'
    badmask = path+'y1a1_gold_1.0.2_wide_badmask_4096.fit'
    # Note that the masks here in in equatorial, ring format.
    gdmask = hp.read_map(goodmask)
    bdmask = hp.read_map(badmask)
    
    ind_good_ring = np.where(( gdmask >= 1) & ((bdmask.astype('int64') & (64+32+8)) == 0) )
    # healpixify the catalog.
    nside=4096


    catpath = '/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/obs/'
    for j in range(40):
        tables = []
        keyword = 'rotated.{:02d}'.format(j)
        
        
        for i in os.listdir(catpath):
            if os.path.isfile(os.path.join(catpath,i)) and keyword in i:
                tables.append(catpath+i)
                print i

        if len(tables)==0: pass
        else :
            bcc = esutil.io.read( tables, combine=True )
            #bcc = fitsio.read('/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/obs/Aardvark_v1.0c_des_rotated.{}.fit')
            phi = bcc['RA'] * np.pi / 180.0
            theta = ( 90.0 - bcc['DEC'] ) * np.pi/180.0

            hpInd = hp.ang2pix(nside,theta,phi,nest=False)
            keep = np.in1d(hpInd,ind_good_ring)
            bcc = bcc[keep]
            filename = 'Aardvark_v1.0c_des_rotated.{:02d}.fit'.format(j)
            fitsio.write('/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/mask/'+filename, bcc)
            print "save fits file to ",'/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/mask/'+filename
    return 0




def CutDESStripe82(keyword = 'Y1A1'):
    
    import os
    import esutil
    #catpath = '/n/des/lee.5922/data/bcc_cat/aardvark_v1.0/mask/'
    catpath = '/n/des/lee.5922/data/y1a1_coadd/'
    tables = []
    
    for j in range(10, 40):
        keyword = 'Y1A1_COADD_OBJECTS_{:05d}'.format(j)
        cat = io.getDESY1A1catalogs(keyword = keyword, sdssmask=False)
        cat_SPT = MatchHPArea(cat, balrog_SPT)
        if len(cat_SPT) == 0: pass
        else :
            tables.append(cat_SPT)
            #fitsio.write(catpath+'SSPT_COADD_OBJECTS_{:05d}.fits'.format(j), cat_SPT)
            print 'SSPT_COADD_OBJECTS_{:05d}.fits'.format(j)

    tables = np.hstack(tables)
    tables = Cuts.doBasicCuts(tables, object='star')
    fitsio.write('result_cat/SSPT_star.fits')
        
        
        
        
    #except IOError : pass

    #tables = np.hstack(tables)
    #fitsio.write(catpath+'SSPT_COADD_OBJECTS.fits', tables)
    return 0




def main():
    
    # large sample
    sdss_data_o = io.getSDSScatalogs(bigSample = True)
    full_des_data = io.getDEScatalogs(bigSample = True)
    des_data_im3 = io.getDEScatalogs(file = '/n/des/huff.791/Projects/CMASS/Data/im3shape_s82_for_xcorr.fits', bigSample = True)
    balrog_data_o = LoadBalrog()
    balrog_truth_o = LoadBalrog( truth = True )
    cmass_catalog_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South.fits.gz')
    cmass_catalog_photo_o = io.getSDSScatalogs(file = '/n/des/lee.5922/data/galaxy_DR11v1_CMASS_South-photoObj_z.fits.gz')
    cmass_rand_o = fitsio.read('/n/des/lee.5922/data/random0_DR11v1_CMASS_South.fits.gz' )

    
    dec = -1.5
    dec2 = 1.5
    ra = 300.0
    ra2 = 360.0
    

    # sdss
    sdss_data = Cuts.SpatialCuts(sdss_data_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    sdss_data = Cuts.doBasicSDSSCuts(sdss_data)
    
    # des
    des_data = Cuts.SpatialCuts(full_des_data,ra = ra, ra2=ra2 , dec= dec, dec2= dec2 )
    des_data = Cuts.doBasicCuts(des_data, object = None)
    des_star = des_data[des_data['MODETYPE'] == 3]
    des_gal = des_data[des_data['MODETYPE'] == 1]
    
    #im3shape
    i3des_data = Cuts.SpatialCuts(des_data_im3,ra = ra, ra2=ra2 , dec= dec, dec2= dec2 )
    i3des_data = i3des_data[i3des_data['INFO_FLAG'] == 0]
    i3des_data = Cuts.keepGoodRegion(i3des_data)
    #des_data = im3shape.im3shape_galprof_mask(i3des_data, des_data)
    
    # balrog
    #balrog_data_o = LoadBalrog()
    balrog_data = Cuts.SpatialCuts(balrog_data_o,ra = ra, ra2=ra2 , dec= dec, dec2= dec2 )
    balrog_data = Cuts.doBasicCuts(balrog_data, balrog=True,object = None)
    balrog_data = balrog_data[balrog_data['MAG_AUTO_I'] < 21]
    
    #balrog_truth = Cuts.SpatialCuts(balrog_truth_o ,ra = ra, ra2=ra2 , dec= dec, dec2= dec2 )
    ra_b = balrog_data['ALPHAWIN_J2000_DET']
    dec_b = balrog_data['DELTAWIN_J2000_DET']
    balrog_data = rf.append_fields(balrog_data, 'RA', ra_b)
    balrog_data = rf.append_fields(balrog_data, 'DEC', dec_b)

    balrog_star = balrog_data[balrog_data['MODETYPE'] == 3]
    balrog_gal = balrog_data[balrog_data['MODETYPE'] == 1]
    #balrogSimMatched, balrogTruthMatched = DES_to_SDSS.matchCatalogsWithTag(balrog_data, balrog_truth , tag = 'ID')
    
    #cmass
    cmass_catalog = Cuts.SpatialCuts(cmass_catalog_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    cmass_catalog = Cuts.keepGoodRegion(cmass_catalog)
    cmass_photo = Cuts.SpatialCuts(cmass_catalog_photo_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    cmass_photo = Cuts.keepGoodRegion(cmass_photo)
    cmass_rand = Cuts.SpatialCuts(cmass_rand_o,ra =ra, ra2=ra2 , dec= dec, dec2= dec2 )
    cmass_rand = Cuts.keepGoodRegion(cmass_rand)

    uniform_rand = make2SphereRandoms(ra_min = ra, ra_max = ra2, dec_min = dec, dec_max = dec2, size = cmass_rand.size)
    uniform_rand = Cuts.keepGoodRegion(uniform_rand)
    
    # -----------------------------------------------------------------------------------
    # cut cmass
    #des_cmass_mask = cmass_criteria(des_data, sdss = None)
    #balrog_dmass_mask = cmass_criteria(balrog_data)
    #balrog_cmass = balrog_data[balrog_dmass_mask]
    
    des_matched, i3des_matched = DES_to_SDSS.match(des_gal, i3des_data)
    des_matched = rf.append_fields(des_matched, 'DESDM_ZP', i3des_matched['DESDM_ZP'])
    des_cmass = des_matched[cmass_criteria(des_matched, sdss = None)]

    des_cmass_mask = cmass_criteria(des_gal)
    des_cmass = des_gal[des_cmass_mask]
    Completeness_Purity(cmass_catalog, des_cmass)
    
    cmass_sdss = sdss_data[SDSS_cmass_criteria(sdss_data)]
    des_cmass = CMASSclassifier(sdss_data, des_gal, train_sample = des_gal )
    balrog_cmass = CMASSclassifier(sdss_data, balrog_gal, train_sample = des_gal )
    
    
    
    
    
    #def mapping(cmass_catalog, balrog_cmass, ra=300, ra2=360, dec=-1.5, dec2=1.5):
    # mapping(cmass_catalog, balrog_cmass, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    # Healpixify -get clean region ------------------------------------------------
    
    properties = ['FWHM', 'AIRMASS', 'SKYSIGMA', 'SKYBRITE','NSTARS']
    filters = ['g','r','i', 'z']
    
    filter = 'g'
    i = 0
    
    nside = 1024
    njack = 30
    
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    ax = ax.ravel()
    
    #for i, property in enumerate(properties):
    property = 'NSTARS'
    #filter = 'g'
    for i, filter in enumerate(filters):
    
        if property == 'NSTARS':
            nside = 512
            filename = 'y1a1_gold_1.0.2_stars_nside1024.fits'
            sysMap = loadSystematicMaps( filename = filename, nside = nside )
        else : sysMap = loadSystematicMaps( property = property, filter = filter, nside = nside)
        
        sysMap = Cuts.SpatialCuts(sysMap, ra = ra, ra2=ra2 , dec=dec , dec2= dec2 )
        
        bins, Cdensity, Cerr = GalaxyDensity_Systematics(cmass_catalog, sysMap, nside = nside, raTag = 'RA', decTag='DEC', property = property)
        bins, Bdensity, Berr = GalaxyDensity_Systematics(balrog_cmass, sysMap, nside = nside, raTag = 'RA', decTag='DEC', property = property)
        
        #bins = bins/np.sum(sysMap['SIGNAL']) *len(sysMap)
        #C_jkerr = jksampling(cmass_catalog, sysMap, nside = nside, njack = 10, raTag = 'RA', decTag = 'DEC' )
        #B_jkerr = jksampling(balrog_cmass, sysMap, nside = nside, njack = 10, raTag = 'RA', decTag = 'DEC' )
        
        zeromask = ( Cdensity != 0.0 )
        Cdensity = Cdensity[zeromask] # = float('nan')
        #C_jkerr = C_jkerr[zeromask]
        Cbins = bins[zeromask]
        Cerr = Cerr[zeromask]
        zeromask = (Bdensity != 0.0 )
        Bdensity = Bdensity[zeromask] # = float('nan')
        #B_jkerr = B_jkerr[zeromask]
        Bbins = bins[zeromask]
        Berr = Berr[zeromask]
        
        #fitting

        Cchi, Cchidof = chisquare_dof( Cbins, Cdensity, Cerr)
        Bchi, Bchidof = chisquare_dof( Bbins, Bdensity, Berr)

        ax[i].errorbar(Cbins-(bins[1]-bins[0])*0.1, Cdensity, yerr = Cerr, color = 'blue', fmt = '.', label='CMASS, chi2/dof={:>2.2f}'.format(Cchidof))
        ax[i].errorbar(Bbins+(bins[1]-bins[0])*0.1, Bdensity, yerr = Berr, color = 'red', fmt= '.',  label='BMASS, chi2/dof={:>2.2f}'.format(Bchidof))
        ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
        ax[i].set_ylabel('n_gal/n_tot '+str(nside))
        ax[i].set_ylim(0.0, 2)
        #ax[i].set_xlim(8.2, 8.55)
        ax[i].axhline(1.0,linestyle='--',color='grey')
        ax[i].legend(loc = 'best')
        
        if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
        #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
        #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
        if property == 'NSTARS': ax[i].set_xlim(0.0, 2.5)


    fig.suptitle('systematic test')
    fig.savefig('../figure/systematic_'+property+'coadd_wFKP')
    #fig.clf()





    # stellar density map as a function of fibermag----------------
    
    filter = 'i'
    nside = 512
    njack = 30
    raTag = 'RA'
    decTag = 'DEC'
    
    # DES stellar density
    path = '/n/des/lee.5922/data/systematic_maps/'
    filename = 'y1a1_gold_1.0.2_stars_nside1024.fits'
    sysMap = loadSystematicMaps( filename = filename, nside = nside )
    sysMap = Cuts.SpatialCuts(sysMap,ra =300, ra2=ra2 , dec=dec , dec2= dec2 )

    # SDSS stellar density
    path = '/n/des/lee.5922/data/systematic_maps/'
    filename = 'allstars17.519.9Healpixall256.dat'
    sysMap = loadSystematicMaps( filename = filename, nside = nside )
    sysMap = Cuts.SpatialCuts(sysMap,ra =300, ra2=ra2 , dec=dec , dec2= dec2 )
    
    
    #bins, Bdensity, Berr = GalaxyDensity_Systematics(balrog_cmass, sysMap, nside = nside, raTag = 'RA', decTag='DEC')
    #C_jkerr = jksampling(cmass_catalog, sysMap, nside = nside, njack = njack, raTag = raTag, decTag = decTag )


    catalog = cmass_photo.copy()
    Tag = 'FIBER2MAG'
    bin_center, binned_cat, binkeep = divide_bins( catalog, Tag = Tag, min = 20., max = 21.5, bin_num = 5, TagIndex = 2 )
    

    fig, ax = plt.subplots(1,1,figsize=(10,10))
    labels = np.linspace(20.0, 21.5, 6)
    labels = ['{} < i_fib2 < {}'.format(labels[i], labels[i+1]) for i in range(len(labels)-1) ]
    for i, cat in enumerate(binned_cat):
        bins, Cdensity, Cerr = GalaxyDensity_Systematics(cat, sysMap, nside = nside, raTag = raTag, decTag= decTag)
        #C_jkerr = jksampling(cmass_catalog, sysMap, nside = nside, njack = njack, raTag = raTag, decTag = decTag )
        zeromask = ( Cdensity != 0.0 )
        Cdensity = Cdensity[zeromask] # = float('nan')
        #C_jkerr = C_jkerr[zeromask]
        Cerr = Cerr[zeromask]
        Cbins = bins[zeromask]
        ax.errorbar(Cbins-(bins[1]-bins[0])*0.05 * i, Cdensity, yerr = Cerr, fmt = 'o', label = labels[i])
        print Cdensity
    
    
    "test"
    for i, cat in enumerate(binned_cat):
        bins, Cdensity, Cerr = GalaxyDensity_Systematics(cat, sysMap, nside = nside, raTag = raTag, decTag= decTag)
        print Cdensity



    bins, Cdensity, Cerr = GalaxyDensity_Systematics(cmass_photo, sysMap, nside = nside, raTag = raTag, decTag= decTag)
    C_jkerr = jksampling(cmass_photo, sysMap, nside = nside, njack = njack, raTag = raTag, decTag = decTag )
    ax.errorbar(bins, Cdensity, yerr = C_jkerr, fmt = '-o', label = 'all')
    ax.set_xlabel('stellar density')
    ax.set_ylabel('n_gal/n_tot '+str(nside))
    ax.set_ylim(0.3, 1.7)
    ax.set_xlim(0, 2.0)
    ax.axhline(1.0,linestyle='--',color='grey')
    ax.legend(loc = 'best')
    ax.set_title('sdss stellar density map')
    #fig.savefig('../figure/stellar_density_fib2_sdss')




    # variation of stellar density across the sky --------------------------

    path = '/n/des/lee.5922/data/systematic_maps/'
    filename = 'y1a1_gold_1.0.2_stars_nside1024.fits'
    n_star_map = loadSystematicMaps( filename = filename, nside = 1024 )
    n_star_map = Cuts.SpatialCuts(n_star_map,ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )

    fig, ax = plt.subplots(1,1,figsize = (20, 5))
    #ax.hexbin(n_star_map['RA'], n_star_map['DEC'], C=n_star_map['SIGNAL'], gridsize = 50, cmap=cm.jet, bins=None)
    idx = n_star_map['SIGNAL'].argsort()
    x = n_star_map['RA'][idx]
    y = n_star_map['DEC'][idx]
    z = n_star_map['SIGNAL'][idx]
    ax.scatter(x[:-1],y[:-1], color=z[:-1], s = 30, edgecolor='')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    fig.savefig('../figure/footprint')
    print 'figsave : ../figure/spatialtest.png'


    # angular correlation of CMASS in st82 and whole CMASS

    dec = -1.5
    dec2 = 1.5
    ra = 315.
    ra2 = 360.

    cmass_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_South.fits.gz')
    cmass_cat_NGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_North.fits.gz')
    #cmass_catalog = np.hstack(( cmass_catalog1, cmass_catalog2 ))	
    rand_cat_SGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz' )
    rand_cat_NGC = fitsio.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_North.fits.gz' )
    #cmass_rand = np.hstack(( cmass_rand1, cmass_rand2 ))
    cmass_catS = Cuts.SpatialCuts(cmass_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )
    rand_catS = Cuts.SpatialCuts(rand_cat_SGC, ra =ra, ra2=ra2 , dec=dec , dec2= dec2 )

    weight_rand_NGC = rand_cat_NGC['WEIGHT_FKP']
    weight_data_NGC = cmass_cat_NGC['WEIGHT_FKP'] * cmass_cat_NGC['WEIGHT_STAR'] * ( cmass_cat_NGC['WEIGHT_CP']+cmass_cat_NGC['WEIGHT_NOZ'] -1. )
    weight_rand_SGC = rand_cat_SGC['WEIGHT_FKP']
    weight_data_SGC = cmass_cat_SGC['WEIGHT_FKP'] * cmass_cat_SGC['WEIGHT_STAR'] * ( cmass_cat_SGC['WEIGHT_CP']+cmass_cat_SGC['WEIGHT_NOZ'] -1. )
    weight_randS = rand_catS['WEIGHT_FKP']
    weight_dataS = cmass_catS['WEIGHT_FKP'] * cmass_catS['WEIGHT_STAR'] * ( cmass_catS['WEIGHT_CP']+cmass_catS['WEIGHT_NOZ'] -1. )
    theta_NGC, w_NGC, _ = angular_correlation(cmass_cat_NGC, rand_cat_NGC, weight = [weight_data_NGC, weight_rand_NGC])
    theta_SGC, w_SGC, _ = angular_correlation(cmass_cat_SGC, rand_cat_SGC, weight = [weight_data_SGC, weight_rand_SGC])
    thetaS, wS, werrS = angular_correlation(cmass_catS, rand_catS, weight = [weight_dataS, weight_randS])
      
    # jk errors

    #from astroML.plotting import setup_text_plots
    #setup_text_plots(fontsize=20, usetex=True)

    njack = 20
    raTag = 'RA'
    decTag = 'DEC'
    _, jkerr_NGC = jk_error( cmass_cat_NGC, njack = njack , target = angular_correlation, jkargs=[cmass_cat_NGC, rand_cat_NGC], jkargsby=[[raTag, decTag],[raTag, decTag]], raTag = raTag, decTag = decTag )
    _, jkerr_SGC = jk_error( cmass_cat_SGC, njack = njack , target = angular_correlation, jkargs=[cmass_cat_SGC, rand_cat_SGC], jkargsby=[[raTag, decTag],[raTag, decTag]], raTag = raTag, decTag = decTag )
    _, Sjkerr = jk_error( cmass_catS, njack = njack , target = angular_correlation, jkargs=[cmass_catS, rand_catS], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )
  

    DAT = np.column_stack(( theta_NGC, w_NGC, jkerr_NGC, theta_SGC, w_SGC, jkerr_SGC, thetaS, wS, Sjkerr  ))
    np.savetxt( 'cmass_acf.txt', DAT, delimiter = ' ', header = ' theta_NGC  w_NGC  jkerr_NGC  theta_SGC  w_SGC   jkerr_SGC   thetaS  wS  Sjkerr ' )
 
    fig, ax = plt.subplots(1,1, figsize = (7, 7))

    ax.errorbar( theta_NGC, w_NGC, yerr = jkerr_NGC, fmt = '.', label = 'NGC')
    ax.errorbar( theta_SGC*0.95, w_SGC, yerr = jkerr_SGC, fmt = '.', label = 'SGC')
    ax.errorbar( thetaS* 1.05, wS, yerr = Sjkerr, fmt = '.', label = 'Stripe82')
    #ax2.errorbar( theta, w - wS, yerr = Sjkerr, fmt = '.')

    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')
    




    # angular correlation with balrog random --------------------------

    _, binned_cmass_cat, binkeep = divide_bins( cmass_catalog, Tag = 'RA', min = 317., max = 360., bin_num = 3 )
    _,binned_rand_cat, binkeep = divide_bins( cmass_rand, Tag = 'RA', min = 317., max = 360., bin_num = 3 )
    _,binned_balrog_cat, binkeep = divide_bins( balrog_cmass, Tag = 'RA', min = 317., max = 360., bin_num = 3 )

    njack = 30
    raTag = 'RA'
    decTag = 'DEC'

    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize=(20,20))

    labels = np.linspace(317., 360., 4)
    labels = ['{:>1.0f} < dec < {:>1.0f}'.format(labels[i], labels[i+1]) for i in range(len(labels)-1)]

    wlist = []
    wBlist = []
    for (i, cmass_cat), rand_cat, bmass_cat in zip( enumerate(binned_cmass_cat), binned_rand_cat, binned_balrog_cat):
        weight_rand = rand_cat['WEIGHT_FKP']
        weight_data = cmass_cat['WEIGHT_FKP'] * cmass_cat['WEIGHT_STAR'] * ( cmass_cat['WEIGHT_CP']+cmass_cat['WEIGHT_NOZ'] -1)
        theta, w, werr = angular_correlation(cmass_cat, rand_cat, weight = [weight_data, weight_rand])
        thetaB, wB, werrB = angular_correlation(cmass_cat, bmass_cat, weight = None)
        wlist.append(w)
        wBlist.append(wB)
        # jk error
        _, Cjkerr = jk_error( cmass_cat, njack = njack , target = angular_correlation, jkargs=[cmass_cat, rand_cat], jkargsby=[[raTag, decTag],[raTag, decTag]], raTag = raTag, decTag = decTag )
        _, Bjkerr = jk_error( cmass_cat, njack = njack , target = angular_correlation, jkargs=[cmass_cat, bmass_cat], jkargsby=[[raTag, decTag],[raTag, decTag]],raTag = raTag, decTag = decTag )
        ax.errorbar( theta*(1+0.1 * i), w, yerr = Bjkerr, fmt = '.', label = labels[i])
        ax2.errorbar( thetaB* (1+0.1 * i), wB, yerr = Bjkerr, fmt = '.', label = labels[i])


    ax.set_xlim(1e-2, 10)
    ax.set_ylim(1e-4, 10)
    ax.set_xlabel('theta[deg]')
    ax.set_ylabel(' w(theta) cmass random')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax2.set_xlim(1e-2, 10)
    ax2.set_ylim(1e-4, 10)
    ax2.set_xlabel('theta[deg]')
    ax2.set_ylabel(' w(theta) balrog random')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.legend(loc = 'best')


    #fig2, (ax3, ax4) = plt.subplots(1,2,figsize=(20,10))
    w1 = wlist[0]
    w2 = wlist[1]
    w3 = wlist[2]
    ax3.errorbar( theta, w1-w3, yerr = Cjkerr, fmt = '.', label = 'w1-w3')
    ax3.errorbar( theta*(1+0.1), w2-w3, yerr = Cjkerr, fmt = '.', label = 'w2-w3')
    ax3.set_xlim(1e-2, 10)
    ax3.set_ylim(-0.5, 0.5)
    ax3.set_xlabel('theta[deg]')
    ax3.set_ylabel(' delta w(theta)  cmass random')
    ax3.set_xscale('log')
    ax3.axhline(0,linestyle='--',color='grey')
    ax3.legend(loc = 'best')
    #ax3.set_yscale('log')
    
    wB1 = wBlist[0]
    wB2 = wBlist[1]
    wB3 = wBlist[2]
    ax4.errorbar( thetaB, wB1-wB3, yerr = Bjkerr, fmt = '.', label = 'wB1-wB3')
    ax4.errorbar( thetaB*(1+0.1), wB2-wB3, yerr = Bjkerr, fmt = '.', label = 'wB2-wB3')
    ax4.set_xlim(1e-2, 10)
    ax4.set_ylim(-0.5, 0.5)
    ax4.set_xlabel('theta[deg]')
    ax4.set_ylabel(' delta w(theta)  cmass random')
    ax4.set_xscale('log')
    ax4.axhline(0,linestyle='--',color='grey')
    ax4.legend(loc = 'best')
    
    fig.savefig('../figure/angular_corr_cmass_balrog')
    fig.savefig('../figure/angular_corr_cmass_balrog_diff')






    fig, ax = plt.subplots(1,1,figsize=(10,10))
    #ax.errorbar( theta, theta * w, yerr = werr * theta, color = 'blue', fmt = '.', label = 'Cmass random')
    ax.errorbar( theta * 0.97, theta * w, yerr = theta*Cjkerr, color = 'red', fmt = '.', label = 'Cmass random')
    ax.errorbar( thetaB * 1.03, thetaB * wB, yerr = thetaB*Bjkerr, color = 'blue', fmt = '.', label = 'Balrog random')
    ax.set_xlim(1e-3, 10)
    #ax.set_ylim(1e-4, 1)
    ax.set_xlabel('theta[deg]')
    ax.set_ylabel('theta * w(theta)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    fig.savefig('../figure/angular_corr')




    # star - galaxy cross correlation ----------------------------------------
    
    bin_center, binned_star, binned_keep = divide_bins( des_star, Tag = 'MAG_PSF_I', min = 17.0, max = 21.0, bin_num = 4 )
    labels = np.linspace(17.0, 21.0, 5)
    labels = ['{} < i < {}'.format(labels[i], labels[i+1]) for i in range(len(labels)-1) ]
    
    #CMASSSystematicMask = SystematicMask( cmass_catalog, nside = 1024, raTag = 'RA', decTag = 'DEC' )
    #clean_cmass_catalog = cmass_catalog[CMASSSystematicMask]
    
    #des_cmass = resampled.copy()
    
    #fig, (ax, ax2, ax3) = plt.subplots(1,3,figsize = (21,7))
    fig, (ax, ax2) = plt.subplots(2,1,figsize = (7,14))
    #rand2 = cmass_rand.copy()
    #rand2 = balrog_cmass.copy()
    rand2 = balrog_star.copy()
    #rand2 = uniform_rand.copy()

    for i, star in enumerate(binned_star):
        data2 = star.copy()
        data1 = resampled_test.copy()
        rand = resampled.copy()
        theta, xi, varxi = cross_angular_correlation(data1, data2, rand, rand2, weight = None)
        ax2.errorbar( theta * (1 + 0.1 * i), xi, yerr = np.sqrt(varxi), fmt = '-', label = labels[i])
        #ax2.errorbar( thetaB, xiB, yerr = np.sqrt(varxiB), label = 'Balrog cmass')

    #ax.errorbar( theta, xi, yerr = np.sqrt(varxi), fmt = '-o', label = 'Sdss cmass')
    #ax.errorbar( thetaB, xiB, yerr = np.sqrt(varxiB), fmt = '-o', label = 'balrog cmass')
    ax.axhline(0,linestyle='--',color='grey')
    ax.set_xlabel('theta [arcmin]')
    ax.set_ylabel('w(theta) stars')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 100)
    ax.set_ylim(-1., 1.)
    ax.legend(loc = 'best')
    ax.set_title('des_cmass, balrog_cmass, des_star, balrog_star')
    #ax.set_yscale('log')
    #ax.set_xlim(-2, 2)
    #fig.savefig('../figure/star_corr_cb')


    ax2.axhline(0,linestyle='--',color='grey')
    ax2.set_xlabel('theta [arcmin]')
    ax2.set_ylabel('w(theta) stars')
    ax2.set_xscale('log')
    ax2.set_xlim(0.1, 100)
    ax2.set_ylim(-1., 1.)
    ax2.legend(loc = 'best')
    ax2.set_title('des_cmass, resampled_balrog, des_star, balrog_star')






    # reweighting DES catalog ------------------------------------------------

    #training = des_cmass.copy()
    #test = balrog_cmass.copy()
    training = balrog_cmass.copy()
    test = des_cmass.copy()

    m1 = training['MAG_AUTO_I']/np.max(training['MAG_AUTO_I'])
    m2 = (training['MAG_MODEL_R'] - training['MAG_MODEL_I']) / np.max(training['MAG_MODEL_R'] - training['MAG_MODEL_I'])
    mt1 = test['MAG_AUTO_I']/np.max(test['MAG_AUTO_I'])
    mt2 = (test['MAG_MODEL_R'] - test['MAG_MODEL_I']) / np.max(test['MAG_MODEL_R'] - test['MAG_MODEL_I'])

    # same volume
    """
    Radius = 0.02
    weight = []
    for i in range(training.size):
        
        m_alpha1 = m1[i]
        m_alpha2 = m2[i]
        distance = np.sqrt( (m1-m_alpha1)**2 + (m2 - m_alpha2)**2 )
        distance_test = np.sqrt( (mt1-m_alpha1)**2 + (mt2 - m_alpha2)**2 )
        
        N = len(distance[distance < Radius])
        N_test = len(distance_test[distance_test < Radius])
        w = N_test * 1.0 / N / test.size
        weight.append(w)
    """
    # same number
    alpha = 70
    weight = []
    for i in range(training.size):
        m_alpha1 = m1[i]
        m_alpha2 = m2[i]
        distance = np.sqrt( (m1-m_alpha1)**2 + (m2 - m_alpha2)**2 )
        distance_test = np.sqrt( (mt1-m_alpha1)**2 + (mt2 - m_alpha2)**2 )
        idx = distance.argsort()
        idx2 = distance_test.argsort()
        d = distance[idx][alpha]
        d_test = distance_test[idx2][alpha]
        w = (d/d_test)**2 / test.size
        weight.append(w)
    

    norm_weight = weight/np.sum(weight)

    resampled = np.random.choice(training, size=test.size/2, p = norm_weight)
    resampled_test = np.random.choice(test, size=test.size/2)

    

    # plotting
    bins1 = np.linspace(18.2, 20.3, 50)
    bins2 = np.linspace(0.34, 2.0, 30)
    
    m1 = training['MAG_AUTO_I']
    m2 = (training['MAG_MODEL_R'] - training['MAG_MODEL_I'])
    mt1 = test['MAG_AUTO_I']
    mt2 = (test['MAG_MODEL_R'] - test['MAG_MODEL_I'])

    N_training, _ = np.histogramdd([m1,m2], bins = [bins1, bins2])
    N_test, _ = np.histogramdd([mt1,mt2], bins = [bins1, bins2])
    norm_N_training = np.log10(N_training/training.size)
    norm_N_test = np.log10(N_test/test.size)

    mr1 = resampled['MAG_AUTO_I']
    mr2 = (resampled['MAG_MODEL_R'] - resampled['MAG_MODEL_I'])
    N_resample, _ = np.histogramdd([mr1,mr2], bins = [bins1, bins2])
    norm_N_resample = np.log10(N_resample/resampled.size)


    mrb1 = resampled_test['MAG_AUTO_I']
    mrb2 = (resampled_test['MAG_MODEL_R'] - resampled_test['MAG_MODEL_I'])
    N_resample_test, _ = np.histogramdd([mrb1,mrb2], bins = [bins1, bins2])
    norm_N_resample_test = np.log10(N_resample_test/resampled_test.size)


    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2,figsize = (10,10))

    im1 = ax1.imshow(np.rot90(norm_N_training), extent=[18.2, 20.3, 0.34, 2.0], aspect = 1)
    im2 = ax2.imshow(np.rot90(norm_N_resample_test), extent=[18.2, 20.3, 0.34, 2.0], aspect = 1)
    im4 = ax4.imshow(np.rot90(norm_N_resample) ,extent=[18.2, 20.3, 0.34, 2.0], aspect = 1)

    cbar = fig.colorbar(im1, ax=ax1, fraction=0.037, pad=0.04)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.037, pad=0.04)
    cbar4 = fig.colorbar(im4, ax=ax4, fraction=0.037, pad=0.04)

    cbar.set_label('log10(N/Ntot)' )
    cbar2.set_label('log10(N/Ntot)' )
    cbar4.set_label('log10(N/Ntot)' )

    #ax.set_title('balrog_cmass')
    ax1.legend(loc='best')
    ax1.set_xlabel('mag_auto_i')
    ax1.set_ylabel('r-i [des]')
    ax2.legend(loc='best')
    ax2.set_xlabel('mag_auto_i')
    ax2.set_ylabel('r-i [balrog]')
    ax4.legend(loc='best')
    ax4.set_xlabel('mag_auto_i')
    ax4.set_ylabel('r-i [des resampled]')


    idx = np.log10(weight).argsort()
    x = m1[idx]
    y = m2[idx]
    z = np.log10(weight)[idx]
    cax3 = ax3.scatter(x,y, color=z, s = 30, edgecolor='')
    cbar3 = fig.colorbar(cax3, ax=ax3 )
    cbar3.set_label('log10(weight)')
    ax3.axis([x.min(), x.max(), y.min(), y.max()])
    ax3.set_xlabel('mag_auto_i')
    ax3.set_ylabel('r-i [des weights]')

    #fig.savefig('../figure/reweight')



    N_resample_1d = np.sum(N_resample, axis = 1)
    N_test_1d = np.sum(N_test, axis = 1)
    #N_balrog_1d = np.sum(N_des, axis=1)

    # projected histogram
    fig,(ax, ax2) = plt.subplots(1,2, figsize=(14,7))
    #ax2.plot( (bins[0:-1] + bins[1:])/2., N3/float(N3.max()), 'b-', label = 'CMASS')
    ax2.plot( (bins1[0:-1] + bins1[1:])/2., N_test_1d/float(N_test_1d.max()), 'g-', label = 'BMASS')
    ax2.plot( (bins1[0:-1] + bins1[1:])/2., N_resample_1d/float(N_resample_1d.max()), 'r-', label = 'DMASS')
    
    #ax.plot( (bins[0:-1] + bins[1:])/2., N3, 'b-',  label = 'CMASS, {}'.format(len(CMASS)))
    ax.plot( (bins1[0:-1] + bins1[1:])/2., N_test_1d, 'g-', label = 'BMASS')
    ax.plot( (bins1[0:-1] + bins1[1:])/2., N_resample_1d, 'r-', label = 'DMASS')
    
    ax.set_title('Histogram')
    ax.legend(loc='best')
    ax.set_xlabel('modelmag_i')
    ax.set_ylabel('N')
    
    ax2.set_title('Normalized Histogram')
    ax2.legend(loc='best')
    ax2.set_xlabel('modelmag_i')
    ax2.set_ylabel('N/N.max')
    ax2.set_ylim(0, 1.1)
    
    #fig.savefig('../figure/bmass_dmass')
    
    #ax.set_xlim(18.5, 20.5)




    # Lensing and Boost Factor -------------------------------------------

    lense = cmass_catalog.copy()
    source = i3des_data.copy()
    rand = cmass_rand.copy()


    r_p_bins, BoostFactor = LensingSignal( lense, source, rand )

    # jk sampling
    import os

    jkfile = './jkregion.txt'
    njack = 30
    raTag = 'RA'
    decTag = 'DEC'
    jk.GenerateJKRegions( lense[raTag], lense[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = LensingSignal, jkargs=[lense, source, rand], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag]], jkargspos=None, regions = jkfile)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    #r_p_bins, BoostFactor, LensSignal, correctedLensSignal = jkresults['full']
    r_p_bins, BoostFactor = jkresults['full']

    # getting jk err
    Boost_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    #LensSignal_i = np.array( [jkresults['jk'][i][2] for i in range(njack)] )
    #correctedLensSignal_i = np.array( [jkresults['jk'][i][3] for i in range(njack)] )


    norm = 1. * (njack-1)/njack
    Boost_cov = 0
    LensSignal_cov = 0
    correctedLensSignal_cov = 0
    for k in range(njack):
        Boost_cov +=  (Boost_i[k] - BoostFactor)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
        #LensSignal_cov +=  (LensSignal_i[k] - LensSignal)**2
        #correctedLensSignal_cov +=  (correctedLensSignal_i[k] - correctedLensSignal)**2

    Boostjkerr = np.sqrt(norm * Boost_cov)
    #LSjkerr = np.sqrt(norm * LensSignal_cov)
    #CLSjkerr = np.sqrt(norm * correctedLensSignal_cov)


    # plotting
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (14,7))
    ax.errorbar(r_p_bins, LensSignal, yerr = LSjkerr, color = 'blue', fmt='o', label = 'z = [{},{}]'.format(z_s_min, z_s_max))
    theory = np.loadtxt('smd_v_theta_cmass.dat')
    rr_the = theory[:,0]
    delta_sigma_the = theory[:,1]
    error_the = theory[:,2] * np.sqrt(5000/120)
    ax.errorbar(10**rr_the, 10**delta_sigma_the, yerr = 10**error_the, color = 'red', fmt='--o', label = 'theory')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim(10, 1e5)
    ax.set_ylim(1e-2,1e3)
    ax.set_xlabel('r (kpc/h)')
    ax.set_ylabel('Delta Sigma'+r'($M_{s}$ h/pc2)')
    ax.set_title('CMASS lensing signal (z_lense = [0.45, 0.55] )'  )
    ax.legend(loc = 'best')

    ax2.errorbar(r_p_bins, correctedLensSignal, yerr = CLSjkerr, color = 'blue', fmt='o', label = 'z = [{},{}]'.format(z_s_min, z_s_max))
    ax2.errorbar(10**rr_the, 10**delta_sigma_the, yerr = 10**error_the, color = 'red', fmt='--o', label = 'theory')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlim(10, 1e5)
    ax2.set_ylim(1e-2,1e3)
    ax2.set_xlabel('r (kpc/h)')
    ax2.set_ylabel('Delta Sigma'+r'($M_{s}$ h/pc2)')
    ax2.set_title('CMASS lensing signal (z_lense = [0.45, 0.55] )'  )
    ax2.legend(loc = 'best')


    # plotting ( difference )
    #fig, (ax, ax2) = plt.subplots(2,1,figsize = (14,14))
    fig, ax = plt.subplots(1,1,figsize = (10,10))
    ax.errorbar(r_p_bins.value * 0.001, BoostFactor, yerr = Boostjkerr, fmt = '.', label = 'B(R)')
    #ax.errorbar(r_p_bins * 0.001 * 0.95, BoostFactor, yerr = Boostjkerr, color = 'black', fmt = '.')
    #ax.plot(r_p_bins * 0.001* 0.95, BoostFactor, 'k-')
    #ax.errorbar(r_p_bins * 0.001, BoostFactor, yerr = Boostjkerr, fmt = '.', label = 'zs=[0.9,1.0]')
    #ax.errorbar(r_p_bins * 0.001, BoostFactor, yerr = Boostjkerr, fmt = '.', label = 'zs=[0.7,1.0]')
    ax.errorbar(r_p_bins.value * 0.001 * 1.1, cross_rp + 1, yerr = np.sqrt(err_varxi), fmt = '.', color = 'green', label = '1 - Xi(R)')
    
    ax.set_xscale('log')
    ax.set_ylim(0.2, 1.8)
    ax.set_xlim(0.0005, 50)
    ax.set_ylabel('Boost Factor')
    ax.set_xlabel('r_p (Mpc/h)')
    ax.axhline(1.0,linestyle='--',color='grey')
    ax.legend(loc = 'best')
    ax.set_title('Boost Factor (No weight)')
    
    #fig.savefig('../figure/boost_weight_all')

    ax2.errorbar(r_p_bins*0.001, correctedLensSignal - LensSignal, yerr = CLSjkerr, fmt = 'o', color = 'blue', label = 'corrected - noncorrected')
    ax2.set_xscale('log')
    ax2.set_ylim(-400.0,400.0)
    ax2.set_xlim(0.001, 100)
    ax2.set_ylabel('deltaSigma difference')
    ax2.set_xlabel('r_p (Mpc/h)')
    ax2.axhline(0.0,linestyle='--',color='grey')
    ax2.legend(loc = 'best')

    ax.set_title('Boost Factor correction')
    fig.savefig('../figure/boost_weight')




