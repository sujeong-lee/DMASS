import sys, os, esutil
from xd import *
from utils import *
import healpy as hp
import numpy as np
from systematics import *
from systematics_module import *


def weightmultiply( weightDic ):
    
    length = weightDic['vetoed'].size
    w0 = np.ones(length)
    names = weightDic.keys()
    print names
    n = len(names)
    for i in range(n):
        print names[i]
        w0 *= weightDic[names[i]]
        
    return w0
    


def fitting_allSP( suffix, properties = None, kind = 'SPT', inputdir = None, plot = True ):

    filters = ['g', 'r', 'i', 'z']

    fitting_SP( property = ['DEPTH', 'EXPTIME', 'AIRMASS', 'FWHM'], filter=filters, kind = kind, 
               suffix=suffix, plot=plot, function = 'linear',
                    path = inputdir )

    fitting_SP( property = ['SKYBRITE'], filter= ['g', 'r', 'z'], kind = kind, 
               suffix=suffix, plot=plot, function = 'sqrt',
                    path = inputdir )

    fitting_SP( property = ['SKYBRITE'], filter=['i'], kind = kind, 
               suffix=suffix, plot=plot, function = 'errftn',
                path = inputdir )
    
    if 'NSTARS_allband' in properties : 
    	fitting_SP( property = ['NSTARS_allband'], filter=['g'], kind = kind, 
	               suffix=suffix, plot=plot, function = 'linear',
	                    path = inputdir )
	if 'GE' in properties :
	    fitting_SP( property = ['GE'], filter=['g'], kind = kind, suffix=suffix, plot=plot, function = 'log',
	                    path = inputdir )
    


# calling catalog
#from systematics import GalaxyDensity_Systematics,loadSystematicMaps, chisquare_dof, MatchHPArea, SysMapBadRegionMask, callingEliGoldMask

def calling_sysMap( properties=None, kind='SPT', nside=4096, path = None ):
    # Calling maps
    from systematics import callingEliGoldMask,callingY1GoldMask
    GoldMask = callingEliGoldMask()
    #GoldMask = callingY1GoldMask(nside)
    
    MaskDic = {}
    for i,p in enumerate(properties):
        if p == 'NSTARS_allband':
            filename =  'y1a1_gold_1.0.2_stars_nside1024.fits'
            sysMap = loadSystematicMaps( filename = filename, property = p, filter = 'g', nside = 1024 , kind = kind, path = path)
            if kind is 'STRIPE82': sysMap = sysMap[sysMap['DEC'] > -3]
            elif kind is 'SPT': sysMap = sysMap[sysMap['DEC'] < -3]
            mapname = 'sys_'+p+'_'+'g'+'_'+kind
            MaskDic[mapname] = sysMap
            
        elif p == 'GE':
            sysMap = loadSystematicMaps( property = p, filter = 'g', nside = 512 , kind = kind, path = path)
            if kind is 'STRIPE82': sysMap = sysMap[sysMap['DEC'] > -3]
            elif kind is 'SPT': sysMap = sysMap[sysMap['DEC'] < -3]
            mapname = 'sys_'+p+'_'+'g'+'_'+kind
            MaskDic[mapname] = sysMap
            
         
        else :
            filter = ['g', 'r', 'i', 'z']
            
            if p =='EXPTIME':
                filename = ['Y1A1NEW_COADD_'+kind+'_band_'+f+'_nside4096_oversamp4_EXPTIME__total.fits.gz'\
                             for f in filter ]
            elif p =='DEPTH' :
                filename = ['Y1A1NEW_COADD_'+kind+'_band_'+f+'_nside4096_oversamp4_maglimit3__.fits.gz'\
                             for f in filter ]
            elif p == 'NSTARS' : 
                filename = ['Y1A1NEW_COADD_'+kind+'_band_'+f+'_nside4096_oversamp4_NSTARS_ACCEPTED_MEAN_coaddweights3_mean.fits.gz'\
                             for f in filter ]
            else : filename = [None for f in filter]

            for j,f in enumerate(filter):
                sysMap = loadSystematicMaps( filename = filename[j], property = p, filter = f, nside = nside , kind = kind, path = path)
                mapname = 'sys_'+p+'_'+f+'_'+kind
                keep = np.in1d(sysMap['PIXEL'], GoldMask['PIXEL'])
                MaskDic[mapname] = sysMap[keep]
                
    return MaskDic



def maskingCatalogSP(catalog=None, sysMap=None):
    
    
    exp_i_hpind = sysMap['sys_EXPTIME_i_SPT']['PIXEL'][(sysMap['sys_EXPTIME_i_SPT']['SIGNAL'] < 500)]
    fwhm_r_hpind = sysMap['sys_FWHM_r_SPT']['PIXEL'][(sysMap['sys_FWHM_r_SPT']['SIGNAL'] < 4.5)]
    #ge_hpind = sysMap['sys_GE_g_SPT']['PIXEL'][(sysMap['sys_GE_g_SPT']['SIGNAL'] < 0.09)]
    #ge_hpind = sysMap['sys_GE_g_SPT']['PIXEL'][(sysMap['sys_GE_g_SPT']['SIGNAL'] < 100000)]
    #skybrite_g_hpind = sysMap['sys_SKYBRITE_g_SPT']['PIXEL'][(sysMap['sys_SKYBRITE_g_SPT']['SIGNAL'] < 170)]
    #skybrite_i_hpind = sysMap['sys_SKYBRITE_i_SPT']['PIXEL'][(sysMap['sys_SKYBRITE_i_SPT']['SIGNAL'] < 1400)]
    skybrite_g_hpind = sysMap['sys_SKYBRITE_g_SPT']['PIXEL'][(sysMap['sys_SKYBRITE_g_SPT']['SIGNAL'] < 1700000)]
    skybrite_i_hpind = sysMap['sys_SKYBRITE_i_SPT']['PIXEL'][(sysMap['sys_SKYBRITE_i_SPT']['SIGNAL'] < 14000000)]  
    
    all_mask1 = np.zeros( hp.nside2npix(4096), dtype=bool )
    all_mask2 = np.zeros( hp.nside2npix(4096),dtype=bool )
    all_mask3 = np.zeros( hp.nside2npix(4096),dtype=bool )
    all_mask4 = np.zeros( hp.nside2npix(4096),dtype=bool )
    
    all_mask512 = np.zeros( hp.nside2npix(512),dtype=bool )
    
    all_mask1[exp_i_hpind] = 1
    all_mask2[fwhm_r_hpind] = 1
    all_mask3[skybrite_g_hpind] = 1
    all_mask4[skybrite_i_hpind] = 1
    
    #all_mask512[ge_hpind] = 1
    all_mask512 = np.ones( hp.nside2npix(512),dtype=bool )
    all_mask4096 = all_mask1 * all_mask2 * all_mask3* all_mask4


    all_ind4096 = np.arange( hp.nside2npix(4096) )
    all_ind512 = np.arange( hp.nside2npix(512) )
    goodindices4096 = all_ind4096[all_mask4096]
    goodindices512 = all_ind512[all_mask512]
  
    
    #goodindices = np.hstack([exp_i_hpind, fwhm_r_hpind, ge_hpind])
    
    #exp_mask =  (sysMap['sys_EXPTIME_i_SPT']['SIGNAL'] < 500) &  (sysMap['sys_EXPTIME_r_SPT']['SIGNAL'] < 500)
    #fwhm_mask = ((sysMap['sys_FWHM_g_SPT']['SIGNAL'] < 500) & (sysMap['sys_FWHM_r_SPT']['SIGNAL'] < 500) 
    #            & (sysMap['sys_FWHM_i_SPT']['SIGNAL'] < 500) & (sysMap['sys_FWHM_z_SPT']['SIGNAL'] < 500))

    fwhm_mask = (sysMap['sys_FWHM_r_SPT']['SIGNAL'] < 4.5) 
    skybrite_mask = (sysMap['sys_SKYBRITE_g_SPT']['SIGNAL'] < 160) & (sysMap['sys_SKYBRITE_i_SPT']['SIGNAL'] < 1400) \
    &(sysMap['sys_SKYBRITE_z_SPT']['SIGNAL'] < 3000) 
    
    #ge_mask = (sysMap['sys_GE_g_SPT']['SIGNAL'] < 0.08)  
    #all_mask = fwhm_mask*exp_mask*skybrite_mask
    #print 'exp mask ', 1. - np.sum(exp_mask) *1./exp_mask.size
    #print 'fwhm mask', 1. - np.sum(fwhm_mask) *1./fwhm_mask.size
    #print 'skybrite mask', 1. - np.sum(skybrite_mask) *1./skybrite_mask.size
    #print 'all mask', 1. - np.sum(fwhm_mask*exp_mask*skybrite_mask) *1./fwhm_mask.size   

    catHpInd4096 = hpRaDecToHEALPixel(catalog['RA'], catalog['DEC'], nside=4096, nest=False)
    catHpInd512 = hpRaDecToHEALPixel(catalog['RA'], catalog['DEC'], nside=512, nest=False)
    HpIdxInsys_mask4096 = np.in1d(catHpInd4096, goodindices4096)
    HpIdxInsys_mask512 = np.in1d(catHpInd512, goodindices512)
    
    HpIdxInsys_mask = HpIdxInsys_mask4096 * HpIdxInsys_mask512
    
    print HpIdxInsys_mask.size, np.sum(HpIdxInsys_mask)
    print 'mask ', np.sum(HpIdxInsys_mask) * 1./catalog.size
    return catalog[HpIdxInsys_mask]


def sys_iteration( nextweight=None, suffix=None, all_weight = None, 
                  cat1=None, cat2=None, rand1 = None, rand2=None,
                  sysMap = None, nside=4096, kind='SPT', function=None, function2=None,
                  properties = None, filters = ['g', 'r', 'i', 'z'], 
                  path=None, plot=True, weightDic=None, FullArea=None ):
    
    #nextprop, nextfil= init(nextweight)   

    nextw = nextweight.split('_')    
    if len(nextw) == 2 : nextprop = nextw[0]
    else : nextprop = nextw[0]+'_'+nextw[1]
    nextfil = nextw[-1]
    #nextprop, nextfil = nextweight.split('_')
    print '----------------------------------'
    print 'initialize function ', nextweight
    print function


    wg = calculate_weight( property = nextprop, filter=nextfil, kind = kind, suffix=suffix, plot=plot, 
                              function = function,
                    path =path, catalog = cat1, sysMap= sysMap, 
                    weight=True, raTag ='RA', decTag='DEC', nside=nside)

    print 'store weight ', nextweight
    os.system('mkdir '+path+'/weights/')
    fitsname = path+'/weights/wg_'+nextweight.lower()+'_'+kind+'.fits'
    print 'save weight to fits', fitsname
    fitsio.write( fitsname, wg, clobber=True )
    weightDic[nextweight] = wg
    
    
    if suffix == 'vetoed': suffix = 'wg_'+nextweight.lower()
    else : suffix = suffix+'_'+nextweight.lower()
    print 'suffix = ', suffix
    all_weight = np.multiply( all_weight, weightDic[nextweight] )

    sys_ngal(cat1 = cat1, cat2=cat2, rand1 = rand1, rand2 = rand2, sysmap = sysMap, 
             FullArea = FullArea, properties = properties, kind=kind, nbins = 15,
             pixelmask = None, reweight = all_weight, 
             suffix=suffix, outdir=path)



def generating_randoms():

	rand = uniform_random_on_sphere(dmass, size = 10 * dmass.size)
	rand = Cuts.keepGoodRegion(rand)
	rand = rand[ rand['DEC'] < -30.0 ]

	rand2 = uniform_random_on_sphere(dmass, size = 50 * dmass.size)
	rand2 = rand2[ rand2['DEC'] < -30.0 ]

	rand_cmass = uniform_random_on_sphere(cmass, size = 200 * cmass.size)
	rand_cmass = Cuts.keepGoodRegion(rand_cmass)
	rand_cmass = rand_cmass[ rand_cmass['DEC'] > -30.0 ]

	rand2_cmass = uniform_random_on_sphere(cmass, size = 500 * cmass.size)
	rand2_cmass = Cuts.keepGoodRegion(rand2_cmass)
	rand2_cmass = rand2_cmass[ rand2_cmass['DEC'] > -30.0 ]

	print rand.size, dmass.size
	print rand_cmass.size, cmass.size

	return rand, rand2, rand_cmass, rand2_cmass


def run_survey_systematics( ):

	# import DMASS and generate randoms
	rootdir = '../output/sfd_train_full2/'
	inputdir = '../output/sfd_train_full2/'
	figoutdir = rootdir+'/figure/'
	sysoutdir = rootdir+'/sys/'
	weightdir = rootdir + '/weight/'
	sig = os.system('mkdir '+rootdir)
	#if sig == 256 : 
	#	print 'Cannot make root directory. Directory already exists or no path'
	#	break
	#else : pass
	os.system('mkdir '+figoutdir)
	os.system('mkdir '+sysoutdir)
	os.system('mkdir '+weightdir)
	#os.system('cp ../data_txt/systematics/4th/systematic*_no_weight.txt '+inputdir+'/.')



	dmass = fitsio.read(rootdir+'dmass_spt_0001.fits')
	#dmass_st82 = fitsio.read(rootdir+'dmass_st82_0001.fits')
	#cmass = fitsio.read(rootdir+'train_sample_des.fits')
	print 'dmass sample size :', dmass.size
	#print 'dmass st82 sample size :', dmass_st82.size
	#print 'cmass sample size :', cmass.size
	rand, rand2, rand_cmass, rand2_cmass = generating_randoms()



	# calling map 
	GoldMask = callingEliGoldMask()
	GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0 ]
	GoldMask_spt = GoldMask[ GoldMask['DEC'] < -3.0 ]

	pixarea = hp.nside2pixarea( 4096, degrees = True)
	sptnpix = GoldMask_spt['PIXEL'].size #hp.get_map_size( GoldMask_spt['PIXEL'] )
	st82npix =  GoldMask_st82['PIXEL'].size # hp.get_map_size( GoldMask_st82 )
	SPTMaparea = pixarea * sptnpix
	ST82Maparea = pixarea * st82npix



	# default setting and calling sysmaps
	kind = 'SPT'
	FullArea = SPTMaparea
	properties = ['DEPTH', 'EXPTIME', 'AIRMASS', 'SKYBRITE', 'FWHM', 'NSTARS_allband', 'GE']
	suffix='no_weight'

	sysMap = calling_sysMap( properties=properties, kind=kind, nside=4096 )



	# masking extreme sys
	dmass_masked = maskingCatalogSP(catalog=dmass, sysMap=sysMap)
	rand_masked = maskingCatalogSP(catalog=rand, sysMap=sysMap)
	rand2_masked = maskingCatalogSP(catalog=rand2, sysMap=sysMap)



	# no weight calculation 
	sys_ngal(cat1 = dmass, cat2=rand, rand1 = rand2, rand2 = rand2, sysmap = sysMap, 
	         FullArea = SPTMaparea, properties = properties, kind=kind, nbins =15, 
	         reweight= None, nside = 4096,
	         suffix='no_weight', outdir=sysoutdir)

	# vetoed 
	sys_ngal(cat1 = dmass_masked, cat2=rand_masked, rand1 = rand2_masked, rand2 = rand2_masked, sysmap = sysMap, 
	         FullArea = SPTMaparea, properties = properties, kind=kind, nbins =15, 
	         reweight= None, nside = 4096,
	         suffix='vetoed', outdir=sysoutdir)

	fitting_allSP( 'vetoed', inputdir = sysoutdir )
	nextwname, nextw = plotting_significance( property = properties, filter=['g', 'r', 'i','z'], kind = kind, suffix='vetoed', path = sysoutdir, deltachi2=True)
	nextweight = nextwname[0]


	# correction start


	weightDic = {}
	weightDic['vetoed'] = np.ones(dmass_masked.size)
	all_weight = weightmultiply(weightDic)
	suffix = 'vetoed'
	#nextweight = 'FWHM_r'

	applied_weight = []
	for ii in range(21):
	    
	    #nextweight = 'AIRMASS_g'
	    print '-------------------------'
	    print ' iteration -', ii
	    print ' nextweight', nextweight
	    print '-------------------------'
	    #if ii < 4 : pp = []
	    #else : pp = properties
	    all_weight = weightmultiply(weightDic)
	    function = 'linear'
	    if nextweight == 'SKYBRITE_i': function = 'errftn'
	    sys_iteration( nextweight=nextweight, suffix=suffix, all_weight = all_weight, 
	                      cat1=dmass_masked, cat2=rand_masked, rand1 = rand2_masked, rand2=rand2_masked,
	                      sysMap = sysMap, nside=4096, kind='SPT', function=function, function2 = None,
	                      properties = properties, filters=['g', 'r', 'i', 'z'],
	                      path=sysoutdir, plot=True, weightDic=weightDic, FullArea=SPTMaparea )

	    if suffix == 'vetoed': suffix = 'wg_'+nextweight.lower()
	    else : suffix = suffix+'_'+nextweight.lower()
	        
	    fitting_allSP( suffix, inputdir = sysoutdir )
	    nextwname, nextw = plotting_significance( property = properties, filter=['g', 'r', 'i','z'], kind = kind, 
	                          suffix=suffix, 
	                    path = sysoutdir, deltachi2=True)
	    
	    nextweight = nextwname[0]
	    for i in range(1,len(nextwname)):
	        if nextweight in applied_weight : nextweight = nextwname[i]
	        else : break
	    applied_weight.append(nextweight)



	print 'plotting galaxy number density with sys_weight, without sys_weight'
	from systematics_module.contCorrection import plot_sysweight
	for p in properties:
	    plot_sysweight(property = p, nside = 4096, kind = kind, 
	               suffix1='vetoed', suffix2=suffix, inputdir1 = sysoutdir, inputdir2 = sysoutdir, outdir=figoutdir)


	print 'calculate correlation function and galaxy bias'


	#tree corr

