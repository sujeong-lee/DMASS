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

    print 'all linear function'
    fitting_SP( property = ['DEPTH', 'EXPTIME', 'AIRMASS', 'FWHM', 'SKYBRITE'], filter=filters, kind = kind, 
               suffix=suffix, plot=plot, function = 'linear',
                    path = inputdir )

    #fitting_SP( property = ['SKYBRITE'], filter= ['g', 'r', 'i', 'z'], kind = kind, 
    #           suffix=suffix, plot=plot, function = 'sqrt',
    #                path = inputdir )

    #fitting_SP( property = ['SKYBRITE'], filter=['i'], kind = kind, 
    #           suffix=suffix, plot=plot, function = 'errftn',
    #            path = inputdir )
    
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




def sys_iteration( nextweight=None, suffix=None, all_weight = None, 
                  cat1=None, cat2=None, rand1 = None, rand2=None,
                  sysMap = None, nside=4096, kind='SPT', nbins=10, function=None, function2=None,
                  properties = None, filters = ['g', 'r', 'i', 'z'], 
                  path=None, plot=True, weightDic=None, weight_rand=None, FullArea=None ):
    
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

    #extremesys_mask_cat1 = maskingCatalogSP(catalog=cat1, sysMap=sysMap, maskonly=True) 
    #extremesys_mask_cat2 = maskingCatalogSP(catalog=cat2, sysMap=sysMap, maskonly=True) 
    #extremesys_mask_rand1 = maskingCatalogSP(catalog=rand1, sysMap=sysMap, maskonly=True) 
    #extremesys_mask_rand2 = maskingCatalogSP(catalog=rand2, sysMap=sysMap, maskonly=True) 

    sys_ngal(cat1 = cat1, #[extremesys_mask_cat1], 
             cat2=cat2, #[extremesys_mask_cat2], 
             rand1 = rand1, #[extremesys_mask_rand1], 
             rand2 = rand2, #[extremesys_mask_rand2], 
             sysmap = sysMap, 
             FullArea = FullArea, properties = properties, kind=kind, nbins = nbins,
             pixelmask = None, reweight_cat1 = all_weight, reweight_rand1 = weight_rand,
             reweight_cat2 = cat2['GOLD_FRAC']*cat2['VETO'], reweight_rand2 = rand2['GOLD_FRAC']*rand2['VETO'],
             suffix=suffix, outdir=path)


def plot_sysweight(property = None, nside = 1024, kind = 'SPT', suffix1='', suffix2='', inputdir1 = '.', inputdir2 = '.', outdir='./', hist=False):

    #import matplotlib.pyplot as plt
    #import numpy as np
    from systematics import SysMapBadRegionMask, loadSystematicMaps, MatchHPArea, chisquare_dof, ReciprocalWeights
    from systematics_module.corr import angular_correlation
    
    
    filters = ['g', 'r', 'i', 'z']
    #filters = ['g']
    
    
    if property is 'NSTARS_allband' :
        nside = 1024
        filters = ['g']
    if property is 'GE':
        nside = 512
        filters = ['g']
        
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    ax = ax.ravel()

    for i, filter in enumerate(filters):
        filename1 = inputdir1+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix1+'.txt'
        filename2 = inputdir2+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix2+'.txt'
        #print filename
        data1 = np.loadtxt(filename1)
        data2 = np.loadtxt(filename2)
        bins, Cdensity, Cerr, Cf_area, _, _, _ = [data1[:,j] for j in range(data1[0].size)]
        bins, Bdensity, Berr, Bf_area, _, _, _ = [data2[:,j] for j in range(data2[0].size)]
        Cf_area = Cf_area * 1./Cf_area.max()/5.

        #zeromaskC, zeromaskB = ( Cdensity != 0.0 )*( Cerr != 0.0 ), (Bdensity != 0.0 )*( Berr != 0.0 )
        zeromaskC = ( Cdensity != 0.0 ) 
        zeromaskB = ( Bdensity != 0.0 )
        #Cdensity, Cbins, Cerr, Cf_area = Cdensity, bins, Cerr, Cf_area #Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
        #C_jkerr = C_jkerr[zeromaskC]
        #Bdensity, Bbins, Berr = Bdensity, bins, Berr #Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
        #B_jkerr = B_jkerr[zeromaskB]

        #fitting
        #Cchi, Cchidof = chisquare_dof( bins[zeromaskC], Cdensity[zeromaskC], Cerr[zeromaskC] )
        #Bchi, Bchidof = chisquare_dof( bins[zeromaskB], Bdensity[zeromaskB], Berr[zeromaskB] )
    
        nCbins = np.sum(zeromaskC)
        nBbins = np.sum(zeromaskB)
        Cchi = np.sum( (Cdensity[zeromaskC] - 1.0 * np.ones(nCbins) )**2/Cerr[zeromaskC]**2 )
        Bchi = np.sum( (Bdensity[zeromaskB] - 1.0 * np.ones(nBbins) )**2/Berr[zeromaskB]**2 )

        #ax[i].errorbar(bins[zeromaskC] , Cdensity[zeromaskC], yerr = Cerr[zeromaskC], 
        #    color = 'grey', fmt = '.-', capsize=3, label='no weight, chi2/dof={:>2.2f}/{:3.0f}'.format(Cchi, nCbins) )
        ax[i].plot(bins[zeromaskC] , Cdensity[zeromaskC], color = 'grey', label='no weight, chi2/dof={:>2.2f}/{:3.0f}'.format(Cchi, nCbins) )
        ax[i].errorbar(bins[zeromaskB]*1.0 , Bdensity[zeromaskB], yerr = Berr[zeromaskB], 
            color = 'red', fmt= '.-', capsize=3, label='weighted, chi2/dof={:>2.2f}/{:3.0f}'.format(Bchi, nBbins) )

        #ax[i].bar(Bbins+(bins[1]-bins[0])*0.1, Bf_area[zeromaskB]+0.7, (bins[1]-bins[0]) ,color = 'red', alpha=0.3 )
        ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
        ax[i].set_ylabel('n_gal/n_tot '+str(nside))

        ymin, ymax = 0.7, 1.3
        if kind is 'SPT' : ymin, ymax = 0.7, 1.3
        ax[i].set_ylim(ymin, ymax)
        if hist : ax[i].bar(bins[zeromaskC], Cf_area[zeromaskC]+ymin,(bins[1]-bins[0]) ,color = 'grey', alpha = 0.1 )

        #ax[i].set_xlim(8.2, 8.55)
        ax[i].axhline(1.0,linestyle='--',color='grey')
        #ax[i].legend(loc = 'best')
        
        #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
        #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
        #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
        if property is 'GE': ax[i].set_xscale('log')
        if property == 'NSTARS': ax[i].set_xlim(0.0, 2.0)

    fig.suptitle('systematic test ({})'.format(kind))
    os.system('mkdir '+outdir)
    figname = outdir+'comparison_systematic_'+property+'_'+kind+'_'+suffix2+'.pdf'
    fig.savefig(figname)
    print "saving fig to ", figname

    return 0


def plot_sysweight_one(property = None, filter = 'g', nside = 1024, kind = 'SPT', xlabel =None, ylabel = 'averaged Ng',
    suffix1='', suffix2='', inputdir1 = '.', inputdir2 = '.', outdir='./', hist=False, hide_yaxis=True):

    #import matplotlib.pyplot as plt
    #import numpy as np
    from systematics import SysMapBadRegionMask, loadSystematicMaps, MatchHPArea, chisquare_dof, ReciprocalWeights
    from systematics_module.corr import angular_correlation
    
    
    #filters = ['g', 'r', 'i', 'z']
    #filters = ['g']
    
    #xlabel = property + '  ' + filter
    print xlabel
    if property is 'NSTARS_allband' :
        nside = 1024
        #xxlabel = property
        #xxlabel = 'stellar density'
        #filters = 'g'
    if property is 'GE':
        nside = 512
        #xxlabel = property
        #xxlabel = 'galactic extinction'
        #filters = 'g'
        
    #fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    extra = 0
    if hide_yaxis is False : extra = 1
    fig, ax = plt.subplots( figsize = (5+extra,4))
    #ax = ax.ravel()

    #for i, filter in enumerate(filters):

    filename1 = inputdir1+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix1+'.txt'
    filename2 = inputdir2+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix2+'.txt'
    #print filename
    data1 = np.loadtxt(filename1)
    data2 = np.loadtxt(filename2)
    bins, Cdensity, Cerr, Cf_area, _, _, _ = [data1[:,j] for j in range(data1[0].size)]
    bins, Bdensity, Berr, Bf_area, _, _, _ = [data2[:,j] for j in range(data2[0].size)]
    Cf_area = Cf_area * 1./Cf_area.max()/5.

    #zeromaskC, zeromaskB = ( Cdensity != 0.0 )*( Cerr != 0.0 ), (Bdensity != 0.0 )*( Berr != 0.0 )
    zeromaskC = ( Cdensity != 0.0 ) 
    zeromaskB = ( Bdensity != 0.0 )
    #Cdensity, Cbins, Cerr, Cf_area = Cdensity, bins, Cerr, Cf_area #Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
    #C_jkerr = C_jkerr[zeromaskC]
    #Bdensity, Bbins, Berr = Bdensity, bins, Berr #Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
    #B_jkerr = B_jkerr[zeromaskB]

    #fitting
    #Cchi, Cchidof = chisquare_dof( bins[zeromaskC], Cdensity[zeromaskC], Cerr[zeromaskC] )
    #Bchi, Bchidof = chisquare_dof( bins[zeromaskB], Bdensity[zeromaskB], Berr[zeromaskB] )

    nCbins = np.sum(zeromaskC)
    nBbins = np.sum(zeromaskB)
    Cchi = np.sum( (Cdensity[zeromaskC] - 1.0 * np.ones(nCbins) )**2/Cerr[zeromaskC]**2 )
    Bchi = np.sum( (Bdensity[zeromaskB] - 1.0 * np.ones(nBbins) )**2/Berr[zeromaskB]**2 )

    #ax[i].errorbar(bins[zeromaskC] , Cdensity[zeromaskC], yerr = Cerr[zeromaskC], 
    #    color = 'grey', fmt = '.-', capsize=3, label='no weight, chi2/dof={:>2.2f}/{:3.0f}'.format(Cchi, nCbins) )
    ax.plot(bins[zeromaskC] , Cdensity[zeromaskC], color = '#006ED5', ls = '-')
    ax.errorbar(bins[zeromaskB]*1.0 , Bdensity[zeromaskB], yerr = Berr[zeromaskB], 
        color = 'k', fmt= '--o', capsize=5 )

    #ax[i].bar(Bbins+(bins[1]-bins[0])*0.1, Bf_area[zeromaskB]+0.7, (bins[1]-bins[0]) ,color = 'red', alpha=0.3 )
    ax.set_xlabel(xlabel, fontsize=30)
    ax.set_ylabel(ylabel, fontsize=30)

    ymin, ymax = 0.7, 1.3
    if kind is 'SPT' : ymin, ymax = 0.7, 1.3
    ax.set_ylim(ymin, ymax)
    if hist : ax.bar(bins[zeromaskC], Cf_area[zeromaskC]+ymin,(bins[1]-bins[0]) ,color = 'grey', alpha = 0.1 )

    #ax[i].set_xlim(8.2, 8.55)
    ax.axhline(1.0,linestyle='--',color='grey')
    #ax[i].legend(loc = 'best')
    
    #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
    #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
    #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
    if property is 'GE': ax.set_xscale('log')
    if property == 'NSTARS': ax.set_xlim(0.0, 2.0)

    if hide_yaxis : ax.set_yticklabels([])
    #fig.suptitle('systematic test ({})'.format(kind))
    os.system('mkdir '+outdir)
    ax.tick_params(labelsize = 26)
    figname = outdir+'comparison_systematic_'+property+'_'+filter+'.pdf'
    fig.tight_layout( rect=[0.0, 0.0, 1.0, 1.0] )
    fig.savefig(figname)
    print "saving fig to ", figname

    return 0


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

