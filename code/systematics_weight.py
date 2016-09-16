

def doVisualization_ngal(property = None, nside = 1024, kind = 'SPT', suffix=''):

    import matplotlib.pyplot as plt
    import numpy as np
    from systematics import SysMapBadRegionMask, loadSystematicMaps, MatchHPArea, chisquare_dof, ReciprocalWeights
    from systematics_module.corr import angular_correlation
    
    
    filters = ['g', 'r', 'i', 'z']
    #filters = ['g']
    
    
    if property is 'NSTARS' :
        nside = 512
        filters = ['g']
        
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    ax = ax.ravel()

    for i, filter in enumerate(filters):
        filename = 'data_txt/systematic_'+property+'_'+filter+'_'+kind+suffix+'.txt'
        print filename
        #print filename
        data = np.loadtxt(filename)
        bins, Cdensity, Cerr, Cf_area, Bdensity, Berr, Bf_area = [data[:,j] for j in range(data[0].size)]
        zeromaskC, zeromaskB = ( Cdensity != 0.0 ), (Bdensity != 0.0 )
        Cdensity, Cbins, Cerr = Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
        #C_jkerr = C_jkerr[zeromaskC]
        Bdensity, Bbins, Berr = Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
        #B_jkerr = B_jkerr[zeromaskB]
        
        #fitting
        Cchi, Cchidof = chisquare_dof( Cbins, Cdensity, Cerr )
        Bchi, Bchidof = chisquare_dof( Bbins, Bdensity, Berr )
    
        #ax[i].errorbar(Cbins-(bins[1]-bins[0])*0.1, Cdensity, yerr = Cerr, color = 'blue', fmt = '.', label='CMASS') #, chi2/dof={:>2.2f}'.format(Cchidof))
        ax[i].errorbar(Bbins+(bins[1]-bins[0])*0.1, Bdensity, yerr = Berr, color = 'red', fmt= '.',  label='DMASS, chi2 = {:>2.2f}, chi2/dof={:>2.2f}'.format(Bchi, Bchidof))
        
        #ax[i].bar(Cbins+(bins[1]-bins[0])*0.1, Cf_area[zeromaskC],(bins[1]-bins[0]) ,color = 'blue', alpha = 0.3 )
        ax[i].bar(Bbins+(bins[1]-bins[0])*0.1, Bf_area[zeromaskB]+0.5, (bins[1]-bins[0]) ,color = 'red', alpha=0.3 )
        ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
        ax[i].set_ylabel('n_gal/n_tot '+str(nside))
        ax[i].set_ylim(0.5, 1.5)
        #ax[i].set_xlim(8.2, 8.55)
        ax[i].axhline(1.0,linestyle='--',color='grey')
        ax[i].legend(loc = 'best')
        
        #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
        #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
        #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
        if property == 'NSTARS': ax[i].set_xlim(0.0, 2.0)

    fig.suptitle('systematic test ({})'.format(kind))
    figname = 'figure/systematic_'+property+'_'+kind+suffix+'.png'
    fig.savefig(figname)
    print "saving fig to ", figname

    return 0




def doVisualization_Angcorr( labels = None ):
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    linestyle = ['-']+['--' for i in labels[:-1]]
    fmt = ['.']+['o' for i in labels[:-2]]+['.']
    color = ['red'] + [None for i in labels[:-1]]
    corr_txt = [np.loadtxt('data_txt/acf_comparison_'+s+'.txt') for s in labels]
    
    corr_txt2 = np.loadtxt('data_txt/acf_comparison_cmass_SGC.txt')
    thetaS, wS, Sjkerr = corr_txt2[:,0], corr_txt2[:,1], corr_txt2[:,2]
    
    fig, (ax, ax2) = plt.subplots(2,1, figsize = (10,15))
    ax.errorbar( thetaS, wS, yerr = Sjkerr, label = 'CMASS SGC', color='black', alpha = 0.5)
    ax2.errorbar( thetaS, wS-wS, yerr = 10 * Sjkerr, label = 'CMASS SGC', color='black', alpha = 0.9)
    
    for i in range(len(labels)):
        
        thetaD, wD, Djkerr = corr_txt[i][:,0], corr_txt[i][:,1], corr_txt[i][:,2]
        
        if labels[i] == 'wnstar+mask': markersize = 12
        else : markersize = 5
        
        ax.errorbar( thetaD*(0.95+0.02*i), wD, yerr = Djkerr, fmt = fmt[i], linestyle = linestyle[i] ,label = labels[i], color = color[i], markersize = markersize)
        ax2.errorbar( thetaD*(0.95+0.05*i), 10 * (wD - wS), yerr = 10 *Djkerr, fmt = fmt[i],  linestyle = linestyle[i], label = labels[i],color = color[i], markersize=markersize)

    ax.set_xlim(1e-2, 50)
    #ax.set_ylim(-0.02 , 0.5)
    ax.set_ylim(0.0001 , 5.0)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')
    
    ax2.axhline(y=0.0, color = 'black')
    ax2.set_xlim(1e-1, 10)
    ax2.set_ylim(-.2, .5)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\theta(deg)$')
    ax2.set_ylabel(r'$10 \times$ $($ ${w}$ - ${w_{\rm{true}}}$ $)$')
    ax2.legend(loc='best')
    figname = 'figure/acf_comparison.png'
    fig.savefig(figname)
    print 'writing plot to ', figname



def main():


    # masking
    # galaxy density for all
    # plot (check)
    # galaxy density for one (plot) -> get weight
    # apply weight and calculate galaxy density for the next property (plot)
    # repeat


    # calling catalog
    from systematics import GalaxyDensity_Systematics,loadSystematicMaps, chisquare_dof, MatchHPArea, SysMapBadRegionMask, callingEliGoldMask

    #balrog_y1a1 = fitsio.read('result_cat/result_balrog_EMHUFF.fits')
    #balrog_SSPT = balrog_y1a1[balrog_y1a1['DEC'] < -3]
    #rand_clean = Cuts.keepGoodRegion(balrog_SSPT)
    
    # dmass
    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    dmass_y1a1, _ = resampleWithPth(result_y1a1)
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    result_gold_stripe82 = fitsio.read(resultpath+'result_gold_stripe82.fits')
    dmass_gold_stripe82, _ = resampleWithPth(result_gold_stripe82)
    

    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    result_gold_stripe82 = fitsio.read(resultpath+'result_gold_2_stripe82.fits')
    dmass_gold_stripe82, _ = resampleWithPth(result_gold_stripe82)
    clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_gold2.fits')
    
    
    # calling Eli mask
    GoldMask = callingEliGoldMask()
    GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0 ]
    angular_correlation(data = dmass_gold_stripe82, rand = GoldMask_st82, weight = [None, GoldMask_st82['FRAC']], suffix = '_gold_stripe82')
    # ----------------------------------------------------------------
    


    kind = 'STRIPE82'
    nside = 4096
    njack = 10

    property = ['AIRMASS', 'SKYBRITE', 'FWHM', 'SKYSIGMA', 'NSTARS']
    filter = ['g', 'r', 'i', 'z']

    # Calling maps
    MaskDic = {}
    for i,p in enumerate(property):
        if p == 'NSTARS':
            filename =  'y1a1_gold_1.0.2_stars_nside1024.fits'
            sysMap = loadSystematicMaps( filename = filename, property = p, filter = 'g', nside = 1024 , kind = kind )
            if kind is 'STRIPE82': sysMap = sysMap[sysMap['DEC'] > -3]
            mapname = 'sys_'+p+'_'+'g'
            MaskDic[mapname] = sysMap
        else :
            nside = 4096
            filter = ['g', 'r', 'i', 'z']
            filename = None
    
            for j,f in enumerate(filter):
                sysMap = loadSystematicMaps( filename = filename, property = p, filter = f, nside = nside , kind = kind)
                #goodIndmask = np.in1d( sysMap['PIXEL'], GoldMask['PIXEL'])
                #sysMap = sysMap[goodIndmask]
                #sysMap = Cuts.SpatialCuts(sysMap, ra =ra, ra2=ra2, dec=dec, dec2=dec2 )
                mapname = 'sys_'+p+'_'+f
                MaskDic[mapname] = sysMap

    # basic cutoff mask
    cutvalue = {'AIRMASS':[1.4, 1.35, 1.35, 1.35],
            'SKYBRITE':[150, 380, 1150, 2500],
            'FWHM':[6.5, 5.5, 5.0, 4.5],
            'SKYSIGMA':[6.5, 9.5, 18, 26],
            'NSTARS':[5, 5, 5, 5 ]}
    
    cutvalue1 = {'AIRMASS':[1.45, 1.45, 1.45, 1.45],
                'SKYBRITE':[3000, 3000, 3000, 3000],
                'FWHM':[6.5, 5.0, 4.5, 4.0],
                'SKYSIGMA':[40, 40, 40, 40],
                'NSTARS':[501, 501, 501, 501 ]}

    for i,p in enumerate(property):
        correctMask = np.ones(dmass_gold_stripe82.size, dtype=bool)
        if p is 'NSTARS':
            nside = 1024
            filter = ['g']
        else :
            nside = 4096
            filter = ['g', 'r', 'i', 'z']
        for j,f in enumerate(filter):
            mapname = 'sys_'+p+'_'+f
            maskedsysMap, mask = SysMapBadRegionMask(dmass_gold_stripe82, MaskDic[mapname], nside = nside, cond = '<=', val = cutvalue[p][j])
            correctMask = correctMask * mask
            MaskDic[mapname+'_masked'] = maskedsysMap

        MaskDic[p+'_mask'] = correctMask
        #angular_correlation(data = dmass_SSPT[correctMask], rand = rand_clean, weight = [None, None], suffix = '_'+kind+'_'+p+'_masked')
        print p+'_mask', np.sum(correctMask)

    TotalMask = MaskDic['AIRMASS_mask'] * MaskDic['FWHM_mask'] # * MaskDic['SKYSIGMA_mask'] * MaskDic['SKYBRITE_mask']# * MaskDic['NSTARS_mask']
    clean_dmass = dmass_gold_stripe82[TotalMask]

    #angular_correlation(data = clean_dmass, rand = GoldMask, weight = [None, None], suffix = '_'+kind+'_masked')





    # pixel fraction weights for each gal

    from systematics import hpRaDecToHEALPixel
    import pandas as pd
    catHpInd = hpRaDecToHEALPixel(clean_dmass['RA'], clean_dmass['DEC'], nside=4096, nest = False)
    catHpInd = pd.DataFrame(catHpInd)
    catHpInd.columns = ['PIXEL']
    GoldMask_pd = pd.DataFrame(GoldMask)
    frac_weight = pd.merge( catHpInd, GoldMask_pd, how = 'left', on = 'PIXEL')
    frac_weight = frac_weight.to_records()


    property = ['AIRMASS', 'SKYBRITE', 'FWHM', 'SKYSIGMA', 'NSTARS']

    # calculating galaxy density and weights iterately
    from systematics import ReciprocalWeights, jksampling
    from systematics_module.corr import angular_correlation

    re_weights = np.ones(clean_dmass.size)
    #re_weights = frac_weight['FRAC']
    #re_weights = GoldMask['FRAC']
    rand_clean = None
    #correctMask = np.ones(clean_dmass.size, dtype=bool)

    re_weights_dic = {}

    for p in property:
        if p is 'NSTARS':
            nside = 1024
            filter = ['g']
        else :
            nside = 4096
            filter = ['g', 'r', 'i', 'z']
        for j,f in enumerate(filter):
            
            mapname = 'sys_'+p+'_'+f+'_masked'
            bins, Bdensity, Berr, Bf_area = GalaxyDensity_Systematics(clean_dmass, MaskDic[mapname], nside = nside, raTag = 'RA', decTag='DEC', property = p, filter = f, weight = re_weights)
            
            #bins = bins/np.sum(sysMap['SIGNAL']) *len(sysMap)
            #B_jkerr = jksampling(clean_dmass, MaskDic[mapname], property = p, nside = nside, njack = 30, raTag = 'RA', decTag = 'DEC' )
            
            filename = 'data_txt/systematic_'+p+'_'+f+'_'+kind+'_masked.txt'
            #DAT = np.column_stack(( bins-(bins[1]-bins[0])*0.1, Bdensity, Berr, Bf_area, Bdensity, Berr, Bf_area  ))
            DAT = np.column_stack(( bins, Bdensity, Berr, Bf_area, Bdensity, Berr, Bf_area  ))
            np.savetxt(filename, DAT, delimiter = ' ', header = 'bins, Cdensity, Cerr, Cfarea, Bdensity, Berr, Bfarea, B_jkerr')
            print "saving data to ", filename
            #doVisualization_ngal(property = p, nside = nside, kind = kind, suffix='_masked')
            """
            cutvalue[p][j] = 1.3
            maskedsysMap, mask = SysMapBadRegionMask(dmass_SSPT, MaskDic[mapname], nside = nside, cond = '<=', val = cutvalue[p][j])
            correctMask = correctMask * mask
            MaskDic[mapname+'_masked'] = maskedsysMap
            clean_dmass = dmass_SSPT[correctMask]
            """
            re_weights = re_weights * ReciprocalWeights( catalog = clean_dmass, sysMap = MaskDic[mapname], property = p, filter = f, nside = nside, kind = kind )
            re_weights_dic['re_weight_'+p] = re_weights
            # check
            bins, Bdensity, Berr, Bf_area = GalaxyDensity_Systematics(clean_dmass, MaskDic[mapname], nside = nside, raTag = 'RA', decTag='DEC', property = p, filter = f, weight = re_weights)
            filename = 'data_txt/systematic_'+p+'_'+f+'_'+kind+'_corrected.txt'
            DAT = np.column_stack(( bins, Bdensity, Berr, Bf_area, Bdensity, Berr, Bf_area  ))
            np.savetxt(filename, DAT, delimiter = ' ', header = 'bins, Cdensity, Cerr, Cfarea, Bdensity, Berr, Bfarea')
            print "saving data to ", filename
            #doVisualization_ngal(property = p, nside = nside, kind = kind, suffix='_corrected')

        #doVisualization_ngal(property = p, nside = nside, kind = kind, suffix='_masked')
        #doVisualization_ngal(property = p, nside = nside, kind = kind, suffix='_corrected')
        #angular_correlation(data = clean_dmass, rand = rand_clean, weight = [re_weights, None], suffix = '_'+kind+'_w_'+p)


    angular_correlation(data = dmass_y1a1, rand = GoldMask, weight = [None, None], suffix = '_'+kind+'_noweight')
    angular_correlation(data = clean_dmass, rand = GoldMask, weight = [None, None], suffix = '_'+kind+'_masked')







    for p in property:
        doVisualization_ngal(property = p, nside = 4096, kind = kind, suffix='_masked')
        #doVisualization_ngal(property = p, nside = nside, kind = kind, suffix='_corrected')
        rw = re_weights_dic['re_weight_'+p]
        angular_correlation(data = clean_dmass, rand = GoldMask, weight = [rw, None], suffix = '_'+kind+'_w_'+p)
    






    # apply weights to correlation function
    property = [ 'AIRMASS', 'SKYBRITE', 'FWHM', 'SKYSIGMA', 'NSTARS']
    kind = 'Y1A1'
    # corr com

    # corr comparison

    labels = ['y1a1']
    
    linestyle = ['-']+['--' for i in labels[:-1]]
    fmt = ['.']+['o' for i in labels[:-2]]+['.']
    color = ['red'] + [None for i in labels[:-1]]
    corr_txt = [np.loadtxt('data_txt/acf_comparison_'+s+'.txt') for s in labels]
    
    corr_txt2 = np.loadtxt('data_txt/acf_comparison_cmass_SGC.txt')
    thetaS, wS, Sjkerr = corr_txt2[:,0], corr_txt2[:,1], corr_txt2[:,2]
    
    fig, (ax, ax2) = plt.subplots(2,1, figsize = (10,15))
    ax.errorbar( thetaS, wS, yerr = Sjkerr, label = 'CMASS SGC', color='black', alpha = 0.5)
    ax2.errorbar( thetaS, wS-wS, yerr = 10 * Sjkerr, label = 'CMASS SGC', color='black', alpha = 0.9)
    
    for i in range(len(labels)):
        
        thetaD, wD, Djkerr = corr_txt[i][:,0], corr_txt[i][:,1], corr_txt[i][:,2]
        
        if labels[i] == 'wnstar+mask': markersize = 12
        else : markersize = 5
        
        ax.errorbar( thetaD*(0.95+0.02*i), wD, yerr = Djkerr, fmt = fmt[i], linestyle = linestyle[i] ,label = labels[i], color = color[i], markersize = markersize)
        ax2.errorbar( thetaD*(0.95+0.05*i), 10 * (wD - wS), yerr = 10 *Djkerr, fmt = fmt[i],  linestyle = linestyle[i], label = labels[i],color = color[i], markersize=markersize)

    ax.set_xlim(1e-2, 50)
    #ax.set_ylim(-0.02 , 0.5)
    ax.set_ylim(0.0001 , 5.0)
    ax.set_xlabel(r'$\theta(deg)$')
    ax.set_ylabel(r'${w(\theta)}$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc = 'best')
    ax.set_title(' angular correlation ')
    
    ax2.axhline(y=0.0, color = 'black')
    ax2.set_xlim(1e-1, 10)
    ax2.set_ylim(-.2, .5)
    ax2.set_xscale('log')
    ax2.set_xlabel(r'$\theta(deg)$')
    ax2.set_ylabel(r'$10 \times$ $($ ${w}$ - ${w_{\rm{true}}}$ $)$')
    ax2.legend(loc='best')
    figname = 'figure/acf_comparison.png'
    fig.savefig(figname)
    print 'writing plot to ', figname




def cont_sys_main():

    from systematics import callingEliGoldMask
    from systematics_module import contCorrection

    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    #result_gold_stripe82 = fitsio.read(resultpath+'result_gold_2_stripe82.fits')
    #dmass_gold_stripe82, dmass_gold_stripe82_ellipse = resampleWithPth(result_gold_stripe82)
    #clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_gold2.fits')

    result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    dmass_y1a1, _ = resampleWithPth(result_y1a1)
    dmass_y1a1_st82 = dmass_y1a1[dmass_y1a1['DEC']> -3.0 ]
    #clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des.fits')
    
    from xd import addphotoz
    gold_photo_z = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_0', gold=True)
    dmass_y1a1_st82 = addphotoz(des = dmass_y1a1_st82, im3shape=gold_photo_z)
    
    
    # calling Eli mask
    GoldMask = callingEliGoldMask()
    GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0]
    
    cmass = getSGCCMASSphotoObjcat()
    SGCmask = (cmass['RA'] < 100 ) | (cmass['RA'] > 300)
    cmass = cmass[SGCmask]
    """ cmass SGC """
    cmass_st82 = Cuts.keepGoodRegion(cmass)
    cmass_rand = fitsio.read('/n/des/lee.5922/data/cmass_cat/'+'random0_DR12v5_CMASS_South.fits.gz')
    
    from systematics import hpRaDecToHEALPixel
    origin_HealInds = hpRaDecToHEALPixel( cmass_rand['RA'],cmass_rand['DEC'], nside= nside, nest= False)
    
    
    
    CC = contCorrection.CorrectContaminant()
    
    CC.obs_data = dmass_y1a1.copy()
    CC.f_c, CC.cont_data, CC.true_data = CC.get_cont_data(obs = dmass_y1a1_st82, true = cmass_st82)
    CC.w_obs_data = np.loadtxt('data_txt/acf_comparison_y1a1.txt')
    #CC.w_true_data = np.loadtxt('data_txt/acf_comparison_cmass_SGC.txt')

    """
    CC.w_cont_data = np.loadtxt('data_txt/acf_comparison_weightedcont.txt')
    CC.w_cross_data = np.loadtxt('data_txt/acf_cross_weightedcont.txt')
    """
    CC.L_obs_data = np.loadtxt('data_txt/Ashley_lensing.txt')
    CC.L_true_data = np.loadtxt('data_txt/Ashley_true_lensing.txt')
    CC.L_cont_data = np.loadtxt('data_txt/lensing_weightedcont_prop.txt')
    
    
    mask_cmass = MatchHPArea(cat = cmass, origin_cat = cmass_rand, nside = 1024)
    #CC.w_obs_data = CC.angular_correlation( data=CC.obs_data, rand=GoldMask, weight = [None, GoldMask['FRAC']], suffix = '')
    CC.w_true_data = CC.angular_correlation( data=mask_cmass, rand = cmass_rand, weight = None, suffix = '_cmass_SGC')
    CC.w_cont_data = CC.angular_correlation( data=CC.cont_data, rand=GoldMask_st82, weight = [ None, GoldMask_st82['FRAC']], suffix = '_coadd_st82_cont')
    CC.w_cross_data = CC.cross_angular_correlation(data = CC.true_data, data2 = CC.cont_data, rand = GoldMask_st82, rand2= GoldMask_st82, weight = [None, GoldMask_st82['FRAC'], None, GoldMask_st82['FRAC']], suffix = '_coadd_st82_cross')

    doVisualization_Angcorr( labels = ['coadd_st82_cont', 'y1a1', 'cmass_NGC'] )

    rmin, rmax = 0.02, 10.
    
    CC.eps82, CC.eps = CC.ang_stat_err(rmin = 0.02, rmax = 10.0)
    CC.eps_len82, CC.eps_len = CC.len_stat_err(rmin = 0.02, rmax = None)
    
    # bias
    r_ang, ang_bias_sys = CC.Angsys_from_bias()
    r_len, len_bias_sys = CC.Lensys_from_bias()
    
    m_fL, i_fL = CC.MaxFc_from_AngSys_L(rmin = rmin, rmax = rmax)
    m_fR, i_fR = CC.MaxFc_from_AngSys_R(rmin = rmin, rmax = rmax)
    m_fL_len, i_fL_len = CC.MaxFc_from_LensSys_L(rmin = rmin, rmax = None)
    m_fR_len, i_fR_len = CC.MaxFc_from_LensSys_R(rmin = rmin, rmax = None)





    """
    fig, (ax, ax2) = plt.subplots(1,2, figsize = (15, 5))

    ax.semilogx(r_ang, ang_bias_sys, label = 'ang_bias_sys' )
    ax.semilogx([r_ang.min(), r_ang.max()], [CC.eps, CC.eps], label = 'eps')
    ax.legend(loc = 'best')
    ax2.semilogx(r_len, len_bias_sys, label = 'len_bias_sys' )
    ax2.semilogx([r_len.min(), r_len.max()], [CC.eps_len, CC.eps_len], label = 'eps')
    ax2.semilogx([r_len.min(), r_len.max()], [CC.eps_len82, CC.eps_len82], label = 'eps82')
    ax2.legend(loc = 'best')
    figname = 'figure/bias_sys.png'
    print 'figsave ', figname
    fig.savefig(figname)
    """


