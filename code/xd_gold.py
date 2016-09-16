




def callingCMASSphotoObj():
    import esutil
    import numpy as np
    
    path = '/n/des/lee.5922/data/cmass_cat/'
    """
    cmass_photo = esutil.io.read(path+'bosstile-final-collated-boss2-boss38.fits.gz', upper=True)
    cmass_specprimary = esutil.io.read(path+'bosstile-final-collated-boss2-boss38-specObj.fits.gz', upper=True)
    cmass_zwarning_noqso = esutil.io.read(path+'bosstile-final-collated-boss2-boss38-photoObj-specObj.fits.gz')
    
    use2 = (( (cmass_photo['BOSS_TARGET1'] & 2) != 0 )
           #(cmass_specprimary['SPECPRIMARY'] == 1) &
           #(cmass_zwarning_noqso['ZWARNING_NOQSO'] == 0 )
           )
    """
    cmass = esutil.io.read(path+'boss_target_selection.fits', upper=True)
    use = (( cmass['BOSS_TARGET1'] & 2 != 0 ) &
           ((cmass['CHUNK'] != 'boss1' ) & (cmass['CHUNK'] != 'boss2')) &
           ((cmass['FIBER2MAG_I'] - cmass['EXTINCTION_I']) < 21.5 )
            )
           
    cmass = cmass[use]
    
    print "Applying Healpix BOSS SGC footprint mask"
    HPboss = esutil.io.read('/n/des/lee.5922/data/cmass_cat/healpix_boss_footprint_SGC_1024.fits')
    from systematics import hpRaDecToHEALPixel
    HealInds = hpRaDecToHEALPixel( cmass['RA'],cmass['DEC'], nside= 1024, nest= False)
    BOSSHealInds = np.in1d( HealInds, HPboss )
    return cmass[BOSSHealInds]


def main():

    import esutil
    from systematics import mergeCatalogsUsingPandas
    from xd import doVisualization_1d
    from systematics_module.corr import angular_correlation, cross_angular_correlation,LensingSignal
    
    ra, ra2, dec, dec2 = 350., 360, -1, 1
    
    
    # callin all des data and make merged cat ===============================
    
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_ALLCOLUMNS_EA', gold=True)
    des_gold = Cuts.keepGoodRegion(des_gold)
    # ------

    """
    for j in np.arange(1,21):
        
        t1 = time.time()
        des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_0000{:02}'.format(j), gold=True)
        merged_list = []
        for i in np.arange(35):
            
            fds = io.getDESY1A1catalogs(keyword = 'Y1A1_COADD_OBJECTS_000{:02}'.format(i), sdssmask=False)
            merged = mergeCatalogsUsingPandas(des=fds, gold=des_gold, how='inner', key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
            merged_list.append(merged)
        merged_list = np.hstack(merged_list)
        fitsio.write('/n/des/lee.5922/data/y1a1_coadd/'+'Y1A1_GOLD_merged_{:02}.fits'.format(j), merged_list)
        print time.time() - t1, ' s'

    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_gold2.fits')
    prefix = 'gold_st82_2_'
    #ra, ra2, dec, dec2 = 350., 355, -1, 1
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    merged_des_s = fitsio.read(resultpath+'merged_des_s.fits')
    
    result_cat = []
    for i in np.arange(1,350):
        fds = io.getDESY1A1catalogs(keyword = 'Y1A1_COADD_OBJECTS_000{:03}'.format(i), sdssmask=False)
        merged_des = mergeCatalogsUsingPandas(des=fds, gold=des_gold, how='inner', key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
        merged_des1 = Cuts.doBasicCuts(merged_des, raTag = 'RA', decTag='DEC', object = None)
        result_gold = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=merged_des_s, test=merged_des1, prefix = prefix, mock=True, gold=True )
        result_cat.append(result_gold)
    
    result_gold_cat = np.hstack(result_cat)
    """
    # ========================================================================
    
    
    full_des_data = io.getDESY1A1catalogs(keyword = 'STRIPE82_COADD', sdssmask=False)
    #des = Cuts.doBasicCuts(full_des_data, object = 'galaxy')
    #des_s = Cuts.SpatialCuts(des, ra =ra, ra2=ra2, dec=dec, dec2=dec2 )
    
    #des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_00', gold=True)
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_ALLCOLUMNS_EA', gold=True)
    des_gold = Cuts.keepGoodRegion(des_gold)
    
    merged_des = mergeCatalogsUsingPandas(des=full_des_data, gold=des_gold, key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
    merged_des1 = Cuts.doBasicCuts(merged_des, raTag = 'RA', decTag='DEC', object = None)
    merged_des_s = Cuts.SpatialCuts(merged_des1, ra =ra, ra2=ra2, dec=dec, dec2=dec2 )
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'merged_des_s.fits', merged_des_s)
    
    cmass = getSGCCMASSphotoObjcat()
    cmass = Cuts.keepGoodRegion(cmass)
    cmass_s = Cuts.SpatialCuts(cmass, ra = ra, ra2=ra2, dec=dec, dec2=dec2 )
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, merged_des1 )

    #prefix = 'gold_small_'
    #prefix = 'gold_st82_'
    #prefix = 'gold_st82_2_'
    #prefix = 'gold_st82_3_'
    prefix = 'gold_st82_SFD98_'

    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(merged_des_s, merged_des_s, [0.5,0.5], random_state=0)
    des_train = merged_des_s[trainInd]
    des_test = merged_des_s[testInd]

    #result_gold_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=des_train, test=des_test, prefix = prefix, mock=True, gold=True )
    result_gold_stripe82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=merged_des_s, test=merged_des1, prefix = prefix, mock=True, gold=True )
    

    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'result_gold_st82_SFD98.fits', result_gold_stripe82)
    fitsio.write(resultpath+'clean_cmass_data_des_gold_SFD98.fits', clean_cmass_data_des)





    # test with coadd obects table ( use slrzeroshift instead of XDCORR ---------
    des = Cuts.doBasicCuts(full_des_data, raTag = 'RA', decTag='DEC', object = 'galaxy')
    des = AddingReddening(des)
    
    prefix = 'coadd_st82_SLR_'
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(des, des, [0.05,0.95], random_state=0)
    des_train = des[trainInd]
    des_test = des[testInd]
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, des )
    result_coadd_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=des_train, test=des_test, prefix = prefix, mock=True, reddening = 'SLR' )
    



    # Test resulted population in various aspects ===================================

    # hist in color spaces
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    result_gold_stripe82 = fitsio.read(resultpath+'result_gold_st82_SFD98.fits')
    dmass_gold_stripe82, dmass_gold_stripe82_ellipse = resampleWithPth(result_gold_stripe82)
    clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_gold_SFD98.fits')
    
    
    #des_im3_st82_o = io.getDEScatalogs(file = '/n/des/lee.5922/data/im3shape_st82.fits')
    #clean_des_im3_st82 = Cuts.keepGoodRegion(des_im3_st82_o)
    #dmass_gold_stripe82 = addphotoz(dmass_gold_stripe82, clean_des_im3_st82)
    #clean_cmass_data_des = addphotoz(clean_cmass_data_des, clean_des_im3_st82)


    As_X, _ = mixing_color( dmass_gold_stripe82, reddening=None)
    Xtrue,_ = mixing_color( clean_cmass_data_des, reddening =None )
    
    labels = ['MAG_MODEL_R', 'MAG_MODEL_I', 'g-r', 'r-i', 'i-z']
    ranges =  [[17,22.5], [17,22.5], [0,2], [-.5,1.5], [0.0,.8]]
    doVisualization_1d( Xtrue, As_X, labels = labels, ranges = ranges, nbins=100, prefix='gold_st82_SFD98_')
    doVisualization( Xtrue, As_X, labels = labels, nbins=100, ranges = ranges, prefix='gold_st82_SFD98_')
    
    # angular
    GoldMask = callingEliGoldMask()
    GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0 ]

    angular_correlation(data = dmass_gold_stripe82, rand = GoldMask_st82, weight = [None, None], suffix = '_gold_st82_SFD98')
    random = fitsio.read('/n/des/lee.5922/data/cmass_cat/'+'random0_DR12v5_CMASS_South.fits.gz')
    clean_random = Cuts.keepGoodRegion(random)
    angular_correlation(data = cmass, rand = clean_random, weight = [None, None ], suffix = '_cmass_st82')
    angular_correlation(data = dmass_gold_stripe82 , rand = clean_random, weight = [None, None ], suffix = '_gold_st82_SFD98_random')

    # corr comparison
    
    labels = [ '_gold_st82_SFD98','_gold_st82_SFD98_random' ,'cmass_st82', 'y1a1']
    #labels = [ 'gold_stripe82_2_randoms','gold_stripe82_3_randoms' ,'cmass_stripe82_randoms', 'y1a1']
    
    linestyle = ['']+['' for i in labels[:-1]]
    fmt = ['.']+['.' for i in labels[:-2]]+['.']
    color = ['cyan'] + [None for i in labels[:-1]]
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

    ax.set_xlim(1e-2, 20)
    #ax.set_ylim(-0.02 , 0.5)
    ax.set_ylim(0.001 , 5.0)
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



    # redshift histogram
    
    z_bin = np.linspace(0.1, 1.0, 200)
    labels = ['cmass', 'gold_st82_SFD98']
    cat = [clean_cmass_data_des, dmass_gold_stripe82]
    
    #noblend_clean_cmass_data_des = addphotoz(noblend_clean_cmass_data_des, des_im3)
    fig, axes = plt.subplots( len(cat), 1, figsize = (8,5*len(cat)))
    for i in range(len(cat)):
        #axes[i].hist( photoz_des['DESDM_ZP'], bins = z_bin,facecolor = 'green', normed = True, label = 'cmass')
        axes[i].hist( clean_cmass_data_des['DESDM_ZP'], bins = z_bin, facecolor = 'green', normed = True, label = 'cmass')
        axes[i].hist( cat[i]['DESDM_ZP'], bins = z_bin, facecolor = 'red',alpha = 0.35, normed = True, label = labels[i])
        axes[i].set_xlabel('photo_z')
        axes[i].set_ylabel('N(z)')
        #ax.set_yscale('log')
        axes[i].legend(loc='best')
    
    axes[0].set_title('redshift hist')
    figname ='figure/hist_z_gold_stripe82.png'
    fig.savefig(figname)
    print 'saving fig to ',figname



    # =============================













