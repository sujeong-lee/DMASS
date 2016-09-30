
def comparison_gold_coadd():
    """
    full_des_data = io.getDESY1A1catalogs(keyword = 'STRIPE82_COADD', sdssmask=False)
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_0', gold=True)
    merged_des = mergeCatalogsUsingPandas(des=full_des_data, gold=des_gold, key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
    merged_des1 = Cuts.doBasicCuts(merged_des.copy(), raTag = 'RA', decTag='DEC', object = None)
    mags = ['MAG_MODEL', 'MAG_DETMODEL']
    merged_des1 = getCorrectedMag( merged_des1, mags = mags, reddening = None )
    merged_des2 = AddingReddening( merged_des1 )
    merged_des1 = 0
    merged_des2 = getCorrectedMag( merged_des2, mags = ['MAG_APER_4'], reddening = 'SLR' )
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'merged_des_st82.fits', merged_des2)

    des = Cuts.doBasicCuts(full_des_data, raTag = 'RA', decTag='DEC', object = 'galaxy')
    des = AddingReddening(des) # add SLR_SHIFT column
    mags = ['MAG_DETMODEL', 'MAG_MODEL', 'MAG_APER_4']
    des = getCorrectedMag( des, mags = mags, reddening = 'SLR' )
    coadd_final = addphotoz(des = des, im3shape=des_gold)
    fitsio.write(resultpath+'coadd_final_st82.fits', coadd_final)

    """
    
    
    full_des_data, merged_des, des_gold = 0, 0, 0
  
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    coadd_final = fitsio.read(resultpath+'coadd_final_st82.fits')
    gold_final = fitsio.read(resultpath+'merged_des_st82.fits')
    
    
    As_gold, _ = mixing_color( gold_final )
    As_coadd,_ = mixing_color( coadd_final )
    
    
    labels = ['MAG_MODEL_R', 'MAG_MODEL_I', 'g-r', 'r-i', 'i-z']
    ranges =  [[17,22], [17,22], [0,2], [-.5,1.5], [0.0,.8]]

    #true : gold, obs : coadd
    doVisualization( As_gold, As_coadd, name=['gold', 'coadd'], labels = labels, nbins=100, ranges = ranges, prefix='test'+'_')
    doVisualization_z( cats = [gold_final, coadd_final], labels = ['gold', 'coadd'] )

    z_faliure = coadd_final[coadd_final['DESDM_ZP'] == 0 ]
    As_zf, _ = mixing_color( z_faliure )
    doVisualization( As_zf, As_zf, name=['z_failure', 'z_failure'], labels = labels, nbins=100, ranges = ranges, prefix='test_zf'+'_')



def gettingDMASSinY1A1():

    #des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_merged_01', gold=True)
    #merged_des = mergeCatalogsUsingPandas(des=full_des_data, gold=des_gold, key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
    #merged_des1 = AddingReddening( merged_des )
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    #clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_'+'gold_st82_9_2'+'.fits')
    gold_final = fitsio.read(resultpath+'merged_des_st82.fits')
    gold_final_s = Cuts.SpatialCuts(  gold_final, ra = 320.0, ra2=330.0 , dec= -1.0 , dec2=1.0 )
    
    cmass = io.getSGCCMASSphotoObjcat()
    cmass = Cuts.keepGoodRegion(cmass)
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, gold_final )
    clean_cmass_data_des_s = Cuts.SpatialCuts(  clean_cmass_data_des, ra = 320.0, ra2=330.0 , dec= -1.0 , dec2=1.0 )
    
    merged_des = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_merged_01', gold=True)
    merged_des = Cuts.doBasicCuts(merged_des, raTag = 'RA', decTag='DEC', object = None)
    mags = ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_APER_4']
    merged_des = getCorrectedMag( merged_des, mags = mags, reddening = None )
    
    #prefix = 'gold_st82_9_'
    prefix = 'gold_st82_11_small_'
    
    result_gold_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des_s, train=gold_final_s, test=merged_des, prefix = prefix, mock=True)
    
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    result = []
    for i in range(1,21):
        merged_des = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_merged_{:02}'.format(i), gold=True)
        merged_des = Cuts.doBasicCuts(merged_des, raTag = 'RA', decTag='DEC', object = None)
    
        mags = ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_APER_4']
        merged_des = getCorrectedMag( merged_des, mags = mags, reddening = None )

        prefix = 'gold_st82_9_'
        result_gold_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=gold_final, test=merged_des, prefix = prefix, mock=True)
        result.append(result_gold_st82)
        fitsio.write('/n/des/lee.5922/data/y1a1_coadd/'+'result_gold_y1a1_{:02}.fits'.format(i), result_gold_st82)
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'result_gold_y1a1_01.fits', result_gold_st82)
    #fitsio.write(resultpath+'clean_cmass_data_des_gold_st82_9_2.fits', clean_cmass_data_des)
    
    #resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    #fitsio.write(resultpath+'merged_des_st82.fits', merged_des2)



def main():

    
    #ra, ra2, dec, dec2 = 350., 355, -1, 1
    
    
    # callin all des data and make merged cat ===============================
    
    #des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_ALLCOLUMNS_EA', gold=True)
    #des_gold = Cuts.keepGoodRegion(des_gold)
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
    
    
    import esutil
    #from systematics import mergeCatalogsUsingPandas
    from xd import doVisualization_1d
    from systematics_module.corr import angular_correlation, cross_angular_correlation,LensingSignal
    from utils import *

    # 1) gold, slr corrected
    """
    full_des_data = io.getDESY1A1catalogs(keyword = 'STRIPE82_COADD', sdssmask=False)
    
    #des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_00', gold=True)
    #des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_ALLCOLUMNS_EA', gold=True)
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_0', gold=True)
    merged_des = mergeCatalogsUsingPandas(des=full_des_data, gold=des_gold, key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
    merged_des1 = Cuts.doBasicCuts(merged_des.copy(), raTag = 'RA', decTag='DEC', object = None)
    mags = ['MAG_MODEL', 'MAG_DETMODEL']
    merged_des1 = getCorrectedMag( merged_des1, mags = mags, reddening = None )
    merged_des2 = AddingReddening( merged_des1 )
    merged_des1 = 0
    merged_des2 = getCorrectedMag( merged_des2, mags = ['MAG_APER_4'], reddening = 'SLR' )
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'merged_des_st82.fits', merged_des2)
    full_des_data, merged_des, des_gold = 0, 0, 0

    """
    
    # galaxy
    #merged_des1 = doBasicCuts(merged_des, raTag = 'RA', decTag = 'DEC', object = None)
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    gold_final = fitsio.read(resultpath+'merged_des_st82.fits')
    gold_final_s = Cuts.SpatialCuts(  gold_final, ra = 320.0, ra2=330.0 , dec= -1.0 , dec2=1.0 )
    
    cmass = io.getSGCCMASSphotoObjcat()
    cmass = Cuts.keepGoodRegion(cmass)
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, gold_final )
    clean_cmass_data_des_s = Cuts.SpatialCuts(  clean_cmass_data_des, ra = 320.0, ra2=330.0 , dec= -1.0 , dec2=1.0 )
    
    #prefix = 'gold_small_'
    #prefix = 'gold_st82_'
    #prefix = 'gold_st82_2_'
    #prefix = 'gold_st82_3_'
    #prefix = 'gold_st82_SFD98_'
    #prefix = 'coadd_st82_SLR_'
    #prefix = 'gold_st82_5_' # the best one so far
    #prefix = 'gold_st82_6_'
    prefix = 'gold_st82_7_'
    prefix = 'gold_st82_9_'
    prefix = 'gold_st82_11_small_'
    
    
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(gold_final, gold_final, [0.9,0.1], random_state=0)
    des_train = gold_final_s[trainInd]
    des_test = gold_final_s[testInd]
    
    result_gold_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=gold_final, test=gold_final, prefix = prefix, mock=True)


    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'result_gold_st82_11_small.fits', result_gold_st82)
    fitsio.write(resultpath+'clean_cmass_data_des_gold_st82_11_small.fits', clean_cmass_data_des_s)
    #fitsio.write(resultpath+'clean_cmass_data_des_gold_st82_9_2.fits', clean_cmass_data_des)


    """
    # 2) dmass, slr (test with coadd obects table ( use slrzeroshift instead of XDCORR )---------
    full_des_data = io.getDESY1A1catalogs(keyword = 'STRIPE82_COADD', sdssmask=False)
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_0', gold=True)
    des = Cuts.doBasicCuts(full_des_data, raTag = 'RA', decTag='DEC', object = 'galaxy')
    des = AddingReddening(des) # add SLR_SHIFT column
    des = getCorrectedMag( des, reddening = 'SLR' )
    des = addphotoz(des = des, im3shape=des_gold)
    
    full_des_data = 0
    
    cmass = io.getSGCCMASSphotoObjcat()
    cmass = Cuts.keepGoodRegion(cmass)
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, des )

    prefix = 'coadd_st82_SLR_'
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(des, des, [0.05,0.95], random_state=0)
    des_train = des[trainInd]
    des_test = des[testInd]
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, des )
    result_coadd_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=des_train, test=des_test, prefix = prefix, mock=True )
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'result_coadd_st82_SLR.fits', result_coadd_st82)
    fitsio.write(resultpath+'clean_cmass_data_des_coadd_st82_SLR.fits', clean_cmass_data_des)
    Xtrue,_ = mixing_color( des  )
    fitsio.write(resultpath+prefix+'X_obs.fits', Xtrue)
    
    
    # 3) gold, SFD98 ---------
    full_des_data = io.getDESY1A1catalogs(keyword = 'STRIPE82_COADD', sdssmask=False)
    #des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_0', gold=True)
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_ALLCOLUMNS_EA', gold=True)
    des_gold = Cuts.keepGoodRegion(des_gold)
    # replace slr corrected gold mag to raw magnitude.
    
    merged_des = mergeCatalogsUsingPandas(des=full_des_data, gold=des_gold, key='COADD_OBJECTS_ID', suffixes = ['_DES',''])
    merged_des1 = Cuts.doBasicCuts(merged_des, raTag = 'RA', decTag='DEC', object = None)
    merged_des1 = getRawMag( merged_des1, reddening = 'SLR')
    merged_des1 = getCorrectedMag( merged_des1, reddening = 'SFD98')
    full_des_data, des_gold = 0, 0
    
    cmass = io.getSGCCMASSphotoObjcat()
    cmass = Cuts.keepGoodRegion(cmass)
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, merged_des1 )


    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(merged_des1, merged_des1, [0.1,0.9], random_state=0)
    des_train = merged_des1[trainInd]
    des_test = merged_des1[testInd]

    result_gold_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=des_train, test=des_test, prefix = 'small_', mock=True)
    
    prefix = 'gold_st82_SFD98_2_'
    presuffix = 'gold_st82_SFD98_2'

    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'result_'+presuffix+'.fits', result_gold_st82)
    fitsio.write(resultpath+'clean_cmass_data_des_'+presuffix+'.fits', clean_cmass_data_des)




    # 4) dmass, SFD98 (test with coadd obects table ( use slrzeroshift instead of XDCORR )---------
    full_des_data = io.getDESY1A1catalogs(keyword = 'STRIPE82_COADD', sdssmask=False)
    des = Cuts.doBasicCuts(full_des_data, raTag = 'RA', decTag='DEC', object = 'galaxy')
    #des = AddingReddening(des) # add SLR_SHIFT column
    des = getCorrectedMag( des, reddening = 'SFD98' )
    des_gold = io.getDESY1A1catalogs(keyword = 'Y1A1_GOLD_STRIPE82_0', gold=True)
    des = addphotoz( des = des, im3shape = des_gold)
    full_des_data = 0
    
    cmass = io.getSGCCMASSphotoObjcat()
    cmass = Cuts.keepGoodRegion(cmass)
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, des )
    
    prefix = 'coadd_st82_SFD98_'
    (trainInd, testInd), (sdsstrainInd,sdsstestInd) = split_samples(des, des, [0.05,0.95], random_state=0)
    des_train = des[trainInd]
    des_test = des[testInd]
    clean_cmass_data, clean_cmass_data_des = DES_to_SDSS.match( cmass, des )
    result_coadd_st82 = XDGMM_model(clean_cmass_data_des, clean_cmass_data_des, train=des_train, test=des_test, prefix = prefix, mock=True )
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    fitsio.write(resultpath+'result_coadd_st82_SFD98.fits', result_coadd_st82)
    fitsio.write(resultpath+'clean_cmass_data_des_coadd_st82_SFD98.fits', clean_cmass_data_des)
    #Xtrue,_ = mixing_color( des  )
    #fitsio.write(resultpath+prefix+'X_obs.fits', Xtrue)

    """

    # Test resulted population in various aspects ===================================

    # hist in color spaces
    presuffix = 'coadd_st82_SLR'
    presuffix = 'gold_st82_SFD98'
    presuffix = 'coadd_st82_SFD98'
    presuffix = 'gold_st82_5'
    presuffix = 'gold_st82_9' # 340 - 360
    presuffix = 'gold_st82_9_2' # stripe82
    prefix = 'gold_st82_11_small_' # 320-330, -1-1
    presuffix = 'gold_st82_11_small'
    
    resultpath = '/n/des/lee.5922/Dropbox/repositories/CMASS/code/result_cat/'
    
    clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_'+'gold_st82_9_2'+'.fits')
    #result_gold_st82 = fitsio.read(resultpath+'result_'+'gold_st82_9_2'+'.fits')
    result_gold_st82 = fitsio.read(resultpath+'result_'+presuffix+'.fits')
    result_gold_st82 = fitsio.read(resultpath+'result_gold_st82_11_small_2.fits')
    #result_gold_st82 = fitsio.read(resultpath+'result_gold_st82_nomax.fits')
    #result_gold_st82_s = Cuts.SpatialCuts(result_gold_st82, ra = 320, ra2 = 330, dec=-1, dec=1)
    probability_calibration( des = result_gold_st82, cmass_des = clean_cmass_data_des, prefix=prefix)



    result_gold_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/result_gold_y1a1_all.fits')
    result_gold_st82 = result_gold_y1a1[result_gold_y1a1['DEC'] > -3 ]
    result_gold_spt = result_gold_y1a1[result_gold_y1a1['DEC'] < -3 ]
    gold_final = fitsio.read(resultpath+'merged_des_st82.fits')
    #result_st82 = addphotoz( des = result_st82, im3shape = gold_final)

    result_gold_st82 = result_gold_y1a1[result_gold_y1a1['DEC'] > -3]
    result_gold_spt = result_gold_y1a1[result_gold_y1a1['DEC'] < -3]
    
    dmass_gold_y1a1, _ = resampleWithPth(result_gold_y1a1, pstart = 0.2 )
    dmass_gold_st82, _ = resampleWithPth(result_gold_st82, pstart = 0.05, pmax = 0.9 )
    dmass_gold_spt, _ = resampleWithPth(result_gold_spt, pstart = 0.125 )


    cmass_des, _= matchCatalogs( result_gold_st82, clean_cmass_data_des , tag = 'COADD_OBJECTS_ID')
    dmass_gold_st82 = resamplewithCmassHist(result_gold_st82, clean_cmass_data_des, des_st82 = result_gold_st82,   pstart = 0.01 )
    dmass_gold_spt = resamplewithCmassHist(result_gold_spt, clean_cmass_data_des, des_st82 = result_gold_st82, pstart = 0.125 )
    
    
    probability_calibration( des = result_gold_st82, cmass_des = clean_cmass_data_des, prefix='gold_st82')
    
    from systematics_module.contCorrection import *
    from systematics_module import contCorrection
    def getavgbias( cat, pstart=0.1 ):
        mag = cat['MAG_MODEL_I_corrected']
        z = cat['DESDM_ZP']
        logL = logL_from_mag( mag = mag, z = z )
        #logL = logL[logL > 9.0]
        avg_b = logL_to_galaxyBias(logL = logL)
        print 'pstart=',pstart, ' avg bias=',avg_b, ' sample size=', mag.size
        return avg_b
    
    avg_g = getavgbias( clean_cmass_data_des )
    avg_g = getavgbias( dmass_gold_st82 )
    avg_g = getavgbias( dmass_gold_spt )
    
    
    ## getting bias for diferent pth_
    
    
    from systematics_module.contCorrection import *
    from systematics_module import contCorrection
    
    #result_gold_st82 = fitsio.read(resultpath+'result_gold_st82_11_small_2.fits')
    
    #mags = ['MAG_DETMODEL', 'MAG_MODEL', 'MAG_APER_4']
    #dmass_coadd_st82 = getCorrectedMag( dmass_coadd_st82, mags = mags, reddening = None , suffix='_corrected')
    
    
    
    def getavgbias( cat, pstart=0.1 ):
        mag = cat['MAG_MODEL_I_corrected']
        z = cat['DESDM_ZP']
        logL = logL_from_mag( mag = mag, z = z )
        #logL = logL[logL > 9.0]
        avg_b = logL_to_galaxyBias(logL = logL)
        print 'pstart=',pstart, ' avg bias=',avg_b, ' sample size=', mag.size
        return avg_b
      
      
      
    import healpy as hp
    # call healpix map
    GoldMask = callingEliGoldMask()
    GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0 ]
    GoldMask_spt = GoldMask[ GoldMask['DEC'] < -3.0 ]

    pixarea = hp.nside2pixarea( 4096, degrees = True)
    sptnpix = GoldMask_spt['PIXEL'].size #hp.get_map_size( GoldMask_spt['PIXEL'] )
    st82npix =  GoldMask_st82['PIXEL'].size # hp.get_map_size( GoldMask_st82 )
    SPTMaparea = pixarea * sptnpix
    ST82Maparea = pixarea * st82npix
      
    
    pbin = np.linspace(0.01, 0.2, 11 )
    n_cmass = clean_cmass_data_des.size *1./ST82Maparea
    avg_bs = np.zeros(pbin.size)
    num_density = np.zeros(pbin.size)
    
    
    dmass_sample = []
    for i, p in enumerate(pbin):
        sample, _ = resampleWithPth(result_gold_st82, pstart = p, pmax = 0.7)
        #sample = resamplewithCmassHist(result_gold_st82, clean_cmass_data_des, des_st82 = result_gold_st82, pstart = p )
        dmass_sample.append(sample)
        avg_bs[i] = getavgbias( dmass_sample[i], pstart = p )
        num_density[i] = dmass_sample[i].size * 1./ST82Maparea
        #print p, avg_bs[i], num_density[i]

        
    DAT = np.column_stack(( pbin, avg_bs, num_density ))
    np.savetxt( resultpath+'bias_pcut.txt', DAT, delimiter = ' ', header = 'pbin, avg_bs, num_density  # dmass spt' )
    
    
    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots()
    data = np.loadtxt(  resultpath+'bias_pcut.txt')
    pbin, avg_bs, num_density = data[:,0], data[:,1], data[:,2]
    for i, p in enumerate(pbin):
        ax.scatter( p, avg_bs[i], marker = 'o', color = 'black')
        ax2.scatter( p, num_density[i], marker = 'o', color = 'black')
        
    #avg_b = getavgbias( dmass_coadd_st82_2, pstart = 0.01 )
    #ax.axhline( y = avg_b, color = 'blue', linestyle = '--', label = 'coadd' )
    #ax2.axhline( y = 11646 * 0.9 * 1./n_cmass, color = 'blue', linestyle = '--', label = 'coadd' )
    avg_g = getavgbias( clean_cmass_data_des )
    ax.axhline( y = 1.17090969569, color = 'red', label = 'cmass' )
    ax2.axhline( y = 72.926946637338176, color = 'red', linestyle = '--', label = 'cmass' )

    ax.set_xlabel('p start')
    ax.set_ylabel('bias')
    ax2.set_xlabel('p start')
    ax2.set_ylabel('number ratio')
    ax.legend(loc='best')
    ax2.legend(loc='best')
    fig.savefig('figure/bias.png')
    fig2.savefig('figure/numdensity.png')
    
    n_dmass = dmass_gold_spt.size * 1./SPTMaparea
    avg = getavgbias( dmass_gold_spt )

    
    m1, m2 = esutil.numpy_util.match(dmass_coadd_st82['COADD_OBJECTS_ID'], gold_final['COADD_OBJECTS_ID'])
    not_in_gold_mask = np.ones(dmass_coadd_st82.size, dtype=bool)
    not_in_gold_mask[m1] = 0
    not_in_gold = dmass_coadd_st82[not_in_gold_mask]
    #not_in_gold_z = addphotoz(des=not_in_gold, im3shape=gold_final)
    ## there must be bugs..

    m, mm = esutil.numpy_util.match(not_in_gold['COADD_OBJECTS_ID'], gold_final['COADD_OBJECTS_ID'])


    clean_cmass_data_des = getRawMag( clean_cmass_data_des, reddening = 'SLR' )
    mags = ['MAG_DETMODEL', 'MAG_MODEL']
    clean_cmass_data_des = getCorrectedMag( clean_cmass_data_des, mags = mags, reddening = 'SFD98' )
    not_in_gold = getCorrectedMag( not_in_gold, mags = mags, reddening = 'SFD98' )
    
    A_not,_ = mixing_color(not_in_gold)
    A_true,_ = mixing_color(clean_cmass_data_des)
    doVisualization( A_not, A_true, labels = labels, nbins=100, ranges = ranges, prefix=presuffix+'test_')
    doVisualization_z( cats = [clean_cmass_data_des, not_in_gold], labels=['cmass', 'not'], suffix = 'test')
    
    
    
    # -----------
    
    probability_calibration( des = result_gold_st82, cmass_des = clean_cmass_data_des, prefix = 'test' )
    probability_calibration( des = result_st82, cmass_des = clean_cmass_data_des, prefix = 'test2' )
    
    #result_y1a1 = fitsio.read('/n/des/lee.5922/data/y1a1_coadd/dmass_y1a1.fits')
    

    As_X, _ = mixing_color( dmass_gold_st82 )
    Xtrue,_ = mixing_color( clean_cmass_data_des )
    #Xall = fitsio.read(resultpath+presuffix+'_noisy_X_sample.fits')
    #X_obs = fitsio.read(resultpath+presuffix+'_X_obs_all.fits')
    
    presuffix = 'st82'
    
    labels = ['MAG_MODEL_R', 'MAG_MODEL_I', 'g-r', 'r-i', 'i-z']
    ranges =  [[17,22], [17,22], [0,2], [-.5,1.5], [0.0,.8]]
    #doVisualization_1d( Xtrue, As_X, labels = labels, ranges = ranges, nbins=100, prefix=presuffix+'_')
    doVisualization( As_X, Xtrue, labels = labels, nbins=100, name = ['dmass spt','cmass st82' ], ranges = ranges, prefix=presuffix+'_')
    #doVisualization( Xall, X_obs, labels = labels, nbins=100, ranges = ranges, prefix=presuffix+'_all_')
    doVisualization_z( cats = [clean_cmass_data_des, dmass_gold_st82], labels=['cmass', 'dmass'], suffix = presuffix)
    

    
    # angular
    from systematics import *
    from systematics_weight import doVisualization_Angcorr
    GoldMask = callingEliGoldMask()
    GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0 ]
    GoldMask_spt = GoldMask[ GoldMask['DEC'] < -3.0 ]
    we = GoldMask_spt['FRAC']
    angular_correlation(data = dmass_gold_spt, rand = GoldMask_spt, weight = [None, we], suffix = '_gold_spt')
    
    clean_cmass_data_des = fitsio.read(resultpath+'clean_cmass_data_des_'+'gold_st82_9_2'+'.fits')
    angular_correlation(data = clean_cmass_data_des, rand = GoldMask_st82, weight = [None, we], suffix = '_gold_st82_9_2_cmass')

    random = fitsio.read('/n/des/lee.5922/data/cmass_cat/'+'random0_DR12v5_CMASS_South.fits.gz')
    clean_random = Cuts.keepGoodRegion(random)
    angular_correlation(data = dmass_gold_st82, rand = clean_random, weight = [None, None], suffix = '_gold_st82_9_2_cmass_random')
    
    
    labels = ['gold_st82_9_2', 'gold_spt', 'gold_st82_9_2_cmass']
    name = ['dmass st82', 'dmass spt', 'cmass st82']
    doVisualization_Angcorr( labels = labels, name = name )


    #angular_correlation(data = cmass, rand = clean_random, weight = [None, None ], suffix = '_cmass_st82')
    #angular_correlation(data = dmass_gold_stripe82 , rand = clean_random, weight = [None, None ], suffix = '_gold_st82_SFD98_random')





    # =============================













