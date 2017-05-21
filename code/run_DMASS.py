from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def priorCut_test(data):
    modelmag_g_des = data['MAG_DETMODEL_G']
    modelmag_r_des = data['MAG_DETMODEL_R']
    modelmag_i_des = data['MAG_DETMODEL_I']
    cmodelmag_g_des = data['MAG_MODEL_G']
    cmodelmag_r_des = data['MAG_MODEL_R']
    cmodelmag_i_des = data['MAG_MODEL_I']
    magauto_des = data['MAG_AUTO_I']

    cut = (((cmodelmag_r_des > 17) & (cmodelmag_r_des <24)) &
           ((cmodelmag_i_des > 17) & (cmodelmag_i_des <24)) &
           ((cmodelmag_g_des > 17) & (cmodelmag_g_des <24)) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) & # 10122 (95%)
           ((modelmag_r_des - modelmag_i_des ) > 0. ) & # 10120 (95%)
           ((modelmag_g_des - modelmag_r_des ) > 0. ) & # 10118 (95%)
           ((modelmag_g_des - modelmag_r_des ) < 2.5 ) & # 10122 (95%)
           (magauto_des < 21. ) #&  10124 (95%)
        )
    return cut



def main():
    
    # calling map --------------------
    GoldMask = callingEliGoldMask()
    GoldMask_st82 = GoldMask[ GoldMask['DEC'] > -3.0 ]
    GoldMask_spt = GoldMask[ GoldMask['DEC'] < -3.0 ]

    pixarea = hp.nside2pixarea( 4096, degrees = True)
    sptnpix = GoldMask_spt['PIXEL'].size #hp.get_map_size( GoldMask_spt['PIXEL'] )
    st82npix =  GoldMask_st82['PIXEL'].size # hp.get_map_size( GoldMask_st82 )
    SPTMaparea = pixarea * sptnpix
    ST82Maparea = pixarea * st82npix

    
    # calling stripe82 gold catalogue -----------------------------
    path = '/n/des/lee.5922/data/gold_cat/'

    columns = ['FLAGS_GOLD', 'FLAGS_BADREGION', 'MAG_MODEL_G', 'MAG_MODEL_R', 'MAG_MODEL_I', 'MAG_MODEL_Z',\
               'MAG_DETMODEL_G', 'MAG_DETMODEL_R', 'MAG_DETMODEL_I', 'MAG_DETMODEL_Z', 'MAGERR_DETMODEL_G',\
               'MAGERR_DETMODEL_R', 'MAGERR_DETMODEL_I', 'MAGERR_DETMODEL_Z', 'MAGERR_MODEL_G', 'MAGERR_MODEL_R',\
               'MAGERR_MODEL_I', 'MAGERR_MODEL_Z', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'RA',\
               'DEC', 'COADD_OBJECTS_ID', 'MODEST_CLASS', 'HPIX', 'DESDM_ZP']

    gold_st82 = io.SearchAndCallFits(path = path, columns = columns, keyword = 'Y1A1_GOLD_STRIPE82_v2')
    gold_st82 = gold_st82[gold_st82['MODEST_CLASS'] == 1]
    gold_st82 = Cuts.keepGoodRegion(gold_st82)

    # flags and color cut
    mask_all = (gold_st82['FLAGS_GOLD'] == 0 )&(priorCut_test(gold_st82))
    gold_st82 = gold_st82[mask_all]


    # calling BOSS cmass and applying dmass goodregion mask ----------------------------
    cmass = io.getSGCCMASSphotoObjcat()
    print 'num of cmass in sgc region', cmass.size
    cmass = Cuts.keepGoodRegion(cmass)
    print 'num of cmass after des veto', cmass.size


    # find cmass in des_gold side --------------------
    mg1, mg2, _ = esutil.htm.HTM(10).match(cmass['RA'], cmass['DEC'], gold_st82['RA'], \
                                         gold_st82['DEC'],2./3600, maxmatch=1)
    cmass_mask = np.zeros(gold_st82.size, dtype=bool)
    cmass_mask[mg2] = 1
    clean_cmass_data_des, nocmass = gold_st82[cmass_mask], gold_st82[~cmass_mask]
    print 'num of cmass in des side', clean_cmass_data_des.size, '({:0.0f}%)'.format(clean_cmass_data_des.size*1./cmass.size * 100)


    # Divide sample into train and test -------------------------
    #(trainInd, testInd), _ = split_samples(merged_des_st82_s, merged_des_st82_s, [0.9,0.1], random_state=0)
    #des_train = merged_des_st82_s[trainInd]
    #des_test = merged_des_st82_s[testInd]
    no_train = 0.2
    #train_ind = np.random.choice( merged_des_st82.size, size = int(merged_des_st82.size * no_train))
    #np.savetxt('data_txt/random_index30.txt', np.array(train_ind))
    train_ind = np.array([ int(i) for i in np.loadtxt('data_txt/random_index30.txt')])
    train_mask = np.zeros(gold_st82.size, dtype = bool)
    train_mask[train_ind] = 1
    des_train = gold_st82[train_mask]
    des_test = gold_st82[~train_mask]

    mt1, mt2 = esutil.numpy_util.match(clean_cmass_data_des['COADD_OBJECTS_ID'], des_train['COADD_OBJECTS_ID'])
    cmass_mask = np.zeros(des_train.size, dtype=bool)
    cmass_mask[mt2] = 1
    cmass_train, nocmass_train = des_train[cmass_mask], des_train[~cmass_mask]

    me1, me2 = esutil.numpy_util.match(clean_cmass_data_des['COADD_OBJECTS_ID'], des_test['COADD_OBJECTS_ID'])
    cmass_mask = np.zeros(des_test.size, dtype=bool)
    cmass_mask[me2] = 1
    cmass_test, _ = des_test[cmass_mask], des_test[~cmass_mask]

    print \
    clean_cmass_data_des.size * 1./gold_st82.size, \
    cmass_train.size*1./des_train.size, \
    cmass_test.size*1./des_test.size # test is always small..why?


    # Fitting ----------------------------------------------
    from xd import _FindOptimalN
    n_cmass, _, _ = _FindOptimalN( np.arange(2, 10, 2), clean_cmass_data_des, pickleFileName = 'pickle/optimal_n_cmass30.pkl', suffix = '')
    n_no,_,_ = _FindOptimalN( np.arange(20,30, 2), nocmass_train, pickleFileName = 'pickle/optimal_n_no30.pkl', suffix = '')

    pickleFileName = 'pickle/gold_st82_30_cut21_XD_no.pkl'                  
    clf_no = XD_fitting( nocmass_train, pickleFileName = pickleFileName, \
                      init_params=None, suffix = '', n_cl = n_no )
    #pickleFileName = 'pickle/gold_st82_28_cut21_XD_cmass_tot.pkl'
    #clf_cmass = XD_fitting( clean_cmass_data_des, pickleFileName = pickleFileName,\
    #                       suffix = '', n_cl = n_cmass )
    pickleFileName = 'pickle/gold_st82_30_cut21_XD_cmass.pkl'
    clf_cmass = XD_fitting( cmass_train, pickleFileName = pickleFileName,\
                           suffix = '', n_cl = n_cmass )


    # assign membership prob ----------------------------------
    cmass_fraction = cmass_train.size*1./des_train.size 
    #cmass_fraction = clean_cmass_data_des.size *1./merged_des_st82_s.size
    print 'cmass_fraction', cmass_fraction
    from xd import assignCMASSProb
    merged_des_st82_s = assignCMASSProb( merged_des_st82, clf_cmass, clf_no, cmass_fraction = cmass_fraction )
    des_train = merged_des_st82_s[train_mask]
    des_test = merged_des_st82_s[~train_mask]



    # resampling with assigned membership probability -------------------
    dmass_train, _ = resampleWithPth( des_train, pstart = 0, pmax = 1.0 )
    print 100. * dmass_train.size/ cmass_train.size, '%'
    dmass_test, _ = resampleWithPth( des_test, pstart = 0, pmax = 1.0 )
    print 100. * dmass_test.size/ cmass_test.size, '%'
    dmass, _ = resampleWithPth( merged_des_st82_s, pstart = 0, pmax = 1.0 )
    print 100. * dmass.size/ clean_cmass_data_des.size, '%'

    print '--------------------------\n Fitting End\n---------------------------'


    # calling spt des_gold ---------------------------------------------
    des_spt = io.SearchAndCallFits(path = path, keyword = 'Y1A1_GOLD_00')
    des_spt = des_spt[des_spt['MODEST_CLASS'] == 1]
    des_spt = Cuts.keepGoodRegion(des_spt)
    des_spt = des_spt[des_spt['DEC'] < -3]
    mask_y1a1 = (des_spt['FLAGS_GOLD'] == 0 )&(priorCut_test(des_spt))
    des_spt = des_spt[mask_y1a1]


    #assign prob to spt ----------------------
    # dmass from spt
    rabin = np.linspace(des_spt['RA'].min(), des_spt['RA'].max(), 15)
    ind_map = np.digitize(des_spt['RA'], bins = rabin)

    des_spt_list = []
    for i in range(1, rabin.size):
        ts = assignCMASSProb(des_spt[ind_map == i] , clf_cmass, clf_no, cmass_fraction = cmass_fraction )
        des_spt_list.append(ts)
        ts = None
    des_spt = np.hstack(des_spt_list)

    # resampling
    dmass_spt, _ = resampleWithPth( des_spt, pstart = 0, pmax = 1.0 )

    #save dmass
    fitsio.write('result_cat/dmass_spt_30.fits', dmass_spt)









