from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

import sys, os
import argparse

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys


def priorCut_test(data):
    print 'CHECK input catalog has only galaxies'
    ## Should add MODEST_CLASS cut later. 
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



def _main_fitting():
    
    """ deprecate later """
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
    print '\n--------------------------------\n applying DES veto mask to CMASS\n--------------------------------'
    cmass = Cuts.keepGoodRegion(cmass)
    print 'num of cmass after des veto', cmass.size

    print '\n--------------------------------\n matching catalogues\n--------------------------------'
    # find cmass in des_gold side --------------------
    mg1, mg2, _ = esutil.htm.HTM(10).match(cmass['RA'], cmass['DEC'], gold_st82['RA'], \
                                         gold_st82['DEC'],2./3600, maxmatch=1)
    cmass_mask = np.zeros(gold_st82.size, dtype=bool)
    cmass_mask[mg2] = 1
    clean_cmass_data_des, nocmass = gold_st82[cmass_mask], gold_st82[~cmass_mask]
    print 'num of cmass in des side', clean_cmass_data_des.size, '({:0.0f}%)'.format(clean_cmass_data_des.size*1./cmass.size * 100)

    print '\n--------------------------------\n Extreme deconvolution fitting\n--------------------------------'
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
    print '\n--------------------------------\n Assign membership prob\n--------------------------------'
    cmass_fraction = cmass_train.size*1./des_train.size 
    #cmass_fraction = clean_cmass_data_des.size *1./merged_des_st82_s.size
    print 'cmass_fraction', cmass_fraction
    from xd import assignCMASSProb
    gold_st82 = assignCMASSProb( gold_st82, clf_cmass, clf_no, cmass_fraction = cmass_fraction )
    des_train = gold_st82[train_mask]
    des_test = gold_st82[~train_mask]



    # resampling with assigned membership probability -------------------
    print '\n--------------------------------\n resampling\n--------------------------------'
    dmass_train, _ = resampleWithPth( des_train, pstart = 0, pmax = 1.0 )
    print 100. * dmass_train.size/ cmass_train.size, '%'
    dmass_test, _ = resampleWithPth( des_test, pstart = 0, pmax = 1.0 )
    print 100. * dmass_test.size/ cmass_test.size, '%'
    dmass, _ = resampleWithPth( gold_st82, pstart = 0, pmax = 1.0 )
    print 100. * dmass.size/ clean_cmass_data_des.size, '%'

    print '\n--------------------------\n End\n---------------------------'


    
def prepare_cmass_sample(params):
    
    cmass = io.getSGCCMASSphotoObjcat()
    print 'num of cmass in sgc region', cmass.size
    print '\n--------------------------------\n applying DES veto mask to CMASS\n--------------------------------'
    cmass = Cuts.keepGoodRegion(cmass)
    print 'num of cmass after des veto', cmass.size

    return cmass   

    
def train_st82(params, param_file):
    
    # calling params 
    output_dir = params['output_dir']
    #cmass_fraction = params['cmass_fraction']
    train_sample_filename = params['train_sample']
    cmass_pickle = output_dir + params['cmass_pickle']
    no_pickle = output_dir + params['no_pickle']
    #out_catname = params['out_catname']
    #out_resampled_cat = params['out_resampled_cat']
    #input_path = params['input_cat_dir']
    #input_keyword = params['input_cat_keyword']
    
    
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
    #cmass = io.getSGCCMASSphotoObjcat()
    train_sample = esutil.io.read(train_sample_filename)
    print 'total num of train', train_sample.size
    print '\n--------------------------------\n applying DES veto mask to CMASS\n--------------------------------'   
    train_sample = Cuts.keepGoodRegion(train_sample)
    print 'num of train_sample after des veto', train_sample.size

    print '\n--------------------------------\n matching catalogues\n--------------------------------'
        
    # find cmass in des_gold side --------------------
    mg1, mg2, _ = esutil.htm.HTM(10).match(train_sample['RA'], train_sample['DEC'], gold_st82['RA'], \
                                         gold_st82['DEC'],2./3600, maxmatch=1)
    cmass_mask = np.zeros(gold_st82.size, dtype=bool)
    cmass_mask[mg2] = 1
    nocmass =  gold_st82[~cmass_mask]

    train_sample = train_sample[mg1]
    clean_cmass_data_des = gold_st82[mg2]

    w_sgc = train_sample['WEIGHT_FKP']*train_sample['WEIGHT_SYSTOT'] *( train_sample['WEIGHT_CP'] + train_sample['WEIGHT_NOZ'] - 1. )
    clean_cmass_data_des = appendColumn(clean_cmass_data_des, name = 'CMASS_WEIGHT', value = w_sgc)

    print 'num of cmass in des side', clean_cmass_data_des.size, '({:0.0f}%)'.format(clean_cmass_data_des.size*1./train_sample.size * 100)
    print 'num of non-cmass in des side ', nocmass.size

    if params['random_sampling'] : 
        random_sampling_ind = np.random.choice(np.arange(nocmass.size), size = nocmass.size/10)
        nocmass = nocmass[random_sampling_ind]
        print 'num of randomly sampled non-cmass ', nocmass.size

    cmass_fraction = clean_cmass_data_des.size *1./gold_st82.size
    print 'cmass_fraction', cmass_fraction
    f = open(output_dir+'cmassfrac', 'w')
    f.write('{0:.10f}'.format(cmass_fraction))

    gold_st82 = None  # initialize called catalogs to save memory

    #params['cmass_fraction'] = cmass_fraction
    print '\n--------------------------------\n Extreme deconvolution fitting\n--------------------------------'
    # Fitting ----------------------------------------------

    n_cmass, n_no = None, None
    if 'n_cmass' in params : n_cmass = params['n_cmass']  
    if 'n_no' in params : n_no = params['n_no'] 
    #clf_cmass = XD_fitting( data = clean_cmass_data_des, pickleFileName = cmass_pickle, n_cl = n_cmass )                 
    #clf_no = XD_fitting( data = nocmass, pickleFileName = no_pickle , n_cl = n_no)


    ncomp, xamp, xmean, xcovar = loadpickle( 'output/n2/gold_st82_XD_cmass.pkl')
    extreme_fitting(clean_cmass_data_des, n_comp = n_cmass, weight=True, xamp=xamp, xmean=xmean, xcovar=xcovar, pickle_name = cmass_pickle, log=output_dir+'xd_log/cmass')
    ncomp, xamp, xmean, xcovar = loadpickle( 'output/n2/gold_st82_XD_no.pkl')
    extreme_fitting(nocmass, n_comp = n_no, xamp=xamp, xmean=xmean, xcovar=xcovar, pickle_name = no_pickle, log=output_dir+'xd_log/no_cmass')

    print '\n--------------------------\n Fitting End\n---------------------------'

    
def main_st82(params):
    
    output_dir = params['output_dir']

    cmass_fraction = params['cmass_fraction']
    cmass_pickle = output_dir + params['cmass_pickle']
    no_pickle = output_dir + params['no_pickle']
    out_catname = output_dir + params['out_catname']
    out_resampled_cat = output_dir + params['out_resampled_cat']
    input_path = params['input_cat_dir']
    input_keyword = params['input_cat_keyword']


    columns = ['FLAGS_GOLD', 'FLAGS_BADREGION', 'MAG_MODEL_G', 'MAG_MODEL_R', 'MAG_MODEL_I', 'MAG_MODEL_Z',\
               'MAG_DETMODEL_G', 'MAG_DETMODEL_R', 'MAG_DETMODEL_I', 'MAG_DETMODEL_Z', 'MAGERR_DETMODEL_G',\
               'MAGERR_DETMODEL_R', 'MAGERR_DETMODEL_I', 'MAGERR_DETMODEL_Z', 'MAGERR_MODEL_G', 'MAGERR_MODEL_R',\
               'MAGERR_MODEL_I', 'MAGERR_MODEL_Z', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'RA',\
               'DEC', 'COADD_OBJECTS_ID', 'MODEST_CLASS', 'HPIX', 'DESDM_ZP']

    gold_st82 = io.SearchAndCallFits(path = input_path, columns = columns, keyword = input_keyword)
    gold_st82 = gold_st82[gold_st82['MODEST_CLASS'] == 1]
    gold_st82 = Cuts.keepGoodRegion(gold_st82)

    # flags and color cut
    mask_all = (gold_st82['FLAGS_GOLD'] == 0 )&(priorCut_test(gold_st82))
    gold_st82 = gold_st82[mask_all]
    
    clf_cmass = XD_fitting( None, pickleFileName = cmass_pickle)               
    clf_no = XD_fitting( None, pickleFileName = no_pickle)
    
    # assign membership prob ----------------------------------
    print '\n--------------------------------\n Assign membership prob\n--------------------------------'
    print 'cmass_fraction', cmass_fraction
    from xd import assignCMASSProb
    gold_st82 = assignCMASSProb( gold_st82, clf_cmass, clf_no, cmass_fraction = cmass_fraction )
    fitsio.write(out_catname, gold_st82)
    
    # resampling with assigned membership probability -------------------
    print '\n--------------------------------\n resampling\n--------------------------------'
    dmass, _ = resampleWithPth( gold_st82, pstart = 0.01, pmax = 1.0 )
    fitsio.write(out_resampled_cat, dmass)

def construct_jk_catalog_ind( cat, njack = 10, root='./', jtype = 'generate', jfile = 'jkregion.txt', suffix = '' ):

    print '\n--------------------------------\n catalog jackknife sampling \n--------------------------------'
    print 'jfile= ', root+jfile
    print 'njack= ', njack
    
    km, jfile = GenerateRegions(cat, cat['RA'], cat['DEC'], root+jfile, njack, jtype)
    ind = AssignIndex(cat, cat['RA'], cat['DEC'], km)
    #ind_rand = AssignIndex(rand, rand['RA'], rand['DEC'], km) 
    
    print '--------------------------------'
    return ind

    
def main(params):
      
    output_dir = params['output_dir']
    cmass_fraction = params['cmass_fraction']
    cmass_pickle = output_dir + params['cmass_pickle']
    no_pickle = output_dir + params['no_pickle']
    out_catname = output_dir + params['out_catname']
    out_resampled_cat = output_dir + params['out_resampled_cat']
    input_path = params['input_cat_dir']
    input_keyword = params['input_cat_keyword']

    # calling spt des_gold ---------------------------------------------
    des_spt = io.SearchAndCallFits(path = input_path, keyword = input_keyword)
    des_spt = des_spt[des_spt['MODEST_CLASS'] == 1]
    des_spt = Cuts.keepGoodRegion(des_spt)
    des_spt = des_spt[des_spt['DEC'] < -3]
    mask_y1a1 = (des_spt['FLAGS_GOLD'] == 0 )&(priorCut_test(des_spt))
    des_spt = des_spt[mask_y1a1]

    clf_cmass = XD_fitting( None, pickleFileName = cmass_pickle)               
    clf_no = XD_fitting( None, pickleFileName = no_pickle)
    
    #assign prob to spt ----------------------
    # dmass from spt
    #rabin = np.linspace(des_spt['RA'].min(), des_spt['RA'].max(), 15)
    #ind_map = np.digitize(des_spt['RA'], bins = rabin)
    njack = 10
    ind_map = construct_jk_catalog_ind( des_spt, njack = njack, root = output_dir, suffix = '' )
    
    
    #des_spt_list = []
    dmass_spt = []
    for i in range(njack):
        ts = assignCMASSProb(des_spt[ind_map == i] , clf_cmass, clf_no, cmass_fraction = cmass_fraction )
        #des_spt_list.append(ts)
        #ts = None
        dm, _ = resampleWithPth( ts, pstart = 0.01, pmax = 1.0 )
        dmass_spt.append(dm)
        outname = out_catname.split('.')[0]+'_jk{:03}.fits'.format(i+1)
        fitsio.write(outname, ts)
        print 'jk sample size :', ts.size
        print 'prob cat save to ', outname
    #des_spt = np.hstack(des_spt_list)
    dmass_spt = np.hstack(dmass_spt)
    
    # resampling
    #dmass_spt, _ = resampleWithPth( des_spt, pstart = 0.01, pmax = 1.0 )
    
    #save dmass
    fitsio.write(out_resampled_cat, dmass_spt)
    
    
if __name__=='__main__':


    parser = argparse.ArgumentParser(description='')
    parser.add_argument("parameter_file", help="YAML configuration file")
    args = parser.parse_args()
    try:
        param_file = args.parameter_file
    except SystemExit:
        sys.exit(1)
        
    params = yaml.load(open(param_file))
    output_dir = params['output_dir']


    if not os.path.exists(output_dir) : os.makedirs(output_dir)
    
    cpconfigname = output_dir+'config.yaml' 
    logname = output_dir + 'log'
    if os.path.exists(cpconfigname):
        cpconfigname = output_dir+'config.yaml' +'.v2'
        logname = output_dir + 'log.v2'
        if os.path.exists(cpconfigname):
           cpconfigname = output_dir+'config.yaml' +'.v3'
           logname = output_dir + 'log.v3'

    import transcript
    transcript.start(logname)


    os.system("cp "+param_file+" "+cpconfigname )
    #params = yaml.load(open('config.yaml'))
    if params['fitting'] : 
        train_st82(params, param_file)
    

    f = open( output_dir+params['cmass_fraction'], 'r')
    cmassfrac = float(f.read())
    params['cmass_fraction'] = cmassfrac
    

    if 'cat_area' in params : 
        if params['cat_area']=='st82': main_st82(params)
        elif params['cat_area'] =='spt' : main(params)
        elif params['cat_area'] =='all' : 
            main_st82(params)
            main(params)
    else : main(params)

    transcript.stop()
