from xd import *
from utils import *
import esutil, yaml, sys, os, argparse
import healpy as hp
from systematics import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



def priorCut_test(data):
    ## Should add MODEST_CLASS cut later.
    modelmag_g_des = data['SOF_CM_MAG_CORRECTED_G']
    modelmag_r_des = data['SOF_CM_MAG_CORRECTED_R']
    modelmag_i_des = data['SOF_CM_MAG_CORRECTED_I']
    cmodelmag_g_des = data['SOF_CM_MAG_CORRECTED_G']
    cmodelmag_r_des = data['SOF_CM_MAG_CORRECTED_R']
    cmodelmag_i_des = data['SOF_CM_MAG_CORRECTED_I']
    magauto_des = data['SOF_CM_MAG_CORRECTED_I']

    cut = (((cmodelmag_r_des > 16) & (cmodelmag_r_des <24)) &
           ((cmodelmag_i_des > 16) & (cmodelmag_i_des <24)) &
           ((cmodelmag_g_des > 16) & (cmodelmag_g_des <24)) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) & # 10122 (95%)
           ((modelmag_r_des - modelmag_i_des ) > 0. ) & # 10120 (95%)
           ((modelmag_g_des - modelmag_r_des ) > 0. ) & # 10118 (95%)
           ((modelmag_g_des - modelmag_r_des ) < 2.5 ) & # 10122 (95%)
           (magauto_des < 21.5 ) #&  10124 (95%)
        )
    return cut


    
#def prepare_cmass_sample(params):
#    
#    cmass = io.getSGCCMASSphotoObjcat()
#    print 'num of cmass in sgc region', cmass.size
#    print '\n--------------------------------\n applying DES veto mask to CMASS\n--------------------------------'
#    cmass = Cuts.keepGoodRegion(cmass)
#    print 'num of cmass after des veto', cmass.size
#    return cmass   

    
def train_st82(params, param_file):
    
    # calling params 
    output_dir = params['output_dir']
    #cmass_fraction = params['cmass_fraction']
    train_sample_filename = params['train_cmass_sample']
    cmass_pickle = output_dir + params['cmass_pickle']
    no_pickle = output_dir + params['no_pickle']

    try:
        train_path = params['train_des_path']
        train_keyword = params['train_des_keyword']
    except : 
        train_path = '/fs/scratch/cond0080/DATA/'
        train_keyword = 'Y1A1_GOLD_STRIPE82_v2'

    #out_catname = params['out_catname']
    #out_resampled_cat = params['out_resampled_cat']
    #input_path = params['input_cat_dir']
    #input_keyword = params['input_cat_keyword']
    
    columns = None
    gold_st82 = io.SearchAndCallFits(path = train_path, columns = columns, keyword = train_keyword )
    #gold_st82 = gold_st82[(gold_st82['MODEST_CLASS'] == 1)&(gold_st82['FLAGS_GOLD'] == 0 )]
    goldY3mask_map = fitsio.read('')
    gold_mask = goldY3_masking(cat=dmassY3, valid_hpind=goldY3mask_map['HPIX'])
    gold_st82 = gold_st82[(gold_st82['FLAGS_GOLD'] == 0 ) & gold_mask ]
    #gold_st82 = Cuts.keepGoodRegion(gold_st82)


    if 'SFD98' in params : 
        if params['SFD98'] : 
            print('change reddening corrections from SLR to SFD98')
            gold_st82 = RemovingSLRReddening(gold_st82)
            gold_st82 = AddingSFD98Reddening(gold_st82, kind='STRIPE82')

    # flags and color cut
    #Su code overlap for y3:
    #train sample=CMASS
    #gold_st82=DES
    
    import esutil
    import numpy as np
    
    mask_all = priorCut_test(gold_st82)
    gold_st82 = gold_st82[mask_all]
    
    # calling BOSS cmass and applying dmass goodregion mask ----------------------------
    #cmass = io.getSGCCMASSphotoObjcat()
    train_sample = esutil.io.read(train_sample_filename)

    print('total num of train', train_sample.size)

    print('\n--------------------------------\n applying DES veto mask to CMASS\n--------------------------------')   
    
    train_sample = Cuts.keepGoodRegion(train_sample)
    fitsio.write( output_dir+'/cmass_in_st82.fits', train_sample)

    print('num of train_sample after des veto', train_sample.size)

    print('\n--------------------------------\n matching catalogues\n--------------------------------')

    # find cmass in des_gold side --------------------
    mg1, mg2, _ = esutil.htm.HTM(10).match(cmass_ra, cmass_dec, ra, \
                                         dec,2./3600, maxmatch=1)
    cmass_mask = np.zeros(ra.size, dtype=bool)
    cmass_mask[mg2] = 1
    clean_cmass_data_des_ra, nocmass_ra = ra[cmass_mask], ra[~cmass_mask]
    clean_cmass_data_des_dec, nocmass_dec = dec[cmass_mask], dec[~cmass_mask]


    print('num of cmass in des side',clean_cmass_data_des_ra.size,'({:0.0f}%)'.format(clean_cmass_data_des_ra.size*1./cmass_ra.size*100))

    if params['random_sampling'] : 
	print('random sampling... ')
	print('sample size :', nocmass.size/100)
        random_sampling_ind = np.random.choice(np.arange(nocmass.size), size = nocmass.size/10)
        nocmass = nocmass[random_sampling_ind]
        print('num of randomly sampled non-cmass ', nocmass.size)

    cmass_fraction = clean_cmass_data_des.size *1./gold_st82.size
    print('cmass_fraction', cmass_fraction)

    fitsio.write( output_dir+'/train_sample_des.fits', clean_cmass_data_des)
    fitsio.write( output_dir+'/train_sample_sdss.fits', train_sample[mg1])

    f = open(output_dir+'cmassfrac', 'w')
    f.write('{0:.10f}'.format(cmass_fraction))

    gold_st82 = None  # initialize called catalogs to save memory

    #params['cmass_fraction'] = cmass_fraction
    print('\n--------------------------------\n Extreme deconvolution fitting\n--------------------------------')
    # Fitting ----------------------------------------------

    n_cmass, n_no = None, None
    tol = 1E-5
    if 'n_cmass' in params : n_cmass = params['n_cmass']  
    if 'n_no' in params : n_no = params['n_no'] 
    if 'tol' in params : tol = float(params['tol'])

    init_params_cmass = None
    init_params_no = None
    if 'continue' in params : 
        if params['continue'] : 

            init_params_cmass = cmass_pickle
            cmass_pickle = cmass_pickle+'.update'
            params['cmass_pickle'] = params['cmass_pickle'] + '.update'

            init_params_no = no_pickle
            no_pickle = no_pickle +'.update'
            params['no_pickle'] = params['no_pickle'] + '.update'

            print('resuming from the existing pkl files')
            print(cmass_pickle)
            print(no_pickle)
    
    mag = ['SOF_CM_MAG_CORRECTED', 'SOF_CM_MAG_CORRECTED' ]
    err = ['SOF_CM_MAG_ERR', 'SOF_CM_MAG_ERR']

    clf_cmass = XD_fitting( data = clean_cmass_data_des, pickleFileName = cmass_pickle, mag=mag, err=err,
        n_cl = n_cmass, n_iter = 10000, tol = tol, verbose = True, init_params= init_params_cmass)                 
    clf_no = XD_fitting( data = nocmass, pickleFileName = no_pickle , 
        n_cl = n_no, n_iter = 10000, tol = tol, verbose = True, init_params = init_params_no)

    
    print('\n--------------------------\n Fitting End\n---------------------------')

    
def main_st82(params):
    
    output_dir = params['output_dir']

    cmass_fraction = params['cmass_fraction']
    cmass_pickle = output_dir + params['cmass_pickle']
    no_pickle = output_dir + params['no_pickle']
    out_catname = output_dir + params['out_catname']
    #out_resampled_cat = output_dir + params['out_resampled_cat']
    input_path = params['input_cat_dir']
    input_keyword = params['input_cat_keyword']
    num_mock = 1
    if 'num_mock' in params : 
        num_mock = params['num_mock']

    if os.path.exists(out_catname): 
        print('probability catalog already exists. Use this for sampling.')
        pass

    else : 
        columns = ['FLAGS_GOLD', 'FLAGS_BADREGION', 'MAG_MODEL_G', 'MAG_MODEL_R', 'MAG_MODEL_I', 'MAG_MODEL_Z',\
                   'MAG_DETMODEL_G', 'MAG_DETMODEL_R', 'MAG_DETMODEL_I', 'MAG_DETMODEL_Z', 'MAGERR_DETMODEL_G',\
                   'MAGERR_DETMODEL_R', 'MAGERR_DETMODEL_I', 'MAGERR_DETMODEL_Z', 'MAGERR_MODEL_G', 'MAGERR_MODEL_R',\
                   'MAGERR_MODEL_I', 'MAGERR_MODEL_Z', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'RA',\
                   'DEC', 'COADD_OBJECTS_ID', 'MODEST_CLASS', 'HPIX', 'DESDM_ZP',
                   'SLR_SHIFT_G', 'SLR_SHIFT_R', 'SLR_SHIFT_I', 'SLR_SHIFT_Z', 'SLR_SHIFT_Y', 'EBV']

        gold_st82 = io.SearchAndCallFits(path = input_path, columns = columns, keyword = input_keyword)
        gold_st82 = gold_st82[ (gold_st82['MODEST_CLASS'] == 1) & (gold_st82['FLAGS_GOLD'] == 0 )]
        gold_st82 = Cuts.keepGoodRegion(gold_st82)

        if 'SFD98' in params : 
            if params['SFD98'] : 
                print('change reddening corrections from SLR to SFD98')
                gold_st82 = RemovingSLRReddening(gold_st82)
                gold_st82 = AddingSFD98Reddening(gold_st82, kind='STRIPE82')

        # flags and color cut
        mask_all = priorCut_test(gold_st82)
        gold_st82 = gold_st82[mask_all]
        

        clf_cmass = XD_fitting( None, pickleFileName = cmass_pickle)               
        clf_no = XD_fitting( None, pickleFileName = no_pickle)
    
    # assign membership prob ----------------------------------
    if os.path.exists(out_catname): gold_st82 = fitsio.read(out_catname)
    else : 
        print('\n--------------------------------\n Assign membership prob\n--------------------------------')

        print('cmass_fraction', cmass_fraction)

        from xd import assignCMASSProb
        gold_st82 = assignCMASSProb( gold_st82, clf_cmass, clf_no, cmass_fraction = cmass_fraction )
        fitsio.write(out_catname, gold_st82, clobber=True)
        
    # resampling with assigned membership probability -------------------

    """
    print '\n--------------------------------\n resampling\n--------------------------------'

    print 'make '+str(num_mock)+' catalogs'
    for ii in range( num_mock ):
        dmass, _ = resampleWithPth( gold_st82, pstart = 0.0, pmax = 1.0 )
        print 'dmass sample size ', out_resampled_cat+'_{:04}.fits'.format(ii+1), dmass.size
        fitsio.write(out_resampled_cat+'_{:04}.fits'.format(ii+1), dmass, clobber=True)
    """

def construct_jk_catalog_ind( cat, njack = 10, root='./', jtype = 'generate', jfile = 'jkregion.txt' ):

    print('\n--------------------------------\n catalog jackknife sampling \n--------------------------------')

    print('jfile= ', root+jfile)
    print('njack= ', njack)
    
    km, jfile = GenerateRegions(cat, cat['RA'], cat['DEC'], root+jfile, njack, jtype)
    ind = AssignIndex(cat, cat['RA'], cat['DEC'], km)
    #ind_rand = AssignIndex(rand, rand['RA'], rand['DEC'], km) 
    
    print('--------------------------------')
    return ind


def main_spt(params):
    
    mag = ['SOF_CM_MAG_CORRECTED', 'SOF_CM_MAG_CORRECTED' ]
    err = ['SOF_CM_MAG_ERR', 'SOF_CM_MAG_ERR']
      
    output_dir = params['output_dir']
    cmass_fraction = params['cmass_fraction']
    cmass_pickle = output_dir + params['cmass_pickle']
    no_pickle = output_dir + params['no_pickle']
    out_catname = output_dir + params['out_catname']
    #out_resampled_cat = output_dir + params['out_resampled_cat']
    input_path = params['input_cat_dir']
    input_keyword = params['input_cat_keyword']
    njack = 8
    num_mock = 1
    if 'num_mock' in params : 
        num_mock = params['num_mock']

    jkoutname = out_catname # +'_jk{:03}.fits'.format(1)
    if os.path.exists(jkoutname): 
        print('probability catalog already exists. Use this for sampling.')
        pass
    else : 
        print('jkoutfile doesnt exist')
        print(jkoutname)

        if 'debug' in params : 
            if params['debug'] : 
                print('debugging mode : small sample for the fast calculation.')
                #input_keyword = 'Y1A1_GOLD_000001'
                des_spt = io.SearchAndCallFits(path = input_path, keyword = input_keyword)
                randind = np.random.choice( np.arange(des_spt.size), size = des_spt.size/100)
                des_spt = des_spt[randind]
            else : des_spt = io.SearchAndCallFits(path = input_path, keyword = input_keyword)
        # calling spt des_gold ---------------------------------------------
        else : des_spt = io.SearchAndCallFits(path = input_path, keyword = input_keyword)


        #des_spt = des_spt[ (des_spt['MODEST_CLASS'] == 1) & (des_spt['FLAGS_GOLD'] == 0 )]
        #des_spt = Cuts.keepGoodRegion(des_spt)
        #des_spt = des_spt[des_spt['DEC'] < -3]
        des_spt = des_spt[priorCut_test(des_spt)]
        des_spt = des_spt[(des_spt['FLAGS_GOLD'] == 0 )]
        #mask_y1a1 = (des_spt['FLAGS_GOLD'] == 0 )&(priorCut_test(des_spt))
        #des_spt = des_spt[mask_y1a1]

        clf_cmass = XD_fitting( None, pickleFileName = cmass_pickle)               
        clf_no = XD_fitting( None, pickleFileName = no_pickle)
        
        #assign prob to spt ----------------------
        # dmass from spt
        #rabin = np.linspace(des_spt['RA'].min(), des_spt['RA'].max(), 15)
        #ind_map = np.digitize(des_spt['RA'], bins = rabin)
        #print 'before', des_spt.size
        #des_spt_list = []



        if 'SFD98' in params : 
            if params['SFD98'] : 
                print('change reddening corrections from SLR to SFD98')
                des_spt = RemovingSLRReddening(des_spt )
                des_spt = des_spt[priorCut_test(des_spt)]
                des_spt = AddingSFD98Reddening(des_spt, kind='SPT')     
            else :
                print('Apply priorCut....SFD98 correction should be applied in advance....')
                des_spt = des_spt[priorCut_test(des_spt)]
        else :
            print('Apply priorCut....SFD98 correction should be applied in advance....')
            des_spt = des_spt[priorCut_test(des_spt)]

        if 'hpix' in params :
            if params['hpix'] :

                ind_map = hpRaDecToHEALPixel(des_spt['RA'], des_spt['DEC'], nside=  8, nest= True)
                valid_hpix = list(set(ind_map))

                print('# of healpix pixels :', len(valid_hpix))

                for hp in valid_hpix:
                    outname = out_catname+'_hpix{:03}.fits'.format(hp)

                    #if hp > 625:
                    if os.path.exists(outname):

                        print('prob exists ', outname)
                        pass

                    else :
                        des_spt_i = des_spt[ind_map == hp]
                        ts = assignCMASSProb(des_spt_i , clf_cmass, clf_no, cmass_fraction = cmass_fraction, mag=mag, err=err )
                        fitsio.write(outname, ts)


                        print('prob cat save to ', outname)

                        ts = None
            else : 
                ind_map = construct_jk_catalog_ind( des_spt, njack = njack, root = output_dir )

        else : 
            ind_map = construct_jk_catalog_ind( des_spt, njack = njack, root = output_dir )
        

            prob_spt = []

            jkoutnames = io.SearchFitsByName(path = output_dir, columns = None, keyword = params['out_catname'])
            if len(jkoutnames) == 0 : 
            
                for i in range(njack):

                    outname = out_catname+'_jk{:03}.fits'.format(i+1)
                    if os.path.exists(outname): ts = fitsio.read(outname)
                    else : 
                        des_spt_i = des_spt[ind_map == i]
                        ts = assignCMASSProb(des_spt_i , clf_cmass, clf_no, cmass_fraction = cmass_fraction, mag=mag, err=err )
                        fitsio.write(outname, ts)

                        print('prob cat save to ', outname)

                
                    prob_spt.append(ts)
                    ts = None
    
            else : 
                prob_spt = [fitsio.read(jko) for jko in jkoutnames] 

            prob_spt = np.hstack(prob_spt)

    
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
    if not os.path.exists(output_dir+'/config/') : os.makedirs(output_dir+'/config/')
    if not os.path.exists(output_dir+'/log/') : os.makedirs(output_dir+'/log/')

    
    cpconfigname = output_dir+'config/config.yaml.v001' 
    logname = output_dir + 'log/log.v001'

    for vind in range(2, 100000):
        if os.path.exists(cpconfigname):
            cpconfigname = output_dir+'config/config.yaml.v{:03}'.format(vind)
            logname = output_dir + 'log/log.v{:03}'.format(vind)
        else : break

    print('log saved to ', logname)

            #if os.path.exists(cpconfigname):
            #   cpconfigname = output_dir+'config.yaml' +'.v3'
            #   logname = output_dir + 'log.v3'

    import transcript
    transcript.start(logname)


    os.system("cp "+param_file+" "+cpconfigname )
    #params = yaml.load(open('config.yaml'))
    if params['fitting'] : 
        train_st82(params, param_file)
    #else : params['cmass_fraction']    

    try :
        float(params['cmass_fraction'])
        #print 'input cmass fraction ', params['cmass_fraction']

    except ValueError:
        #if type(params['cmass_fraction']) == 'str' : 
        f = open( params['cmass_fraction'], 'r')
        cmassfrac = float(f.read())
        params['cmass_fraction'] = cmassfrac

        #pass

    if 'cat_area' in params : 
        if params['cat_area'] in ['st82', 'stripe82', 'ST82', 'STRIPE82']: main_st82(params)
        elif params['cat_area'] in ['spt', 'SPT'] : main_spt(params)
        #elif params['cat_area'] =='all' : 
        #    main_st82(params)
        #    main(params)
    else : main_spt(params)

    transcript.stop()
