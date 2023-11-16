import numpy as np
import healpy as hp
import os, sys
sys.path.append('/n/des/lee.5922/programs/cosmolike/DMASS-analysis/measurements/code/')
from utils import hpRaDecToHEALPixel
import treecorr

def treecorr_nn(lens, source, random=None, filename=None, cat_type='im3',
		nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05):
    import treecorr
    
    if cat_type=='im3' : w_src = source['WEIGHT']
    elif cat_type == 'mcal': w_src=None
    
    cat_l = treecorr.Catalog(ra=lens['RA'], dec=lens['DEC'], w=lens['WEIGHT'], ra_units='deg', dec_units='deg')
    cat_s = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], 
                                  g1=source['E1'], g2=source['E2'], 
                                  w = w_src, 
                                  ra_units='deg', dec_units='deg')
    
    ls = treecorr.NNCorrelation(nbins=nbins, min_sep=min_sep,
                                max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop,
                                num_threads=30, verbose=1)
    ls.process(cat_l, cat_s)
    
    rs=None
    if random is not None:
        cat_r = treecorr.Catalog(ra=random['RA'], dec=random['DEC'], w=random['WEIGHT'], 
                                 ra_units='deg', dec_units='deg')
        rs = treecorr.NNCorrelation(nbins=nbins, min_sep=min_sep,
                                max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop,
                                    num_threads=30, verbose=1)    
        rs.process(cat_r, cat_s)

    ls.write(filename, rr=rs)

def treecorr_nk(lens, source, scalar, filename=None,
		nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05):
    import treecorr
    """
    Uses TreeCorr to compute the NK correlation between lens and source.
    Used to compute scale dependece responses.
    scalar: array with scalar values of some quantity of which we want to compute the
           average in angular bins.
    Returns theta and R_nk.
    """
    nk = treecorr.NKCorrelation(nbins=nbins, min_sep=min_sep,
                                max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop,
                                num_threads=30, verbose=2)

    cat_l = treecorr.Catalog(ra=lens['RA'], dec=lens['DEC'], w=lens['WEIGHT'], ra_units='deg', dec_units='deg')
    cat_s = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], k=scalar, ra_units='deg', dec_units='deg')
    nk.process(cat_l, cat_s)
    nk.write(filename)

def treecorr_ng(lens, source, filename=None, cat_type='im3',
		nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05):
    import treecorr
    
    if cat_type=='im3' : w_src = source['WEIGHT']
    elif cat_type == 'mcal': w_src=None
    
    ng = treecorr.NGCorrelation(nbins=nbins, min_sep=min_sep,
                                max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop, 
                                num_threads=30, verbose=2)

    cat_l = treecorr.Catalog(ra=lens['RA'], dec=lens['DEC'], w=lens['WEIGHT'], ra_units='deg', dec_units='deg')
    cat_s = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], 
                                  g1=source['E1'], g2=source['E2'], 
                                  w = w_src, 
                                  ra_units='deg', dec_units='deg')
    
    ng.process(cat_l, cat_s)
    ng.write(filename)


def run_nk_jk( lens=None, source = None, scalar = None, 
	        nside=8, dir=None,
		nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05):
    
    hpind_lens = hpRaDecToHEALPixel(lens['RA'], lens['DEC'], nside=nside )
    hpind_src = hpRaDecToHEALPixel( source['RA'], source['DEC'], nside=nside )
    
    njk = len( set(hpind_lens) )
    #njk_src = len( set(hpind_src) )
    print('njk=',njk)
    os.system('mkdir '+dir+'/jk/')
    
    n = 1
    for i in set(hpind_lens):

        neighsjk = hp.get_all_neighbours(nside, i)
        neighsjk = np.append(neighsjk, i)
        mask_src = np.in1d(hpind_src, neighsjk)
        mask_lens = hpind_lens == i
        njk_src = len(neighsjk)
        #ngal = np.sum(hpind_lens == i)
        if 'WEIGHT' in lens.dtype.names : ngal = np.sum(lens[mask_lens]['WEIGHT'])
        else : ngal = np.sum(mask_lens)
        nsrc = np.sum(mask_src)
        print('njk/ntot={}/{}, nsrc={}, ngal={}'.format(n,njk, nsrc, ngal))
        #filename = dir+'/jk_onejk_cross/ggl_hpix_lens{:05}_src{:05}.txt'.format(i, j)
        filename = dir+'/jk/nk_hpix{:05}.txt'.format(i)
        if os.path.exists(filename): 
            print('file exists.. ', filename)
        else : 
            if nsrc == 0 : pass
            elif ngal == 0 : pass
            else : 

                lens_i = lens[hpind_lens == i]
                src_i = source[mask_src]
                scalar_i = scalar[mask_src]
                treecorr_nk(lens_i, src_i, scalar_i, filename=filename)
                lens_i, src_i = None, None
        n += 1


def run_ng_jk( lens=None, source = None, 
           nside=8, dir=None, cat_type='im3', 
	   nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05):
    
    hpind_lens = hpRaDecToHEALPixel(lens['RA'], lens['DEC'], nside=nside )
    hpind_src = hpRaDecToHEALPixel( source['RA'], source['DEC'], nside=nside )
    
    njk = len( set(hpind_lens) )
    #njk_src = len( set(hpind_src) )
    print('njk=',njk)
    os.system('mkdir '+dir+'/jk/')
    
    n = 1
    for i in set(hpind_lens):

        neighsjk = hp.get_all_neighbours(nside, i)
        neighsjk = np.append(neighsjk, i)
        mask_src = np.in1d(hpind_src, neighsjk)
        mask_lens = hpind_lens == i
        njk_src = len(neighsjk)
        #ngal = np.sum(hpind_lens == i)
        if 'WEIGHT' in lens.dtype.names : 
            ngal = np.sum(lens[mask_lens]['WEIGHT'])
        else : 
            ngal = np.sum(mask_lens)
        nsrc = np.sum(mask_src)
        print('njk/ntot={}/{}, nsrc={}, ngal={}'.format(n,njk, nsrc, ngal))
        #filename = dir+'/jk_onejk_cross/ggl_hpix_lens{:05}_src{:05}.txt'.format(i, j)
        filename = dir+'/jk/ng_hpix{:05}.txt'.format(i)

        if os.path.exists(filename): 
            print('file exists.. ', filename)
        else : 
            if nsrc == 0 : pass
            elif ngal == 0 : pass
            else : 
                lens_i = lens[hpind_lens == i]
                src_i = source[mask_src]
                treecorr_ng(lens_i, src_i, 
			            nbins=nbins, min_sep=min_sep,
                        max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop,
			            filename=filename, cat_type=cat_type)
                lens_i, src_i = None, None
        n += 1

def run_nn_jk( lens=None, source = None, random=None, 
           nside=8, dir=None, cat_type='im3', 
           nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05):
    
    hpind_lens = hpRaDecToHEALPixel(lens['RA'], lens['DEC'], nside=nside )
    hpind_src = hpRaDecToHEALPixel( source['RA'], source['DEC'], nside=nside )
    
    if random is not None : 
        hpind_rand = hpRaDecToHEALPixel( random['RA'], random['DEC'], nside=nside )
    else : 
        rand_i = None
        
    njk = len( set(hpind_lens) )
    #njk_src = len( set(hpind_src) )
    print('njk=',njk)
    os.system('mkdir '+dir+'/jk/')
    
    n = 1
    for i in set(hpind_lens):

        neighsjk = hp.get_all_neighbours(nside, i)
        neighsjk = np.append(neighsjk, i)
        mask_src = np.in1d(hpind_src, neighsjk)
        mask_lens = hpind_lens == i
        njk_src = len(neighsjk)
        #ngal = np.sum(hpind_lens == i)
        if 'WEIGHT' in lens.dtype.names : ngal = np.sum(lens[mask_lens]['WEIGHT'])
        else : ngal = np.sum(mask_lens)
        nsrc = np.sum(mask_src)
        nrand = -1.0
        if random is not None : nrand = np.sum(hpind_rand ==i)
        print('njk/ntot={}/{}, nsrc={}, ngal={}, nrand={}'.format(n,njk, nsrc, ngal, nrand))
        #filename = dir+'/jk_onejk_cross/ggl_hpix_lens{:05}_src{:05}.txt'.format(i, j)
        filename = dir+'/jk/nn_hpix{:05}.txt'.format(i)
        if os.path.exists(filename): 
            print('file exists.. ', filename)
        else : 
            if nsrc == 0 : pass
            elif ngal == 0 : pass
            elif nrand == 0 : pass
            else : 
                #lens_i = lens[hpind_lens == i]
                lens_i = lens[mask_lens]
                src_i = source[mask_src]
                if random is not None : rand_i = random[hpind_rand == i]
                treecorr_nn(lens_i, src_i, random=rand_i, 
                            nbins=nbins, min_sep=min_sep,
                            max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop,
                            filename=filename, cat_type=cat_type)
                lens_i, src_i = None, None
        n += 1


def compute_eix_jk(lens, source_unsheared, nside=8, 
		   nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05, dir=None):
    #e_ix = {}
    components = ['1p', '1m', '2p', '2m']
    for i, comp in enumerate(components):
        subdir = dir+'/eix/'+comp+'/'
        os.system('mkdir '+subdir)
        source_comp = source_unsheared[comp] #['cat']
        scalar = source_comp['E'+comp[0]]
        #_nkcorr_eix(lens, source_comp, scalar, filename=_filename)
        run_nk_jk(nside=nside, lens=lens, source=source_comp, scalar=scalar, 
		nbins=nbins, min_sep=min_sep,
                max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop, dir=subdir )

def compute_Rgamma_jk(lens, source, nside=8, 
		nbins=20, min_sep=2.5, max_sep=250, sep_units='arcmin', bin_slop=0.05, dir=None):
    R11 = source['R11']
    R22 = source['R22']
    scalar = 0.5 * (R11+R22)
    #filename_R = dir+'/jk/Rgamma_hpix{:05}.txt'.format(i)
    subdir = dir+'/Rgamma/'
    run_nk_jk(nside=nside, lens=lens, source=source, scalar=scalar, 
		nbins=nbins, min_sep=min_sep,
                max_sep=max_sep, sep_units=sep_units, bin_slop=bin_slop,
		dir=subdir )

def compute_Rs(e_ix):
    mean_e1_1p = e_ix['1p'] #np.mean(source_unsheared['1p']['cat']['E1'])
    mean_e1_1m = e_ix['1m']#np.mean(source_unsheared['1m']['cat']['E1'])
    mean_e2_2p = e_ix['2p']#np.mean(source_unsheared['2p']['cat']['E2'])
    mean_e2_2m = e_ix['2m']#np.mean(source_unsheared['2m']['cat']['E2'])
    
    dgamma = 0.01*2
    Rs = np.zeros(2)
    Rs11 = ( mean_e1_1p + mean_e1_1m )/dgamma
    Rs22 = ( mean_e2_2p + mean_e2_2m )/dgamma
    
    Rs = 0.5 * (Rs11 + Rs22)
    return Rs


def construct_jk(dir, nbins=20, gammax=False, boost=False):
    
    dir = dir+'/jk/'
    import glob
    filenamelist = glob.glob(dir+'*.txt')
    filenamelist.sort()

    n_jk = len(filenamelist)
    print('n_jk=',n_jk)
    print('calling files from ', dir)

    gts = np.zeros((n_jk, nbins) )
    #gxs = np.zeros((n_jk, 20) )
    wt = np.zeros((n_jk, nbins) )
    #npairs = np.zeros((20,n_jk) )
    
    for i,f in enumerate(filenamelist):
	#print f
        dat = np.genfromtxt(f)
        meanr = dat[:,0]
        gts[i,:] = dat[:,3]
        wt[i,:] = dat[:,-2]
	
        if gammax: gts[i,:] = dat[:,4]
        if boost: 
            gts[i,:] = dat[:,-3]

    if boost:
        gtnum = np.array([np.sum(np.delete(np.array(gts), i, axis = 0), axis = 0) for i in range(len(gts))])
        wnum = np.array([np.sum(np.delete(np.array(wt), i, axis = 0), axis = 0) for i in range(len(gts))]) 
    else :        
        gts = gts * wt
        gtnum = np.array([np.sum(np.delete(np.array(gts), i, axis = 0), axis = 0) for i in range(len(gts))])
        wnum = np.array([np.sum(np.delete(np.array(wt), i, axis = 0), axis = 0) for i in range(len(gts))])
    
    return meanr, gtnum, wnum

def construct_jk_onebin(dir, nbins=20, gammax=False, boost=False):
    
    dir = dir+'/jk/'
    import glob
    filenamelist = glob.glob(dir+'*.txt')
    filenamelist.sort()

    n_jk = len(filenamelist)
    print('n_jk=',n_jk)
    print('calling files from ', dir)

    gts = np.zeros((n_jk, nbins) )
    #gxs = np.zeros((n_jk, 20) )
    wt = np.zeros((n_jk, nbins) )
    #npairs = np.zeros((20,n_jk) )
    
    for i,f in enumerate(filenamelist):
	#print f
        dat = np.genfromtxt(f)
        meanr = np.sum([dat[0,0],dat[-1,0]])*0.5
        gts[i,:] = dat[:,3]
        wt[i,:] = dat[:,-2]
	
        if gammax: gts[i,:] = dat[:,4]
        if boost: 
            gts[i,:] = dat[:,-3]

    if boost:
        gtnum = np.array([np.sum(np.delete(np.array(gts), i, axis = 0), axis = 0) for i in range(len(gts))])
        wnum = np.array([np.sum(np.delete(np.array(wt), i, axis = 0), axis = 0) for i in range(len(gts))]) 
    else :        
        gts = gts*wt
        gts = np.sum(gts, axis = 1)
        wt = np.sum(wt, axis=1)
        gtnum = np.array([np.sum(np.delete(np.array(gts), i, axis = 0), axis = 0) for i in range(len(gts))])
        wnum = np.array([np.sum(np.delete(np.array(wt), i, axis = 0), axis = 0) for i in range(len(gts))])
    
    return meanr, gtnum, wnum


def compute_jkcov( gt_all, nbins=20 ):

    Njk = len(gt_all)
    gt_mean = np.mean(gt_all, axis=0)
    Np = gt_mean.size
    Hartrap = (Njk - Np - 2)*1./(Njk-1)
    gt_cov = np.zeros((nbins,nbins))
    for i in range(nbins):
        for j in range(nbins):
            gt_cov[i,j] = np.sum((gt_all[:,i]-gt_mean[i])*(gt_all[:,j]-gt_mean[j])) * (Njk-1.)*1./Njk /Hartrap

    return gt_cov


def compute_jkcov_onebin( gt_all, nbins=1 ):

    Njk = len(gt_all)
    gt_mean = np.mean(gt_all, axis=0)
    Np = gt_mean.size
    Hartrap = (Njk - Np - 2)*1./(Njk-1)
    gt_cov = np.zeros((nbins,nbins))
    #for i in range(nbins):
    #    for j in range(nbins):
    #        gt_cov[i,j] = np.sum((gt_all[:,i]-gt_mean[i])*(gt_all[:,j]-gt_mean[j])) * (Njk-1.)*1./Njk /Hartrap
    gt_cov[0,0] = np.sum((gt_all-gt_mean)*(gt_all-gt_mean)) * (Njk-1.)*1./Njk /Hartrap

    #print gt_mean.shape, gt_cov.shape, gt_all.shape
    #print gt_cov
    return gt_cov


def delete_unnecessary_random_jk(savedir, lensdir='/lens/', randomdir='/random/',
    subdirs = ['/jk/*', '/eix/1m/jk/*','/eix/1p/jk/*',
        '/eix/2m/jk/*','/eix/2p/jk/*', '/Rgamma/jk/*'] ):    

    import glob

    for subs in subdirs:
        filelist_l = np.array(glob.glob(savedir+lensdir+subs))
        filelist_r = np.array(glob.glob(savedir+randomdir+subs))
        filelist_r_ = np.array([r.split('/')[-1] for r in filelist_r])
        filelist_l_ = np.array([l.split('/')[-1] for l in filelist_l])
        #print subs, len(filelist_l), len(filelist_r)
        mask = np.in1d(filelist_r_, filelist_l_)
        mask2 = np.in1d(filelist_l_, filelist_r_)
    	#if len(mask) == 0: pass
        #elif len(mask2) == 0 : return 0
            #else :

        #print subs, np.sum(mask), np.sum(mask2)
        if len(mask)!= 0 :
            filename = filelist_r[~mask]
            for fd in filename:
                print('delete unnecessary random file ', fd)
                os.system('rm '+fd)

        if len(mask2)!=0:
            filename = filelist_l[~mask2]
            for fd in filename:
                print('delete unnecessary lens file ', fd)
                os.system('rm '+fd)

    return 0


def delete_unnecessary_random_jk_boost(savedir, lensdir='/lens/', randomdir='/random/',
    subdirs = ['../boostfacor/jk/*'] ):    

    import glob
    lenspairdir = '/jk/*'

    for subs in subdirs:
        filelist_l = np.array(glob.glob(savedir+lensdir+lenspairdir))
        filelist_r = np.array(glob.glob(savedir+randomdir+subs))
        filelist_r_ = np.array([r.split('/')[-1].split('_')[-1] for r in filelist_r])
        filelist_l_ = np.array([l.split('/')[-1].split('_')[-1] for l in filelist_l])
        #print subs, len(filelist_l), len(filelist_r)
        mask = np.in1d(filelist_r_, filelist_l_)
        mask2 = np.in1d(filelist_l_, filelist_r_)
    	#if len(mask) == 0: pass
        #elif len(mask2) == 0 : return 0
            #else :

        #print subs, np.sum(mask), np.sum(mask2)
        if len(mask)!= 0 :
            filename = filelist_r[~mask]
            for fd in filename:
                print('delete unnecessary random file ', fd)
                os.system('rm '+fd)

        if len(mask2)!=0:
            filename = filelist_l[~mask2]
            for fd in filename:
                print('delete unnecessary lens file ', fd)
                os.system('rm '+fd)

    return 0

def save_results( meanr, jk_all, jk_cov, dir=None, filename=None  ):
    os.system('mkdir -p '+dir+'/finalize/')
    print('save results in ', dir+'finalize/')
    jk_mean = np.mean(jk_all, axis=0)

    if jk_cov is None:
        DAT = np.column_stack(( meanr, jk_mean))
    else : 
        np.savetxt( dir+'finalize/covjk_'+filename+'.txt', jk_cov)
        err_jk = np.sqrt( jk_cov.diagonal() )
        DAT = np.column_stack(( meanr, jk_mean, err_jk))

    header= '\nmeanr   jkmean  jkerr'
    np.savetxt( dir+'finalize/meanjk_'+filename+'.txt', DAT, header=header )
    
    DAT2 = np.column_stack(( jk_all))
    np.savetxt( dir+'finalize/alljk_'+filename+'.txt', DAT2, header=header )



def calculate_chi2_null( data, cov, cut=5, total=False ):
    
    ma = np.zeros(20, dtype=bool)
    ma[cut:] = 1
    
    if total: mask = np.hstack([ma,ma,ma,ma])
    else : mask = ma.copy()
    Nma = np.sum(mask)
    
    d1, d2 = np.mgrid[0:mask.size, 0:mask.size]
    md1 = mask[d1]
    md2 = mask[d2]
    mask2d = md1 * md2

    cov_masked = cov[mask2d].reshape(Nma, Nma)
    data_masked = data[mask]
    
    #fig, ax = plt.subplots()
    #ax.imshow( np.log10(cov_masked)  )
    
    covinv = np.linalg.inv(cov_masked)
    chi2 = np.dot( np.dot( data_masked, covinv), data_masked.T) 
    print('chi2=', chi2, '  dof=', Nma, '   chi2/dof=', chi2/Nma)
    return chi2



def plotting_measurement( rootdir, blind=True, type='gammat', ylabel='', yscale='log', yaxhline=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    labels = [ '$0.20 < z_s < 0.43$' ,  '$0.43 < z_s < 0.63$' ,  '$0.63 < z_s < 0.90$' ,  '$0.90 < z_s < 1.30$'  ]
    for i in range(4):
        savedir = rootdir+'/source'+str(i+1)+'/'
        meanr, gt, err_gt=\
        np.genfromtxt(savedir+'/finalize/meanjk_'+type+'.txt', unpack=True)
        ax.errorbar(meanr, gt, yerr=err_gt, fmt='o', label=labels[i], capsize=5 )

    if yaxhline is not None:
        ax.axhline(y=yaxhline, ls='--', color='grey')

    ax.axvspan(1, 27.0, alpha=0.2, color='grey')
    ax.set_xscale('log')
    ax.set_yscale(yscale)
    ax.set_xlabel('$\\theta$ [arcmin]')
    ax.set_ylabel(ylabel)
    ax.set_xlim(2.3, 300)
    if blind: 
        ax.set_yticklabels([])
        ax.tick_params( which='major', axis='y', length = 0, width = 0, color = 'blue')
        ax.tick_params( which='minor', axis='y', length = 0, width = 0, color = 'blue')
    ax.legend()
    os.system('mkdir '+rootdir+'/fig/')
    plt.tight_layout()
    fig.savefig(rootdir+'/fig/'+type+'.pdf')
    print('fig saved to', rootdir+'/fig/'+type+'.pdf')


def plotting_measurement_split( rootdir, blind=True, types='gammat', input_err=None,
ylabel='', yscale='log', yaxhline=None, scalecut=True, thetagamma=False,
yextent = None, data_labels = None, mask_ang=[27,27,27,27]):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,4, figsize=(16,3.5) )
    ax = ax.ravel()

    labels = [ '$0.20 < z_s < 0.43$' ,  '$0.43 < z_s < 0.63$' ,  '$0.63 < z_s < 0.90$' ,  '$0.90 < z_s < 1.30$'  ]
    xlabel = '$\\theta$ ${\\rm [arcmin]}$'
    #mask_mg = [28, 28, 35,35]
    #mask_lcdm = [27, 27, 27,27]
    
    #mask_ang = mask_mg
    #mask_ang = mask_lcdm
    fmts = ['o', 'd', 's']
    mfc = [None, 'w', 'w']
    lc = [None, 'tomato', None]
        

    for i in range(4):

        for ti, type in enumerate(types):
    
            savedir = rootdir+'/source'+str(i+1)+'/'
            meanr, gt, err_gt=\
            np.genfromtxt(savedir+'/finalize/meanjk_'+type+'.txt', unpack=True)

	    #covgt = np.genfromtxt(savedir+'/finalize/covjk_'+type+'.txt')
	    #chi2_null_4 = calculate_chi2_null( gt, covgt, cut=6, total=False )
	    #chi2_null_12 = calculate_chi2_null( gt, covgt, cut=10, total=False )
	    
            if thetagamma : 
                gt = gt*meanr
                err_gt = err_gt * meanr

            if input_err is not None :
                err_gt = input_err[meanr.size*i: meanr.size*(i+1)]

            dlabel=None
            if data_labels != None:
                #print data_labels[ti]
                if i == 0 : dlabel= data_labels[ti]
            #if ti != 0 : label=None
            ax[i].errorbar(meanr*(1 + 0.05*ti), gt, yerr=err_gt, fmt=fmts[ti], markerfacecolor=mfc[ti],  color=lc[ti], capsize=5, label=dlabel)

	    

        if yaxhline is not None:
            ax[i].axhline(y=yaxhline, ls='--', color='grey')

        ax[i].text(0.95, 0.95, labels[i], transform=ax[i].transAxes,fontsize=17, horizontalalignment='right', verticalalignment='top' )
        ax[i].axvspan(1, mask_ang[i], alpha=0.2, color='grey')
        #ax[i].axvline(x=meanr, color='green', lw=27, alpha=0.5)
        ax[i].set_xscale('log')
        ax[i].set_yscale(yscale, fontsize=17)
        ax[i].set_xlabel(xlabel,fontsize=17)
        ax[i].set_xlim(2.3, 300)
        ax[i].tick_params(labelsize=15)

        if yextent is not None : ax[i].set_ylim(yextent)
        #ax[i].set_ylabel(ylabel)

        if i != 0: ax[i].set_yticklabels([])
        if blind: 
            ax[i].tick_params( which='major', axis='y', length = 0, width = 0, color = 'blue')
            ax[i].tick_params( which='minor', axis='y', length = 0, width = 0, color = 'blue')
            ax[i].set_yticklabels([])
        

        ax[i].legend(loc=4)
    ax[0].set_ylabel(ylabel,fontsize=17)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    
     # labels along the bottom edge are off

    os.system('mkdir '+rootdir+'/fig/')
    fig.savefig(rootdir+'/fig/'+type+'_split.pdf')
    print('fig saved to', rootdir+'/fig/'+type+'_split.pdf')


def main_jk():

    lens, randoms = calling_lens_catalog()
    source = calling_source_mcal_catalog()

    rootdir = '../ggl_mcal/'
    for i in range(4):
        savedir = rootdir+'/source'+str(i+1)+'/'
        os.system('mkdir '+savedir)

        # ng lens
        run_ng_jk( lens=lens, source = source['sheared']['bin'+str(i+1)]['cat'], 
            dir=savedir+'/lens/', cat_type='mcal')  
        
        # ng rand
        run_ng_jk( lens=randoms[randomsind], source = source['sheared']['bin'+str(i+1)]['cat'], 
            dir=savedir+'/random/', cat_type='mcal')  
        
        # response functions
        compute_eix_jk(lens, source['unsheared']['bin'+str(i+1)], dir=savedir)
        compute_Rgamma_jk(lens, source['sheared']['bin'+str(i+1)]['cat'], dir=savedir)
        

        # boostfactor
        run_nn_jk( lens=lens, source = source['sheared']['bin'+str(i+1)]['cat'],
                random = randoms[randomsind],
            dir=savedir+'/boostfactor/', cat_type='mcal') 


    for i in range(4):
        savedir = rootdir+'/source'+str(i+1)+'/'

        # response function
        meanr, Rgnum, Rwnum = construct_jk(savedir+'/Rgamma/')
        Rgamma_all = Rgnum/Rwnum
        Rgamma_mean = np.mean(Rgamma_all, axis=0)

        eix={}
        components = ['1p', '1m', '2p', '2m']
        for i, comp in enumerate(components):
            meanr, eixnum, weix = construct_jk(savedir+'/eix/'+comp+'/')
            eix_all = eixnum/weix
            #eix_mean = np.mean(eix_all, axis=0)
            eix[comp] = eix_all

        Rs_all = compute_Rs(eix)
        R_all = Rgamma_all + Rs_all
        R_cov = compute_jkcov( R_all )
        save_results( meanr, R_all, R_cov, dir=savedir, filename='response' )


        # gammat
        delete_unnecessary_random_jk(savedir)
        meanr, gtnum, wnum = construct_jk(savedir+'/lens/')
        meanr, grnum, wrnum = construct_jk(savedir+'/random/')
        
        gt_all = gtnum/wnum/R - grnum/wrnum/R
        gt_mean = np.mean(gt_all, axis=0)
        gt_cov = compute_jkcov( gt_all )
        save_results( meanr, gt_all, gt_cov, dir=savedir, filename='gammat' )

        # gammax
        meanr, gxnum, wnum = construct_jk(savedir+'/lens/', gammax=True)
        meanr, gxrnum, wrnum = construct_jk(savedir+'/random/', gammax=True)
        gx_all = gxnum/wnum/R - gxrnum/wrnum/R
        gx_mean = np.mean(gx_all, axis=0)
        gx_cov = compute_jkcov( gx_all )
        save_results( meanr, gx_all, gx_cov, dir=savedir, filename='gammax' )

        # BoostFactor
        meanr, DD, RR = construct_jk(savedir+'/boostfactor/', boost=True)
        boostfactor = DD/RR
        boostfactor_mean = np.mean(boostfactor, axis=0)
        boostfactor_cov = compute_jkcov( boostfactor )
        save_results( meanr, boostfactor, boostfactor_cov, dir=savedir, filename='boostfactor' )

        #boostfactor_mean = np.mean(boostfactor, axis=0)
        gt_all_boosted = boostfactor * gtnum/wnum/R - grnum/wrnum/R
        gt_mean_boosted = np.mean(gt_all_boosted, axis=0) 
        gt_boosted_cov = compute_jkcov( gt_all_boosted )
        save_results( meanr, gt_all_boosted, gt_boosted_cov, dir=savedir, filename='gammat_boosted' )

    # plotting results
    #from ggl_mcal import plotting_measurement
    plotting_measurement( rootdir, blind=True, type='gammat', ylabel='$\gamma_t$')
    plotting_measurement( rootdir, blind=True,type='gammat_boosted', ylabel='$\gamma_t$, boosted')
    plotting_measurement( rootdir, blind=False,type='gammax', ylabel='$\gamma_x$', yscale='linear', yaxhline=0)
    plotting_measurement( rootdir, blind=False,type='boostfactor', ylabel='$B(\\theta)$', yscale='linear', yaxhline=1)


def main_systematics():
    #from ggl_2 import run_ng_jk, run_nn_jk, run_nk_jk, compute_eix_jk, compute_Rgamma_jk, construct_jk, compute_Rs
    from ggl_sys import divide_sample_in_sysmap
    from calling_catalogs import calling_lens_catalog, calling_source_mcal_catalog

    rootdir_sys = '../ggl_mcal/observing_condition/'
    lens, randoms = calling_lens_catalog()
    source = calling_source_mcal_catalog(
        outputcatdir = '../output_cats/', combine=True, unsheared=True )

    randoms = appendColumn( randoms, value=np.ones(randoms.size), name='WEIGHT')
    randomsind = np.random.choice( randoms.size, size=randoms.size/20)
    
    columns = ['coadd_objects_id', 'z_mc']
    mof_photoz_cat = esutil.io.read('/n/des/lee.5922/data/photoz_cat/y1a1-gold-mof-badregion_BPZ.fits', columns=columns, upper=True)

    mapdir = '../../sysmaps/'
    sysmap = {}
    sysmap['airmass']= esutil.io.read(mapdir+'Y1A1GOLD_band_r_nside4096_AIRMASS_coaddweights3_mean.fits.gz')
    sysmap['skybrite'] = esutil.io.read(mapdir+'Y1A1GOLD_band_r_nside4096_SKYBRITE_coaddweights3_mean.fits.gz')
    sysmap['maglim'] = esutil.io.read(mapdir+'Y1A1GOLD_band_r_nside4096_maglimit3__.fits.gz')
    sysmap['fwhm'] = esutil.io.read(mapdir+'Y1A1GOLD_band_r_nside4096_FWHM_MEAN_coaddweights3_mean.fits.gz')


    labels_ = ['airmass', 'skybrite', 'maglim', 'fwhm']
    cutvalue = [1.26, 285, 23.92, 3.68]
    suffix_ = ['low', 'hi']
    labels=[ l1+'_'+l2 for l1 in labels_ for l2 in suffix_ ]



    # store unsheared cat in dictionary
    # compute eix with unsheared cat

    components = ['1p', '1m', '2p', '2m']
    for i, la in enumerate(labels_):
        
        source['unsheared'][la+'_low'] = {}
        source['unsheared'][la+'_hi'] = {}   
        for comp in components:
            print(comp, la)
            source_low, source_hi \
            = divide_sample_in_sysmap( cat=source['unsheared'][comp], sysmap=sysmap[la], cutvalue=cutvalue[i]  )

            source['unsheared'][la+'_low'][comp] = source_low
            source['unsheared'][la+'_hi'][comp] = source_hi
            
        source_low, source_hi = None, None
        for su in suffix_:
            savedir_sys = rootdir_sys +'/'+la+'_'+su+'/'
            compute_eix_jk(lens, source['unsheared'][la+'_'+su], dir=savedir_sys)        
            source['unsheared'][la+'_'+su] = None
    source['unsheared'] = None



    # callig sheared cat 
    source = calling_source_mcal_catalog(
            outputcatdir = '../output_cats/', combine=True, unsheared=False )

    for i, la in enumerate(labels_):
        print(la)
        source_low, source_hi \
        = divide_sample_in_sysmap( cat=source['sheared']['cat'], sysmap=sysmap[la], cutvalue=cutvalue[i]  )
        #for su in suffix_:
        source['sheared'][la+'_low'] = source_low
        source['sheared'][la+'_hi'] = source_hi
        source_low, source_hi = None, None
        source['sheared']['cat'] = None
    

       
    for i, la in enumerate(labels):
        print(la)
        src = source['sheared'][la]
        savedir_sys = rootdir_sys +'/'+la +'/'
        os.system('mkdir '+ savedir_sys)

        # ng lens
        run_ng_jk( lens=lens, source = src, 
            dir=savedir_sys+'/lens/', cat_type='mcal')  

        # ng rand
        run_ng_jk( lens=randoms[randomsind], source = src, 
            dir=savedir_sys+'/random/', cat_type='mcal')  
        
        # boostfactor
        run_nn_jk( lens=lens, source = src,
                random = randoms[randomsind],
            dir=savedir_sys+'/boostfactor/', cat_type='mcal') 
        
        # response functions
        compute_Rgamma_jk(lens, src, dir=savedir_sys)

    # construct jk 
    for i, la in enumerate(labels):
        savedir_sys = rootdir_sys +'/'+la +'/'
        # response function
        meanr, Rgnum, Rwnum = construct_jk(savedir_sys+'/Rgamma/')
        Rgamma_all = Rgnum/Rwnum
        Rgamma_mean = np.mean(Rgamma_all, axis=0)


        eix={}
        components = ['1p', '1m', '2p', '2m']
        for i, comp in enumerate(components):
            meanr, eixnum, weix = construct_jk(savedir_sys+'/eix/'+comp+'/')
            eix_all = eixnum/weix
            #eix_mean = np.mean(eix_all, axis=0)
            eix[comp] = eix_all

        Rs_all = compute_Rs(eix)
        R_all = Rgamma_all + Rs_all
        R_cov = compute_jkcov( R_all )
        R_mean = np.mean(R_all, axis=0)
        save_results( meanr, R_all, R_cov, dir=savedir_sys, filename='response' )

        # gammat
        delete_unnecessary_random_jk(savedir_sys)
        meanr, gtnum, wnum = construct_jk(savedir_sys+'/lens/')
        meanr, grnum, wrnum = construct_jk(savedir_sys+'/random/')
        gt_all = gtnum/wnum/R_all - grnum/wrnum/R_all
        gt_mean = np.mean(gt_all, axis=0)
        gt_cov = compute_jkcov( gt_all )
        save_results( meanr, gt_all, gt_cov, dir=savedir_sys, filename='gammat' )

        # gammax
        meanr, gxnum, wnum = construct_jk(savedir_sys+'/lens/', gammax=True)
        meanr, gxrnum, wrnum = construct_jk(savedir_sys+'/random/', gammax=True)
        gx_all = gxnum/wnum/R_all - gxrnum/wrnum/R_all
        gx_mean = np.mean(gx_all, axis=0)
        gx_cov = compute_jkcov( gx_all )
        save_results( meanr, gx_all, gx_cov, dir=savedir_sys, filename='gammax' )

        # BoostFactor
        meanr, DD, RR = construct_jk(savedir_sys+'/boostfactor/', boost=True)
        boostfactor = DD/RR
        boostfactor_mean = np.mean(boostfactor, axis=0)
        boostfactor_cov = compute_jkcov( boostfactor )
        save_results( meanr, boostfactor, boostfactor_cov, dir=savedir_sys, filename='boostfactor' )

        #boostfactor_mean = np.mean(boostfactor, axis=0)
        gt_all_boosted = boostfactor * gtnum/wnum/R_all - grnum/wrnum/R_all
        gt_mean_boosted = np.mean(gt_all_boosted, axis=0) 
        gt_boosted_cov = compute_jkcov( gt_all_boosted )
        save_results( meanr, gt_all_boosted, gt_boosted_cov, dir=savedir_sys, filename='gammat_boosted' )


    # fitting ratio amplitude
    from ggl_sys import fitting_gammat_amplitude_jk, read_cov
    cov_combined_theory=\
    read_cov( filename = '/n/des/lee.5922/programs/cosmolike/lighthouse_cov/output_source_combined/cov_im3combined_dmass_pcut_sysweight_20bins_NG_lsls_cov_Ntheta20_Ntomo1_5')
    cov_combined_theory=cov_combined_theory[40:60, 40:60]

    FITSDATA_combined = fitsio.FITS('/n/des/lee.5922/programs/cosmolike/DMASS-analysis/simulated_data/simulated_y1_dmass_3x2pt_source_combined.fits')
    gammat_combined_theory = FITSDATA_combined['galaxy_shear_xi']['VALUE'].read()
    theta = FITSDATA_combined['galaxy_shear_xi']['ANG'].read()

    R_amp_source_values = {}
    cov=cov_combined_theory*2
    for la in labels_:
        for su in suffix_:
            full_la = la +'_'+su
            savedir_sys = rootdir_sys +'/'+full_la +'/'
            
            datajk = np.genfromtxt(savedir_sys+'/finalize/alljk_gammat_boosted.txt')
            #covjk = np.genfromtxt(savedir_sys+'/construct_jk/jk_cov.txt')
            mean_R_amp_source=\
            fitting_gammat_amplitude_jk(datajk=datajk, theory= gammat_combined_theory , cov=cov)
            
            mean_R_amp_source = np.array(mean_R_amp_source)
            R_amp_source_values['mean_'+full_la] = np.mean(mean_R_amp_source)
            R_amp_source_values['std_'+full_la] = np.std(mean_R_amp_source) * np.sqrt((len(mean_R_amp_source) -1 ) )
            
        Rhi = R_amp_source_values['mean_'+la+'_hi']
        Rlo = R_amp_source_values['mean_'+la+'_low']
        stdhi = R_amp_source_values['std_'+la+'_hi']
        stdlo = R_amp_source_values['std_'+la+'_low']
        
        ratio = Rlo/Rhi    
        err = ratio * np.sqrt( stdlo**2/Rlo**2 + stdhi**2/Rhi**2 )
        
        R_amp_source_values['ratio_'+la] = ratio
        R_amp_source_values['err_ratio_'+la] = err
        
    #import pickle
    f2 = open(rootdir_sys+'/R_amp_ratio.pkl', "wb")
    pickle.dump(R_amp_source_values, f2)
    f2.close()




    # Sigma Crit 
    # 1) redshift distributions
    bins, bs = np.linspace(0, 2, 201, retstep=True)
    z_s = bins[:-1]+bs/2.
    bin_low = bins[:-1]

    for i, la in enumerate(labels):
        print('observing condition - nz ', la)
        savedir_sys = rootdir_sys +'/'+la +'/'
        src = source['sheared'][la]
        
        src_matched, photoz_matched = matchCatalogs(cat1=src, cat2=mof_photoz_cat,tag='COADD_OBJECTS_ID')
        src_matched=None
        nz_src, _ = np.histogram(photoz_matched['Z_MC'], bins=bins, density=True)
        DAT = np.column_stack(( bin_low, nz_src ))
        filename = savedir_sys+'/'+la+'.nz'
        np.savetxt(filename, DAT, header='zlow, nz')


    # 2) sigma crit

    from ggl import DeltaSigma
    DS = DeltaSigma()

    z_l, nz_l = np.loadtxt('/n/des/lee.5922/programs/cosmolike/cosmosis/dmass_cat/cmass_sgc.nz_2', unpack=True)
    z_l_step = 0.5*(z_l[3]-z_l[2])
    z_l += z_l_step

    sig_crit_dics = {}

    for i, la in enumerate(labels):
        print('observing condition - sigma_crit ', la)
        savedir_sys = rootdir_sys +'/'+la +'/'
        
        z_src_low, nz_src = np.genfromtxt(savedir_sys+'/'+la+'.nz', unpack=True)
        z_src_step = 0.5*(z_src_low[3]-z_src_low[2])
        z_s = z_src_low + z_src_step
    
        sig_crit_inv_eff = DS.sig_crit_inv_eff(z_l, nz_l, z_s, nz_src)
        sig_crit_dics[la] = sig_crit_inv_eff
        
    import pickle
    f = open(rootdir_sys+'/sig_crit_inv_eff.pkl', "wb")
    pickle.dump(sig_crit_dics, f)
    f.close()

    f = open(rootdir_sys+'/sig_crit_inv_eff.pkl')
    sig_crit_dics = pickle.load(f)
    f2 = open(rootdir_sys+'/R_amp_ratio.pkl')
    R_amp_source_values = pickle.load(f2)


    # plotting
    nn = 1
    fig, ax = plt.subplots()
    #for la in labels:
    for la in labels_:
        sig_crit_dics['ratio_'+la] = sig_crit_dics[la+'_low']/sig_crit_dics[la+'_hi']
        ax.plot(nn, sig_crit_dics['ratio_'+la], 'sk', markersize=10)
        ax.errorbar(nn, R_amp_source_values['ratio_'+la], yerr=R_amp_source_values['err_ratio_'+la], fmt='ro')
        nn += 1
    ax.set_xticks([0,1,2,3,4,5])
    ax.set_ylabel( '$\gamma_t^{\\rm low} / \gamma_t^{\\rm high}$' )
    ax.set_xticklabels(['']+labels_+[''])
    os.system('mkdir '+rootdir_sys + '/fig/')
    fig.savefig(rootdir_sys + '/fig/sys_ratio.png')
