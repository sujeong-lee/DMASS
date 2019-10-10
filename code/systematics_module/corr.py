# tools for calculting clustering signals and lensing signals


#import easyaccess as ea
import esutil
import sys
import os
import healpy as hp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from suchyta_utils import jk

import fitsio
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts




def adding_dc(catalog, zlabel = 'Z', dclabel = 'DC', h=0.7, Om0=0.286, Ob0=0.046, Neff=3.046):
    from astropy.cosmology import FlatLambdaCDM
    # Cosmology Ibanez
    #h = 0.7
    #Om0 = 0.286
    #Ob0 = 0.046
    cosmo = FlatLambdaCDM(H0=h*100, Om0=Om0, Ob0=Ob0, Neff=Neff)
    print 'Calculate comoving distance with cosmology \nH0={}, Om0={}'.format(100*h, Om0)

    r = cosmo.comoving_distance(catalog[zlabel]).value *h
    #import numpy.lib.recfuctions as rf
    from numpy.lib import recfunctions as rf
    print 'Adding Comoving distance column'
    sys.stdout.flush()
    catalog = rf.append_fields( catalog, dclabel, data = r )
    return catalog


def rp_pi_counts_to_s( Ncounts, sbin ):
    rp = Ncounts['rpavg']
    pi = Ncounts['pimax'] - 0.5
    npairs = Ncounts['npairs']
    s = np.sqrt(rp**2 + pi**2)
    digi_ind = np.digitize(s, sbin)
    
    new_npairs = np.zeros(sbin.size-1)
    for i in range(sbin.size-1):
        new_npairs[i] = np.sum(npairs[digi_ind == i+1])
    return new_npairs
    
def cf_s( DD_counts, DR_counts, RR_counts, sbin, N, Nrand):        

    fN = np.float(Nrand)/np.float(N)
    DD_new = rp_pi_counts_to_s( DD_counts, sbin )
    DR_new = rp_pi_counts_to_s( DR_counts, sbin )
    RR_new = rp_pi_counts_to_s( RR_counts, sbin )
    
    xi = (fN**2*DD_new - 2*fN* DR_new + RR_new)/(RR_new)
    return xi

def rp_pi_counts_to_smu( Ncounts, sbin, mubin ):
    #rp = Ncounts['rpavg']
    rp = (Ncounts['rmin']+Ncounts['rmax'])/2.
    dpi = Ncounts['pimax'][1] - Ncounts['pimax'][0]
    pi = Ncounts['pimax'] - dpi/2.

    npairs = Ncounts['npairs'] * Ncounts['weightavg']
    s = np.sqrt(rp**2 + pi**2)
    mu = pi/s
   
    new_npairs = np.zeros((sbin.size-1, mubin.size))
    wsqr = np.zeros((sbin.size-1, mubin.size))
    for i in range(s.size-1):
        ind_s = np.digitize( s[i], sbin )
        ind_mu = np.digitize( mu[i], mubin )
        if ind_s == 0 or ind_s == sbin.size: pass
        elif ind_mu == 0 or ind_mu == mubin.size: pass
        else : new_npairs[ind_s-1, ind_mu-1] += npairs[i]

    return new_npairs


def _convert_3d_counts_to_cf(ND1, ND2, NR1, NR2,
                            D1D2, D1R2, D2R1, R1R2,
                            estimator='LS'):
    import numpy as np
    pair_counts = dict()
    fields = ['D1D2', 'D1R2', 'D2R1', 'R1R2']
    arrays = [D1D2, D1R2, D2R1, R1R2]
    for (field, array) in zip(fields, arrays):
        try:
            npairs = array['npairs']* array['weightavg']
            pair_counts[field] = npairs
        except IndexError:
            pair_counts[field] = array

    nbins = len(pair_counts['D1D2'])
    if (nbins != len(pair_counts['D1R2'])) or \
       (nbins != len(pair_counts['D2R1'])) or \
       (nbins != len(pair_counts['R1R2'])):
        msg = 'Pair counts must have the same number of elements (same bins)'
        raise ValueError(msg)

    nonzero = pair_counts['R1R2'] > 0
    if 'LS' in estimator or 'Landy' in estimator:
        fN1 = np.float(NR1) / np.float(ND1)
        fN2 = np.float(NR2) / np.float(ND2)
        cf = np.zeros(nbins)
        cf[:] = np.nan
        cf[nonzero] = (fN1 * fN2 * pair_counts['D1D2'][nonzero] -
                       fN1 * pair_counts['D1R2'][nonzero] -
                       fN2 * pair_counts['D2R1'][nonzero] +
                       pair_counts['R1R2'][nonzero]) / pair_counts['R1R2'][nonzero]
        if len(cf) != nbins:
            msg = 'Bug in code. Calculated correlation function does not '\
                  'have the same number of bins as input arrays. Input bins '\
                  '={0} bins in (wrong) calculated correlation = {1}'.format(
                      nbins, len(cf))
            raise RuntimeError(msg)
    else:
        msg = "Only the Landy-Szalay estimator is supported. Pass estimator"\
              "='LS'. (Got estimator = {0})".format(estimator)
        raise ValueError(msg)

    return cf


def cf_smu( DD_counts, DR_counts, RR_counts, sbin, mubin, N, Nrand):        

    #fN = np.float(Nrand)/np.float(N)
    DD_counts_smu = rp_pi_counts_to_smu( DD_counts, sbin, mubin )
    DR_counts_smu = rp_pi_counts_to_smu( DR_counts, sbin, mubin )
    RR_counts_smu = rp_pi_counts_to_smu( RR_counts, sbin, mubin )

    #fN = np.sqrt( np.sum(RR_counts_smu)*1./np.sum(DD_counts_smu) )
 

    Ndd = N*N #np.sum(DD_counts['npairs'])
    Ndr = N*Nrand#np.sum(DR_counts['npairs'])
    Nrr = Nrand*Nrand #np.sum(RR_counts['npairs'])

    # normalization
    norm_DD = DD_counts_smu  * Nrr*1./Ndd #**2
    norm_DR = DR_counts_smu  * Nrr*1./Ndr #**2
    norm_RR = RR_counts_smu 

    # total weight
    #wd = np.sqrt(np.sum(DD_counts_smu))
    #wdr = np.sqrt(np.sum(DR_counts_smu))
    #wr = np.sqrt(np.sum(RR_counts_smu))

    # normalization
    #norm_DD = DD_counts_smu * 1./wd**2
    #norm_DR = DR_counts_smu * 1./wdr**2
    #norm_RR = RR_counts_smu * 1./wr**2

    zeromask = (RR_counts_smu == 0.0)
    #norm_RR[zeromask] = 0.0  # to avoid zero divide error. will be excluded at the end.

    #xi = (fN**2*norm_DD - 2*fN* norm_DR + norm_RR)/(norm_RR)
    xi = (norm_DD - 2*norm_DR + norm_RR)/(norm_RR)

    xi[zeromask] = 0.0
    
    return xi, norm_DD, norm_DR, norm_RR




def direct_cf_smu( DD_counts, DR_counts, RR_counts, scenter, mubin, N, Nrand):        


    """
    N : weighted total number of galaxies
    Nrand : weighted total number of randoms

    """
    #fN = np.float(Nrand)/np.float(N)
    DD = DD_counts['npairs'] * DD_counts['weightavg'] # rp_pi_counts_to_smu( DD_counts, sbin, mubin )
    DR = DR_counts['npairs'] * DR_counts['weightavg'] # rp_pi_counts_to_smu( DR_counts, sbin, mubin )
    RR = RR_counts['npairs'] * RR_counts['weightavg'] # rp_pi_counts_to_smu( RR_counts, sbin, mubin )

    #fN = np.sqrt( np.sum(RR_counts_smu)*1./np.sum(DD_counts_smu) )
 
    #N = DD_counts['npairs']
    # total weight
    Ndd = N*N #np.sum(DD_counts['npairs'])
    Ndr = N*Nrand#np.sum(DR_counts['npairs'])
    Nrr = Nrand*Nrand #np.sum(RR_counts['npairs'])

    # normalization
    norm_DD = DD  * Nrr*1./Ndd #**2
    norm_DR = DR  * Nrr*1./Ndr #**2
    norm_RR = RR 

    zeromask = (RR == 0.0)
    #norm_RR[zeromask] = 0.0  # to avoid zero divide error. will be excluded at the end.

    #xi = (fN**2*norm_DD - 2*fN* norm_DR + norm_RR)/(norm_RR)
    xi = (norm_DD - 2.*norm_DR + norm_RR) *1./(norm_RR)

    xi[zeromask] = 0.0
    
    return xi, norm_DD, norm_DR, norm_RR #.reshape( scenter.size, mubin.size)

# Specify that an autocorrelation is wanted

def _cfz(data, rand, zlabel = 'Z', weight = None):
    import sys
    #print 'cfz running'
    sys.stdout.flush()
    import treecorr
    
    weight_data = None
    weight_rand = None
    
    if weight is not None:
        weight_data = weight[0]
        weight_rand = weight[1]
    
    """
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    print 'Calculate comoving distance with FlatLambdaCDM cosmology \nH0=70, Om0=0.3'
    
    h = 0.7
    r = cosmo.comoving_distance(data[zlabel]).value *h
    r_rand = cosmo.comoving_distance(rand[zlabel]).value *h
    """
    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], r=data['DC'],  w = weight_data, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], r=rand['DC'], is_rand=True, w = weight_rand, ra_units='deg', dec_units='deg')

    nbins = 35
    min_sep = 10
    max_sep = 180
    
    print '.',

    #dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    #dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    #rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, verbose=1)
    dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, verbose=1)
    rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, verbose=1)
    
    dd.process(cat)
    dr.process(cat,cat_rand)
    rr.process(cat_rand)
    
    xi, varxi = dd.calculateXi(rr,dr)
    
    return dd.meanr, xi, varxi 


def correlation_function_3d(data = None, rand = None, zlabel = 'Z', njack = 30, weight = None, mpi=True, suffix = '', out = None):
    # jk sampling
    import os, sys
    from suchyta_utils import jk
    print 'calculate 3d correlation function'
    #r, xi, xierr = _acf( data, rand, weight = weight )
    
    if 'DC' not in data.dtype.names : 
    
        from astropy.cosmology import FlatLambdaCDM

        h = 0.6777

        cosmo = FlatLambdaCDM(H0=h*100, Om0=0.3)
        print 'Calculate comoving distance with Planck cosmology \nH0=0.6777, Om0=0.3'
        sys.stdout.flush()

        r = cosmo.comoving_distance(data[zlabel]).value *h
        r_rand = cosmo.comoving_distance(rand[zlabel]).value *h

        from numpy.lib import recfunctions as rf
        print 'Adding Comoving distance column'
        sys.stdout.flush()
        data = rf.append_fields( data, 'DC', data = r )
        rand = rf.append_fields( rand, 'DC', data = r_rand )
    
    print 'JK sampling'
    sys.stdout.flush()
    
    jkfile = './jkregion.txt'

    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cfz, jkargs=[ data, rand ], jkargsby=[[raTag, decTag],[raTag, decTag]], regions = jkfile, nojkkwargs = {'weight':weight, 'zlabel':zlabel})
    jktest.DoJK( regions = jkfile, mpi=mpi )
    jkresults = jktest.GetResults(jk=True, full = False)
    os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    # getting jk err
    norm = (njack-1.) *1./njack

    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    xi_i = xi_i.reshape(njack, r.size)
    xi = np.mean( xi_i, axis=0)
    xi_dat = np.vstack(( r, xi_i)).T


    """
    xi_cov = 0
    for k in range(njack):
        xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)

    filename = 'data_txt/cfz_comparison_'+suffix+'.txt'
    header = 'r (Mpc/h), xi(r), jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename

    if out is True : return DAT
    else : return 0
    """


    xi_cov = np.zeros((r.size, r.size))
    for i in range(r.size):
        for j in range(r.size):
            #covkikj = 0
            #for ki in range(njack):
            #    for kj in range(njack):
            #        covkikj += (xi_i[ki, i] - xi[i]) * (xi_i[kj,j]-xi[j] )
            #xi_cov[i][j] = norm * np.sum( (xi_i[:, i][m1] - xi[i]) * (xi_i[:,j][m2]-xi[j] ))
            xi_cov[i][j] = np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))
            #xi_cov[i][j] = covkikj
    #inv = np.linalg.inv(xi_cov)
    
    xijkerr = np.sqrt(norm * xi_cov.diagonal())


    filename = 'data_txt/cfz_comparison_'+suffix+'.txt'
    header = 'r (Mpc/h), xi(r), jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt('data_txt/cfz_comparison_'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt('data_txt/cfz_comparison_'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "saving data file to : ",filename
    if out is True : return DAT




def pickle_data( data = None, root = 'data_txt/pair_counting/', pickle_name = None ):
    import pickle

    """
    i = 0
    _root = root + '_{}/'.format(i) 

    for i in range(100): 
        _root = root + '_{}/'.format(i) 
        if os.path.exists(_root) : 
            print _root, 'exist'
            i += 1
        else : break

    #_root = root + '_{}/'.format(i) 
    """

    _root = root+'/'
    if not os.path.exists(_root) : 
        os.makedirs(_root)
    #if not os.path.exists(dir) : os.mkdir(dir)
    for i in range(100):
        if os.path.exists(_root+pickle_name+'_jk'+str(i)+'.pkl') : 
            i+=1
        else : break

    pickle_name_i = pickle_name+'_jk'+str(i)+'.pkl'
    print _root+pickle_name_i
    output = open( _root + pickle_name_i, 'wb')
    pickle.dump(data, output)
    output.close()



def load_pickle_data_single( root = 'data_txt/pair_counting/', pickle_name = None ):
    import pickle

    """
    i = 0
    _root = root + '_{}/'.format(i) 

    for i in range(100): 
        _root = root + '_{}/'.format(i) 
        if os.path.exists(_root) : 
            print _root, 'exist'
            i += 1
        else : break

    #_root = root + '_{}/'.format(i) 
    """

    _root = root+'/'
    #if not os.path.exists(_root) : 
    #    os.makedirs(_root)
    #if not os.path.exists(dir) : os.mkdir(dir)

    pickle_name_i = pickle_name+'.pkl'

    if os.path.exists(_root+pickle_name_i):
        print 'Loading stored paircounts... :', _root+pickle_name_i
        file = open( _root + pickle_name_i)
        picklefile = pickle.load(file)
        return picklefile
    else : return None


def pickle_data_single( data = None, root = 'data_txt/pair_counting/', pickle_name = None ):
    import pickle

    """
    i = 0
    _root = root + '_{}/'.format(i) 

    for i in range(100): 
        _root = root + '_{}/'.format(i) 
        if os.path.exists(_root) : 
            print _root, 'exist'
            i += 1
        else : break

    #_root = root + '_{}/'.format(i) 
    """

    _root = root+'/'
    if not os.path.exists(_root) : 
        os.makedirs(_root)
    #if not os.path.exists(dir) : os.mkdir(dir)

    pickle_name_i = pickle_name+'.pkl'
    print _root+pickle_name_i
    output = open( _root + pickle_name_i, 'wb')
    pickle.dump(data, output)
    output.close()




def _cfz_multipoles( data, rand, weight = None, nthreads = 20, suffix = 'test' ):
    import Corrfunc

    cosmology = 2

    # Create the bins array

    #transverse separation
    rmin = 0.01
    rmax = 250.0
    nbins = 500
    
    rbins, rstep = np.linspace(rmin, rmax, nbins+1, retstep=True)
    # Specify the distance to integrate along line of sight
    pimax = 100.0


    RA = data['RA']
    DEC = data['DEC']
    DC = data['DC']

    RAND_RA = rand['RA']
    RAND_DEC = rand['DEC']
    RAND_DC = rand['DC']
    

    weight_data = None
    weight_rand = None
    
    if weight is not None:
        #weight_data = weight[0]
        #weight_rand = weight[1]
        weight_data = data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*( data['WEIGHT_CP'] + data['WEIGHT_NOZ'] - 1.)
        weight_rand = rand['WEIGHT_FKP']

        if weight_data.dtype != RA.dtype: 
            weight_data = weight_data.astype(RA.dtype)
            weight_rand = weight_rand.astype(RA.dtype)


    #print '.',

    from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
    import pickle

    DD_counts, apitimeDD = DDrppi_mocks(1, cosmology, nthreads, pimax, rbins, RA, DEC, DC, 
        weights1=weight_data, 
        c_api_timer=True, is_comoving_dist=True, output_rpavg=False, weight_type="pair_product")

    root = 'data_txt/pair_counting/'+suffix
    pickle_data( data = DD_counts, root = root, pickle_name = 'dd' )
    DR_counts, apitimeDR = DDrppi_mocks(0, cosmology, nthreads, pimax, rbins,
                              RA, DEC, DC, weights1=weight_data, 
                              RA2=RAND_RA, DEC2=RAND_DEC, CZ2=RAND_DC, weights2=weight_rand
                              ,c_api_timer=True, is_comoving_dist=True, verbose=False, weight_type="pair_product",
                                     output_rpavg = False)
    pickle_data( data = DR_counts, root = root, pickle_name = 'dr' )
    RR_counts, apitimeRR = DDrppi_mocks(1, cosmology, nthreads, pimax, rbins, 
                             RAND_RA, RAND_DEC, RAND_DC, weights1=weight_rand
                             ,c_api_timer=True, is_comoving_dist=True, output_rpavg=False,weight_type="pair_product")
    pickle_data( data = RR_counts, root = root, pickle_name = 'rr' )
    #print '\relapsed time ', apitimeDD + apitimeDR + apitimeRR

    #for r in RR_counts[10:]: print("{0:10.6f} {1:10.6f} {2:10.6f} {3:10.1f}"
    #                           " {4:10d} {5:10.6f}".format(r['rmin'], r['rmax'],
    #                           r['rpavg'], r['pimax'], r['npairs'], r['weightavg']))

    
    #stop
    N = data.size
    rand_N = rand.size
    
    smin = 40
    smax = 180
    snbin = 29
    
    sbin, step = np.linspace(smin, smax, snbin, retstep=True)
   
    sbin_center = sbin[:-1] + step/2.
    mubin, mus = np.linspace(0.0, 1.0, 101, retstep = True)
    mucenter = mubin[:-1]+mus/2.
    
    xi_smu = cf_smu( DD_counts, DR_counts, RR_counts, sbin, mubin, N, rand_N)
    
    import scipy
    legendre0 = np.array([scipy.special.eval_legendre(0,m) for m in mucenter]).ravel() 
    legendre2 = np.array([scipy.special.eval_legendre(2,m) for m in mucenter]).ravel() 

    xi_monopole = np.sum(xi_smu * legendre0.T, axis = 1)/mucenter.size
    xi_quadrupole = 5*np.sum(xi_smu * legendre2.T, axis = 1)/mucenter.size

    
    return sbin_center, np.hstack([xi_monopole, xi_quadrupole])



def correlation_function_multipoles_rppi_single( data, rand, weight_data = None, weight_rand = None, nthreads = 20, dir = './', 
    smin=5,smax=200, snbin=40, verbose=True ):
    import Corrfunc

    cosmology = 2

    # Create the bins array

    #transverse separation
    rmin = 0.01
    rmax = 250.0
    nbins = 500
    
    rbins, rstep = np.linspace(rmin, rmax, nbins+1, retstep=True)
    # Specify the distance to integrate along line of sight
    pimax = 100.0
    mumax = 1.0
    nmubin = 101
    mubin, mus = np.linspace(0.0, 1.0, nmubin, retstep = True)
    mucenter = mubin[:-1]+mus/2.

    sbin, step = np.linspace(smin, smax, snbin, retstep=True)
    sbin_center = sbin[:-1] + step/2.

    RA = data['RA']
    DEC = data['DEC'].astype(RA.dtype)
    DC = data['DC'].astype(RA.dtype)

    RAND_RA = rand['RA'].astype(RA.dtype)
    RAND_DEC = rand['DEC'].astype(RA.dtype)
    RAND_DC = rand['DC'].astype(RA.dtype)
    

    #weight_data = None
    #weight_rand = None
    N = data.size
    Nrand = rand.size
    if weight_data is not None:
        N = np.sum(weight_data)
    if weight_rand is not None:
        Nrand = np.sum(weight_rand)

    if weight_data.dtype != RA.dtype: 
        weight_data = weight_data.astype(RA.dtype)
        weight_rand = weight_rand.astype(RA.dtype)


    #print '.',

    from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
    import pickle

    DD_counts = load_pickle_data_single( root = dir, pickle_name = 'dd' )
    DR_counts = load_pickle_data_single( root = dir, pickle_name = 'dr' )
    RR_counts = load_pickle_data_single( root = dir, pickle_name = 'rr' )

    if DD_counts is None : 
        DD_counts, apitimeDD = DDrppi_mocks(1, cosmology, nthreads, pimax, rbins, RA, DEC, DC, 
            weights1=weight_data, verbose=verbose,
            c_api_timer=True, is_comoving_dist=True, output_rpavg=False, weight_type="pair_product")
        pickle_data_single( data = DD_counts, root = dir, pickle_name = 'dd' )

    if DR_counts is None : 
        DR_counts, apitimeDR = DDrppi_mocks(0, cosmology, nthreads, pimax, rbins,
                                  RA, DEC, DC, weights1=weight_data, 
                                  RA2=RAND_RA, DEC2=RAND_DEC, CZ2=RAND_DC, weights2=weight_rand
                                  ,c_api_timer=True, is_comoving_dist=True, verbose=verbose, weight_type="pair_product",
                                         output_rpavg = False)
        pickle_data_single( data = DR_counts, root = dir, pickle_name = 'dr' )

    if RR_counts is None : 
        RR_counts, apitimeRR = DDrppi_mocks(1, cosmology, nthreads, pimax, rbins, 
                                 RAND_RA, RAND_DEC, RAND_DC, weights1=weight_rand, verbose=verbose
                                 ,c_api_timer=True, is_comoving_dist=True, output_rpavg=False,weight_type="pair_product")
        pickle_data_single( data = RR_counts, root = dir, pickle_name = 'rr' )


    """
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    import pickle

    DD_counts, apitimeDD = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubin, sbin, RA, DEC, DC, 
        weights1=weight_data, verbose=verbose, 
        c_api_timer=True, is_comoving_dist=True, weight_type="pair_product")
    pickle_data_single( data = DD_counts, root = dir, pickle_name = 'dd' )
    DR_counts, apitimeDR = DDsmu_mocks(0, cosmology, nthreads, mumax, nmubin, sbin, 
                              RA, DEC, DC, weights1=weight_data, 
                              RA2=RAND_RA, DEC2=RAND_DEC, CZ2=RAND_DC, weights2=weight_rand
                              ,c_api_timer=True, is_comoving_dist=True, verbose=verbose, weight_type="pair_product")
    pickle_data_single( data = DR_counts, root = dir, pickle_name = 'dr' )
    RR_counts, apitimeRR = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubin, sbin, 
                             RAND_RA, RAND_DEC, RAND_DC, weights1=weight_rand,verbose=verbose 
                             ,c_api_timer=True, is_comoving_dist=True, weight_type="pair_product")
    pickle_data_single( data = RR_counts, root = dir, pickle_name = 'rr' )
    #print '\relapsed time ', apitimeDD + apitimeDR + apitimeRR
    """

    #for r in RR_counts[10:]: print("{0:10.6f} {1:10.6f} {2:10.6f} {3:10.1f}"
    #                           " {4:10d} {5:10.6f}".format(r['rmin'], r['rmax'],
    #                           r['rpavg'], r['pimax'], r['npairs'], r['weightavg']))

    
    #stop
    #N = data.size
    #rand_N = rand.size
    
    #smin = 40
    #smax = 180
    #snbin = 29
    
    #sbin, step = np.linspace(smin, smax, snbin, retstep=True)
   
    #sbin_center = sbin[:-1] + step/2.
    #mubin, mus = np.linspace(0.0, 1.0, 101, retstep = True)
    #mucenter = mubin[:-1]+mus/2.
    
    #xi_smu = Corrfunc.utils.convert_3d_counts_to_cf(N,N,rand_N,rand_N, DD_counts, DR_counts, DR_counts, RR_counts )


    #xi_smu, DD, DR, RR = direct_cf_smu( DD_counts, DR_counts, RR_counts, sbin, mubin, N, Nrand)
    xi_smu, DD, DR, RR = cf_smu( DD_counts, DR_counts, RR_counts, sbin, mubin, N, Nrand)
    xi_smu_reshape = xi_smu.reshape(sbin_center.size, mubin.size)
    DAT0 = np.column_stack(( DD, DR, RR, xi_smu ))
    DATs = np.column_stack(( sbin[:-1], sbin_center, sbin[1:] ))
    np.savetxt(dir + 'npairs.txt', DAT0, header='DD, DR, RR, xi_smu')
    np.savetxt(dir + 'sbin.txt', DATs, header='smin, smid, smax (Mpc/h)')
    np.savetxt(dir + 'mubin.txt', mubin )
    
    import scipy
    legendre0 = np.array([scipy.special.eval_legendre(0,m) for m in mubin]).ravel() 
    legendre2 = np.array([scipy.special.eval_legendre(2,m) for m in mubin]).ravel() 

    xi_monopole = np.sum(xi_smu_reshape * legendre0.T, axis = 1)/mubin.size
    xi_quadrupole = 5.0*np.sum(xi_smu_reshape * legendre2.T, axis = 1)/mubin.size


    DAT = np.column_stack(( sbin[:-1], sbin_center, sbin[1:], xi_monopole, xi_quadrupole ))
    np.savetxt(dir + 'cfz_multipole_single.txt', DAT)
    print 'file saved to ', dir + 'cfz_multipole_single.txt'


def correlation_function_multipoles_single( data, rand, weight_data = None, weight_rand = None, nthreads = 20, dir = './', 
    smin=5,smax=200, snbin=40, verbose=True ):
    import Corrfunc

    cosmology = 2

    # Create the bins array

    #transverse separation
    rmin = 0.01
    rmax = 250.0
    nbins = 500
    
    rbins, rstep = np.linspace(rmin, rmax, nbins+1, retstep=True)
    # Specify the distance to integrate along line of sight
    #pimax = 100.0
    mumax = 1.0
    nmubin = 101
    mubin, mus = np.linspace(0.0, 1.0, nmubin, retstep = True)
    mucenter = mubin[:-1]+mus/2.

    sbin, step = np.linspace(smin, smax, snbin, retstep=True)
    sbin_center = sbin[:-1] + step/2.

    RA = data['RA']
    DEC = data['DEC'].astype(RA.dtype)
    DC = data['DC'].astype(RA.dtype)

    RAND_RA = rand['RA'].astype(RA.dtype)
    RAND_DEC = rand['DEC'].astype(RA.dtype)
    RAND_DC = rand['DC'].astype(RA.dtype)
    


    #weight_data = None
    #weight_rand = None
    N = data.size
    Nrand = rand.size
    if weight_data is not None:
        N = np.sum(weight_data)
    if weight_rand is not None:
        Nrand = np.sum(weight_rand)

        #weight_data = weight[0]
        #weight_rand = weight[1]
    #    weight_data = data['WEIGHT_FKP']*data['WEIGHT_SYSTOT']*( data['WEIGHT_CP'] + data['WEIGHT_NOZ'] - 1.)
    #    weight_rand = rand['WEIGHT_FKP']


    if weight_data.dtype != RA.dtype: 
        weight_data = weight_data.astype(RA.dtype)
        weight_rand = weight_rand.astype(RA.dtype)


    #print '.',

    DD_counts = load_pickle_data_single( root = dir, pickle_name = 'dd' )
    DR_counts = load_pickle_data_single( root = dir, pickle_name = 'dr' )
    RR_counts = load_pickle_data_single( root = dir, pickle_name = 'rr' )

    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    import pickle

    if DD_counts is None : 
        DD_counts, apitimeDD = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubin, sbin, RA, DEC, DC, 
            weights1=weight_data, verbose=verbose, 
            c_api_timer=True, is_comoving_dist=True, weight_type="pair_product")
        pickle_data_single( data = DD_counts, root = dir, pickle_name = 'dd' )

    if DR_counts is None : 
        DR_counts, apitimeDR = DDsmu_mocks(0, cosmology, nthreads, mumax, nmubin, sbin, 
                                  RA, DEC, DC, weights1=weight_data, 
                                  RA2=RAND_RA, DEC2=RAND_DEC, CZ2=RAND_DC, weights2=weight_rand
                                  ,c_api_timer=True, is_comoving_dist=True, verbose=verbose, weight_type="pair_product")
        pickle_data_single( data = DR_counts, root = dir, pickle_name = 'dr' )

    if RR_counts is None : 
        RR_counts, apitimeRR = DDsmu_mocks(1, cosmology, nthreads, mumax, nmubin, sbin, 
                                 RAND_RA, RAND_DEC, RAND_DC, weights1=weight_rand,verbose=verbose 
                                 ,c_api_timer=True, is_comoving_dist=True, weight_type="pair_product")
        pickle_data_single( data = RR_counts, root = dir, pickle_name = 'rr' )
    #print '\relapsed time ', apitimeDD + apitimeDR + apitimeRR

    #for r in RR_counts[10:]: print("{0:10.6f} {1:10.6f} {2:10.6f} {3:10.1f}"
    #                           " {4:10d} {5:10.6f}".format(r['rmin'], r['rmax'],
    #                           r['rpavg'], r['pimax'], r['npairs'], r['weightavg']))

    
    #stop
    #N = data.size
    #rand_N = rand.size
    
    #smin = 40
    #smax = 180
    #snbin = 29
    
    #sbin, step = np.linspace(smin, smax, snbin, retstep=True)
   
    #sbin_center = sbin[:-1] + step/2.
    #mubin, mus = np.linspace(0.0, 1.0, 101, retstep = True)
    #mucenter = mubin[:-1]+mus/2.
    
    #xi_smu = Corrfunc.utils.convert_3d_counts_to_cf(N,N,rand_N,rand_N, DD_counts, DR_counts, DR_counts, RR_counts )


    xi_smu, DD, DR, RR = direct_cf_smu( DD_counts, DR_counts, RR_counts, sbin, mubin, N, Nrand)
    xi_smu_reshape = xi_smu.reshape(sbin_center.size, mubin.size)
    DAT0 = np.column_stack(( DD, DR, RR, xi_smu ))
    DATs = np.column_stack(( sbin[:-1], sbin_center, sbin[1:] ))
    np.savetxt(dir + 'npairs.txt', DAT0, header='DD, DR, RR, xi_smu')
    np.savetxt(dir + 'sbin.txt', DATs, header='smin, smid, smax (Mpc/h)')
    np.savetxt(dir + 'mubin.txt', mubin )
    
    import scipy
    legendre0 = np.array([scipy.special.eval_legendre(0,m) for m in mubin]).ravel() 
    legendre2 = np.array([scipy.special.eval_legendre(2,m) for m in mubin]).ravel() 

    xi_monopole = np.sum(xi_smu_reshape * legendre0.T, axis = 1)/mubin.size
    xi_quadrupole = 5.0*np.sum(xi_smu_reshape * legendre2.T, axis = 1)/mubin.size


    DAT = np.column_stack(( sbin[:-1], sbin_center, sbin[1:], xi_monopole, xi_quadrupole ))
    np.savetxt(dir + 'cfz_multipole_single.txt', DAT)
    print 'file saved to ', dir + 'cfz_multipole_single.txt'

    
    #return sbin_center, np.hstack([xi_monopole, xi_quadrupole])
    


def correlation_function_multipoles(data = None, rand = None, zlabel = 'Z', njack = 30, weight = None, suffix = '', out = None, nthreads = 20):
    # jk sampling
    import os, sys
    from suchyta_utils import jk
    print 'calculate correlation function multipoles 0 and 2'

    dir = 'data_txt/pair_counting/'+suffix+'/'
    if not os.path.exists(dir) : os.makedirs(dir)
    filename = dir+'cfz_multipole.txt'
    print '# of jackknife sample :', njack
    print '# of threads :', nthreads
    print 'Ndata :', data.size, ' Nrand :', rand.size
    print 'corr file will be saved to '+filename
    #r, xi, xierr = _acf( data, rand, weight = weight )
    

    if 'DC' not in data.dtype.names : 
        from astropy.cosmology import FlatLambdaCDM

        h = 0.68
        Om0 = 0.305
        cosmo = FlatLambdaCDM(H0=h*100, Om0=0.305)
        print 'Calculate comoving distance with Planck cosmology \nH0={}, Om0={}'.format(100*h, Om0)
        sys.stdout.flush()

        r = cosmo.comoving_distance(data[zlabel]).value *h
        r_rand = cosmo.comoving_distance(rand[zlabel]).value *h

        from numpy.lib import recfunctions as rf
        print 'Cannot find comoving distance column... Adding Comoving distance column'
        sys.stdout.flush()
        data = rf.append_fields( data, 'DC', data = r )
        rand = rf.append_fields( rand, 'DC', data = r_rand )
    
    print 'JK sampling'
    sys.stdout.flush()
    
    jkfile = './.jkregion.'+suffix

    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cfz_multipoles, jkargs=[ data, rand ], jkargsby=[[raTag, decTag],[raTag, decTag]], regions = jkfile, 
        nojkkwargs={'weight':weight, 'nthreads':nthreads, 'suffix':suffix})
    jktest.DoJK( regions = jkfile, mpi=False ) ## never make mpi True!!! ( mpi is already implemented in sub function )
    jkresults = jktest.GetResults(jk=True, full = False)
    os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    # getting jk err
    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    xi_mean = np.mean(xi_i, axis = 0)
    norm = 1. * (njack-1)/njack


    xi_cov = 0
    for k in range(njack):
        xi_cov +=  (xi_i[k] - xi_mean)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)



    #filename = 'data_txt/cfz_multipole_comparison_'+suffix+'.txt'
    header = 'r (Mpc/h), monopole(r), quadrupole(r), jkerr0, jkerr2'
    DAT = np.column_stack((r, xi_mean[:r.size], xi_mean[r.size:], xijkerr[:r.size], xijkerr[r.size:] ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename

    if out is True : return DAT
    else : return 0

    
def _corrfunc_acf_auto( data1, random1, tmin = 2.5/60, tmax = 250/60., nbins = 20, nthreads=30 ):

    from Corrfunc.utils import convert_3d_counts_to_cf
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

    RA, DEC = data1['RA'], data1['DEC']
    RAND_RA, RAND_DEC = random1['RA'], random1['DEC']
    
    N = RA.size
    rand_N = RAND_RA.size

    tbins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)

    DD_counts = DDtheta_mocks(1, nthreads, tbins, data1['RA'], data1['DEC'], verbose=1)
    print 'DD'
    DR_counts = DDtheta_mocks(0, nthreads, tbins,
                              RA, DEC,
                              RA2=RAND_RA, DEC2=RAND_DEC)
    print 'DR'
    RR_counts = DDtheta_mocks(1, nthreads, tbins, RAND_RA, RAND_DEC)
    print 'RR'
    wtheta = convert_3d_counts_to_cf(N, N, rand_N, rand_N,
                                     DD_counts, DR_counts,
                                     DR_counts, RR_counts)
    
    return tbins[:-1], wtheta
    

def corrfunc_angular_correlation(data = None, rand = None, njack = 30, 
				nbins = 20, min_sep = 2.5/60, max_sep = 250./60,
				weight = None, suffix = '', out = None, dir = './'):
    # jk sampling
    import os
    from suchyta_utils import jk
    print 'calculate angular correlation function'
    #r, xi, xierr = _acf( data, rand, weight = weight )

    jkfile = './jkregion.txt'

    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _corrfunc_acf_auto, jkargs=[ data, rand ],
	nojkkwargs = {'nbins':nbins, 'tmin':min_sep, 'tmax':max_sep},
        jkargsby=[[raTag, decTag],[raTag, decTag]],
        regions = jkfile, mpi=False)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = False)
    #os.remove(jkfile)
    norm = 1. * (njack-1)/njack

    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    xi_i = xi_i.reshape(njack, r.size)
    xi = np.mean( xi_i, axis=0)
    xi_dat = np.vstack(( r, xi_i)).T

    xi_cov = np.zeros((r.size, r.size))
    for i in range(r.size):
        for j in range(r.size):
            xi_cov[i][j] = norm* np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))

    xijkerr = np.sqrt(xi_cov.diagonal())


    filename = dir+'/corrfunc_acf_auto'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt(dir+'/corrfunc_acf_auto'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt(dir+'/corrfunc_acf_auto'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "saving data file to : ",filename
    if out is True : return DAT



def _corrfunc_acf_cross( data1, data2, random1, random2, tmin = 2.5/60, tmax = 250/60., nbins = 20, nthreads=30 ):

    from Corrfunc.utils import convert_3d_counts_to_cf
    from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks

    RA, DEC = data1['RA'], data1['DEC']
    RA2, DEC2 = data2['RA'], data2['DEC']
    RAND_RA, RAND_DEC = random1['RA'], random1['DEC']
    RAND_RA2, RAND_DEC2 = random2['RA'], random2['DEC']
    
    N = RA.size
    rand_N = RAND_RA.size
    
    N2 = RA2.size
    rand_N2 = RAND_RA2.size

    tbins = np.logspace(np.log10(tmin), np.log10(tmax), nbins + 1)

    DD_counts = DDtheta_mocks(0, nthreads, tbins,
                              RA, DEC,
                              RA2=RA2, DEC2=DEC2)
    DR_counts = DDtheta_mocks(0, nthreads, tbins,
                              RA, DEC,
                              RA2=RAND_RA2, DEC2=RAND_DEC2)
    RD_counts = DDtheta_mocks(0, nthreads, tbins,
                          RAND_RA, RAND_DEC,
                          RA2=RA2, DEC2=DEC2 )
    
    RR_counts = DDtheta_mocks(0, nthreads, tbins,
                      RAND_RA, RAND_DEC,
                      RA2=RAND_RA2, DEC2=RAND_DEC2)

    print '.\r',
    wtheta = convert_3d_counts_to_cf(N, N2, rand_N, rand_N2,
                                     DD_counts, DR_counts,
                                     RD_counts, RR_counts)
    
    return tbins[:-1], wtheta


def corrfunc_cross_angular_correlation(data = None, data2 = None, rand = None, rand2= None,
                            njack = 30,  nbins = 20, min_sep = 2.5/60, max_sep = 250/60, weight = None,
                            suffix = '', out=None, dir = './'):
    # jk sampling
    import os
    from suchyta_utils import jk

    jkfile = './jkregion.txt'

    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _corrfunc_acf_cross, jkargs=[data, data2, rand, rand2],
        jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag],[raTag, decTag]],
        nojkkwargs = {'nbins':nbins, 'tmin':min_sep, 'tmax':max_sep}, jkargspos=None, regions = jkfile, mpi=False)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = False)
    #os.remove(jkfile)

    norm = 1. * (njack-1)/njack
    # getting jk err

    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    xi_i = xi_i.reshape(njack, r.size)
    #xi_dat = np.column_stack((r, xi_i))
    xi_dat = np.vstack(( r, xi_i)).T


    xi = np.mean( xi_i, axis=0)

    m1, m2 = np.mgrid[0:njack, 0:njack]
    xi_cov = np.zeros((r.size, r.size))

    for i in range(njack):
        xi_cov += norm * (xi_i[i,:][m1] - xi[m1]) * (xi_i[i,:][m2] - xi[m2])

    xijkerr = np.sqrt( xi_cov.diagonal())


    filename = dir+'/corrfunc_acf_cross'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt(dir+'/corrfunc_acf_cross'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt(dir+'/corrfunc_acf_cross'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "\nsaving data file to : ",filename
    if out is True : return DAT



def _acf(data, rand, weight = None, nbins = 20, min_sep = 2.5/60., max_sep = 250/60.):
    # cmass and balrog : all systematic correction except obscuration should be applied before passing here
    
    import treecorr
    
    weight_data = None
    weight_rand = None
    
    if weight is not None:
        weight_data = weight[0]
        weight_rand = weight[1]
        if weight_data is True : 
            weight_data = data['WEIGHT']
        else : weight_data = None

        if weight_rand is True : 
            weight_rand = rand['WEIGHT_RAND']
        else : weight_rand = None

        #weight_data = data['WEIGHT']#*data['WEIGHT_SYSTOT']*( data['WEIGHT_CP'] + data['WEIGHT_NOZ'] - 1.)
        #weight_rand = rand['WEIGHT']

    #print weight_data.size
    #print rand.size
    
    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand, ra_units='deg', dec_units='deg')

    #nbins = 20
    #bin_size = 0.5
    #min_sep = 2.5/60.
    #max_sep = 250/60.
    sep_units = 'degree'
    
    print '.',

    #dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    #dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    #rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    
    dd.process(cat)
    dr.process(cat,cat_rand)
    rr.process(cat_rand)
    
    xi, varxi = dd.calculateXi(rr,dr)
    
    return dd.meanr, xi, np.sqrt(varxi)
    #return dd, dr, rr

def angular_correlation_poisson(data, rand, weight_data = None, weight_rand = None, 
    nbins = 20, min_sep = 2.5/60., max_sep = 250/60., dir = './', suffix=''):
    # cmass and balrog : all systematic correction except obscuration should be applied before passing here
    
    import treecorr
    
    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand, ra_units='deg', dec_units='deg')

    #nbins = 20
    #bin_size = 0.5
    #min_sep = 2.5/60.
    #max_sep = 250/60.
    sep_units = 'degree'

    dd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    
    dd.process(cat)
    dr.process(cat,cat_rand)
    rr.process(cat_rand)
    
    xi, varxi = dd.calculateXi(rr,dr)
    errxi = np.sqrt(varxi)

    filename = dir+'/acf_auto'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((dd.meanr, xi, errxi ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    #np.savetxt(dir+'/acf_auto'+suffix+'.cov', xi_cov, header=''+str(r))
    #np.savetxt(dir+'/acf_auto'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "saving data file to : ",filename


def angular_correlation(data = None, rand = None, njack = 30, nbins=20, min_sep = 2.5/60, max_sep=250/60., weight = None, mpi=True, suffix = '', out = None, dir = './'):
    # jk sampling
    import os
    from suchyta_utils import jk
    print 'calculate angular correlation function'
    #r, xi, xierr = _acf( data, rand, weight = weight )
    #
    #os.system('rm -rf ./jkregion.txt')
    jkfile = './jkregion.txt'

    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _acf, jkargs=[ data, rand ], 
        jkargsby=[[raTag, decTag],[raTag, decTag]], 
        regions = jkfile, 
	nojkkwargs = {'weight':weight, 'nbins':nbins, 'min_sep':min_sep, 'max_sep':max_sep},
	mpi=mpi)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = False)
    os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    # getting jk err
    #xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    #xi_mean = np.mean(xi_i, axis = 1)
    norm = 1. * (njack-1)/njack

    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    xi_i = xi_i.reshape(njack, r.size)
    xi = np.mean( xi_i, axis=0)
    xi_dat = np.vstack(( r, xi_i)).T

    #xi_cov = 0
    #for k in range(njack):
    #    xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    #xijkerr = np.sqrt(norm * xi_cov)
    """
    xi_cov = 0
    for ki in range(njack):
        for kj in range(njack):
            xi_cov +=  (xi_i[ki] - xi)*(xi_i[kj] - xi) # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)

    filename = 'data_txt/acf_comparison'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename

    if out is True : return DAT
    else : return 0
    """

    xi_cov = np.zeros((r.size, r.size))
    for i in range(r.size):
        for j in range(r.size):
            #covkikj = 0
            #for ki in range(njack):
            #    for kj in range(njack):
            #        covkikj += (xi_i[ki, i] - xi[i]) * (xi_i[kj,j]-xi[j] )
            #xi_cov[i][j] = norm * np.sum( (xi_i[:, i][m1] - xi[i]) * (xi_i[:,j][m2]-xi[j] ))
            xi_cov[i][j] = norm* np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))
            #xi_cov[i][j] = covkikj
    #inv = np.linalg.inv(xi_cov)
    
    xijkerr = np.sqrt(xi_cov.diagonal())


    filename = dir+'/acf_auto'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt(dir+'/acf_auto'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt(dir+'/acf_auto'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "saving data file to : ",filename
    if out is True : return DAT




def _cross_acf(data, data2, rand, rand2, nbins = 20, min_sep = 2.5/60., max_sep = 250/60., weight = None):
    
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
        if weight_data1 == True : weight_data1 = data['WEIGHT']#weight[0]
        if weight_rand1 == True : weight_rand1 = rand['WEIGHT_RAND']#weight[1]
        if weight_data2 == True : weight_data2 = data2['WEIGHT']#weight[2]
        if weight_rand2 == True : weight_rand2 = rand2['WEIGHT_RAND']#weight[3]


    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data1, ra_units='deg', dec_units='deg')
    cat2 = treecorr.Catalog(ra=data2['RA'], dec=data2['DEC'], w= weight_data2, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand1, ra_units='deg', dec_units='deg')
    cat_rand2 = treecorr.Catalog(ra=rand2['RA'], dec=rand2['DEC'], is_rand=True, w = weight_rand2, ra_units='deg', dec_units='deg')

    #nbins = 30
    #bin_size = 0.2
    #min_sep = 0.1
    #sep_units = 'arcmin'

    #nbins = 20
    #bin_size = 0.5
    #min_sep = 2.5/60.
    #max_sep = 250/60.
    sep_units = 'degree'

    dd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    rd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
 

    # Randy-Szalay estimator
    # [ D1D2 - D1R2 - D2R1 + R1R2 ]/ R1R2

    print '.',
    dd.process(cat, cat2) 
    dr.process(cat, cat_rand2)
    rd.process(cat2, cat_rand)
    rr.process(cat_rand, cat_rand2)
    
    xi, varxi = dd.calculateXi(rr,dr,rd)

    return dd.meanr, xi, varxi


def cross_angular_correlation(data = None, data2 = None, rand = None, rand2= None, 
                            njack = 30,  nbins = 20, min_sep = 2.5/60, max_sep = 250/60, weight = None, 
                            mpi=True, suffix = '', out=None, dir = './'):
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
 
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cross_acf, jkargs=[data, data2, rand, rand2], 
        jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag],[raTag, decTag]], 
        nojkkwargs = {'weight':weight, 'nbins':nbins, 'min_sep':min_sep, 'max_sep':max_sep}, jkargspos=None, regions = jkfile, mpi=mpi)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = False)
    #os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    #r, xi, varxi = _cross_acf(data, data2, rand, rand2, weight = weight )
    norm = 1. * (njack-1)/njack
    # getting jk err
    
    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    xi_i = xi_i.reshape(njack, r.size)
    #xi_dat = np.column_stack((r, xi_i))
    xi_dat = np.vstack(( r, xi_i)).T


    xi = np.mean( xi_i, axis=0)
    #xi_cov = 0
    #for k in range(njack):
    #    xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    #xijkerr = np.sqrt(norm * xi_cov)

    #m1, m2 = np.mgrid[0:njack, 0:njack]
    #xi_cov = np.zeros((r.size, r.size))
    # for i in range(njack):
    #	xi_cov += norm * (xi_i[i,:][m1] - xi[m1]) * (xi_i[i,:][m2] - xi[m2])
   
    #for i in range(r.size):
    #    for j in range(r.size):
    #        xi_cov[i][j] = norm * np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))
    xi_cov = np.zeros((r.size, r.size))
    for i in range(r.size):
        for j in range(r.size):
            xi_cov[i][j] = norm* np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))
    
    xijkerr = np.sqrt( xi_cov.diagonal())


    filename = dir+'/acf_cross'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt(dir+'/acf_cross'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt(dir+'/acf_cross'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "\nsaving data file to : ",filename
    if out is True : return DAT

    



def _cross_kg_acf(data, data2, weight = None, kappa = None):
    
    import treecorr
    if weight is not None : weight = data['WEIGHT']
    if kappa is not None : kappa = data2['KAPPA']


    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight, ra_units='deg', dec_units='deg')
    cat2 = treecorr.Catalog(ra=data2['RA'], dec=data2['DEC'], k= kappa, ra_units='deg', dec_units='deg')
    #cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand1, ra_units='deg', dec_units='deg')
    #cat_rand2 = treecorr.Catalog(ra=rand2['RA'], dec=rand2['DEC'], is_rand=True, w = weight_rand2, ra_units='deg', dec_units='deg')

    #nbins = 30
    #bin_size = 0.2
    #min_sep = 0.1
    #sep_units = 'arcmin'

    nbins = 30
    #bin_size = 0.5
    min_sep = 50/60.
    max_sep = 1000/60.
    sep_units = 'degree'

    nk = treecorr.NKCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    #dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    #rd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    #rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
 

    # Randy-Szalay estimator
    # [ D1D2 - D1R2 - D2R1 + R1R2 ]/ R1R2

    print '.',
    nk.process(cat, cat2) 
    #dr.process(cat, cat_rand2)
    #rd.process(cat2, cat_rand)
    #rr.process(cat_rand, cat_rand2)
    #xi = nk.xi
    xi, varxi = nk.calculateXi()

    return nk.meanr, xi, varxi


def cross_kg_correlation(data = None, data2 = None, njack = 30,  weight = None, kappa = True, mpi=True, suffix = '', out=None, dir='../data_txt/'):
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
 
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cross_kg_acf, jkargs=[data, data2], 
        jkargsby=[[raTag, decTag],[raTag, decTag]], 
        nojkkwargs = {'weight':weight, 'kappa':kappa}, jkargspos=None, regions = jkfile, mpi=mpi)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = False)
    #os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    #r, xi, varxi = _cross_acf(data, data2, rand, rand2, weight = weight )
    norm = 1. * (njack-1)/njack
    # getting jk err
    
    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )

    xi_i = xi_i.reshape(njack, r.size)
    #xi_dat = np.column_stack((r, xi_i))
    xi_dat = np.vstack(( r, xi_i)).T


    xi = np.mean( xi_i, axis=0)
    #xi_cov = 0
    #for k in range(njack):
    #    xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    #xijkerr = np.sqrt(norm * xi_cov)

    #m1, m2 = np.mgrid[0:njack, 0:njack]

    xi_cov = np.zeros((r.size, r.size))
    for i in range(r.size):
        for j in range(r.size):
            #covkikj = 0
            #for ki in range(njack):
            #    for kj in range(njack):
            #        covkikj += (xi_i[ki, i] - xi[i]) * (xi_i[kj,j]-xi[j] )
            #xi_cov[i][j] = norm * np.sum( (xi_i[:, i][m1] - xi[i]) * (xi_i[:,j][m2]-xi[j] ))
            xi_cov[i][j] = np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))
            #xi_cov[i][j] = covkikj
    #inv = np.linalg.inv(xi_cov)
    
    xijkerr = np.sqrt(norm * xi_cov.diagonal())


    filename = dir+'/acf_cross_kg'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt(dir+'/acf_cross_kg'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt(dir+'/acf_cross_kg'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "saving data file to : ",filename
    if out is True : return DAT






def _cross_kk_acf(data, data2, kappa1 = None, kappa2 = None, weight1 = None, weight2 = None):
    
    import treecorr
    if kappa1 is not None : kappa1 = data['KAPPA']
    if kappa2 is not None : kappa2 = data2['KAPPA']
    if weight1 is not None : weight1 = data['WEIGHT']
    if weight2 is not None : weight2 = data2['WEIGHT']


    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], k = kappa1, w=weight1, ra_units='deg', dec_units='deg')
    cat2 = treecorr.Catalog(ra=data2['RA'], dec=data2['DEC'], k= kappa2, w=weight2, ra_units='deg', dec_units='deg')
    #cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand1, ra_units='deg', dec_units='deg')
    #cat_rand2 = treecorr.Catalog(ra=rand2['RA'], dec=rand2['DEC'], is_rand=True, w = weight_rand2, ra_units='deg', dec_units='deg')

    #nbins = 30
    #bin_size = 0.2
    #min_sep = 0.1
    #sep_units = 'arcmin'

    nbins = 30
    #bin_size = 0.5
    min_sep = 2.5/60.
    max_sep = 1000/60.
    sep_units = 'degree'

    kk = treecorr.KKCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    #dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    #rd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    #rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
 

    # Randy-Szalay estimator
    # [ D1D2 - D1R2 - D2R1 + R1R2 ]/ R1R2

    print '.',
    kk.process(cat, cat2) 

    #kk.process_cross(cat, cat2)
    #kk.write('cross_kk.txt')
    rr = np.logspace(np.log(min_sep), np.log(max_sep), nbins)
    #print kk.npairs
    #print kk.weight
    #print kk.xi
    #dr.process(cat, cat_rand2)
    #rd.process(cat2, cat_rand)
    #rr.process(cat_rand, cat_rand2)
    #xi = nk.xi
    #xi, varxi = kk.calculateXi()

    return kk.meanr, kk.xi, kk.varxi


def cross_kk_correlation(data = None, data2 = None, njack = 30,  kappa1 = None, kappa2 = True, weight1 = None, weight2=None, mpi=True, suffix = '', out=None):
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
 
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cross_kk_acf, jkargs=[data, data2], 
        jkargsby=[[raTag, decTag],[raTag, decTag]], 
        nojkkwargs = {'kappa1':kappa1, 'kappa2':kappa2, 'weight1':weight1, 'weight2':weight2}, jkargspos=None, regions = jkfile, mpi=mpi)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = False)
    #os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    #r, xi, varxi = _cross_acf(data, data2, rand, rand2, weight = weight )
    norm = 1. * (njack-1)/njack
    # getting jk err
    
    r = jkresults['jk'][0][0]
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )

    xi_i = xi_i.reshape(njack, r.size)
    #xi_dat = np.column_stack((r, xi_i))
    xi_dat = np.vstack(( r, xi_i)).T


    xi = np.mean( xi_i, axis=0)
    #xi_cov = 0
    #for k in range(njack):
    #    xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    #xijkerr = np.sqrt(norm * xi_cov)

    #m1, m2 = np.mgrid[0:njack, 0:njack]

    xi_cov = np.zeros((r.size, r.size))
    for i in range(r.size):
        for j in range(r.size):
            #covkikj = 0
            #for ki in range(njack):
            #    for kj in range(njack):
            #        covkikj += (xi_i[ki, i] - xi[i]) * (xi_i[kj,j]-xi[j] )
            #xi_cov[i][j] = norm * np.sum( (xi_i[:, i][m1] - xi[i]) * (xi_i[:,j][m2]-xi[j] ))
            xi_cov[i][j] = np.sum( (xi_i[:, i] - xi[i]) * (xi_i[:,j]-xi[j] ))
            #xi_cov[i][j] = covkikj
    #inv = np.linalg.inv(xi_cov)
    
    xijkerr = np.sqrt(norm * xi_cov.diagonal())


    filename = 'data_txt/acf_cross_kk'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    np.savetxt('data_txt/acf_cross_kk'+suffix+'.cov', xi_cov, header=''+str(r))
    np.savetxt('data_txt/acf_cross_kk'+suffix+'.jk_corr', xi_dat, header='r  jksamples')
    print "saving data file to : ",filename
    if out is True : return DAT


    
def _twocf(data, rand, ztag = None, weight = None):
    # cmass and balrog : all systematic correction except obscuration should be applied before passing here
    
    import treecorr
    import astropy
    
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


    r = cosmo.comoving_distance(data[ztag]).value# * cosmo.h
    r_rand = cosmo.comoving_distance(rand[ztag]).value# * cosmo.h
    
    weight_data = None
    weight_rand = None
    
    if weight is not None:
        weight_data = weight[0]
        weight_rand = weight[1]
    
    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], r = r, w = weight_data, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], r = r_rand, is_rand=True, w = weight_rand, ra_units='deg', dec_units='deg')

    nbins = 50
    min_sep = 1 #2.5/ 60.
    max_sep = 250/60.
    sep_units = 'degree'
    
    #min_sep = 1
    #max_sep = 200
    #nbins = 20
    
    #dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    #dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    #rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dd = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, max_sep = max_sep, min_sep= min_sep, sep_units=sep_units)
    
    dd.process(cat)
    dr.process(cat,cat_rand)
    rr.process(cat_rand)
    
    xi, varxi = dd.calculateXi(rr,dr)
    
    return dd.meanr, xi, varxi



def two_point_function(data = None, rand = None, njack = 30, ztag = None, weight = None, suffix = '', out = None):
    # jk sampling
    import os
    from suchyta_utils import jk
    print 'calculate angular correlation function'
    #r, xi, xierr = _acf( data, rand, weight = weight )
    
    jkfile = './jkregion.txt'
 
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _twocf, jkargs=[ data, rand ], jkargsby=[[raTag, decTag],[raTag, decTag]], regions = jkfile, nojkkwargs = {'weight':weight, 'ztag':ztag })
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    r, xi, varxi = jkresults['full']
    
    # getting jk err
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    norm = 1. * (njack-1)/njack
    #xi_cov = np.sum((xi_i - mean_xi)**2) * norm
    
    xi_cov = 0
    for k in range(njack):
        xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)

    filename = 'data_txt/acf_comparison'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename

    if out is True : return DAT
    else : return 0


    


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
    
    norm = (njack-1)*1./njack
    #cov = np.std(it_j - full_j)**2 * norm
    cov = np.sum((it_j - it_j.mean())**2) * norm
    
    os.remove(jkfile)
    return np.mean(it_j), np.sqrt(cov)


def _Sigma_crit(zl, zs):
    
    # angular diameter
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70,Om0=0.274)
    
    from astropy import constants as const
    from astropy import units as u
    
    h = 0.7
    c = const.c.to(u.megaparsec / u.s)
    G = const.G.to(u.megaparsec**3 / (u.M_sun * u.s**2))
    
    dA_l = cosmo.angular_diameter_distance(zl) * h # unit Mpc
    dA_s = cosmo.angular_diameter_distance(zs) * h
    try:dA_ls = cosmo.angular_diameter_distance_z1z2(zl,zs) * h
    except ValueError:
        if zs < zl: dA_ls = 0
        else : print "ValueError"
    
    Sigma_crit = c**2 / (4 * np.pi * G) * dA_s/(dA_l * dA_ls)
    return Sigma_crit
 
    
def _LS_tomo(lense, source, rand, weight = None):
    
    import treecorr
    """
    gamma_t(theta)
    -------------------------
    rand : random sample. only needed when booster correction needed
    
    : Calculate angular lensing signal for one lensing bin and one source bin as a function of theta
    
    """
    weight_lense, weight_source, weight_rand = [ None, None, None ]
    if weight is not None:
        weight_lense, weight_source, weight_rand =  weight[0], weight[1], weight[2]
        
    # lensing Signal
    lense_cat = treecorr.Catalog(ra=lense['RA'], dec=lense['DEC'], \
                                     w = weight_lense, ra_units='deg', dec_units='deg')
    source_cat = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], \
                                     w = weight_source, g1=source['E1'], g2 = source['E2'],\
                                     ra_units='deg', dec_units='deg')
        
    min_sep = 2.5
    max_sep = 250
    nbins = 20
    
    ng = treecorr.NGCorrelation(nbins = nbins, \
                                min_sep= min_sep, max_sep=max_sep, sep_units='arcmin')
        
    ng.process(lense_cat, source_cat)
    return ng.meanr, ng.xi



def LensingSignal_tomo(lense = None, source = None, rand = None, weight = None, suffix = '', out=None, njack = 10):
    
    print "Calculating lensing signal "
    
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( lense[raTag], lense[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _LS_tomo, jkargs=[lense, source, rand], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag]], nojkkwargs={'weight':weight}, regions = jkfile)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    theta, LensSignal = jkresults['full']
    
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

    filename = 'data_txt/ggl_gammat_'+suffix+'.txt'
    header = 'r_p_bins, gamma_t, jkerr'
    DAT = np.column_stack((theta, LensSignal, LSjkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename
    if out is True : return DAT



    
    
    

def _LS(lense, source, rand, zbin = None, weight = None, Boost = False):
    
    from utils import divide_bins
    
    """
        Calculate Lensing Signal deltaSigma, Boost Factor, Corrected deltaSigma
        
        parameter
        ---------
        lense : lense catalog
        source : source catalog (shear catalog that contains e1, e2)
        rand : random catalog
        
        """
    
    import treecorr
    
    z_l_min, z_l_max, z_s_min, z_s_max = zbin
    weight_lense, weight_source, weight_rand = [ None, None, None ]
    
    if weight is not None:
        weight_lense, weight_source, weight_rand =  weight[0], weight[1], weight[2]

    num_lense_bin = 1
    num_source_bin = 3
    z_l_bins = np.linspace(z_l_min, z_l_max, num_lense_bin)
    z_s_bins = np.linspace(z_s_min, z_s_max, num_source_bin)
    #matrix1, matrix2 = np.mgrid[0:z_l_bins.size, 0:z_s_bins.size]
    source = source[ (source['DESDM_ZP'] > z_s_min ) & (source['DESDM_ZP'] < z_s_max )]
    #lense = lense[ (lense['DESDM_ZP'] > 0.45) & (lense['DESDM_ZP'] < 0.55) ]
    #rand = rand[ (rand['DESDM_ZP'] > 0.45) & (rand['DESDM_ZP'] < 0.55) ]
    
    z_l_bincenter, lense_binned_cat,_ = divide_bins( lense, Tag = 'DESDM_ZP', min = z_l_min, max = z_l_max, bin_num = num_lense_bin)
    z_s_bincenter, source_binned_cat,_ = divide_bins( source, Tag = 'DESDM_ZP', min = z_s_min, max = z_s_max, bin_num = num_source_bin)
    z_r_bincenter, rand_binned_cat,_ = divide_bins( rand, Tag = 'DESDM_ZP', min = z_l_min, max = z_l_max, bin_num = num_lense_bin)
    
    # angular diameter
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70,Om0=0.274)
    
    from astropy import constants as const
    from astropy import units as u
    
    h = 0.7
    c = const.c.to(u.megaparsec / u.s)
    G = const.G.to(u.megaparsec**3 / (u.M_sun * u.s**2))
    
    dA_l = cosmo.angular_diameter_distance(z_l_bincenter) * h # unit Mpc
    dA_s = cosmo.angular_diameter_distance(z_s_bincenter) * h
    matrix1, matrix2 = np.mgrid[0:z_l_bincenter.size, 0:z_s_bincenter.size]
    dA_ls = cosmo.angular_diameter_distance_z1z2(z_l_bincenter[matrix1],z_s_bincenter[matrix2]) * h
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


    # lensing Signal
    
    lense_cat_tot = treecorr.Catalog(ra=lense['RA'], dec=lense['DEC'], w = weight_lense, ra_units='deg', dec_units='deg')
    source_cat_tot = treecorr.Catalog(ra=source['RA'], dec=source['DEC'], w = weight_source, g1=source['E1'], g2 = source['E2'], ra_units='deg', dec_units='deg')
    
    gamma_matrix = []
    #varxi_matrix = []
    mean_rp = []
    
    for dA in dA_l:
        min_sep = theta[0] * dA_l[0]/dA
        ng = treecorr.NGCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
        ng.process(lense_cat_tot, source_cat_tot)
        gamma_matrix.append(ng.xi)
        #varxi_matrix.append(ng.varxi)
        mean_rp.append(ng.meanr * dA)
        
    #varxi = np.sum(varxi_matrix, axis= 0)/len(dA_l)
    gamma_matrix = np.array(gamma_matrix)
    gamma_avg = np.sum(np.array(gamma_matrix), axis = 0)/len(z_l_bins)
    mean_rp = np.array(mean_rp)   
    mean_rp = np.mean(mean_rp, axis = 0)
    
    
    
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

    delta_sigma = np.array(delta_sigma)/(len(z_l_bincenter) * len(z_s_bincenter))
    
    if Boost is True :
                                        
        Corrected_delta_sigma = np.array(BoostFactor) * delta_sigma
        return mean_rp, np.array(BoostFactor), delta_sigma, Corrected_delta_sigma

    else : pass
    #delta_sigma_1bin = np.array(delta_sigma_1bin)/len(z_s_bins)
    #summed_sigma_crit = np.array(summed_sigma_crit)/(len(z_l_bins) * len(z_s_bins))
    #error_tot = np.sqrt(summed_sigma_crit**2 * varxi)
    

    return mean_rp, delta_sigma


def LensingSignal(lense = None, source = None, rand = None, zbin = [0.45, 0.55, 0.7, 1.0], weight = None, suffix = '', out=None, njack = 10):
    
    print "Calculating lensing signal "
    
    # jk sampling
    import os
    from suchyta_utils import jk
    
    z_l_min, z_l_max, z_s_min, z_s_max = zbin
    
    print 'z_l = ({}, {})'.format(z_l_min, z_l_max),
    print ' z_s = ({}, {})'.format(z_s_min, z_s_max)
    
    jkfile = './jkregion.txt'
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( lense[raTag], lense[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _LS, jkargs=[lense, source, rand], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag]], nojkkwargs={'weight':weight, 'zbin':zbin}, regions = jkfile)
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
    if out is True : return DAT




