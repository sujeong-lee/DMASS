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


def _acf(data, rand, weight = None):
    # cmass and balrog : all systematic correction except obscuration should be applied before passing here
    
    import treecorr
    
    weight_data = None
    weight_rand = None
    
    if weight is not None:
        weight_data = weight[0]
        weight_rand = weight[1]
    
    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand, ra_units='deg', dec_units='deg')

    nbins = 20
    bin_size = 0.5
    min_sep = .01
    sep_units = 'degree'
    dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dd.process(cat)
    dr.process(cat,cat_rand)
    rr.process(cat_rand)
    
    xi, varxi = dd.calculateXi(rr,dr)
    
    return dd.meanr, xi, varxi


def angular_correlation(data = None, rand = None, weight = None, suffix = '', out = None):
    # jk sampling
    import os
    from suchyta_utils import jk
    print 'calculate angular correlation function'
    #r, xi, xierr = _acf( data, rand, weight = weight )
    
    jkfile = './jkregion.txt'
    njack = 30
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _acf, jkargs=[ data, rand ], jkargsby=[[raTag, decTag],[raTag, decTag]], regions = jkfile, nojkkwargs = {'weight':weight})
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    r, xi, varxi = jkresults['full']
    
    # getting jk err
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    norm = 1. * (njack-1)/njack
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





def _cross_acf(data, data2, rand, rand2, weight = None):
    
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


    cat = treecorr.Catalog(ra=data['RA'], dec=data['DEC'], w = weight_data1, ra_units='deg', dec_units='deg')
    cat2 = treecorr.Catalog(ra=data2['RA'], dec=data2['DEC'], w= weight_data2, ra_units='deg', dec_units='deg')
    cat_rand = treecorr.Catalog(ra=rand['RA'], dec=rand['DEC'], is_rand=True, w = weight_rand1, ra_units='deg', dec_units='deg')
    cat_rand2 = treecorr.Catalog(ra=rand2['RA'], dec=rand2['DEC'], is_rand=True, w = weight_rand2, ra_units='deg', dec_units='deg')

    #nbins = 30
    #bin_size = 0.2
    #min_sep = 0.1
    #sep_units = 'arcmin'

    nbins = 20
    bin_size = 0.5
    min_sep = .001
    sep_units = 'degree'
    
    dd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    dr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    rd = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    rr = treecorr.NNCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units=sep_units)
    
    dd.process(cat, cat2)
    dr.process(cat, cat_rand)
    rd.process(cat2, cat_rand2)
    rr.process(cat_rand, cat_rand2)
    
    xi, varxi = dd.calculateXi(rr,dr,rd)
    return dd.meanr, xi, varxi


def cross_angular_correlation(data = None, data2 = None, rand = None, rand2= None, weight = None, suffix = '', out=None):
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
    njack = 10
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( data[raTag], data[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _cross_acf, jkargs=[data, data2, rand, rand2], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag],[raTag, decTag]], nojkkwargs = {'weight':weight}, jkargspos=None, regions = jkfile)
    jktest.DoJK( regions = jkfile )
    jkresults = jktest.GetResults(jk=True, full = True)
    os.remove(jkfile)
    #r, xi, varxi = jkresults['full']
    
    r, xi, varxi = _cross_acf(data, data2, rand, rand2, weight = weight )
    
    # getting jk err
    xi_i = np.array( [jkresults['jk'][i][1] for i in range(njack)] )
    
    norm = 1. * (njack-1)/njack
    xi_cov = 0
    for k in range(njack):
        xi_cov +=  (xi_i[k] - xi)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    xijkerr = np.sqrt(norm * xi_cov)

    filename = 'data_txt/acf_cross'+suffix+'.txt'
    header = 'r, xi, jkerr'
    DAT = np.column_stack((r, xi, xijkerr ))
    np.savetxt( filename, DAT, delimiter=' ', header=header )
    print "saving data file to : ",filename
    if out is True : return DAT


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
    
    njack = len(it_j) #redefine njack
    norm = 1. * (njack-1)/njack
    
    cov = 0
    for k in range(len(it_j)):cov +=  (it_j[k] - full_j)**2 # * (it_j[k][matrix2]- full_j[matrix2] )
    
    cov = norm * cov
    os.remove(jkfile)
    jkerr = np.sqrt(cov)
    
    return full_j, jkerr




def _LS(lense, source, rand, weight = None, Boost = False):
    """
        Calculate Lensing Signal deltaSigma, Boost Factor, Corrected deltaSigma
        
        parameter
        ---------
        lense : lense catalog
        source : source catalog (shear catalog that contains e1, e2)
        rand : random catalog
        
        """
    
    import treecorr
    
    weight_lense, weight_source, weight_rand = weight
    
    z_s_min = 0.7
    z_s_max = 1.0
    
    num_lense_bin = 3
    num_source_bin = 3
    z_l_bins = np.linspace(0.45, 0.55, num_lense_bin)
    z_s_bins = np.linspace(z_s_min, z_s_max, num_source_bin)
    #matrix1, matrix2 = np.mgrid[0:z_l_bins.size, 0:z_s_bins.size]
    source = source[ (source['DESDM_ZP'] > z_s_min ) & (source['DESDM_ZP'] < z_s_max )]
    lense = lense[ (lense['DESDM_ZP'] > 0.45) & (lense['DESDM_ZP'] < 0.55) ]
    rand = rand[ (rand['Z'] > 0.45) & (rand['Z'] < 0.55) ]
    
    z_l_bincenter, lense_binned_cat,_ = divide_bins( lense, Tag = 'DESDM_ZP', min = 0.45, max = 0.55, bin_num = num_lense_bin)
    z_r_bincenter, rand_binned_cat,_ = divide_bins( rand, Tag = 'Z', min = 0.45, max = 0.55, bin_num = num_lense_bin)
    z_s_bincenter, source_binned_cat,_ = divide_bins( source, Tag = 'DESDM_ZP', min = z_s_min, max = z_s_max, bin_num = num_source_bin)
    
    # angular diameter
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70,Om0=0.274)
    
    from astropy import constants as const
    from astropy import units as u
    
    h = 0.7
    c = const.c.to(u.megaparsec / u.s)
    G = const.G.to(u.megaparsec**3 / (u.M_sun * u.s**2))
    
    dA_l = cosmo.angular_diameter_distance(z_l_bins) * h
    dA_s = cosmo.angular_diameter_distance(z_s_bins) * h
    matrix1, matrix2 = np.mgrid[0:z_l_bins.size, 0:z_s_bins.size]
    dA_ls = cosmo.angular_diameter_distance_z1z2(z_l_bins[matrix1],z_s_bins[matrix2]) * h
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
    varxi_matrix = []
    
    for dA in dA_l:
        min_sep = theta[0] * dA_l[0]/dA
        ng = treecorr.NGCorrelation(nbins = nbins, bin_size = bin_size, min_sep= min_sep, sep_units='arcmin')
        ng.process(lense_cat_tot, source_cat_tot)
        gamma_matrix.append(ng.xi)
        varxi_matrix.append(ng.varxi)

    varxi = np.sum(varxi_matrix, axis= 0)/len(dA_l)
    gamma_matrix = np.array(gamma_matrix)
    gamma_avg = np.sum(np.array(gamma_matrix), axis = 0)/len(z_l_bins)


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

    delta_sigma = np.array(delta_sigma)/(len(z_l_bins) * len(z_s_bins))
    
    if Boost is True :
        Corrected_delta_sigma = np.array(BoostFactor) * delta_sigma
        return r_p_bins, np.array(BoostFactor), delta_sigma, Corrected_delta_sigma

    else : pass
    #delta_sigma_1bin = np.array(delta_sigma_1bin)/len(z_s_bins)
    #summed_sigma_crit = np.array(summed_sigma_crit)/(len(z_l_bins) * len(z_s_bins))
    #error_tot = np.sqrt(summed_sigma_crit**2 * varxi)
    
    return r_p_bins, delta_sigma


def LensingSignal(lense = None, source = None, rand = None, weight = None, suffix = '', out=None):
    
    # jk sampling
    import os
    from suchyta_utils import jk
    
    jkfile = './jkregion.txt'
    njack = 10
    raTag, decTag = 'RA', 'DEC'
    jk.GenerateJKRegions( lense[raTag], lense[decTag], njack, jkfile )
    jktest = jk.SphericalJK( target = _LS, jkargs=[lense, source, rand], jkargsby=[[raTag, decTag],[raTag, decTag],[raTag, decTag]], nojkkwargs={'weight':weight}, regions = jkfile)
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




