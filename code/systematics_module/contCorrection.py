#!/usr/bin/env python
#from __future__ import print_function, division

import sys
import os
import numpy as np
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from scipy import linalg



def L_nu_from_magAB(magAB = None, z = None):
    import numpy as np
    
    """Convert absolute magnitude into luminosity (erg s^-1 Hz^-1)."""
    
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    
    m_zero = 30.
    d = cosmo.comoving_distance( z ).value
    factor = 4. * np.pi * d**2.
    L_nu =  factor * 10.**((magAB - m_zero)/(-2.5))
    
    return L_nu

def SW_halobias(mass, z, h=None, Om_M=None, Om_L=None):
    
    """
        from astropy.cosmology import Planck13 as cosmo
        # default parameters
        h = cosmo.h
        Om_M = cosmo.Om0
        Om_L = 1. - Om_M
        """
    """Calculate halo bias, from Seljak & Warren 2004.
        Parameters
        ----------
        mass : ndarray or float
        Halo mass to calculate bias for.
        z : ndarray or float
        Halo z, same type and size as mass.
        h : float, optional
        Hubble parameter, defaults to astropy.cosmology.Planck13.h
        Om_M : float, optional
        Fractional matter density, defaults to
        astropy.cosmology.Planck13.Om0
        Om_L : float, optional
        Fractional dark energy density, defaults to 1-Om_M.
        Returns
        ----------
        ndarray or float
        Returns the halo bias, of same type and size as input mass_halo and
        z_halo. Calculated according to Seljak & Warren 2004 for z = 0.
        For halo z > 0, the non-linear mass is adjusted using the input
        cosmological parameters.
        References
        ----------
        Based on fitting formula derived from simulations in:
        U. Seljak and M.S. Warren, "Large-scale bias and stochasticity of
        haloes and dark matter," Monthly Notices of the Royal Astronomical
        Society, Volume 355, Issue 1, pp. 129-136 (2004).
        """
    M_nl_0 = (8.73 / h) * (10. ** 12.)         # nonlinear mass today [M_sun]
    M_nl = M_nl_0 * (Om_M + Om_L / ((1. + z) ** 3.))  # scaled to z_lens
    x = mass / M_nl
    b = 0.53 + 0.39 * (x ** 0.45) + 0.13 / (40. * x +
                                            1.) + (5.e-4) * (x ** 1.5)
    return b

def haloBiasTinker10(M_halo, z = None):
    
    from colossus.cosmology.cosmology import setCosmology
    cosmo = setCosmology('planck15')
    
    #sigma = cosmo.sigma(R, z, j=0, inverse=False, derivative=False, Pk_source='eh98', filt='tophat')
    # http://bdiemer.bitbucket.org/cosmology_cosmology.html#cosmology.cosmology.Cosmology.sigma
    #delta_c = cosmo.collapseOverdensity()
    
    from colossus.halo import bias
    #HaloModel = bias.MODELS
    HaloBias = bias.haloBias(M_halo, z, 'vir', model='tinker10')
    # http://bdiemer.bitbucket.org/halo_bias.html#halo.bias.haloBias
    # mdef parameter : http://bdiemer.bitbucket.org/halo_mass.html#spherical-overdensity-basics
    
    nu = cosmo.peakHeight(M_halo, z, filt='tophat', Pk_source='eh98', deltac_const=True)
    # http://bdiemer.bitbucket.org/cosmology_cosmology.html#cosmology.cosmology.Cosmology.peakHeight
    
    return nu, HaloBias

def Mag_to_galaxyBias(magAB = None, z = 0.0):
    
    #from astropy.constants import L_sun, M_sun
    #from systematics_weight import L_nu_from_magAB
    """
        Calculate Occupation number of galaxies in haloes by using mass-luminosity relation
    """
    # change magnitude to luminosity
    # binning in luminosity bin
    # replace x axis to halo mass bin
    # calculate weight for each galaxies by using bias-halo relation (Tinker et al. 2010)
    # summing over all sample
    
    Luminosity = L_nu_from_magAB(magAB = magAB, z = z)
    
    # binning in Mag bin
    
    magbin, step = np.linspace(-1 * magAB.max(), -1 * magAB.min(), 200, retstep =True )
    N, _ = np.histogram( -1. * magAB, bins = magbin )
    magbin_center = magbin[:-1] + step/2.
    
    Lbin = np.linspace( np.log10(Luminosity.min()), 13, 200 )
    N_L, _ = np.histogram( np.log10(Luminosity), bins=Lbin )
    
    
    # the mass luminosity relation (Vale et al 2004 )
    A = 5.7e+9
    m = 1e+11
    b,c,d,k = 4., 0.57, 3.72, 0.23
    Mbin2 = np.logspace(16, 22, 100, base=10.)
    Lbin2 = np.log10( A * (Mbin2/m)**b * 1./ (c + (Mbin2/m)**(d*k))**(1./k) )
    
    # get Ocupation number as a ftn of halo mass
    N_gal, _ = np.histogram( np.log10(Luminosity), bins=Lbin2 )
    
    # calculating bias
    # Halo bias as weight ( equation 5) in Seljak et al. 2004 )
    
    Mbin2_center = np.log10( (Mbin2[:-1] + Mbin2[1:])/2. )
    nu, haloBias = haloBiasTinker10(Mbin2_center, z = 0)
    # assign weights to each gal in catalog
    nu_avg = np.mean(nu)
    N_sample = len(magAB)
    bias_g = np.sum( haloBias * N_gal ) *1./N_sample
    
    """
    fig, ax = plt.subplots()
    N_sample = catalog.size
    zbin = np.linspace(0.0, 20, 50)
    bias_g, nu_avg = np.zeros( zbin.size ), np.zeros( zbin.size )
    for i in range(zbin.size):
    nu, haloBias = haloBiasTinker10(Mbin2_center, z = zbin[i])
    nu_avg[i] = np.mean(nu)
    bias_g[i] = np.sum( haloBias * N_gal ) *1./N_sample
    ax.scatter( np.log10(nu), haloBias, marker = '.', alpha = 0.1)
    
    ax.plot(np.log10(nu_avg), bias_g, '--', label='galaxy bias' )
    ax.set_title( 'DMASS galaxy bias')
    ax.set_xlabel(' log10( nu )')
    ax.set_ylabel('galaxy bias')
    #ax.set_xlim(-0.5, 1 )
    #ax.set_ylim(0, 8)
    ax.legend(loc='best')
    figname = 'figure/haloBias.png'
    print 'savefig : ', figname
    fig.savefig(figname)
    #
    # plot
    
    fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2,figsize = (15,15))
    ax.hist( -1 * magAB, bins = magbin, histtype='bar', label='magnitude' )
    ax2.hist(np.log10(Luminosity), bins = Lbin, label = 'Luminosity' )
    ax.set_ylabel('N_gal')
    ax.set_xlabel('MAG_MODEL_I')
    ax2.set_ylabel('N_gal')
    ax2.set_xlabel('log L (L_0)')
    #ax2.set_xscale('log')
    ax.legend()
    ax2.legend()
    
    Mbin = np.logspace(9, 30, 100, base=10.)
    L = np.log10(A * (Mbin/m)**b * 1./ (c + (Mbin/m)**(d*k))**(1./k))
    ax3.plot(np.log10(Mbin), L, )
    ax3.set_xlabel( 'log M_halo ( M_0 )')
    ax3.set_ylabel( 'log L (L_0)')
    ax3.set_xlim(9,20)
    ax3.set_title('The mass luminosity relation (Vale et al 2004)')
    
    ax4.plot( np.log10(Mbin2[:-1]), N_gal)
    ax4.set_xlabel('log M_halo (M_0)')
    ax4.set_ylabel('N_gal')
    figname = 'figure/OccupationNumber.png'
    print 'savefig : ', figname
    fig.savefig(figname)
    
    
    # large scale structure bias plot
    
    fig, ax = plt.subplots()
    ax.plot(np.log10(nu_avg), bias_g )
    #ax.set_xlim(-0.5, 0.6)
    ax.set_title( 'DMASS galaxy bias')
    ax.set_xlabel(' log10( nu )')
    ax.set_ylabel('galaxy bias')
    figname = 'figure/haloBias.png'
    print 'savefig : ', figname
    fig.savefig(figname)
    
    """
    
    return bias_g


def _loadndarray(data):
    r, w, err = data[:,0], data[:,1], data[:,2]
    return r, w, err


class CorrectContaminant():

    def __init__(self):


        # catalog
        self.obs_data = None
        self.true_data = None
        self.cont_data = None
        
        # angular corr ftns
        self.w_obs_data = None
        self.w_true_data = None
        self.w_cross_data = None
        self.w_cont_data = None

        # lensing
        self.L_obs_data = None
        self.L_true_data = None
        self.L_cont_data = None

        # params
        self.f_c = None
        self.eps82, self.eps = None, None #self.ang_stat_err()
        self.eps_len82, self.eps_len = None, None # self.len_stat_err()


    def angular_correlation(self, data=None, rand=None, weight = None, suffix = ''):
        from systematics_module import corr
        w_data = corr.angular_correlation(data = data, rand = rand, weight = weight, suffix = suffix, out = True)
        return w_data
    
    def cross_angular_correlation(self, data = None, data2 = None, rand = None, rand2= None, weight = None, suffix = ''):
        from systematics_module import corr
        w_data = corr.cross_angular_correlation(data = data, data2 = data2, rand = rand, rand2= rand2, weight = weight, suffix = suffix, out =True)
        return w_data

    def LensingSignal(self, lense = None, source = None, rand = None, weight = None, suffix = ''):
        from systematics_module import corr
        L_data = corr.LensingSignal(lense = lense, source = source, rand = rand, weight = weight, suffix = suffix, out=True)
        return L_data
    
    def get_cont_data(self, obs = None, true = None):
        import esutil
        m_dmass, m_true, _ = esutil.htm.HTM(10).match( obs['RA'], obs['DEC'], true['RA'], true['DEC'], 1./3600, maxmatch = 1)
        true_mask = np.zeros( obs.size, dtype = bool )
        true_mask[m_dmass] = 1
        f_c = np.sum(~true_mask) * 1./obs.size
        
        cont_data = obs[~true_mask]
        true_overlap_data = obs[true_mask]
        
        return f_c, cont_data, true_overlap_data
   
   
    def ang_stat_err(self, rmin = 0.02, rmax = 10.0):
        
        r, w, err = _loadndarray(self.w_obs_data)
        
        keep = (r > rmin) & (r < rmax)
        eps82 = np.mean(err[keep]/w[keep])/np.sqrt(np.sum(keep))
        eps = eps82  * np.sqrt(180./1000)
        print 'angular stat err (st82, full) ', eps82, eps
        return eps82, eps
   
   
    def len_stat_err(self, rmin = 0.02, rmax = 10.0):
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)
        
        keep = (r > rmin) & (~(L == 0)) & (~(L_c == 0)) & (~(L_t == 0))
        
        # ideal fraction
        eps_len82 = np.mean(err[keep]/L[keep])/np.sqrt(np.sum(keep))
        eps_len = eps_len82 * np.sqrt(180./1000)
        
        print 'lensing stat err (st82, full) ', eps_len82, eps_len
        return eps_len82, eps_len
    
    
    def contFraction(self):
        import esutil
        m_dmass, m_true, _ = esutil.htm.HTM(10).match( self.obs_data['RA'], self.obs_data['DEC'], self.true_data['RA'], self.true_data['DEC'], 1./3600, maxmatch = 1)
        true_mask = np.zeros( self.obs_data.size, dtype = bool )
        true_mask[m_dmass] = 1
        f_c = np.sum(~true_mask) * 1./self.obs_data.size
        self.f_c = f_c
        return f_c
    

    def MaxFc_from_AngSys_L(self, rmin = 0.02, rmax = 10):
        r, w, err = _loadndarray(self.w_obs_data)
        rt, w_t, err_t = _loadndarray(self.w_true_data)
        f_c = self.f_c
        ang_L = 1. + w - (1. -f_c)**2 * (1. + w_t)
    
        keep = (r > rmin) & (r < rmax)
        fractionL = 1. - np.sqrt( (1. + w - ang_L) * 1./(1. + w_t) )
        m_fL = np.mean(fractionL) # current max cont fraction
        
        # stat error
        #eps82 = np.mean(err[keep]/w[keep])/np.sqrt(np.sum(keep))
        #eps = eps82  * np.sqrt(180./1000)
        i_fL = np.abs(1. - np.sqrt( (1 + w - self.eps) * 1./(1 + w_t) ))


        self.m_i_fL = np.mean(np.ma.masked_invalid(i_fL[keep]))

        print 'Angular max fc ', m_fL, ' ideal fc (simple mean) ', self.m_i_fL
        return m_fL, self.m_i_fL


    def MaxFc_from_AngSys_R(self, rmin = 0.02, rmax = 10):
        
        r, w, err = _loadndarray(self.w_obs_data)
        rtc, w_tc, err_tc = _loadndarray(self.w_cross_data)
        rc, w_c, err_c = _loadndarray(self.w_cont_data)
        f_c = self.f_c
        ang_R = 2. * self.f_c * (1. -self.f_c)*(1. + w_tc) + self.f_c**2 * (1+w_c)

        keep = (r > rmin) & (r < rmax)
        fractionR = (-(1.+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* ang_R) ) * 1./(w_c-1.-2* w_tc)
        m_fR = np.mean(fractionR) # current max cont fraction
        
        # stat error
        #eps82 = np.mean(err[keep]/w[keep])/np.sqrt(np.sum(keep))
        #eps = eps82  * np.sqrt(100./1000)
        i_fR = np.abs((-(1+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* self.eps) ) * 1./(w_c-1.-2* w_tc))
        
        #square weight
        self.m_i_fR = np.mean(np.ma.masked_invalid(i_fR[keep]))
        
        print 'Angular max fc ', m_fR, ' ideal fc (simple mean) ', self.m_i_fR
        return m_fR, self.m_i_fR


    def MaxFc_from_LensSys_L(self, rmin = 0.02, rmax = 10):
        
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)
    
        keep = (r > rmin) & (~(L == 0)) & (~(L_c == 0)) & (~(L_t == 0))
        len_L = L - (1 - self.f_c) * L_t
    
        fractionL = 1 - (L - len_L)/L_t

        m_fL_len = np.mean(fractionL[keep])
        
        # ideal fraction
        #eps_len82 = np.mean(err[keep]/L[keep])/np.sqrt(np.sum(keep))
        #eps_len = eps_len82 * np.sqrt(100./1000)
        i_fL_len = 1. - (L - self.eps_len)/L_t
        
        maskL = np.ma.getmask(np.ma.masked_inside(i_fL_len, 0.0, 1.0))
    
        self.m_i_fL_len = np.mean(i_fL_len[maskL * keep])
    
        wel = 1./( err_t**2 + err_c**2 )
        weight_len = wel /np.sum(wel[keep * maskL])
        i_f_weighted_len = weight_len * i_fL_len
        self.f_weighted_len_L = np.sum(i_f_weighted_len[keep * maskL])
        
        print 'Lening max fc ', m_fL_len, ' ideal fc (weighted mean) ', self.f_weighted_len_L
        return m_fL_len, self.f_weighted_len_L


    def MaxFc_from_LensSys_R(self, rmin = 0.02, rmax = 10):
    
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)

        keep = (r > rmin) & (~(L == 0)) & (~(L_c == 0)) & (~(L_t == 0))
        len_R = L_c * self.f_c
        
        fractionR = len_R/L_c
        m_fR_len = np.mean(fractionR[keep])

        # ideal fraction
        #eps_len82, eps_len = self.len_stat_err()
        #eps_len82 = np.mean(err[keep]/L[keep])/np.sqrt(np.sum(keep))
        #eps_len = eps_len82 * np.sqrt(100./1000)
        i_fR_len = self.eps_len/L_c

        maskR = np.ma.getmask(np.ma.masked_inside(i_fR_len, 0.0, 1.0))
    
        m_i_fractionR_len = np.mean(i_fR_len[maskR * keep])
    
        wel = 1./( err_t**2 + err_c**2 )
        weight_len = wel /np.sum(wel[keep * maskR])
        i_f_weighted_len = weight_len * i_fR_len
        self.f_weighted_len_R = np.sum(i_f_weighted_len[keep * maskR])
        
        print 'Lening max fc ', m_fR_len, ' ideal fc (weighted mean) ', self.f_weighted_len_R
        return m_fR_len, self.f_weighted_len_R



    def Angsys_from_bias(self, rmin = 0.02, rmax = 10.0):
        
        r, w, err = _loadndarray(self.w_obs_data)
        rt, w_t, err_t = _loadndarray(self.w_true_data)
        #f_c = np.min( [self.m_i_fR, self.m_i_fL] ) #self.f_c
        f_c = self.f_c
        
        keep = (r > rmin) & (r < rmax)
        magc = self.cont_data['MAG_MODEL_I'] - self.cont_data['XCORR_SFD98_I']
        magt = self.true_data['MAG_MODEL_I'] - self.true_data['XCORR_SFD98_I']
        z_c = self.cont_data['DESDM_ZP']
        z_t = self.true_data['DESDM_ZP']
        b_c = Mag_to_galaxyBias( magAB = magc, z = z_c )
        b_t = Mag_to_galaxyBias( magAB = magt, z = z_t )
        
        sys = w_t * 2 * f_c * (b_c/b_t -1 )
        
        print 'b_c ', b_c, ' b_t ', b_t, ' mean sys ', np.mean(np.abs(sys[keep]))
        return r, sys



    def Lensys_from_bias(self, rmin = 0.02, rmax = 10.0):
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)
        #f_c = np.min( [self.f_weighted_len_R, self.f_weighted_len_L] ) #self.f_c
        f_c = self.f_c
        
        keep = (r > rmin) & (~(L == 0)) & (~(L_c == 0)) & (~(L_t == 0))
        
        magc = self.cont_data['MAG_MODEL_I'] #- self.obs_data['XCORR_SFD98']
        magt = self.true_data['MAG_MODEL_I'] #- self.obs_data['XCORR_SFD98']
        z_c = self.cont_data['DESDM_ZP'] #- self.obs_data['XCORR_SFD98']
        z_t = self.true_data['DESDM_ZP'] #- self.obs_data['XCORR_SFD98']
        
        b_c = Mag_to_galaxyBias( magAB = magc, z = z_c )
        b_t = Mag_to_galaxyBias( magAB = magt, z = z_t )
    
        sys = L_t * f_c * (b_c/b_t - 1)
        
        #self.len_stat_err(rmin = rmin, rmax = rmax)
        print 'b_c ', b_c, ' b_t ', b_t, ' mean sys ', np.mean(np.abs(sys[keep]))
        return r, sys



"""
def EstimateContaminantFraction( dmass = None, cmass = None ):
    
    #  check difference is smaller than stat error
    
    import esutil
    m_dmass, m_true, _ = esutil.htm.HTM(10).match( dmass['RA'], dmass['DEC'], cmass['RA'], cmass['DEC'], 1./3600, maxmatch = 1)
    true_mask = np.zeros( dmass.size, dtype = bool )
    true_mask[m_dmass] = 1
    
    #cross_data = np.loadtxt('data_txt/acf_cross_noweightcont.txt')
    #cross_data = np.loadtxt('data_txt/acf_cross_weightedcont_prop.txt')
    cross_data = np.loadtxt('data_txt/acf_cross_weightedcont.txt')
    r_tc, w_tc, err_tc = cross_data[:,0], cross_data[:,1], cross_data[:,2]
    auto_data = np.loadtxt('data_txt/acf_comparison_photo_dmass.txt')
    true_data = np.loadtxt('data_txt/acf_comparison_photo_true.txt')
    #cont_data = np.loadtxt('data_txt/acf_comparison_noweightcont.txt')
    #cont_data = np.loadtxt('data_txt/acf_comparison_weightedcont_prop.txt')
    cont_data = np.loadtxt('data_txt/acf_comparison_weightedcont.txt')
    r, w, err, w_t, err_t,  w_c, err_c = auto_data[:,0], auto_data[:,1],  auto_data[:,2],  true_data[:,1],  true_data[:,2], cont_data[:,1], cont_data[:,2]
    
    f_c = np.sum(~true_mask) * 1./dmass.size
    print "current f_c = {:>0.2f} %".format( f_c * 100)
    ang_L = 1. + w - (1. -f_c)**2 * (1. + w_t)
    ang_R = 2. * f_c * (1. -f_c)*(1. + w_tc) + f_c**2 * (1+w_c)
    
    keep = (r > 0.001) & (r < 10)
    fractionL = 1. - np.sqrt( (1. + w - ang_L) * 1./(1. + w_t) )
    fractionR = (-(1.+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* ang_R) ) * 1./(w_c-1.-2* w_tc)
    m_fL, m_fR = np.mean(fractionL), np.mean(fractionR) # current max cont fraction
    
    #print "current f_cont (LHS, RHS) : ", m_fL, m_fR
    
    # ideal fraction
    eps82 = np.mean(err[keep]/w[keep])/np.sqrt(np.sum(keep))
    eps = eps82  * np.sqrt(100./1000)
    i_fractionL = np.abs(1. - np.sqrt( (1 + w - eps) * 1./(1 + w_t) ))
    i_fractionR = np.abs((-(1+w_tc)+np.sqrt((1+w_tc)**2 + (w_c-1.-2*w_tc)* eps) ) * 1./(w_c-1.-2* w_tc))
    
"""
        # calculate everything for each point, and then remove invalid values (negative or higher
        # values than 1) and get mean.
"""
    
    m_i_fractionL, m_i_fractionR = np.mean(np.ma.masked_invalid(i_fractionL[keep])), np.mean(np.ma.masked_invalid(i_fractionR[keep]))
    
    #square weight
    we = 1./(err_t**2 + err_c**2 )
    weight = we/np.sum(we[keep])
    i_f_weighted = weight * i_fractionR
    f_weighted = np.sum(np.ma.masked_invalid(i_f_weighted[keep]))
    
    print "angular >> "
    print "- amp of systematic terms ", np.mean(np.abs(ang_R[keep]))
    print "- stat err (y1a1, st82) ", eps, eps82
    print "- ideal f_c (LHS, RHS): {:>0.4f} %, {:>0.4f} %".format(m_i_fractionL * 100, m_i_fractionR * 100)
    print "- f_weighted : {:>0.4f} %".format(f_weighted*100)
    
    filename = 'data_txt/fcont_estimation.txt'
    DAT = np.column_stack((r, i_fractionL, i_fractionR, weight, i_f_weighted, err, err_t, err_c))
    np.savetxt(filename, DAT[keep], delimiter = ' ', header = 'r, i_fractionL, i_fractionR, weight, f_weighted, err, err_t, err_c')
    print 'data file writing to ',filename
    
    
    dmass_lensing = np.loadtxt('data_txt/Ashley_lensing.txt')
    true_lensing = np.loadtxt('data_txt/Ashley_true_lensing.txt')
    #cont_lensing = np.loadtxt('data_txt/lensing_noweightcont.txt')
    cont_lensing = np.loadtxt('data_txt/lensing_weightedcont_prop.txt')
    #cont_lensing = np.loadtxt('data_txt/lensing_weightedcont.txt')
    
    
    r_p, LS, err_LS, LS_t, err_LS_t, LS_c, err_LS_c = dmass_lensing[:,0], dmass_lensing[:,1], dmass_lensing[:,2], true_lensing[:,1], true_lensing[:,2], cont_lensing[:,1], cont_lensing[:,2]
    
    keep = (r_p > 0.0) & (~(LS == 0)) & (~(LS_c == 0)) & (~(LS_t == 0))
    len_L = LS - (1 - f_c) * LS_t
    len_R = LS_c * f_c
    
    fractionL = 1 - (LS - len_L)/LS_t
    fractionR = len_R/LS_c
    m_fL_len, m_fR_len = np.mean(fractionL[keep]), np.mean(fractionR[keep])
    #print "current f_cont (LHS, RHS) : ", m_fL_len, m_fR_len
    
    # ideal fraction
    eps_len82 = np.mean(err_LS[keep]/LS[keep])/np.sqrt(np.sum(keep))
    eps_len = eps_len82 * np.sqrt(100./1000)
    i_fractionL_len = 1. - (LS - eps_len)/LS_t
    i_fractionR_len = eps_len82/LS_c
    
    maskL = np.ma.getmask(np.ma.masked_inside(i_fractionL_len, 0.0, 1.0))
    maskR = np.ma.getmask(np.ma.masked_inside(i_fractionR_len, 0.0, 1.0))
    
    m_i_fractionL_len, m_i_fractionR_len = np.mean(i_fractionL_len[maskL * keep]), np.mean(i_fractionR_len[maskR * keep])
    
    wel = 1./( err_LS_t**2 + err_LS_c**2 )
    weight_len = wel /np.sum(wel[keep * maskR])
    i_f_weighted_len = weight_len * i_fractionR_len
    f_weighted_len = np.sum(i_f_weighted_len[keep * maskR])
    
    print "Lensing >>"
    print "- amp of systematic terms ", np.mean(np.abs(len_R[maskR * maskL * keep]))
    print "- stat err (y1a1, st82) ", eps_len, eps_len82
    print "- ideal f_c (LHS, RHS): {:>0.4f} %, {:>0.4f} %".format(m_i_fractionL_len * 100, m_i_fractionR_len * 100)
    print "- f_weighted : {:>0.4f} %".format(f_weighted_len*100 )
    
    filename2 = 'data_txt/fcont_estimation_lense.txt'
    DAT2 = np.column_stack((r_p, i_fractionL_len, i_fractionR_len, weight_len, i_f_weighted_len, err_LS,err_LS_t, err_LS_c))
    np.savetxt(filename2, DAT2[keep * maskR], delimiter = ' ', header = 'r_p, i_fractionL_len, i_fractionR_len, weight_len, f_weighted_len, err_LS, err_LS_t, err_LS_c')
    print 'data file writing to ', filename2

    """






