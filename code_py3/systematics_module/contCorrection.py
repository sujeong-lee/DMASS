#!/usr/bin/env python
#from __future__ import print_function, division

import sys, os, fitsio
import numpy as np
from cmass_modules import io, DES_to_SDSS, im3shape, Cuts
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from scipy import linalg




def getavgbias( mag, z , pstart=0.0 ):
    #mag_i = cat['MAG_MODEL_I_corrected']
    #z = cat['DESDM_ZP']
    logL = logL_from_mag( mag = mag, z = z )
    avg_b = logL_to_galaxyBias(logL = logL)
    print('avg bias=',avg_b, ' sample size=', mag.size)
    return avg_b


def ABmag_from_mag( mag = None, z = None ):

    from astropy.cosmology import FlatLambdaCDM
    from astropy.constants import L_sun, M_sun
    from astropy import units as u
    from cosmolopy import magnitudes
    
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d = cosmo.comoving_distance( z ) * u.Mpc.to(u.pc)
    magAB = mag + 5 - 5*np.log10(d.value)
    return magAB


def logL_from_mag( mag = None, z = None ):
    """
    log(L/L_sun) from apparent magnitude.
    """

    from astropy.cosmology import FlatLambdaCDM
    from astropy.constants import L_sun, M_sun
    from astropy import units as u
    from cosmolopy import magnitudes

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    d = cosmo.comoving_distance( z ) * u.Mpc.to(u.pc)
    magAB = mag + 5 - 5*np.log10(d.value)

    magnitudes.MAB0 = 30 # des zeropoint
    L = magnitudes.L_nu_from_magAB(magAB)
    LogLuminosity = np.log10(L/L_sun.value)

    return LogLuminosity



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
    
    from colossus.lss import bias
    #HaloModel = bias.MODELS
    HaloBias = bias.haloBias(M_halo, z, 'vir', model='tinker10')
    # http://bdiemer.bitbucket.org/halo_bias.html#halo.bias.haloBias
    # mdef parameter : http://bdiemer.bitbucket.org/halo_mass.html#spherical-overdensity-basics
    
    #nu = cosmo.peakHeight(M_halo, z, filt='tophat', Pk_source='eh98', deltac_const=True)

    from colossus.lss import peaks
    nu = peaks.peakHeight(M_halo, z, ps_args={'model':'eisenstein98'})
    # http://bdiemer.bitbucket.org/cosmology_cosmology.html#cosmology.cosmology.Cosmology.peakHeight
    
    return nu, HaloBias



def massLuminosity( M_halo ):
    """
    the mass luminosity relation (Vale et al 2004 )
    M_halo unit :  M (M_sun)
    return  L (L_sun)
    """
    A = 5.7e+9
    m = 1e+11
    b,c,d,k = 4., 0.57, 3.72, 0.23
    #Mbin2 = np.logspace(11, 15, 200, base=10.)
    L = A * (M_halo/m)**b * 1./ (c + (M_halo/m)**(d*k))**(1./k)

    return L


def __Mag_to_galaxyBias(mag = None, z = None):

    Mbin = np.logspace(10, 20, 1000, base=10.)
    Lbin = massLuminosity( Mbin )
    _, bhbin = haloBiasTinker10(Mbin, z = 0.0 )

    logL = logL_from_mag(mag=mag, z=z )

    
    N, _ = np.histogram( logL, bins = np.log10(Lbin) )
    N_sample = mag.size
    bias_g = np.sum( bhbin[:-1] * N ) *1./N_sample
    #print N.size, np.sum(N!=0)
    return bias_g


def logL_to_galaxyBias( logL = None ):
   
    Mbin = np.logspace(10, 20, 1000, base=10.)
    Lbin = massLuminosity( Mbin )
    _, bhbin = haloBiasTinker10(Mbin, z = 0.0 )
    
    #logL = logL_from_mag(mag=mag, z=z )  
    logL = np.ma.masked_invalid(logL)
    logL = logL[~logL.mask]
    N, _ = np.histogram( logL, bins = np.log10(Lbin) )
    
    N_sample = logL.size
    bias_g = np.sum( bhbin[:-1] * N ) *1./N_sample
    #print N.size, np.sum(N!=0)
    return bias_g



def _loadndarray(data):
    r, w, err = data[:,0], data[:,1], data[:,2]
    return r, w, err



def doVisualization_ngal(property = None, nside = 1024, kind = 'SPT', suffix='', inputdir = '.', outdir='./'):

    import matplotlib.pyplot as plt
    import numpy as np
    from systematics import SysMapBadRegionMask, loadSystematicMaps, MatchHPArea, chisquare_dof, ReciprocalWeights
    from systematics_module.corr import angular_correlation
    
    
    filters = ['g', 'r', 'i', 'z']
    #filters = ['g']
    
    
    if property is 'NSTARS_allband' :
        nside = 1024
        filters = ['g']
    elif property is 'GE' :
        nside = 512
        filters = ['g']
    elif property in ['SKYBRITEpca0','SKYBRITEpca1','SKYBRITEpca2','SKYBRITEpca3'] :
        nside = 4096
        filters = ['g']     

    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    ax = ax.ravel()

    for i, filter in enumerate(filters):
        filename = inputdir+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix+'.txt'
        #print filename
        data = np.loadtxt(filename)
        bins, Cdensity, Cerr, Cf_area, Bdensity, Berr, Bf_area = [data[:,j] for j in range(data[0].size)]
        Cf_area = Cf_area * 1./Cf_area.max()/5.
        
        
        #zeromaskC, zeromaskB = ( Cdensity != 0.0 )*( Cerr != 0.0 ), (Bdensity != 0.0 )*( Berr != 0.0 )
        zeromaskC = ( Cdensity != 0.0 ) 
        zeromaskB = ( Bdensity != 0.0 )
        #Cdensity, Cbins, Cerr, Cf_area = Cdensity, bins, Cerr, Cf_area #Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
        #C_jkerr = C_jkerr[zeromaskC]
        #Bdensity, Bbins, Berr = Bdensity, bins, Berr #Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
        #B_jkerr = B_jkerr[zeromaskB]

        #fitting
        #Cchi, Cchidof = chisquare_dof( bins[zeromaskC], Cdensity[zeromaskC], Cerr[zeromaskC] )
        #Bchi, Bchidof = chisquare_dof( bins[zeromaskB], Bdensity[zeromaskB], Berr[zeromaskB] )
    
        nCbins = np.sum(zeromaskC)
        nBbins = np.sum(zeromaskB)
        Cchi = np.sum( (Cdensity[zeromaskC] - 1.0 * np.ones(nCbins) )**2/Cerr[zeromaskC]**2 )
        Bchi = np.sum( (Bdensity[zeromaskB] - 1.0 * np.ones(nBbins) )**2/Berr[zeromaskB]**2 )


        ax[i].errorbar(bins[zeromaskC] , Cdensity[zeromaskC], yerr = Cerr[zeromaskC], 
            color = 'blue', fmt = '.-', capsize=3, label='DMASS, chi2/dof={:>2.2f}/{:3.0f}'.format(Cchi, nCbins) )
        ax[i].errorbar(bins[zeromaskB]*1.0 , Bdensity[zeromaskB], yerr = Berr[zeromaskB], 
            color = 'red', fmt= '.-', capsize=3, label='Random, chi2/dof={:>2.2f}/{:3.0f}'.format(Bchi, nBbins) )

        #ax[i].bar(Bbins+(bins[1]-bins[0])*0.1, Bf_area[zeromaskB]+0.7, (bins[1]-bins[0]) ,color = 'red', alpha=0.3 )
        
        ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
        ax[i].set_ylabel('n_gal/n_tot '+str(nside))
        
        ymin, ymax = 0.7, 1.3
        if kind is 'SPT' : ymin, ymax = 0.7, 1.3
        ax[i].set_ylim(ymin, ymax)

        barwidth = (bins[1]-bins[0])
        #if property is 'GE' : 
        #    barwidth = bins[zeromaskC][1:] - bins[zeromaskC][:-1]
        #    print barwidth.size, bins[zeromaskC].size
        ax[i].bar(bins[zeromaskC], Cf_area[zeromaskC]+ymin,barwidth ,color = 'blue', alpha = 0.1 )

        #ax[i].set_xlim(8.2, 8.55)
        ax[i].axhline(1.0,linestyle='--',color='grey')
        ax[i].legend(loc = 'best')
        
        #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
        #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
        #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
        if property == 'NSTARS_allband': ax[i].set_xlim(0.0, 2.0)
        if property == 'NSTARS': ax[i].set_xlim(0.0, 1000)
        if property == 'GE' : 
            ax[i].set_xscale('log')
            #ax[i].set_xlim(0.004, 0.2)
    fig.suptitle('systematic test ({})'.format(kind))
    os.system('mkdir '+outdir)
    figname = outdir+'systematic_'+property+'_'+kind+'_'+suffix+'.png'
    fig.savefig(figname)
    print("saving fig to ", figname)

    return 0



def plot_sysweight(property = None, nside = 1024, kind = 'SPT', suffix1='', suffix2='', inputdir1 = '.', inputdir2 = '.', outdir='./'):

    import matplotlib.pyplot as plt
    import numpy as np
    from systematics import SysMapBadRegionMask, loadSystematicMaps, MatchHPArea, chisquare_dof, ReciprocalWeights
    from systematics_module.corr import angular_correlation
    
    
    filters = ['g', 'r', 'i', 'z']
    #filters = ['g']
    
    
    if property is 'NSTARS_allband' :
        nside = 1024
        filters = ['g']
    if property is 'GE':
        nside = 512
        filters = ['g']
        
    fig, ax = plt.subplots(2, 2, figsize = (15, 10))
    ax = ax.ravel()

    for i, filter in enumerate(filters):
        filename1 = inputdir1+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix1+'.txt'
        filename2 = inputdir2+'/systematic_'+property+'_'+filter+'_'+kind+'_'+suffix2+'.txt'
        #print filename
        data1 = np.loadtxt(filename1)
        data2 = np.loadtxt(filename2)
        bins, Cdensity, Cerr, Cf_area, _, _, _ = [data1[:,j] for j in range(data1[0].size)]
        bins, Bdensity, Berr, Bf_area, _, _, _ = [data2[:,j] for j in range(data2[0].size)]
        Cf_area = Cf_area * 1./Cf_area.max()/5.

        #zeromaskC, zeromaskB = ( Cdensity != 0.0 )*( Cerr != 0.0 ), (Bdensity != 0.0 )*( Berr != 0.0 )
        zeromaskC = ( Cdensity != 0.0 ) 
        zeromaskB = ( Bdensity != 0.0 )
        #Cdensity, Cbins, Cerr, Cf_area = Cdensity, bins, Cerr, Cf_area #Cdensity[zeromaskC], bins[zeromaskC], Cerr[zeromaskC]
        #C_jkerr = C_jkerr[zeromaskC]
        #Bdensity, Bbins, Berr = Bdensity, bins, Berr #Bdensity[zeromaskB],bins[zeromaskB],Berr[zeromaskB]
        #B_jkerr = B_jkerr[zeromaskB]

        #fitting
        #Cchi, Cchidof = chisquare_dof( bins[zeromaskC], Cdensity[zeromaskC], Cerr[zeromaskC] )
        #Bchi, Bchidof = chisquare_dof( bins[zeromaskB], Bdensity[zeromaskB], Berr[zeromaskB] )
    
        nCbins = np.sum(zeromaskC)
        nBbins = np.sum(zeromaskB)
        Cchi = np.sum( (Cdensity[zeromaskC] - 1.0 * np.ones(nCbins) )**2/Cerr[zeromaskC]**2 )
        Bchi = np.sum( (Bdensity[zeromaskB] - 1.0 * np.ones(nBbins) )**2/Berr[zeromaskB]**2 )

        #ax[i].errorbar(bins[zeromaskC] , Cdensity[zeromaskC], yerr = Cerr[zeromaskC], 
        #    color = 'grey', fmt = '.-', capsize=3, label='no weight, chi2/dof={:>2.2f}/{:3.0f}'.format(Cchi, nCbins) )
        ax[i].plot(bins[zeromaskC] , Cdensity[zeromaskC], color = 'grey', label='no weight, chi2/dof={:>2.2f}/{:3.0f}'.format(Cchi, nCbins) )
        ax[i].errorbar(bins[zeromaskB]*1.0 , Bdensity[zeromaskB], yerr = Berr[zeromaskB], 
            color = 'red', fmt= '.-', capsize=3, label='weighted, chi2/dof={:>2.2f}/{:3.0f}'.format(Bchi, nBbins) )

        #ax[i].bar(Bbins+(bins[1]-bins[0])*0.1, Bf_area[zeromaskB]+0.7, (bins[1]-bins[0]) ,color = 'red', alpha=0.3 )
        ax[i].set_xlabel('{}_{} (mean)'.format(property, filter))
        ax[i].set_ylabel('n_gal/n_tot '+str(nside))

        ymin, ymax = 0.7, 1.3
        if kind is 'SPT' : ymin, ymax = 0.7, 1.3
        ax[i].set_ylim(ymin, ymax)
        ax[i].bar(bins[zeromaskC], Cf_area[zeromaskC]+ymin,(bins[1]-bins[0]) ,color = 'grey', alpha = 0.1 )

        #ax[i].set_xlim(8.2, 8.55)
        ax[i].axhline(1.0,linestyle='--',color='grey')
        ax[i].legend(loc = 'best')
        
        #if property == 'FWHM' : ax[i].set_ylim(0.6, 1.4)
        #if property == 'AIRMASS': ax[i].set_ylim(0.0, 2.0)
        #if property == 'SKYSIGMA': ax[i].set_xlim(12, 18)
        if property is 'GE': ax[i].set_xscale('log')
        if property == 'NSTARS': ax[i].set_xlim(0.0, 2.0)

    fig.suptitle('systematic test ({})'.format(kind))
    os.system('mkdir '+outdir)
    figname = outdir+'comparison_systematic_'+property+'_'+kind+'_'+suffix2+'.png'
    fig.savefig(figname)
    print("saving fig to ", figname)

    return 0



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
        eps82 = np.mean(np.fabs(err[keep]/w[keep]))/np.sqrt(np.sum(keep))
        eps = eps82  * np.sqrt(180./1000)
        print('angular stat err (st82, full) ', eps82, eps)
        return eps82, eps
   
   
    def len_stat_err(self, rmin = 0.02, rmax = 10.0):
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)
        
        keep = (r > rmin) & (r < rmax) & (~(L == 0)) # & (~(L_c == 0)) & (~(L_t == 0))
        
        # ideal fraction
        eps_len82 = np.mean(np.abs(err[keep]/L[keep]))/np.sqrt(np.sum(keep))
        eps_len = eps_len82 * np.sqrt(180./1000)
        
        print('lensing stat err (st82, full) ', eps_len82, eps_len)
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

        i_fL = np.abs(1. - np.sqrt( (1 + w - self.eps) * 1./(1 + w_t) ))
        self.m_i_fL = np.mean(np.ma.masked_invalid(i_fL[keep]))

        print('Angular max fc ', m_fL, ' ideal fc (simple mean) ', self.m_i_fL, ' mean sys ', np.mean(ang_L[keep]))
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
        
        print('Angular max fc ', m_fR, ' ideal fc (simple mean) ', self.m_i_fR, ' mean sys ', np.mean(ang_R[keep]))
        return m_fR, self.m_i_fR


    def MaxFc_from_LensSys_L(self, rmin = 0.02, rmax = 10):
        
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)
    
        keep = (r > rmin) &(r < rmax)& (~(L == 0)) & (~(L_t == 0))
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
        
        print('Lensing max fc ', m_fL_len, ' ideal fc (weighted mean) ', self.f_weighted_len_L, ' mean sys ', np.mean(len_L[keep]))
        return m_fL_len, self.f_weighted_len_L


    def MaxFc_from_LensSys_R(self, rmin = 0.02, rmax = 10):
    
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)

        keep = (r > rmin) &(r < rmax) & (~(L_c == 0)) #& (~(L == 0)) & (~(L_t == 0))
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
        
        print('Lening max fc ', m_fR_len, ' ideal fc (weighted mean) ', self.f_weighted_len_R, ' mean sys ', np.mean(len_R[keep]))
        return m_fR_len, self.f_weighted_len_R



    def Angsys_from_bias(self, rmin = 0.02, rmax = 10.0):
        r, w, err = _loadndarray(self.w_obs_data)
        rt, w_t, err_t = _loadndarray(self.w_true_data)
        #f_c = np.min( [self.m_i_fR, self.m_i_fL] ) #self.f_c
        f_c = self.f_c
        
        keep = (r > rmin) & (r < rmax)
        magc = self.cont_data['MAG_MODEL_I_corrected']
        magt = self.true_data['MAG_MODEL_I_corrected']
        z_c = self.cont_data['DESDM_ZP']
        z_t = self.true_data['DESDM_ZP']
        logLt = logL_from_mag( mag = magt, z = z_t )
        logLt = logLt[logLt > 9.0]
        logLc = logL_from_mag( mag = magc, z = z_c )
        logLc = logLc[logLc > 9.0]

        b_c = logL_to_galaxyBias(logL = logLc)
        b_t = logL_to_galaxyBias(logL = logLt)
        
        sys = w_t * 2 * f_c * (b_c/b_t -1 )
        
        print('b_c ', b_c, ' b_t ', b_t, ' mean sys ', np.mean(np.abs(sys[keep])))
        return r, sys



    def Lensys_from_bias(self, rmin = 0.02, rmax = 10000.0):
        r, L, err = _loadndarray(self.L_obs_data)
        r_t, L_t, err_t = _loadndarray(self.L_true_data)
        r_c, L_c, err_c = _loadndarray(self.L_cont_data)
        #f_c = np.min( [self.f_weighted_len_R, self.f_weighted_len_L] ) #self.f_c
        f_c = self.f_c
        
        keep = (r > rmin) & (~(L == 0)) & (~(L_c == 0)) & (~(L_t == 0))
        
        magc = self.cont_data['MAG_MODEL_I_corrected']# - self.cont_data['XCORR_SFD98_I']
        magt = self.true_data['MAG_MODEL_I_corrected']# - self.true_data['XCORR_SFD98_I']
        z_c = self.cont_data['DESDM_ZP']
        z_t = self.true_data['DESDM_ZP']
        logLt = logL_from_mag( mag = magt, z = z_t )
        logLt = logLt[logLt > 9.0]
        logLc = logL_from_mag( mag = magc, z = z_c )
        logLc = logLc[logLc > 9.0]
        b_c = logL_to_galaxyBias(logL = logLc)
        b_t = logL_to_galaxyBias(logL = logLt)
        
        #b_c = Mag_to_galaxyBias( mag = magc, z = z_c )
        #b_t = Mag_to_galaxyBias( mag = magt, z = z_t )
    
        sys = L_t * f_c * (b_c/b_t - 1)
        
        #self.len_stat_err(rmin = rmin, rmax = rmax)
        print('b_c ', b_c, ' b_t ', b_t, ' mean sys ', np.mean(np.abs(sys[keep])))
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






