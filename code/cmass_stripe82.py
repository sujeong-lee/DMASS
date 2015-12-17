import easyaccess as ea
import esutil
import sys
import os
#import healpy as hp
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
#import seaborn as sns

from ang2stripe import *
import fitsio
from fitsio import FITS, FITSHDR


def CmassGal_in_stripe82(data):
    
    list = []
    for i in range(0, len(data)):
        ra, dec = data[i]['RA'], data[i]['DEC']
        stripe_num = ang2stripe(ra,dec)
    
        if stripe_num == 82:
            list.append(data[i])
    
        else : pass
    
    list = np.array(list)
    #selected_data = np.array(list, dtype=data.dtype)
    
    return list

def CMASS_plotting():
    
    stripe82_data = CmassGal_in_stripe82(data)
    fig = plt.figure()
    for i in range(0, len(stripe82_data)):
        ra, dec = stripe82_data[i]['RA'], stripe82_data[i]['DEC']
        plt.scatter(ra, dec, marker='+')
    plt.xlim(310, 370)
    plt.ylim(-2.0, 2.0)
    plt.title('CMASS Galaxy in Stripe82')
    plt.show()
    fig.savefig('figure/Cmass_st82_2.png')


# DES easyaccess-----------------------------------------

def getDEScatalogs( file = '../data/stripe82_des_cut_000001.fits' ):
    """
    if file is False :
        
        #get file from server and save
        connection=ea.connect()
        query=connection.loadsql('../query/stripe82_des_cut.sql')

        #data = connection.query_to_pandas(query)
        data = connection.query_and_save(query,'../data/stripe82_des_cut.fits')
        data = fitsio.read(file)
    
    else:
    """
    #file = '../data/stripe82_des_cut_000001.fits'
    #file2 = '../data/stripe82_des_cut_000002.fits'
    #file = '/n/des/huff.791/Projects/combined_i.fits'
    
    data = fitsio.read(file)
    #data2 = fitsio.read(file2)
    
    #data = np.hstack((data, data2))
    #data = data[data['CLEAN'] == 1] #check later
    data.dtype.names = tuple([ data.dtype.names[i].upper() for i in range(len(data.dtype.names))])
    data.sort(order = 'COADD_OBJECTS_ID')
    return data


def getSDSScatalogs():
    
    #file1 = '../data/galaxy_DR11v1_CMASS_South-photoObj.fits'
    #file2 = '../data/galaxy_DR11v1_LOWZ_South-photoObj.fits'
    file1 = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit'
    #file1 = '/Users/SJ/Dropbox/repositories/CMASS/data/test_emhuff.fit'
    #file3 = '../data/sdss_clean_galaxy_350_360_m05_0.fits'
    #file4 = '../data/sdss_clean_galaxy_350_351_m05_0.fits'
    #file1 = '../data/SDSS_clean_galaxy.fits'
    data = fitsio.read(file1)
    #data = esutil.io.read_header(file1,ext=1)
    #data2 = fitsio.read(file2)
    #data3 = fitsio.read(file3)
    #data4 = fitsio.read(file4)
    
    data.dtype.names = tuple([ data.dtype.names[i].upper() for i in range(len(data.dtype.names))])
    
    return data


def getSDSScatalogsCMASSLOWZ():
    
    file1 = '../data/galaxy_DR11v1_CMASS_South-photoObj.fits'
    file2 = '../data/galaxy_DR11v1_LOWZ_South-photoObj.fits'
    #file1 = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit'
    #file3 = '../data/sdss_clean_galaxy_350_360_m05_0.fits'
    #file4 = '../data/sdss_clean_galaxy_350_351_m05_0.fits'
    
    data1 = fitsio.read(file1)
    data2 = fitsio.read(file2)
    #data3 = fitsio.read(file3)
    #data4 = fitsio.read(file4)
    
    data = np.hstack((data1, data2))
    
    data.dtype.names = tuple([ data.dtype.names[i].upper() for i in range(len(data.dtype.names))])
    
    return data



def modestify(data):
    #from Eric's code
    
    modest = np.zeros(len(data), dtype=np.int32)
    
    galcut = (data['FLAGS_I'] <=3) & -( ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'] < 18.0)) | ((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) | ((data['MAG_PSF_I'] > 30.0) & (data['MAG_AUTO_I'] < 21.0)))
    modest[galcut] = 1
    
    starcut = (data['FLAGS_I'] <=3) & ((data['CLASS_STAR_I'] > 0.3) & (data['MAG_AUTO_I'] < 18.0) & (data['MAG_PSF_I'] < 30.0) | (((data['SPREAD_MODEL_I'] + 3*data['SPREADERR_MODEL_I']) < 0.003) & ((data['SPREAD_MODEL_I'] +3*data['SPREADERR_MODEL_I']) > -0.003)))
    modest[starcut] = 3
    
    neither = -(galcut | starcut)
    modest[neither] = 5
    
    data = rf.append_fields(data, 'modtype', modest)
    print len(data), np.sum(galcut), np.sum(starcut), np.sum(neither)
    return data


def whichGalaxyProfile(sdss):

    exp_L = np.exp(np.array([sdss['LNLEXP_G'],sdss['LNLEXP_R'],sdss['LNLEXP_I'],sdss['LNLEXP_Z']])).T
    dev_L = np.exp(np.array([sdss['LNLDEV_G'],sdss['LNLDEV_R'],sdss['LNLDEV_I'],sdss['LNLDEV_Z']])).T
    star_L = np.exp(np.array([sdss['LNLSTAR_G'],sdss['LNLSTAR_R'],sdss['LNLSTAR_I'],sdss['LNLSTAR_Z']])).T

    expfracL = exp_L /(exp_L + dev_L + star_L)
    devfracL = dev_L /(exp_L + dev_L + star_L)
    
    modelmode = np.zeros((len(sdss), 4), dtype=np.int32)

    expmodel = (expfracL > 0.5)
    modelmode[expmodel] = 0
    devmodel = (devfracL > 0.5)
    modelmode[devmodel] = 1
    neither = - (expmodel | devmodel)
    modelmode[neither] = 2
    
    sdss = rf.append_fields(sdss, 'BESTPROF_G', modelmode[:,0])
    sdss = rf.append_fields(sdss, 'BESTPROF_R', modelmode[:,1])
    sdss = rf.append_fields(sdss, 'BESTPROF_I', modelmode[:,2])
    sdss = rf.append_fields(sdss, 'BESTPROF_Z', modelmode[:,3])
    
    #print ' exp :', np.sum(expmodel),' dev :', np.sum(devmodel), 'neither :', np.sum(neither)
    return sdss


def simplewhichGalaxyProfile(sdss):
    
    modelmode = np.zeros((len(sdss), 4), dtype=np.int32)

    exp_L = np.exp(np.array([sdss['LNLEXP_G'],sdss['LNLEXP_R'],sdss['LNLEXP_I'],sdss['LNLEXP_Z']])).T
    dev_L = np.exp(np.array([sdss['LNLDEV_G'],sdss['LNLDEV_R'],sdss['LNLDEV_I'],sdss['LNLDEV_Z']])).T
    star_L = np.exp(np.array([sdss['LNLSTAR_G'],sdss['LNLSTAR_R'],sdss['LNLSTAR_I'],sdss['LNLSTAR_Z']])).T
    
    expmodel = ( exp_L > dev_L )
    modelmode[expmodel] = 0
    devmodel = ( exp_L < dev_L )
    modelmode[devmodel] = 1
    neither = - (expmodel | devmodel)
    modelmode[neither] = 2

    sdss = rf.append_fields(sdss, 'BESTPROF_G', modelmode[:,0])
    sdss = rf.append_fields(sdss, 'BESTPROF_R', modelmode[:,1])
    sdss = rf.append_fields(sdss, 'BESTPROF_I', modelmode[:,2])
    sdss = rf.append_fields(sdss, 'BESTPROF_Z', modelmode[:,3])
    
    #print 'tot :', len(sdss),' exp :', np.sum(expmodel),' dev :', np.sum(devmodel)
    return sdss


def addMagforim3shape(des):

    zeromag = 25.
    mag = zeromag - 2.5 * np.log10((des['DISC_FLUX'] + des['BULGE_FLUX']) * des['MEAN_FLUX'])
    des = rf.append_fields(des, 'MAG_MODEL_I', mag)
    return des


def addPogsonMag(sdss):

    F_0 = 1.
    zeromag = 25.
    mag_g = zeromag - 2.5 * np.log10(sdss['EXPFLUX_G']/ F_0)
    mag_r = zeromag - 2.5 * np.log10(sdss['EXPFLUX_R']/ F_0)
    mag_i = zeromag - 2.5 * np.log10(sdss['EXPFLUX_I']/ F_0)
    mag_z = zeromag - 2.5 * np.log10(sdss['EXPFLUX_Z']/ F_0)
    
    sdss = rf.append_fields(sdss, 'POGSONMAG_G', mag_g)
    sdss = rf.append_fields(sdss, 'POGSONMAG_R', mag_r)
    sdss = rf.append_fields(sdss, 'POGSONMAG_I', mag_i)
    sdss = rf.append_fields(sdss, 'POGSONMAG_Z', mag_z)
    return sdss



def transform_DES_to_SDSS(g_des, r_des, i_des, z_des):
    # Transform DES colors to SDSSdes
    # Transformation equations for stars bluer than (g-r)_sdss = 1.2:
    # Coefficients from Doug Tucker's and Anne Bauer's work, here:
    # https://cdcvs.fnal.gov/redmine/projects/descalibration/wiki
    Fmatrix = np.array([[1. - 0.104,  0.104,0., 0.],
                        [0., 1. - 0.102, 0.102, 0.],
                        [0., 0., 1 - 0.256, 0.256,],
                        [0., 0., -0.086, 1 + 0.086]])
                        
    const = np.array( [0.01, 0.02, 0.02,0.01] )
    
    des_mag =  np.array( [g_des, r_des, i_des, z_des] )
    Finv = np.linalg.inv(Fmatrix)
    
    sdss_mag = np.dot(Finv, (des_mag.T - const).T ) 
    return sdss_mag

def add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL'):
    print "Doing des->sdss color transforms for "+magTag_template
    filters = ['G','R','I','Z']
    magTags = []

    desMags = np.empty([len(filters),len(data)])
    for i,thisFilter in enumerate(filters):
        magTag = magTag_template+'_'+thisFilter
        desMags[i,:] = data[magTag]
    
    sdssMags = transform_DES_to_SDSS(desMags[0,:], desMags[1,:], desMags[2,:], desMags[3,:])
    
    data = rf.append_fields(data, magTag_template+'_G_SDSS', sdssMags[0,:])
    data = rf.append_fields(data, magTag_template+'_R_SDSS', sdssMags[1,:])
    data = rf.append_fields(data, magTag_template+'_I_SDSS', sdssMags[2,:])
    data = rf.append_fields(data, magTag_template+'_Z_SDSS', sdssMags[3,:])
    return data


def do_CMASS_cuts(data):
    
    data = add_SDSS_colors(data, magTag_template = 'MAG_DETMODEL')
    data = add_SDSS_colors(data, magTag_template = 'MAG_MODEL')
    data = add_SDSS_colors(data, magTag_template = 'MAG_PSF')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_4')
    data = add_SDSS_colors(data, magTag_template = 'MAG_APER_5')
    # Aperture magnitudes are in apertures of:
    # PHOT_APERTURES  1.85   3.70   5.55   7.41  11.11   14.81   18.52   22.22   25.93   29.63   44.44  66.67
    # (MAG_APER aperture diameter(s) are in pixels)
    # SDSS fiber and fiber2 mags are (sort of) in apertures of 3" and 2", respectively, which correspond to
    # 7.41 pix and 11.11 pix (in DES pixels) respectively, so we'll use mag_aper_4 and mag_aper_5 for the two fiber mags.
    print "Calculating/applying CMASS object cuts."
    dperp = ( (data['MAG_MODEL_R_SDSS'] - data['MAG_MODEL_I_SDSS']) -
             (data['MAG_MODEL_G_SDSS'] - data['MAG_MODEL_R_SDSS'])/8.0 )
             
    keep_cmass = ( (dperp > 0.55) &
                   (data['MAG_DETMODEL_I_SDSS'] < (19.86 + 1.6*(dperp - 0.8) )) &
                   (data['MAG_DETMODEL_I_SDSS'] > 17.5 ) & (data['MAG_DETMODEL_I_SDSS'] < 19.9) &
                   ((data['MAG_MODEL_R_SDSS'] - data['MAG_MODEL_I_SDSS'] ) < 2 ) &
                   (data['MAG_APER_4_I_SDSS'] < 21.5 ) &
                   ((data['MAG_PSF_I_SDSS'] - data['MAG_MODEL_I_SDSS']) > (0.2 + 0.2*(20.0 - data['MAG_MODEL_I_SDSS'] ) ) ) &
                   ((data['MAG_PSF_Z_SDSS'] - data['MAG_MODEL_Z_SDSS']) > (9.125 - 0.46 * data['MAG_MODEL_Z_SDSS'])) )
     
    return data[keep_cmass], data

def make_cmass_plots(data):
    fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(9,13))
    dperp = ( (data['MAG_MODEL_R_SDSS'] - data['MAG_MODEL_I_SDSS']) -
             (data['MAG_MODEL_G_SDSS'] - data['MAG_MODEL_R_SDSS'])/8.0 )
             
             
    ax1.plot(data['MAG_MODEL_G_SDSS'] - data['MAG_MODEL_R_SDSS'],
              data['MAG_MODEL_R_SDSS'] - data['MAG_MODEL_I_SDSS'],'.')
    ax1.set_xlabel("(g-r)_sdss (model)")
    ax1.set_ylabel("(r-i)_sdss (model)")
    ax1.set_xlim(0,3)
    ax1.set_ylim(0.5,1.6)
      
    ax2.plot(data['MAG_DETMODEL_I_SDSS'],dperp,'.',markersize=10)
    ax2.set_xlabel("i_sdss (model)")
    ax2.set_ylabel("dperp")
    ax2.set_xlim(17.0,20.0)
    ax2.set_ylim(0.5,1.6)

    figname ="../figure/cmass_plots"
    fig.savefig(figname)
    print "fig saved : "+figname+".png"


def makeMatchedPlotsCMASSLOWZ(sdss, des, figname):
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG'][:,1],des['MAG_MODEL_G'],'.')
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG'][:,2],des['MAG_MODEL_R'],'.')
    ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG'][:,3],des['MAG_MODEL_I'],'.')
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    ax4.plot(sdss['FIBER2MAG'][:,3],des['MAG_APER_4_I'],'.',label='aper4')
    ax4.plot(sdss['FIBERMAG'][:,3],des['MAG_APER_5_I'],'.',label='aper5',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)

    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlotsCMASSLOWZ(sdss, des, sdss2, des2, figname):
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG_G'],des['MAG_MODEL_G'],'b.')
    ax1.plot(sdss2['MODELMAG'][:,1],des2['MAG_MODEL_G'],'r.')
    ax1.plot([15,24],[15,24],'--',color='green')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG_R'],des['MAG_MODEL_R'],'b.')
    ax2.plot(sdss2['MODELMAG'][:,2],des2['MAG_MODEL_R'],'r.')
    ax2.plot([15,24],[15,24],'--',color='green')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG_I'],des['MAG_MODEL_I'],'b.')
    ax3.plot(sdss2['MODELMAG'][:,3],des2['MAG_MODEL_I'],'r.')
    ax3.plot([15,24],[15,24],'--',color='green')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    #ax4.plot(sdss2['FIBER2MAG'][:,3],des2['MAG_APER_4_I'],'.',label='aper4')
    #ax4.plot(sdss2['FIBERMAG'][:,3],des2['MAG_APER_5_I'],'.',label='aper5',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='green')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlotsALLSDSSGALS(sdss, des, sdss2, des2, figname):
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG_G'],des['MAG_MODEL_G'],'b.')
    ax1.plot(sdss2['MODELMAG_G'],des2['MAG_MODEL_G'],'r.')
    ax1.plot([15,24],[15,24],'--',color='green')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG_R'],des['MAG_MODEL_R'],'b.')
    ax2.plot(sdss2['MODELMAG_R'],des2['MAG_MODEL_R'],'r.')
    ax2.plot([15,24],[15,24],'--',color='green')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG_I'],des['MAG_MODEL_I'],'b.')
    ax3.plot(sdss2['MODELMAG_I'],des2['MAG_MODEL_I'],'r.')
    ax3.plot([15,24],[15,24],'--',color='green')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    #ax4.plot(sdss2['FIBER2MAG'][:,3],des2['MAG_APER_4_I'],'.',label='aper4')
    #ax4.plot(sdss2['FIBERMAG'][:,3],des2['MAG_APER_5_I'],'.',label='aper5',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='green')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlots(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG_G'],des['MAG_MODEL_G_SDSS'],'.')
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG_R'],des['MAG_MODEL_R_SDSS'],'.')
    ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG_I'],des['MAG_MODEL_I_SDSS'],'.')
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    ax4.plot(sdss['FIBER2MAG_I'],des['MAG_APER_4_I_SDSS'],'.',label='aper4')
    ax4.plot(sdss['FIBERMAG_I'],des['MAG_APER_5_I_SDSS'],'.',label='aper5',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlotsim3shape(sdss, des, figname = 'test', figtitle = 'test_title'):

    #sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    #print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'total :', len(sdss)
    fig,((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    #fig, (ax1,ax3) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    
    expcut =(des['DISC_FLUX'] == 0)
    devcut = (des['BULGE_FLUX'] == 0)
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    
    ax1.plot(sdss_exp['POGSONMAG_I'],des_exp['MAG_MODEL_I'],'r.', label = 'exp', alpha = 0.9)
    ax1.plot(sdss_dev['POGSONMAG_I'],des_dev['MAG_MODEL_I'],'b.', label = 'dev', alpha = 0.3)
    ax1.plot([0,30],[0,30],'--',color='red')
    ax1.set_xlabel('sdss i (model)')
    ax1.set_ylabel('im3shape i (model)')
    ax1.legend(loc='best')
    ax1.set_xlim(10,30)
    ax1.set_ylim(5,30)
    
    ax2.plot(sdss_exp['POGSONMAG_I'],des_exp['MAG_AUTO_I'] + 2.5,'r.', label = 'exp', alpha = 0.9)
    ax2.plot(sdss_dev['POGSONMAG_I'],des_dev['MAG_AUTO_I'] + 2.5,'b.', label = 'dev', alpha = 0.3)
    ax2.plot([0,30],[0,30],'--',color='red')
    ax2.set_xlabel('sdss i (model)')
    ax2.set_ylabel('des mag_auto i (model)')
    ax2.set_xlim(17,25)
    ax2.set_ylim(17,25)
    ax2 .legend(loc='best')
    
    ax3.plot(des_exp['MAG_AUTO_I'],des_exp['MAG_MODEL_I'],'r.', label = 'exp', alpha = 0.9)
    ax3.plot(des_dev['MAG_AUTO_I'],des_dev['MAG_MODEL_I'],'b.', label = 'dev', alpha = 0.3)
    ax3.plot([0,30],[0,30],'--',color='red')
    ax3.set_xlabel('des  MAG_AUTO_I')
    ax3.set_ylabel('im3shape i (model)')
    ax3.set_xlim(15,30)
    ax3.set_ylim(0,30)
    ax3.legend(loc='best')
    
    #ax4.plot(sdss['FIBER2MAG_I'],des['MAG_APER_4_I_SDSS'],'.',label='aper4')
    #ax4.plot(sdss['FIBERMAG_I'],des['MAG_APER_5_I_SDSS'],'.',label='aper5',alpha=0.33)
    ax3.plot(des_exp['MAG_AUTO_I'],des_exp['MAG_MODEL_I'],'r.', label = 'exp', alpha = 0.9)
    ax3.plot(des_dev['MAG_AUTO_I'],des_dev['MAG_MODEL_I'],'b.', label = 'dev', alpha = 0.3)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'



def makeMatchedPlots(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG_G'],des['MAG_MODEL_G_SDSS'],'.')
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG_R'],des['MAG_MODEL_R_SDSS'],'.')
    ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG_I'],des['MAG_MODEL_I_SDSS'],'.')
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    ax4.plot(sdss['FIBER2MAG_I'],des['MAG_APER_4_I_SDSS'],'.',label='aper4')
    ax4.plot(sdss['FIBERMAG_I'],des['MAG_APER_5_I_SDSS'],'.',label='aper5',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'



def makeMatchedPlotsMatching(sdss_up, des_up, sdss_down, des_down, figname = 'test', figtitle = 'test_title'):
    
    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss_up['MODELMAG_G'],des_up['MAG_MODEL_G_SDSS'],'b.')
    ax1.plot(sdss_down['MODELMAG_G'],des_down['MAG_MODEL_G_SDSS'],'r.')
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss_up['MODELMAG_R'],des_up['MAG_MODEL_R_SDSS'],'b.')
    ax2.plot(sdss_down['MODELMAG_R'],des_down['MAG_MODEL_R_SDSS'],'r.')
    ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss_up['MODELMAG_I'],des_up['MAG_MODEL_I_SDSS'],'b.')
    ax3.plot(sdss_down['MODELMAG_I'],des_down['MAG_MODEL_I_SDSS'],'r.')
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    ax4.plot(sdss_up['MODELMAG_Z'],des_up['MAG_MODEL_Z_SDSS'],'b.')
    ax4.plot(sdss_down['MODELMAG_Z'],des_down['MAG_MODEL_Z_SDSS'],'r.')
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss z (model)')
    ax4.set_ylabel('des z_sdss (model)')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlots_NoColorCorrection(sdss, des, figname):
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG_G'],des['MAG_MODEL_G'],'.')
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG_R'],des['MAG_MODEL_R'],'.')
    ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG_I'],des['MAG_MODEL_I'],'.')
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    ax4.plot(sdss['FIBER2MAG_I'],des['MAG_APER_4_I'],'.',label='aper4')
    ax4.plot(sdss['FIBERMAG_I'],des['MAG_APER_5_I'],'.',label='aper5',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def ModelClassifier(sdss, des, filter = 'G' ):
    parameter = 'BESTPROF_'+filter
    expmodel = ( sdss[parameter] == 0 )
    devmodel = ( sdss[parameter] == 1 )
    neither = - (expmodel|devmodel)
    
    return sdss[expmodel], des[expmodel], sdss[devmodel], des[devmodel], sdss[neither], des[neither]

def ModelClassifier2(data, filter = 'G' ):
    parameter = 'BESTPROF_'+filter
    expmodel = ( data[parameter] == 0 )
    devmodel = ( data[parameter] == 1 )
    neither = - (expmodel|devmodel)
    
    return data[expmodel], data[devmodel], data[neither]


def makeMatchedPlotsGalaxyProfile(sdss, des, figname = 'test', figtitle = 'test_title'):

    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'G')

    fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss_other['MODELMAG_G'],des_other['MAG_MODEL_G_SDSS'],'g.', alpha=0.33, label = 'other')
    ax1.plot(sdss_exp['MODELMAG_G'],des_exp['MAG_MODEL_G_SDSS'],'b.', label = 'exp')
    ax1.plot(sdss_dev['MODELMAG_G'],des_dev['MAG_MODEL_G_SDSS'],'r.', alpha = 0.33, label = 'dev')
    
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    ax1.legend(loc = 'best')
    
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'R')
    ax2.plot(sdss_other['MODELMAG_R'],des_other['MAG_MODEL_R_SDSS'],'g.', alpha=0.33)
    ax2.plot(sdss_exp['MODELMAG_R'],des_exp['MAG_MODEL_R_SDSS'],'b.')
    ax2.plot(sdss_dev['MODELMAG_R'],des_dev['MAG_MODEL_R_SDSS'],'r.', alpha = 0.33)
    ax2.plot([15, 24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    
    ax3.plot(sdss_other['MODELMAG_I'],des_other['MAG_MODEL_I_SDSS'],'g.', alpha=0.33)
    ax3.plot(sdss_exp['MODELMAG_I'],des_exp['MAG_MODEL_I_SDSS'],'b.')
    ax3.plot(sdss_dev['MODELMAG_I'],des_dev['MAG_MODEL_I_SDSS'],'r.', alpha = 0.33)
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'Z')
    
    ax4.plot(sdss_other['MODELMAG_Z'],des_other['MAG_MODEL_Z_SDSS'],'g.', alpha=0.33)
    ax4.plot(sdss_exp['MODELMAG_Z'],des_exp['MAG_MODEL_Z_SDSS'],'b.')
    ax4.plot(sdss_dev['MODELMAG_Z'],des_dev['MAG_MODEL_Z_SDSS'],'r.', alpha = 0.33)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss z (model)')
    ax4.set_ylabel('des z_sds s (model)')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'



def match(sdss, des):
    h = esutil.htm.HTM(10)
    matchDist = 1/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(des['RA'], des['DEC'], sdss['RA'],sdss['DEC'],matchDist,maxmatch=1)
    
    print 'matched object ',len(m_des)
    return sdss[m_sdss],des[m_des]



def SpatialCuts( data ):
    cut = ((data['RA'] < 355.0) &
           (data['DEC'] < 0.0) &
           (data['RA'] > 350.0) &
           (data['DEC'] > - 0.5))
    return data[cut]


def ColorCuts( sdssdata ):
    
    sdsscut_G = ((sdssdata['MODELMAG_G'] < 18.0) & (sdssdata['MODELMAG_G'] > 15.0))
    sdsscut_R = ((sdssdata['MODELMAG_R'] < 18.0) & (sdssdata['MODELMAG_R'] > 15.0))
    sdsscut_I = ((sdssdata['MODELMAG_I'] < 18.0) & (sdssdata['MODELMAG_I'] > 15.0))
    sdsscut_Z = ((sdssdata['MODELMAG_Z'] < 18.0) & (sdssdata['MODELMAG_Z'] > 15.0))

    return sdssdata[sdsscut_G], sdssdata[sdsscut_R], sdssdata[sdsscut_I], sdssdata[sdsscut_Z]


def plotPosition( sdss_data_G, sdss_data_R, sdss_data_I, sdss_data_Z, figname ):

    fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot( sdss_data_G['RA'], sdss_data_G['DEC'], 'b.' ,alpha=0.33)
    ax1.set_xlim(350, 355)
    ax1.set_ylim(-0.5, 0.0)
    ax1.set_xlabel('sdss_modelmag_G_RA')
    ax1.set_ylabel('sdss_DEC')
    
    ax2.plot( sdss_data_R['RA'], sdss_data_R['DEC'], 'r.',alpha=0.53 )
    ax2.set_xlim(350, 355)
    ax2.set_ylim(-0.5, 0.0)
    ax2.set_xlabel('sdss_modelmag_R_RA')
    ax2.set_ylabel('sdss_DEC')

    ax3.plot( sdss_data_I['RA'], sdss_data_I['DEC'], 'g.',alpha=0.53 )
    ax3.set_xlim(350, 355)
    ax3.set_ylim(-0.5, 0.0)
    ax3.set_xlabel('sdss_modelmag_I_RA')
    ax3.set_ylabel('sdss_DEC')
    
    ax4.plot( sdss_data_Z['RA'], sdss_data_Z['DEC'], 'c.',alpha=0.53 )
    ax4.set_xlim(350, 355)
    ax4.set_ylim(-0.5, 0.0)
    ax4.set_xlabel('sdss_modelmag_Z_RA')
    ax4.set_ylabel('sdss_DEC')
    
    
    plt.title('sdss_des color cut 15-19')
    
    
    print "save fig : "+figname+'.png'
    fig.savefig(figname)


def makeColorCutPositionPlot(sdss_data, des_data):
    #color cut
    sdss_data_G,sdss_data_R, sdss_data_I, sdss_data_Z = ColorCuts(sdss_data)
    sdss_matched_G, des_matched = match(sdss_data_G, des_data)
    sdss_matched_R, des_matched = match(sdss_data_R, des_data)
    sdss_matched_I, des_matched = match(sdss_data_I, des_data)
    sdss_matched_Z, des_matched = match(sdss_data_Z, des_data)
    plotPosition(sdss_matched_G, sdss_matched_R, sdss_matched_I, sdss_matched_Z, '../figure/sdss_des_comparison_colorcut')
    return sdss_matched_G, sdss_matched_R, sdss_matched_I, sdss_matched_Z



def makeMatchedPlots2(sdss,des, figname = 'test'):
    """ color difference plot """
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['MODELMAG_G'] - sdss['MODELMAG_R'] ,des['MAG_MODEL_G']-sdss['MODELMAG_G'],'.')
    #ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('g_s-r_s')
    ax1.set_ylabel('g_d-g_s')
    #ax1.set_xlim(15,24)
    #ax1.set_ylim(15,24)
    
    ax2.plot(sdss['MODELMAG_R'] - sdss['MODELMAG_I'],des['MAG_MODEL_R'] - sdss['MODELMAG_R'],'.')
    #ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('r_s-i_s')
    ax2.set_ylabel('r_d-r_s')
    #ax2.set_xlim(15,24)
    #ax2.set_ylim(15,24)
    
    ax3.plot(sdss['MODELMAG_I'] - sdss['MODELMAG_Z'],des['MAG_MODEL_I'] - sdss['MODELMAG_I'],'.')
    #ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('i_s-z_s')
    ax3.set_ylabel('z_d-z_s')
    #ax3.set_xlim(15,24)
    #ax3.set_ylim(15,24)
    
    ax4.plot(sdss['MODELMAG_I']- sdss['MODELMAG_Z'] ,des['MAG_MODEL_Z'] - sdss['MODELMAG_Z'],'.')
    #ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('i_s-z_s')
    ax4.set_ylabel('i_d-i_s')
    ax4.legend(loc='best')
    #ax4.set_xlim(15,24)
    #ax4.set_ylim(15,24)
    
    fig.savefig(figname)
    print "fig saved :", figname+'.png'




def doBasicCuts(des):
    # 25 is the faintest object detected by DES
    # objects smaller than 25 considered as Noise
    use = ((des['MAG_MODEL_G'] < 23) &
           (des['MAG_MODEL_R'] < 23) &
           (des['MAG_MODEL_I'] < 21) &
           (des['MAG_MODEL_Z'] < 23) &
           (des['FLAGS_G'] < 3) &
           (des['FLAGS_R'] < 3) &
           (des['FLAGS_I'] < 3) &
           (des['FLAGS_Z'] < 3))
    return des[use]


def R_ColorCuts( sdssdata ):
    sdsscut_R = ((sdssdata['MODELMAG_R'] < 20.0) & (sdssdata['MODELMAG_R'] > 18.0))
    blue_star = (sdssdata['MODELMAG_G'] - sdssdata['MODELMAG_R'] < 1.2 )
    return sdssdata[sdsscut_R]


def DESclassifier(des):
    #des = modestify(des)
    gal_cut = (des['modtype'] == 1)
    star_cut = (des['modtype'] == 3)
    return des[gal_cut], des[star_cut]


def SDSSclassifier(sdss):
    star_cut = ((sdss['TYPE_G'] == 6) &
                (sdss['TYPE_R'] == 6) &
                (sdss['TYPE_I'] == 6) &
                (sdss['TYPE_Z'] == 6))
                
    gal_cut = ((sdss['TYPE_G'] == 3) &
               (sdss['TYPE_R'] == 3) &
               (sdss['TYPE_I'] == 3) &
               (sdss['TYPE_Z'] == 3) & (sdss['CLEAN'] == 1))
    return sdss[gal_cut], sdss[star_cut]


def SortNotBlended(sdss):
    # 0 value means blended flags 3 for SDSS is not set. = Not a composite object
    
    notblended = ((sdss['FLAGS_G'] & 2**3 == 0) &
           (sdss['FLAGS_R'] & 2**3 == 0) &
           (sdss['FLAGS_I'] & 2**3 == 0) &
           (sdss['FLAGS_Z'] & 2**3 == 0))
           
    blended = ((sdss['FLAGS_G'] & 2**3 == 0) &
                 (sdss['FLAGS_R'] & 2**3 == 0) &
                 (sdss['FLAGS_I'] & 2**3 == 0) &
                 (sdss['FLAGS_Z'] & 2**3 == 0))
       
    print 'Not Blended Obj in SDSS',np.sum(use)
    return sdss[notblended] #, sdss[blended]


def parallelogramColorCut(des_matched, sdss_matched, color = 'G'):
    # G color
    descolor = 'MAG_MODEL_'+color+'_SDSS'
    sdsscolor = 'MODELMAG_'+color
    
    up = ((sdss_matched[sdsscolor] < 19.0 )& (sdss_matched[sdsscolor] > 10.0 ) &
          (des_matched[descolor] - sdss_matched[sdsscolor] > 0.2 ) &
          (des_matched[descolor] - sdss_matched[sdsscolor] < 1.0 ))

    down = ((sdss_matched[sdsscolor] < 19.0 )& (sdss_matched[sdsscolor] > 10.0 ) &
          (des_matched[descolor] - sdss_matched[sdsscolor] < 0.1 ) &
          (des_matched[descolor] - sdss_matched[sdsscolor] > - 0.1 ))

    print 'parallel cut ' + color
    return des_matched[up], sdss_matched[up], des_matched[down], sdss_matched[down]






def MakePlotsDeltaVSdistance(sdss, des, figname = 'test', figtitle = 'test title'):
    from numpy import sqrt
    distance = sqrt((des['RA'] - sdss['RA'])**2 + (des['DEC'] - sdss['DEC'])**2)

    """ color difference plot """
    
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(distance ,des['MAG_MODEL_G_SDSS']-sdss['MODELMAG_G'],'.')
    #ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('distance')
    ax1.set_ylabel('g_d-g_s')
    ax1.set_xlim(0.0,0.0002)
    ax1.set_ylim(-0.4,0.4)
    
    ax2.plot(distance ,des['MAG_MODEL_R_SDSS'] - sdss['MODELMAG_R'],'.')
    #ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_ylabel('r_d-r_s')
    ax2.set_xlabel('distance')
    ax2.set_xlim(0.0,0.0002)
    ax2.set_ylim(-0.4,0.4)
    
    ax3.plot(distance, des['MAG_MODEL_I_SDSS'] - sdss['MODELMAG_I'],'.')
    #ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_ylabel('i_d-i_s')
    ax3.set_xlabel('distance')
    ax3.set_xlim(0.0,0.0002)
    ax3.set_ylim(-0.4,0.4)
    
    ax4.plot(distance, des['MAG_MODEL_Z_SDSS'] - sdss['MODELMAG_Z'],'.')
    #ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_ylabel('z_d-z_s')
    ax4.set_xlabel('distance')
    ax4.legend(loc='best')
    ax4.set_xlim(0.0,0.0002)
    ax4.set_ylim(-0.4,0.4)
    
    fig.suptitle(figtitle, fontsize = 20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'

def main():
    
    sdss_data = getSDSScatalogs() #Eric's catalog
    sdss_data2 = getSDSScatalogsCMASSLOWZ()
    des_data = getDEScatalogs()
    des_data = doBasicCuts(des_data)
    
    des_data = SpatialCuts(des_data)
    #sdss_data= R_ColorCuts(sdss_data)
    sdss_data = SpatialCuts(sdss_data)
    sdss_data2 = SpatialCuts(sdss_data2)
    sdss_data = whichGalaxyProfile(sdss_data)
    des_data = modestify(des_data)
    #makeColorCutPositionPlot(sdss_data, des_data)
    
    #make_cmass_plots(cmass)
    
    cmass, des_data = do_CMASS_cuts(des_data)
    des_gals, des_stars = DESclassifier(des_data)
    #sdss_gals, sdss_stars = SDSSclassifier(sdss_data)
    
    sdss_matched, des_matched = match(sdss_data, des_stars)
    figname = "../figure/sdss_des_comparison_onlystars"
    makeMatchedPlots(sdss_matched, des_matched, figname=figname)
    figname = "../figure/sdss_des_comparison_onlystars_CMASSLOWZ"
    makeMatchedPlots_NoColorCorrection(sdss_matched, des_matched, figname = figname)
    figname2 = "../figure/sdss_des_comparison2_onlystars_CMASSLOWZ"
    makeMatchedPlots2(sdss_matched, des_matched, figname = figname2)
    

    sdss_matched, des_matched = match(sdss_data, des_gals)
    figname = "../figure/sdss_des_comparison_onlygals"
    makeMatchedPlots(sdss_matched, des_matched, figname = figname)
    figname = "../figure/sdss_des_comparison_onlygals"
    makeMatchedPlots_NoColorCorrection(sdss_matched, des_matched, figname = figname)
    figname2 = "../figure/sdss_des_comparison2_onlygals"
    makeMatchedPlots2(sdss_matched, des_matched, figname = figname2)


    sdss_matched2, des_matched2 = match(sdss_data2, des_gals)
    figname = "../figure/sdss_des_comparison_onlygals_CMASSLOWZ"
    makeMatchedPlotsCMASSLOWZ(sdss_matched, des_matched, sdss_matched2, des_matched2, figname = figname)

    sdss_matched3, des_matched3 = match(sdss_gals, des_gals)
    figname = "../figure/sdss_des_comparison_onlygals_ALLSDSSGALS"
    makeMatchedPlotsALLSDSSGALS(sdss_matched, des_matched, sdss_matched3, des_matched3, figname = figname)





def makeHistogram( des_data ):

    import numpy as np
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))

    
    x = des_data['RA_AS']
    y = des_data['DEC_AS']

    ax.hist(x, 30, log = 'True', normed=1, facecolor='green', alpha=0.75)
    ax.set_xlabel(' ra_as ')
    ax.set_ylabel(' N ')
    
    ax2.hist(y, 30, log = 'True', normed=1, facecolor='green', alpha=0.75)
    ax2.set_xlabel(' ra_as ')
    ax2.set_ylabel(' N ')

    plt.title('Histogram')
    #plt.axis([40, 160, 0, 0.03])
    #ax.set_grid(True)
    fig.savefig('../figure/histogram')
    print 'figsave : ../figure/histogram.png'




def main2():

    sdss_data_o = getSDSScatalogs() #Eric's catalog
    sdss_data = SpatialCuts(sdss_data_o)
    sdss_data = addPogsonMag(sdss_data)
    
    #sdss_data = simplewhichGalaxyProfile(sdss_data)


    des_data_o = getDEScatalogs(file = '/n/des/huff.791/Projects/CMASS/Scripts/main_combined.fits')
    des_data = SpatialCuts(des_data_o)
    des_data = addMagforim3shape(des_data) # add magnitude
    
    
    sdss_matched, des_matched = match(sdss_data, des_data)
    makeMatchedPlotsim3shape(sdss_matched, des_matched)
    
    
    #des_data = modestify(des_data)
    #cmass, des_data = do_CMASS_cuts(des_data)
    #des_gals, des_stars = DESclassifier(des_data)
    
    
    
    figname = "../figure/simpleGalaxyProfile_colorcomparison2"
    figtitle = 'Galaxy Profile'
    makeMatchedPlotsGalaxyProfile(sdss_matched, des_matched, figtitle = figtitle, figname = figname)
    
    figname = "../figure/Sortout_blended_colorcomparison"
    figtitle = 'Only NonBlended Objects (SDSS, DES)'
    makeMatchedPlots(sdss_matched, des_matched, figname = figname, figtitle = figtitle)


    figname = '../figure/ColorVsDistance'
    MakePlotsDeltaVSdistance(sdss_matched, des_matched, figname = figname, figtitle = None )
    
    
    
    # G color cut
    des_matched_up, sdss_matched_up, des_matched_down, sdss_matched_down, = parallelogramColorCut(des_matched, sdss_matched, color = 'G')
    figname = '../figure/ColorVsDistance_G_cut_up'
    MakePlotsDeltaVSdistance(sdss_matched_up, des_matched_up, figname = figname, figtitle = 'mag_g cut 10-19, upper branch' )
    figname = '../figure/ColorVsDistance_G_cut_down'
    MakePlotsDeltaVSdistance(sdss_matched_down, des_matched_down, figname = figname, figtitle = 'mag_g cut 10-19, lower branch'  )
    figname = '../figure/ColorVsDistance_G_matching'
    makeMatchedPlotsMatching(sdss_matched_up, des_matched_up, sdss_matched_down, des_matched_down, figname = figname, figtitle = 'mag_g cut 10-19, lower branch')

    # R color cut
    des_matched_up, sdss_matched_up, des_matched_down, sdss_matched_down, = parallelogramColorCut(des_matched, sdss_matched, color = 'R')
    figname = '../figure/ColorVsDistance_R_cut_up'
    MakePlotsDeltaVSdistance(sdss_matched_up, des_matched_up, figname = figname, figtitle = 'mag_r cut 10-19, upper branch' )
    figname = '../figure/ColorVsDistance_R_cut_down'
    MakePlotsDeltaVSdistance(sdss_matched_down, des_matched_down, figname = figname, figtitle = 'mag_r cut 10-19, lower branch'  )
    figname = '../figure/ColorVsDistance_R_matching'
    makeMatchedPlotsMatching(sdss_matched_up, des_matched_up, sdss_matched_down, des_matched_down, figname = figname, figtitle = 'mag_r cut 10-19, lower branch')

    
    # I color cut
    des_matched_up, sdss_matched_up, des_matched_down, sdss_matched_down, = parallelogramColorCut(des_matched, sdss_matched, color = 'R')
    figname = '../figure/ColorVsDistance_I_cut_up'
    MakePlotsDeltaVSdistance(sdss_matched_up, des_matched_up, figname = figname, figtitle = 'mag_i cut 10-19, upper branch' )
    figname = '../figure/ColorVsDistance_I_cut_down'
    MakePlotsDeltaVSdistance(sdss_matched_down, des_matched_down, figname = figname, figtitle = 'mag_i cut 10-19, lower branch'  )
    figname = '../figure/ColorVsDistance_I_matching'
    makeMatchedPlotsMatching(sdss_matched_up, des_matched_up, sdss_matched_down, des_matched_down, figname = figname, figtitle = 'mag_i cut 10-19, lower branch')
    
    
    mainfigname = "../figure/test"
    makeMatchedPlots(sdss_matched_G_up, des_matched_G_up, figname)



def im3shape_galprof_mask( im3shape, fulldes ):
    """ add mode to distingush galaxy profiles used in im3shape """
    """ return only full des """

    im3galprofile = np.zeros(len(im3shape), dtype=np.int32)
    
    expcut = (im3shape['BULGE_FLUX'] == 0)
    devcut = (im3shape['DISC_FLUX'] == 0)
    
    des_exp = im3shape[expcut]
    des_dev = im3shape[devcut]

    expID = des_exp['COADD_OBJECTS_ID']
    devID = des_dev['COADD_OBJECTS_ID']
    fullID = fulldes['COADD_OBJECTS_ID']
    
    expmask = np.in1d(fullID, expID)
    devmask = np.in1d(fullID, devID)

    im3galprofile[expmask] = 1
    im3galprofile[devmask] = 2

    data = rf.append_fields(fulldes, 'IM3_GALPROF', im3galprofile)
    return data



def im3shape_add_radius( im3shape, fulldes ):
    
    im3radius = np.zeros(len(im3shape), dtype=np.int32)
    
    common_cut = np.in1d(fulldes['COADD_OBJECTS_ID'], im3shape['COADD_OBJECTS_ID'])
    im3radius[common_cut] = im3shape['RADIUS']
    
    data = rf.append_fields(fulldes, 'IM3_RADIUS', im3radius)
    return data




def makeMatchedPlotsWithim3shapeMask(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    """ return corrected color comparison plots and residual plots """
    
    
    #sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    fig,((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    fig2,((ax5,ax6),(ax7, ax8)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    fig3,((ax9,ax10),(ax11, ax12)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    fig4,((ax13,ax14),(ax15, ax16)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    
    print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'neither :', len(sdss) -len(sdss_exp)-len(sdss_dev)

    Scolorkind = 'MODELMAG_'
    Dcolorkind = 'MAG_MODEL_'
    colorCorrection = ''#'_SDSS'
    

    
    filter = 'G'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #ax1.plot(sdss_exp[sdssColor],des_exp[desColor],'b.', label = 'exp', alpha = 0.9)
    #ax1.plot(sdss_dev[sdssColor],des_dev[desColor],'r.', label = 'dev', alpha = 0.33)
    #ax1.plot(sdss_ne[sdssColor],des_ne[desColor],'g.', label = 'neither', alpha = 0.3)

    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19) & (sdss_exp[sdssColor] > 15) & (des_exp[desColor] > 15)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19) & (sdss_dev[sdssColor] > 15) & (des_dev[desColor] > 15)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    #ax1.plot(sdss_exp[sdssColor], ys_exp, 'c', label=polynomial_exp)
    
    alpha, beta = coef_exp
    exp_des_corrected = (des_exp[desColor] - beta)/alpha
    ax1.plot(sdss_exp[sdssColor], exp_des_corrected,'r.', label = 'exp', alpha = 1.0)
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    #ax1.plot(sdss_dev[sdssColor], ys_dev, 'y',label=polynomial_dev)
    
    alpha, beta = coef_dev
    dev_des_corrected = (des_dev[desColor] - beta)/alpha
    ax1.plot(sdss_dev[sdssColor],dev_des_corrected,'c.', label = 'dev', alpha = 0.3)

    ax1.plot([0,30],[0,30],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('sdss_colored_des g')
    ax1.legend(loc='best', fontsize = 10)
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)
    
    # residuals
    residuals_e = exp_des_corrected - sdss_exp[sdssColor]
    residuals_d = dev_des_corrected - sdss_dev[sdssColor]
    std_e = np.sqrt( np.sum(residuals_e**2)/len(residuals_e) )
    std_d = np.sqrt( np.sum(residuals_d**2)/len(residuals_d) )
    
    ax5.plot([0,30],[0,0],'--',color='red')
    ax5.plot( des_exp[desColor], residuals_e, 'b.', label = 'exp, std :'+str(std_e))
    ax5.plot( des_dev[desColor], residuals_d, 'r.', label = 'dev, std :'+str(std_d), alpha=0.3)
    ax5.set_xlim(15,24)
    ax5.set_ylim(-5,5)
    ax5.set_xlabel('sdss g (model)')
    ax5.set_ylabel('residual = des_color - fit')
    ax5.legend(loc='best')
    
    #Histogram
    n, bins, patches = ax9.hist(residuals_e, 100, facecolor='red', alpha=0.75)
    ax9.grid(True)
    ax9.set_xlabel('residual_e')
    ax9.set_ylabel('N')
    ax9.set_xlim(-1,1)
    
    
    filter = 'R'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #ax2.plot(sdss_exp[sdssColor],des_exp[desColor],'b.', label = 'exp', alpha = 0.9)
    #ax2.plot(sdss_dev[sdssColor],des_dev[desColor],'r.', label = 'dev', alpha = 0.33)
    #ax2.plot(sdss_ne[sdssColor],des_ne[desColor],'g.', label = 'neither', alpha = 0.3)
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20) & (sdss_exp[sdssColor] > 15) & (des_exp[desColor] > 15)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20) & (sdss_dev[sdssColor] > 15) & (des_dev[desColor] > 15)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    #ax2.plot(sdss_exp[sdssColor], ys_exp, 'c', label=polynomial_exp)
    
    alpha, beta = coef_exp
    exp_des_corrected = (des_exp[desColor] - beta)/alpha
    ax2.plot(sdss_exp[sdssColor], exp_des_corrected,'r.', label = 'exp', alpha = 1.0)

    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    #ax2.plot(sdss_dev[sdssColor], ys_dev, 'y',label=polynomial_dev)
    
    alpha, beta = coef_dev
    dev_des_corrected = (des_dev[desColor] - beta)/alpha
    ax2.plot(sdss_dev[sdssColor],dev_des_corrected,'c.', label = 'dev', alpha = 0.3)


    ax2.plot([0,30],[0,30],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('sdss_colored_des r')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    ax2.legend(loc='best', fontsize = 10)
    
    # residuals
    residuals_e = exp_des_corrected - sdss_exp[sdssColor]
    residuals_d = dev_des_corrected - sdss_dev[sdssColor]
    std_e = np.sqrt( np.sum(residuals_e**2)/len(residuals_e) )
    std_d = np.sqrt( np.sum(residuals_d**2)/len(residuals_d) )
    
    ax6.plot([0,30],[0,0],'--',color='red')
    ax6.plot( des_exp[desColor], residuals_e, 'b.', label = 'exp, std :'+str(std_e))
    ax6.plot( des_dev[desColor], residuals_d, 'r.', label = 'dev, std :'+str(std_d), alpha=0.3)
    ax6.set_xlim(15,24)
    ax6.set_ylim(-5,5)
    ax6.set_xlabel('sdss g (model)')
    ax6.set_ylabel('residual = des_color - fit')
    ax6.legend(loc='best')
    
    #Histogram
    n, bins, patches = ax10.hist(residuals_e, 100, facecolor='red', alpha=0.75)
    ax10.grid(True)
    ax10.set_xlabel('residual')
    ax10.set_ylabel('N')
    ax10.set_xlim(-1,1)
    
    
    filter = 'I'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #ax3.plot(sdss_exp[sdssColor],des_exp[desColor],'b.', label = 'exp', alpha = 0.9)
    #ax3.plot(sdss_dev[sdssColor],des_dev[desColor],'r.', label = 'dev', alpha = 0.33)
    #ax3.plot(sdss_ne[sdssColor],des_ne[desColor],'g.', label = 'neither', alpha = 0.3)
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19) & (sdss_exp[sdssColor] > 15) & (des_exp[desColor] > 15)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19) & (sdss_dev[sdssColor] > 15) & (des_dev[desColor] > 15)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    #ax3.plot(sdss_exp[sdssColor], ys_exp, 'c', label=polynomial_exp)
    
    alpha, beta = coef_exp
    exp_des_corrected = (des_exp[desColor] - beta)/alpha
    ax3.plot(sdss_exp[sdssColor], exp_des_corrected,'r.', label = 'exp', alpha = 1.0)

    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    #ax3.plot(sdss_dev[sdssColor], ys_dev, 'y',label=polynomial_dev)
    
    alpha, beta = coef_dev
    dev_des_corrected = (des_dev[desColor] - beta)/alpha
    ax3.plot(sdss_dev[sdssColor], dev_des_corrected,'c.', label = 'dev', alpha = 0.3)
    
    ax3.plot([0,30],[0,30],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('sdss_colored_des i')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    ax3.legend(loc='best', fontsize = 10)

    # residuals
    residuals_e = exp_des_corrected - sdss_exp[sdssColor]
    residuals_d = dev_des_corrected - sdss_dev[sdssColor]
    std_e = np.sqrt( np.sum(residuals_e**2)/len(residuals_e) )
    std_d = np.sqrt( np.sum(residuals_d**2)/len(residuals_d) )
    
    ax7.plot([0,30],[0,0],'--',color='red')
    ax7.plot( des_exp[desColor], residuals_e, 'b.', label = 'exp, std :'+str(std_e))
    ax7.plot( des_dev[desColor], residuals_d, 'r.', label = 'dev, std :'+str(std_d), alpha=0.3)
    ax7.set_xlim(15,24)
    ax7.set_ylim(-5,5)
    ax7.set_xlabel('sdss g (model)')
    ax7.set_ylabel('residual = des_color - fit')
    ax7.legend(loc='best')

    #Histogram
    n, bins, patches = ax11.hist(residuals_e, 100, facecolor='red', alpha=0.75)
    ax11.grid(True)
    ax11.set_xlabel('residual')
    ax11.set_ylabel('N')
    ax11.set_xlim(-1,1)
    

    filter = 'Z'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #ax4.plot(sdss_exp[sdssColor],des_exp[desColor],'b.', label = 'exp', alpha = 0.9)
    #ax4.plot(sdss_dev[sdssColor],des_dev[desColor],'r.', label = 'dev', alpha = 0.9)
    #ax4.plot(sdss_ne[sdssColor],des_ne[desColor],'g.', label = 'neither', alpha = 0.3)
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19) & (sdss_exp[sdssColor] > 15) & (des_exp[desColor] > 15)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19) & (sdss_dev[sdssColor] > 15) & (des_dev[desColor] > 15)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    #ax4.plot(sdss_exp[sdssColor], ys_exp, 'c', label=polynomial_exp)
    
    alpha, beta = coef_exp
    exp_des_corrected = (des_exp[desColor] - beta)/alpha
    ax4.plot(sdss_exp[sdssColor], exp_des_corrected,'r.', label = 'exp', alpha = 1.0)

    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    ax4.plot(sdss_dev[sdssColor], ys_dev, 'y',label=polynomial_dev)

    alpha, beta = coef_dev
    dev_des_corrected = (des_dev[desColor] - beta)/alpha
    ax4.plot(sdss_dev[sdssColor],dev_des_corrected,'c.', label = 'dev', alpha = 0.3)

    ax4.plot([0,30],[0,30],'--',color='red')
    ax4.set_xlabel('sdss z (model)')
    ax4.set_ylabel('sdss_colored_des z')
    ax4.legend(loc='best', fontsize = 10)
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)
    
    # residuals
    residuals_e = exp_des_corrected - sdss_exp[sdssColor]
    residuals_d = dev_des_corrected - sdss_dev[sdssColor]
    std_e = np.sqrt( np.sum(residuals_e**2)/len(residuals_e) )
    std_d = np.sqrt( np.sum(residuals_d**2)/len(residuals_d) )
    
    ax8.plot([0,30],[0,0],'--',color='red')
    ax8.plot( des_exp[desColor], residuals_e, 'b.', label = 'exp, std :'+str(std_e))
    ax8.plot( des_dev[desColor], residuals_d, 'r.', label = 'dev, std :'+str(std_d), alpha=0.3)
    ax8.set_xlim(15,24)
    ax8.set_ylim(-5,5)
    ax8.set_xlabel('sdss g (model)')
    ax8.set_ylabel('residual = des_color - fit')
    ax8.legend(loc='best')

    #Histogram
    n, bins, patches = ax12.hist(residuals_e, 100, facecolor='red', alpha=0.75)
    ax12.grid(True)
    ax12.set_xlabel('residual')
    ax12.set_ylabel('N')
    ax12.set_xlim(-1,1)

    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    
    fig2.suptitle(figtitle+'_residual', fontsize = 20)
    fig2.savefig(figname+'_residual')
    
    fig3.suptitle(figtitle+'_hist', fontsize = 20)
    fig3.savefig(figname+'_hist')
    
    print "fig saved :", figname+'.png'
    print "fig saved :",   figname+'_residual'+'.png'
    print "fig saved :",   figname+'_hist'+'.png'



def makeMatchedPlotsWithim3shapeMask_GR(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    #sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    
    print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'neither :', len(sdss) -len(sdss_exp)-len(sdss_dev)
    
    Scolorkind = 'MODELMAG_'
    Dcolorkind = 'MAG_MODEL_'
    colorCorrection = ''#'_SDSS'
    
    
    
    filter = 'G'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #ax1.plot(sdss_exp[sdssColor],des_exp[desColor],'b.', label = 'exp', alpha = 0.9)
    #ax1.plot(sdss_dev[sdssColor],des_dev[desColor],'r.', label = 'dev', alpha = 0.33)
    #ax1.plot(sdss_ne[sdssColor],des_ne[desColor],'g.', label = 'neither', alpha = 0.3)
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20) & (sdss_exp[sdssColor] > 15) & (des_exp[desColor] > 15)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20) & (sdss_dev[sdssColor] > 15) & (des_dev[desColor] > 15)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    #ax1.plot(sdss_exp[sdssColor], ys_exp, 'c', label=polynomial_exp)
    
    alpha, beta = coef_exp
    exp_des_corrected_G = (des_exp[desColor] - beta)/alpha
    ax1.plot(sdss_exp[sdssColor],exp_des_corrected_G,'c.', label = 'exp', alpha = 0.3)

    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    #ax1.plot(sdss_dev[sdssColor], ys_dev, 'y',label=polynomial_dev)
    
    print coef_dev
    alpha, beta = coef_dev
    dev_des_corrected_G = (des_dev[desColor] - beta)/alpha
    ax1.plot(sdss_dev[sdssColor],dev_des_corrected_G,'c.', label = 'dev', alpha = 0.3)
    
    #stop
    
    
    ax1.plot([-30,30],[-30,30],'--',color='red')
    ax1.set_xlabel('des_g - des_r (model, corrected)')
    ax1.set_ylabel('sdss_g - sdss_r')
    ax1.set_xlim(-5,5)
    ax1.set_ylim(-5,5)
    ax1.legend(loc='best', fontsize = 10)
    
    
    filter = 'R'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #ax2.plot(sdss_exp[sdssColor],des_exp[desColor],'b.', label = 'exp', alpha = 0.9)
    #ax2.plot(sdss_dev[sdssColor],des_dev[desColor],'r.', label = 'dev', alpha = 0.33)
    #ax2.plot(sdss_ne[sdssColor],des_ne[desColor],'g.', label = 'neither', alpha = 0.3)

    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    #ax2.plot(sdss_exp[sdssColor], ys_exp, 'c', label=polynomial_exp)
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    #ax2.plot(sdss_dev[sdssColor], ys_dev, 'y',label=polynomial_dev)
    
    alpha, beta = coef_exp
    exp_des_corrected_R = (des_exp[desColor] - beta)/alpha

    alpha, beta = coef_dev
    dev_des_corrected_R = (des_dev[desColor] - beta)/alpha
    #ax2.plot(sdss_dev[sdssColor],dev_des_corrected_R,'c.', label = 'dev', alpha = 0.3)
    ax1.plot(exp_des_corrected_G - exp_des_corrected_R ,sdss_exp[Scolorkind+'G']- sdss_exp[Scolorkind+'R'], 'r.', label = 'exp')
    ax2.plot(dev_des_corrected_G - dev_des_corrected_R ,sdss_dev[Scolorkind+'G']- sdss_dev[Scolorkind+'R'], 'b.', label = 'dev', alpha = 1.0)

    
    ax2.plot([-30,30],[-30,30],'--',color='red')
    ax2.set_xlabel('des_g - des_r (model, corrected)')
    ax2.set_ylabel('sdss_g - sdss_r')
    ax2.set_xlim(-5,5)
    ax2.set_ylim(-5,5)
    ax2.legend(loc='best', fontsize = 10)
    
    

    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlotsWithim3shapeMask_Pogson(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    #sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    #print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'total :', len(sdss)
    fig,((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    #fig, (ax1,ax3) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    
    print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'neither :', len(sdss) -len(sdss_exp)-len(sdss_dev)
    
    ax1.plot(sdss_exp['POGSONMAG_G'],des_exp['MAG_MODEL_G_SDSS'] + 2.5,'b.', label = 'exp', alpha = 0.9)
    ax1.plot(sdss_dev['POGSONMAG_G'],des_dev['MAG_MODEL_G_SDSS']+ 2.5,'r.', label = 'dev', alpha = 0.33)
    ax1.plot(sdss_ne['POGSONMAG_G'],des_ne['MAG_MODEL_G_SDSS']+ 2.5,'g.', label = 'neither', alpha = 0.3)
    ax1.plot([0,30],[0,30],'--',color='red')
    ax1.set_xlabel('sdss g : 25 - 2.5 * np.log10(sdss[EXPFLUX_G])')
    ax1.set_ylabel('sdss_colored_des g')
    ax1.legend(loc='best')
    ax1.set_xlim(15,26)
    ax1.set_ylim(15,26)
    
    ax2.plot(sdss_exp['POGSONMAG_R'],des_exp['MAG_MODEL_R_SDSS']+ 2.5,'b.', label = 'exp', alpha = 0.9)
    ax2.plot(sdss_dev['POGSONMAG_R'],des_dev['MAG_MODEL_R_SDSS']+ 2.5 ,'r.', label = 'dev', alpha = 0.33)
    ax2.plot(sdss_ne['POGSONMAG_R'],des_ne['MAG_MODEL_R_SDSS']+ 2.5,'g.', label = 'neither', alpha = 0.3)
    ax2.plot([0,30],[0,30],'--',color='red')
    ax2.set_xlabel('sdss r : 25 - 2.5 * np.log10(sdss[EXPFLUX_R])')
    ax2.set_ylabel('sdss_colored_des r')
    ax2.set_xlim(15,26)
    ax2.set_ylim(15,26)
    ax2 .legend(loc='best')
    
    ax3.plot(sdss_exp['POGSONMAG_I'],des_exp['MAG_MODEL_I_SDSS']+ 2.5,'b.', label = 'exp', alpha = 0.9)
    ax3.plot(sdss_dev['POGSONMAG_I'],des_dev['MAG_MODEL_I_SDSS']+ 2.5,'r.', label = 'dev', alpha = 0.33)
    ax3.plot(sdss_ne['POGSONMAG_I'],des_ne['MAG_MODEL_I_SDSS']+ 2.5,'g.', label = 'neither', alpha = 0.3)
    ax3.plot([0,30],[0,30],'--',color='red')
    ax3.set_xlabel('sdss i : 25 - 2.5 * np.log10(sdss[EXPFLUX_I])')
    ax3.set_ylabel('sdss_colored_des i')
    ax3.set_xlim(15,26)
    ax3.set_ylim(15,26)
    ax3.legend(loc='best')
    
    ax4.plot(sdss_exp['POGSONMAG_Z'],des_exp['MAG_MODEL_Z_SDSS']+ 2.5,'b.', label = 'exp', alpha = 0.9)
    ax4.plot(sdss_dev['POGSONMAG_Z'],des_dev['MAG_MODEL_Z_SDSS']+ 2.5,'r.', label = 'dev', alpha = 0.33)
    ax4.plot(sdss_ne['POGSONMAG_Z'],des_ne['MAG_MODEL_Z_SDSS']+ 2.5,'g.', label = 'neither', alpha = 0.3)
    ax4.plot([0,30],[0,30],'--',color='red')
    ax4.set_xlabel('sdss z : 25 - 2.5 * np.log10(sdss[EXPFLUX_Z])')
    ax4.set_ylabel('sdss_colored_des z')
    ax4.legend(loc='best')
    ax4.set_xlim(15,26)
    ax4.set_ylim(15,26)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeMatchedPlotsWithim3shapeMask_residu_size(sdss, des, im3shape, figname = 'test', figtitle = 'test_title'):
    
    #sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    fig,((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    
    
    
    # cross match
    im3ID = im3shape['COADD_OBJECTS_ID']
    fullID = des['COADD_OBJECTS_ID']
    mask = np.in1d(im3ID, fullID)
    crossmatch_im3shape = im3shape[mask]
    
    des = rf.append_fields(des, 'IM3_RADIUS', crossmatch_im3shape['RADIUS'])
    
    # model classification
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    

    print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'neither :', len(sdss) -len(sdss_exp)-len(sdss_dev)
    
    Scolorkind = 'MODELMAG'
    Dcolorkind = 'MAG_MODEL'
    colorCorrection = '_SDSS'
    Sizekind =  'FLUX_RADIUS'
    
    filter = '_G'
    Size = Sizekind+filter
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
   
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = sdss_dev[sdssColor] - ys_dev
    
    ax1.plot(des_exp[Size], residue_exp, 'b.', label = 'exp')
    ax1.plot(des_dev[Size], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax1.plot([0,30],[0,0],'--',color='green')
    ax1.set_xlabel( Size )
    ax1.set_ylabel('sdss - sdss_fit_'+filter)
    ax1.legend(loc='best')
    ax1.set_xlim(0,10)
    ax1.set_ylim(-5,5)
    
    
    filter = '_R'
    Size = Sizekind+filter
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = sdss_dev[sdssColor] - ys_dev
    
    ax2.plot(des_exp[Size], residue_exp, 'b.', label = 'exp')
    ax2.plot(des_dev[Size], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax2.plot([0,30],[0,0],'--',color='green')
    ax2.set_xlabel( Size )
    ax2.set_ylabel('sdss - sdss_fit_'+filter)
    ax2.legend(loc='best')
    ax2.set_xlim(0,10)
    ax2.set_ylim(-5,5)
    
    filter = '_I'
    Size = Sizekind+filter
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = sdss_dev[sdssColor] - ys_dev
    
    ax3.plot(des_exp[Size], residue_exp, 'b.', label = 'exp')
    ax3.plot(des_dev[Size], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax3.plot([0,30],[0,0],'--',color='green')
    ax3.set_xlabel( Size )
    ax3.set_ylabel('sdss - sdss_fit_'+filter)
    ax3.legend(loc='best')
    ax3.set_xlim(0,10)
    ax3.set_ylim(-5,5)
    
    
    filter = '_Z'
    Size = Sizekind+filter
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 20) & (des_exp[desColor] < 20)
    keepdev = (sdss_dev[sdssColor] < 20) & (des_dev[desColor] < 20)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = sdss_dev[sdssColor] - ys_dev
    
    ax4.plot(des_exp[Size], residue_exp, 'b.', label = 'exp')
    ax4.plot(des_dev[Size], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax4.plot([0,30],[0,0],'--',color='green')
    ax4.set_xlabel( Size )
    ax4.set_ylabel('sdss - sdss_fit_'+filter)
    ax4.legend(loc='best')
    ax4.set_xlim(0,10)
    ax4.set_ylim(-5,5)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makeDS9list( tilename, ra, dec, radius, modelname = 'exp' ):
    
    f = file('../figure/outlier_'+modelname+'.reg','w')
    f.write('# Region file format: DS9 version 4.0 \n')
    f.write('# ra dec radius \n')
    for i in range(len(ra)):
        f.write('fk5; circle ' + str(ra[i]) + ' ' + str(dec[i]) + ' ' + str(radius[i]/1000.)+'\n')
        f.write('fk5; circle ' + str(ra[i]) + ' ' + str(dec[i]) + ' ' + str(radius[i]/10000.)+'\n')
        print tilename[i], ra[i], dec[i]
    f.close()


def makeMatchedPlotsWithim3shapeMask_residu(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    #sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other = ModelClassifier(sdss, des, filter = 'I')
    fig,((ax1,ax2),(ax3, ax4)) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    
    expcut = (des['IM3_GALPROF'] == 1)
    devcut = (des['IM3_GALPROF'] == 2)
    neither = (des['IM3_GALPROF'] == 0)
    
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    
    print 'exp :', len(sdss_exp), 'dev :', len(sdss_dev), 'neither :', len(sdss) -len(sdss_exp)-len(sdss_dev)
    
    Scolorkind = 'MODELMAG'
    Dcolorkind = 'MAG_MODEL'
    colorCorrection = ''#_SDSS'
    
    filter = '_G'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    #coef_exp = [1.0, 0.0]
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = des_dev[desColor] - ys_dev
    print coef_exp, coef_dev
    
    
    ax1.plot(des_exp[desColor], residue_exp, 'b.', label = 'exp')
    ax1.plot(des_dev[desColor], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax1.plot([0,30],[0,0],'--',color='green')
    ax1.set_xlabel('sdss_colored_des'+filter)
    ax1.set_ylabel('des - fit')
    ax1.legend(loc='best')
    ax1.set_xlim(15,24)
    ax1.set_ylim(-5,5)
    
    
    # outlier selection ---------------
    """
    expoutlier_mask = ( residue_exp > -5.0 ) &  (residue_exp < -2.0 ) & (des_exp[desColor] >17.0) & (des_exp[desColor] < 22.0)
    devoutlier_mask = ( residue_dev > 0.2 ) &  (residue_dev < 0.7 ) & (des_dev[desColor] >18.0) & (des_dev[desColor] < 21.5)
    
    SDSS_exp_out = sdss_exp[expoutlier_mask]
    DES_exp_out = des_exp[expoutlier_mask]
    SDSS_dev_out = sdss_dev[devoutlier_mask]
    DES_dev_out = des_dev[devoutlier_mask]
    
    
    #make outlier data list
    
    makeDS9list(DES_exp_out['TILENAME'], DES_exp_out['RA'],DES_exp_out['DEC'],DES_exp_out['KRON_RADIUS'], modelname = 'exp')
    makeDS9list(DES_dev_out['TILENAME'], DES_dev_out['RA'],DES_dev_out['DEC'],DES_dev_out['KRON_RADIUS'], modelname = 'dev')

    stop

    """
    """
    fig2, ax5 = plt.subplots(nrows=1,ncols=1,figsize=(7,7))
    ax5.plot(des_exp[desColor], residue_exp, 'b.', label = 'exp')
    ax5.plot(des_dev[desColor], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax5.plot(DES_exp_out[desColor], residue_exp[expoutlier_mask], 'g.', label = 'exp_outlier' )
    ax5.plot(DES_dev_out[desColor], residue_dev[devoutlier_mask], 'y.', label = 'dev_outlier' )
    ax5.plot([0,30],[0,0],'--',color='green')
    ax5.set_xlabel('sdss_colored_des'+filter)
    ax5.set_ylabel('sdss - sdss_fit')
    ax5.legend(loc='best')
    ax5.set_xlim(15,24)
    ax5.set_ylim(-5,5)
    fig2.suptitle('Outlier selection', fontsize=20)
    fig2.savefig('../figure/outlier_selection')
    print 'fig saved :', '../figure/outlier_selection'+'.png'
    """
    #---------------------------------
    
    
    filter = '_R'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = des_dev[desColor] - ys_dev
    
    ax2.plot(des_exp[desColor], residue_exp, 'b.', label = 'exp')
    ax2.plot(des_dev[desColor], residue_dev, 'r.', label = 'dev',alpha = 0.33)
    ax2.plot([0,30],[0,0],'--',color='green')
    ax2.set_xlabel('sdss_colored_des'+filter)
    ax2.set_ylabel('des - fit')
    ax2.legend(loc='best')
    ax2.set_xlim(15,24)
    ax2.set_ylim(-5,5)
    
    filter = '_I'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = des_dev[desColor] - ys_dev
    
    ax3.plot(des_exp[desColor], residue_exp, 'b.', label = 'exp')
    ax3.plot(des_dev[desColor], residue_dev, 'r.', label = 'dev',alpha = 0.33)
    ax3.plot([0,30],[0,0],'--',color='green')
    ax3.set_xlabel('sdss_colored_des'+filter)
    ax3.set_ylabel('des - fit')
    ax3.legend(loc='best')
    ax3.set_xlim(15,24)
    ax3.set_ylim(-5,5)
    
    
    filter = '_Z'
    sdssColor = Scolorkind+filter
    desColor = Dcolorkind+filter+colorCorrection
    
    #fitting curve
    keepexp = (sdss_exp[sdssColor] < 19) & (des_exp[desColor] < 19)
    keepdev = (sdss_dev[sdssColor] < 19) & (des_dev[desColor] < 19)
    
    coef_exp = np.polyfit(sdss_exp[sdssColor][keepexp], des_exp[desColor][keepexp], 1)
    polynomial_exp = np.poly1d(coef_exp)
    ys_exp = polynomial_exp(sdss_exp[sdssColor])
    residue_exp = des_exp[desColor] - ys_exp
    
    coef_dev = np.polyfit(sdss_dev[sdssColor][keepdev], des_dev[desColor][keepdev], 1)
    polynomial_dev = np.poly1d(coef_dev)
    ys_dev = polynomial_dev(sdss_dev[sdssColor])
    residue_dev = des_dev[desColor] - ys_dev
    
    ax4.plot(des_exp[desColor], residue_exp, 'b.', label = 'exp')
    ax4.plot(des_dev[desColor], residue_dev, 'r.', label = 'dev', alpha = 0.33)
    ax4.plot([0,30],[0,0],'--',color='green')
    ax4.set_xlabel('sdss_colored_des'+filter)
    ax4.set_ylabel('des - des_fit')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(-5,5)
    
    fig.suptitle(figtitle, fontsize=20)
    fig.savefig(figname)
    print "fig saved :", figname+'.png'


def makePlotIM3_SDSS_profileAgreement(sdss, des ):
    
    #IM3 Profile mask
    im3expcut = (des['IM3_GALPROF'] == 1)
    im3devcut = (des['IM3_GALPROF'] == 2)
    im3neither = (des['IM3_GALPROF'] == 0)
    """
    des_exp = des[expcut]
    des_dev = des[devcut]
    sdss_exp = sdss[expcut]
    sdss_dev = sdss[devcut]
    des_ne = des[neither]
    sdss_ne = sdss[neither]
    """
    #SDSS profile mask (compare likelihood)
    #def ModelClassifier2(data, filter = 'G' ):
    #parameter = 'BESTPROF_'+filter
    #sdssexpcut = ( data['BESTPROF_G'] == 0 )
    #sdssdevcut = ( data['BESTPROF_G'] == 1 )
    #sdssneither = - (expmodel|devmodel)

    #same expcut
    ee_cut = ((des['IM3_GALPROF'] == 1) & ( sdss['BESTPROF_G'] == 0 ))
    dd_cut = ((des['IM3_GALPROF'] == 2) & ( sdss['BESTPROF_G'] == 1 ))
    same_cut = (ee_cut | dd_cut)
    
    #opposite cut
    ed_cut = ((des['IM3_GALPROF'] == 1) & ( sdss['BESTPROF_G'] == 1 ))
    de_cut = ((des['IM3_GALPROF'] == 2) & ( sdss['BESTPROF_G'] == 0 ))
    oppo_cut = (ed_cut | de_cut)

    # masked DES data
    ee_des = des[ee_cut]
    dd_des = des[dd_cut]
    same_des = des[same_cut]
    
    ed_des = des[ed_cut]
    de_des = des[de_cut]
    oppo_des = des[oppo_cut]
    
    
    # plotting histogram
    fig,((ax1,ax2),(ax3,ax4),(ax5, ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(14,21))
    #fig,(ax1,ax2) = plt.subplots(nrows=1,ncols=2,figsize=(14,7))
    
    x = same_des['MAG_MODEL_G']
    ee = ee_des['MAG_MODEL_G']
    dd = dd_des['MAG_MODEL_G']
    
    y = oppo_des['MAG_MODEL_G']
    ed = ed_des['MAG_MODEL_G']
    de = de_des['MAG_MODEL_G']
    
    ax1.hist(x, 30, facecolor='green', label = 'same', histtype='step')
    ax1.hist(ee, 30, facecolor='blue', label = 'ee',histtype='step')
    ax1.hist(dd, 30, facecolor='red', label = 'dd',histtype='step')
    ax1.set_xlabel(' des_mag_G ')
    ax1.set_ylabel(' N ')
    ax1.legend(loc=2)
    ax1.axis([17, 24, 0, 300])
    
    ax2.hist(y, 30, facecolor='yellow', alpha=0.5, label='different',histtype='step')
    ax2.hist(ed, 30, facecolor='blue', label = 'ed',histtype='step')
    ax2.hist(de, 30,  facecolor='red', label = 'de',histtype='step')
    ax2.set_xlabel(' des_mag_G ')
    ax2.set_ylabel(' N ')
    ax2.legend(loc=2)
    ax2.axis([17, 24, 0, 60])
    
    
    import matplotlib.mlab as mlab
    mu = 21.5
    sigma = 1.
    
    
    n, bins, patches = ax3.hist(x, 30, facecolor='green', histtype='step', alpha = 0.0)
    nn = n / len(sdss)
    ya = mlab.normpdf(bins, mu, sigma)
    #ax3.plot(bins, ya,'r--')
    n2, bins2, _ = ax3.hist(y, 30, facecolor='blue',histtype='step', alpha = 0.0)
    nn2 = n2/len(sdss)
    ax3.plot(bins[0:-1]+0.1, nn, 'r-',label = 'same')
    ax3.plot(bins2[0:-1]+0.1, nn2, 'b-',label='different')
    #ax3.hist(ed, 30, facecolor='blue', label = 'ed',histtype='step')
    #ax3.hist(de, 30,  facecolor='red', label = 'de',histtype='step')
    ax3.set_xlabel(' des_mag_G ')
    ax3.set_ylabel(' n/N ')
    ax3.legend(loc=2)
    ax3.axis([17, 24, 0, 0.03])


    fraction = n2/n
    ax4.plot(bins[0:-1]+0.1, fraction, 'b.')
    #ax4.hist(x, 30, facecolor='green', label = 'same', histtype='step')
    #ax4.hist(y, 30, facecolor='blue', label='different',histtype='step')
    #ax5.hist(ed, 30, facecolor='blue', label = 'ed',histtype='step')
    #ax5.hist(de, 30,  facecolor='red', label = 'de',histtype='step')
    ax4.set_xlabel(' des_mag_G ')
    ax4.set_ylabel(' fraction = n_different/n_same ')
    ax4.legend(loc=2)


    #ax3.hist(x, 30, facecolor='green', label = 'same', histtype='step')
    ax5.hist(ee, 30, normed=1, facecolor='blue', label = 'ee',histtype='step')
    ax5.hist(ed, 30, normed=1, facecolor='green', label = 'ed',histtype='step')
    ax5.hist(de, 30, normed=1, facecolor='red', label = 'de',histtype='step')
    ax5.hist(dd, 30, normed=1, facecolor='yellow', label = 'dd',histtype='step')
    #ax3.hist(dd, 30, facecolor='red', label = 'dd',histtype='step')
    ax5.set_xlabel(' des_mag_G ')
    ax5.set_ylabel(' N ')
    ax5.legend(loc=2)
    ax5.axis([17, 24, 0, 1.0])
    
    ax6.hist(x, 30, facecolor='green', label = 'same', histtype='step')
    ax6.hist(y, 30, facecolor='blue', label='different',histtype='step')
    #ax5.hist(ed, 30, facecolor='blue', label = 'ed',histtype='step')
    #ax5.hist(de, 30,  facecolor='red', label = 'de',histtype='step')
    ax6.set_xlabel(' des_mag_G ')
    ax6.set_ylabel(' N ')
    ax6.legend(loc=2)


    
    
    #ax4.axis([17, 24, 0, 300])
    
    #plt.set_grid(True)
    fig.suptitle('Histogram')
    fig.savefig('../figure/agreement')
    
    print 'figsave : ../figure/agreement.png'





    #print len(des[same_cut]), len(des[oppo_cut])




def main3(sdss_data_o, full_des_data, des_data_im3):
    """ mag_auto -> mag_model in full DES, with im3shape mask """
    #calling all data
    """
    sdss_data_o = getSDSScatalogs()
    full_des_data = getDEScatalogs(file = '../data/y1a1_stripe82_000001.fits') #full
    des_data_im3 = getDEScatalogs(file = '/n/des/huff.791/Projects/CMASS/Scripts/main_combined.fits') #im3shape
    
    """
    """
    tilenamelist = list(set(full_des_data['TILENAME']))
    tilenamelist.sort()
    tilenamelist = np.array(tilenamelist)
    np.savetxt('../figure/lee_y1a1_tilename.txt', tilenamelist, fmt='%s' ,  delimiter=' ', header = 'DES Y1A1 STRIPE82 COADD / RA 350 355 / DEC -0.5 0.0')
    
    """
    #sdss
    sdss_data = SpatialCuts(sdss_data_o)
    #sdss_data = addPogsonMag(sdss_data)
    sdss_data = whichGalaxyProfile(sdss_data_o) # add galaxy profile
    
    #full des catalog
    #full_des_data.sort(order = 'COADD_OBJECTS_ID') # sort full data by order
    des_data_f = SpatialCuts(full_des_data)
    des_data_f = doBasicCuts(des_data_f)
    des_data_f = modestify(des_data_f)
    
    #im3shape catalog
    #des_data_im3.sort(order = 'COADD_OBJECTS_ID') # sort by order
    des_data_im3 = SpatialCuts(des_data_im3)
    #des_data_im3 = addMagforim3shape(des_data_im3) # add magnitude
    
    #match im3shape and full des
    des_data_f = im3shape_add_radius(des_data_im3, des_data_f)
    masked_des_f = im3shape_galprof_mask(des_data_im3, des_data_f) # add galaxy profile mode

    # add sdss color
    #cmass, des_data = do_CMASS_cuts(masked_des_f)
    
    # pick up only des_galaxies
    des_gals, des_stars = DESclassifier(masked_des_f)
    
    
    sdss_matched, des_matched = match(sdss_data, des_gals)
    
    
    #makePlotIM3_SDSS_profileAgreement(sdss_matched, des_matched)
    
    
    figname = "../figure/colorcomparison_im3maskedDesmag_sdss_gals"
    figtitle = 'im3shape masked DES_SDSS color comparison (galaxy)'
    makeMatchedPlotsWithim3shapeMask(sdss_matched, des_matched, figname = figname, figtitle = figtitle)
    #makeMatchedPlotsWithim3shapeMask_Pogson(sdss_matched, des_matched, figname = figname, figtitle = figtitle)



    figname = "../figure/colorcomparison_im3maskedDesmag_sdss_gals_GR"
    figtitle = 'im3shape masked DES_SDSS color comparison (galaxy)'
    makeMatchedPlotsWithim3shapeMask_GR(sdss_matched, des_matched, figname = figname, figtitle = figtitle)


    figname = "../figure/im3maskedResidu"
    figtitle = 'im3shape masked DES_SDSS color comparison (galaxy)'
    makeMatchedPlotsWithim3shapeMask_residu(sdss_matched, des_matched, figname = figname, figtitle = figtitle)

    figname = "../figure/im3maskedResidu_kron"
    figtitle = 'im3shape masked DES_SDSS color comparison (galaxy)'
    makeMatchedPlotsWithim3shapeMask_residu_size(sdss_matched, des_matched, des_data_im3, figname = figname, figtitle = figtitle)


    sdss_matched, des_matched = match(sdss_data, des_data)
    
    figname = "../figure/colorcomparison_im3maskedDesmag_sdss_all"
    figtitle = 'im3shape masked DES_SDSS color comparison (galaxy + star)'
    makeMatchedPlotsWithim3shapeMask(sdss_matched, des_matched, figname = figname, figtitle = figtitle)
    #makeMatchedPlotsWithim3shapeMask_Pogson(sdss_matched, des_matched, figname = figname, figtitle = figtitle)








if __name__ == '__main__':
    main()