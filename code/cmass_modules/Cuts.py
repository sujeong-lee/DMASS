#import easyaccess as ea
import esutil
import sys
import os
#import healpy as hp
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
#import seaborn as sns

from ang2stripe import *
import fitsio
from fitsio import FITS, FITSHDR

def CmassGal_in_stripe82(  data):
    
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

def doBasicCuts( des):
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
        
    taglist = ['MAG_APER_3', 'MAG_APER_4', 'MAG_APER_5', 'MAG_APER_6']
    filters = ['G','R','I','Z']
           
    for tag in taglist:
       for thisfilter in filters:
           thistag = tag+'_'+thisfilter
           use = use & ((des[thistag] < 25) & (des[thistag] > 15))

    print 'do Basic Cut', np.sum(use)
    return des[use]
        
def SpatialCuts(  data, ra = 350.0, ra2=355.0 , dec= 0.0 , dec2=1.0 ):
    
    cut = ((data['RA'] < ra2) &
           (data['DEC'] < dec2) &
           (data['RA'] > ra) &
           (data['DEC'] > dec))
    print 'Spatial Cut ', np.sum(cut)
    return data[cut]


def RedshiftCuts(  data, z_min=0.43, z_max=0.7 ):
    
    cut = ((data['Z'] < z_max ) &
           (data['Z'] > z_min))
    print 'Redshift Cut ', np.sum(cut)
    return data[cut]


def ColorCuts(  sdssdata ):
    
    sdsscut_G = ((sdssdata['MODELMAG_G'] < 18.0) & (sdssdata['MODELMAG_G'] > 15.0))
    sdsscut_R = ((sdssdata['MODELMAG_R'] < 18.0) & (sdssdata['MODELMAG_R'] > 15.0))
    sdsscut_I = ((sdssdata['MODELMAG_I'] < 18.0) & (sdssdata['MODELMAG_I'] > 15.0))
    sdsscut_Z = ((sdssdata['MODELMAG_Z'] < 18.0) & (sdssdata['MODELMAG_Z'] > 15.0))
    
    return sdssdata[sdsscut_G], sdssdata[sdsscut_R], sdssdata[sdsscut_I], sdssdata[sdsscut_Z]


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


def ModelClassifier(  sdss, des, filter = 'G' ):
    parameter = 'BESTPROF_'+filter
    expmodel = ( sdss[parameter] == 0 )
    devmodel = ( sdss[parameter] == 1 )
    neither = - (expmodel|devmodel)
    
    return sdss[expmodel], des[expmodel], sdss[devmodel], des[devmodel], sdss[neither], des[neither]

def ModelClassifier2(  data, filter = 'G' ):
    parameter = 'BESTPROF_'+filter
    expmodel = ( data[parameter] == 0 )
    devmodel = ( data[parameter] == 1 )
    neither = - (expmodel|devmodel)
    
    return data[expmodel], data[devmodel], data[neither]

def makeMatchedPlotsGalaxyProfile(sdss, des, figname = 'test', figtitle = 'test_title'):
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other =  ModelClassifier(sdss, des, filter = 'G')
    
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
    
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other =  ModelClassifier(sdss, des, filter = 'R')
    ax2.plot(sdss_other['MODELMAG_R'],des_other['MAG_MODEL_R_SDSS'],'g.', alpha=0.33)
    ax2.plot(sdss_exp['MODELMAG_R'],des_exp['MAG_MODEL_R_SDSS'],'b.')
    ax2.plot(sdss_dev['MODELMAG_R'],des_dev['MAG_MODEL_R_SDSS'],'r.', alpha = 0.33)
    ax2.plot([15, 24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other =  ModelClassifier(sdss, des, filter = 'I')
    
    ax3.plot(sdss_other['MODELMAG_I'],des_other['MAG_MODEL_I_SDSS'],'g.', alpha=0.33)
    ax3.plot(sdss_exp['MODELMAG_I'],des_exp['MAG_MODEL_I_SDSS'],'b.')
    ax3.plot(sdss_dev['MODELMAG_I'],des_dev['MAG_MODEL_I_SDSS'],'r.', alpha = 0.33)
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)
    
    sdss_exp, des_exp, sdss_dev, des_dev, sdss_other, des_other =  ModelClassifier(sdss, des, filter = 'Z')
    
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