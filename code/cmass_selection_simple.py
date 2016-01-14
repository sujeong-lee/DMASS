#!/usr/bin/env python

import desdb
import numpy as np
import esutil
import sys
import healpy as hp
import os
import numpy.lib.recfunctions as rf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def getDESCatalogs(goldDB = False,rows= None):
    if goldDB is False:
        cfile = '../Data/DES_Y1_S82.fits'
        '''
        Note: This fits table is everything in Y1A1_COADD_OBJECTS between (300 < ra < 60) and (dec > -10) with i_auto < 21.
        '''
        print "Getting fits catalog"
        data,thing = esutil.io.read_header(cfile,ext=1,rows=rows)
    else:
        cur = desdb.connect()
        #query = '''
        #             SELECT *
        #             FROM 
        #               Y1A1_COADD_OBJECTS y1
        #             WHERE 
        #               ((y1.ra > 300) or (y1.ra < 60)) AND 
        #               (y1.dec > -10) AND 
        #               (y1.mag_auto_i < 21.5) AND 
        #               (COADD_OBJECTS_ID IN (SELECT COADD_OBJECTS_ID FROM nsevilla.y1a1_gold_1_0_1 n WHERE n.flags_badregion=0)) AND
        #               rownum < 1000
        #        '''
        query = '''
                     SELECT *
                     FROM 
                       Y1A1_COADD_OBJECTS y1
                     WHERE 
                       ((y1.dec < 2.7) and (y1.dec > -1.9) and ((y1.ra > 316 ) or (y1.ra < 4.8))) and 
                       (COADD_OBJECTS_ID IN (SELECT COADD_OBJECTS_ID FROM nsevilla.y1a1_gold_1_0_1 n WHERE n.flags_badregion=0)) AND
                       rownum < 1000
                '''
        data = cur.quick(query,array=True)
    return data


def transform_DES_to_SDSS(g_des, r_des, i_des, z_des):
    # Transform DES colors to SDSS
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

def add_SDSS_colors(data, magTag_template = 'mag_detmodel'):
    print "Doing des->sdss color transforms for "+magTag_template
    filters = ['g','r','i','z']
    magTags = []
    desMags = np.empty([len(filters),len(data)])
    for i,thisFilter in enumerate(filters):
        magTag = magTag_template+'_'+thisFilter
        desMags[i,:] = data[magTag]

    sdssMags = transform_DES_to_SDSS(desMags[0,:], desMags[1,:], desMags[2,:], desMags[3,:])

    data = rf.append_fields(data, magTag_template+'_g_sdss', sdssMags[0,:])
    data = rf.append_fields(data, magTag_template+'_r_sdss', sdssMags[1,:])
    data = rf.append_fields(data, magTag_template+'_i_sdss', sdssMags[2,:])
    data = rf.append_fields(data, magTag_template+'_z_sdss', sdssMags[3,:])


    return data

def do_CMASS_cuts(data):
    data = add_SDSS_colors(data, magTag_template = 'mag_detmodel')
    data = add_SDSS_colors(data, magTag_template = 'mag_model')
    data = add_SDSS_colors(data, magTag_template = 'mag_psf')
    data = add_SDSS_colors(data, magTag_template = 'mag_aper_4')
    data = add_SDSS_colors(data, magTag_template = 'mag_aper_2')
    # Aperture magnitudes are in apertures of:
    # PHOT_APERTURES  1.85   3.70   5.55   7.41  11.11   14.81   18.52   22.22   25.93   29.63   44.44  66.67  
    # (MAG_APER aperture diameter(s) are in pixels)
    # SDSS fiber and fiber2 mags are (sort of) in apertures of 3" and 2", respectively, which correspond to
    # 7.41 pix and 11.11 pix (in DES pixels) respectively, so we'll use mag_aper_4 and mag_aper_5 for the two fiber mags.
    print "Calculating/applying CMASS object cuts."
    dperp = ( (data['mag_model_r_sdss'] - data['mag_model_i_sdss']) - 
              (data['mag_model_g_sdss'] - data['mag_model_r_sdss'])/8.0 )
    
    keep_cmass = ( (dperp > 0.55) &
                   (data['mag_detmodel_i_sdss'] < (19.86 + 1.6*(dperp - 0.8) )) & 
                   (data['mag_detmodel_i_sdss'] > 17.5 ) & (data['mag_detmodel_i_sdss'] < 19.9) & 
                   ((data['mag_model_r_sdss'] - data['mag_model_i_sdss'] ) < 2 ) & 
                   (data['mag_aper_4_i_sdss'] < 21.5 ) & 
                   ((data['mag_psf_i_sdss'] - data['mag_model_i_sdss']) > (0.2 + 0.2*(20.0 - data['mag_model_i_sdss'] ) ) ) &
                   ((data['mag_psf_z_sdss'] - data['mag_model_z_sdss']) > (9.125 - 0.46 * data['mag_model_z_sdss'])) )

    

    return data[keep_cmass], data

def make_cmass_plots(data):
    fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(9,13))
    dperp = ( (data['mag_model_r_sdss'] - data['mag_model_i_sdss']) - 
              (data['mag_model_g_sdss'] - data['mag_model_r_sdss'])/8.0 )


    ax1.plot(data['mag_model_g_sdss'] - data['mag_model_r_sdss'],
             data['mag_model_r_sdss'] - data['mag_model_i_sdss'],'.')
    ax1.set_xlabel("(g-r)_sdss (model)")
    ax1.set_ylabel("(r-i)_sdss (model)")
    ax1.set_xlim(0,3)
    ax1.set_ylim(0.5,1.6)

    ax2.plot(data['mag_detmodel_i_sdss'],dperp,'.',markersize=10)
    ax2.set_xlabel("i_sdss (model)")
    ax2.set_ylabel("dperp")
    ax2.set_xlim(17.0,20.0)
    ax2.set_ylim(0.5,1.6)

    fig.savefig("cmass_plots")


def getSDSScatalog():
    file1 = '../Data/desoverlap_1_danielgruen.fit'
    file5 = '../Data/desoverlap_5_danielgruen.fit'
    
    data1 = esutil.io.read(file1)
    data5 = esutil.io.read(file5)
    data = np.hstack((data1,data5))

    # Apply what are probably the basic quality cuts.
    data = data[data['clean'] == 1]

    return data


def match(sdss, des):
    h = esutil.htm.HTM(10)
    matchDist = 1.0/3600. # match distance (degrees) -- default to 1 arcsec
    m_des, m_sdss, d12 = h.match(des['ra'], des['dec'], sdss['ra'],sdss['dec'],matchDist,maxmatch=1)

    return sdss[m_sdss],des[m_des]

def makeMatchedPlots(sdss,des):
    fig,((ax1,ax2),(ax3,ax4) ) = plt.subplots(nrows=2,ncols=2,figsize=(14,14))
    ax1.plot(sdss['modelMag_g'],des['mag_model_g_sdss'],'.')
    ax1.plot([15,24],[15,24],'--',color='red')
    ax1.set_xlabel('sdss g (model)')
    ax1.set_ylabel('des g_sdss (model)')
    ax1.set_xlim(15,24)
    ax1.set_ylim(15,24)

    ax2.plot(sdss['modelMag_r'],des['mag_model_r_sdss'],'.')
    ax2.plot([15,24],[15,24],'--',color='red')
    ax2.set_xlabel('sdss r (model)')
    ax2.set_ylabel('des r_sdss (model)')
    ax2.set_xlim(15,24)
    ax2.set_ylim(15,24)

    ax3.plot(sdss['modelMag_i'],des['mag_model_i_sdss'],'.')
    ax3.plot([15,24],[15,24],'--',color='red')
    ax3.set_xlabel('sdss i (model)')
    ax3.set_ylabel('des i_sdss (model)')
    ax3.set_xlim(15,24)
    ax3.set_ylim(15,24)

    ax4.plot(sdss['fiber2Mag_i'],des['mag_aper_4_i_sdss'],'.',label='aper4')
    ax4.plot(sdss['fiber2Mag_i'],des['mag_aper_2_i_sdss'],'.',label='aper2',alpha=0.33)
    ax4.plot([15,24],[15,24],'--',color='red')
    ax4.set_xlabel('sdss fiber2mag_i')
    ax4.set_ylabel('des mag_aper_[*]_i_sdss')
    ax4.legend(loc='best')
    ax4.set_xlim(15,24)
    ax4.set_ylim(15,24)

    fig.savefig("sdss-des_mag_model_comparison")


def doBasicCuts(des):
    use = ((des['mag_model_g'] < 25) & 
           (des['mag_model_r'] < 25) & 
           (des['mag_model_i'] < 25) & 
           (des['mag_model_z'] < 25) &
           (des['flags_g'] < 3) &
           (des['flags_r'] < 3) & 
           (des['flags_i'] < 3) & 
           (des['flags_z'] < 3))
    return des[use]


def main(argv):
    cmass_file = "../Data/cmass_simple.fits"
    rows =  np.arange(10000)
    alldata  = getDESCatalogs(goldDB= False,rows= rows)
    alldata = doBasicCuts(alldata)
    cmass, data = do_CMASS_cuts(alldata)
    #    print str(cmass.size)+" objects survive, out of "+str(data.size)
    cmass = esutil.io.read(cmass_file)
    make_cmass_plots(cmass)
    sdss = getSDSScatalog()
    sdss_matched, des_matched = match(sdss, data)
    makeMatchedPlots(sdss_matched, des_matched)
    print "writing candidate cmass table to "+cmass_file
    #esutil.io.write(cmass_file,cmass,clobber=True)

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
    
             
