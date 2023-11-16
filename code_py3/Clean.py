import sys
sys.path.append('../')
from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
from systematics_module import *
import os
from numpy.lib.recfunctions import append_fields
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
from run_systematics import sys_iteration, weightmultiply, fitting_allSP, calling_sysMap


dmass_spt = calling_lens_catalog('/fs/scratch/PCON0008/warner785/bwarner/dmass_spt.fits')
#dmass_val = esutil.io.read('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')
# check prob cut
random_spt = uniform_random_on_sphere(dmass_spt, size = 10*int(np.sum(dmass_spt['WEIGHT']))) 
#random_val = uniform_random_on_sphere(dmass_val, size = 10*int(np.sum(dmass_val['WEIGHT'])))#larger size of randoms
# applying LSS mask 
random_spt = keepGoodRegion(random_spt)
#random_val = keepGoodRegion(random_val)

random_spt = appendColumn(random_spt, value=np.ones(random_spt.size), name='WEIGHT')
#random_val = appendColumn(random_val, value=np.ones(random_val.size), name='WEIGHT')

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')

phi = random_spt['RA'] * np.pi / 180.0
theta = ( 90.0 - random_spt['DEC'] ) * np.pi/180.0
#phi = random_val['RA'] * np.pi / 180.0
#theta = ( 90.0 - random_val['DEC'] ) * np.pi/180.0
random_pix = hp.ang2pix(4096, theta, phi)

frac = np.zeros(hp.nside2npix(4096))
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
frac[fracDet['PIXEL']] = fracDet['SIGNAL']
fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']

frac_obj = frac[random_pix]

u = np.random.rand(len(random_pix))
#select random points with the condition u < frac_obj
random_spt_fracselected = random_spt[u < frac_obj]
#random_val_fracselected = random_val[u < frac_obj]

def cutPCA(sysMap):
    
    RA, DEC = hp.pix2ang(4096, sysMap['PIXEL'], lonlat=True)
    sysMap = append_fields(sysMap, 'RA', RA, usemask=False)
    sysMap = append_fields(sysMap, 'DEC', DEC, usemask=False)
    #print(sysMap.dtype.names)

    sysMap = keepGoodRegion(sysMap)
    
    # for SPT region
    mask_spt = (sysMap['RA']>295)&(sysMap['RA']<360)|(sysMap['RA']<105)
    mask_spt = mask_spt & (sysMap['DEC']>-68) & (sysMap['DEC']<-10)
    sysMap = sysMap[mask_spt]

    # for validation region
#    mask4 =(sysMap['RA']>18)&(sysMap['RA']<43)
#    mask4 = mask4 & (sysMap['DEC']>-10) & (sysMap['DEC']<10)
#    sysMap = sysMap[mask4]
    
    # for training region
#    mask = (sysMap['RA']>310) & (sysMap['RA']<360)|(sysMap['RA']<7)
#    mask = mask & (sysMap['DEC']>-10) & (sysMap['DEC']<10)
#    sysMap = sysMap[mask]
    
    return sysMap

def number_gal(sysMap, dmass_chron, sys_weights = False): # apply systematic weights here
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)
    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2

    x = np.full(hp.nside2npix(4096), hp.UNSEEN)
    x[sysMap['PIXEL']] = sysMap['SIGNAL']

    #systematic value at galaxy location:
    sysval_gal = x[dmass_chron['HPIX_4096']].copy()
    
    if sys_weights == True:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["WEIGHT"][sysval_gal != hp.UNSEEN]*dmass_chron["SYS_WEIGHT"][sysval_gal != hp.UNSEEN])
    else:
        h,_ = np.histogram(sysval_gal[sysval_gal != hp.UNSEEN], bins=pbin, weights = dmass_chron["WEIGHT"][sysval_gal != hp.UNSEEN]) # -- density of dmass sample, not gold sample
    
    return h, sysval_gal

def area_pixels(sysMap, frac_weight):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2
    
# number of galaxies in each pixel:
    if custom == True:
        sys_signal = sysMap['SIGNAL']
        area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = frac_weight)
    else:   
        sys_signal = sysMap['SIGNAL']
        area,_ = np.histogram(sys_signal[sys_signal != hp.UNSEEN] , bins=pbin , weights = sysMap['FRACDET'][sys_signal != hp.UNSEEN])

    return area

def number_density(sysMap, h, area):
    
    minimum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 1)
    maximum = np.percentile(sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN], 99)

    pbin, pstep = np.linspace( minimum, maximum, 13, retstep=True)
    pcenter = pbin[:-1] + pstep/2

    number_density = []
    for x in range(len(h)):
        den = h[x]/area[x]
        number_density.append(den)

    total_area = 0
    #Normalize based on total number density of used footprint:
    for x in range(len(area)):
        total_area += area[x]

    # total galaxies:
    total_h = 0
    for x in range(len(h)):
        total_h += h[x]

    #normalization: 
    total_num_density = total_h/total_area
    
    # apply normalization: 
    norm_number_density = number_density/total_num_density

    fracerr = np.sqrt(h) #1 / sqrt(number of cmass galaxies in each bin)
    fracerr_norm = (fracerr/area)/total_num_density
    
    return pcenter, norm_number_density, fracerr_norm

def func(pcenter,m,c):
    return m*pcenter+c

def chi2(norm_number_density, x2_value, fracerr_norm, n, covariance):
    x1 = norm_number_density
    x2 = x2_value
    cov = covariance
    if type(cov) != int:
        inv_cov = np.linalg.inv(cov)
    else:
        inv_cov = cov
    X = x1-x2
    Matrix = np.matrix(X)
    X_T = np.transpose(Matrix)
    chi2 = (Matrix)*inv_cov*X_T
    #import pdb
    #pdb.set_trace()
    sum_chi2=float(chi2)
    chi2_reduced = sum_chi2/(len(norm_number_density)-n)
    # n = 2 for linear fit, 3 for quad.
    
    return chi2, chi2_reduced

#--------------------------different loaded files:------------------------------------------#

n_pca = 57
input_path = '/users/PCON0003/warner785/DMASSy3/systematics/pca_SP107_SPT_v2_cformat/'
#input_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_SP107_validationregion/'
final_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_maps_jointmask_no_stars1623/'
#y3/band_z/
keyword_template = 'pc{0}_'
final_template = 'pca{0}_'

#chi2_randoms = []
#chi2_dmassi = []
#chi2_trend1 = []
#chi2_trend2 = []
#chi2_dmassf = []
#trend = []

mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/'

cov = []
m_pca = 57
cov_template = 'cov{0}'
for i_pca in range(m_pca): #n_pca
    cov_input= cov_template.format(i_pca)
    cov.append(cov_input)

#cov_template = 'covariance{0}'
cov_template = 'covar107_{0}'
for i_pca in range(m_pca): #n_pca
    cov_keyword = cov_template.format(i_pca+50)
    print(cov_keyword)
    with open(mock_outdir + cov_keyword + '.txt') as mocks:
        array1 = [x.split() for x in mocks]
        array2 = np.array(array1)
        cov[i_pca] = array2.astype(float)
    mocks.close() 

custom = False
for i_pca in range(n_pca): #50
    if i_pca > -1:
        if custom == True:
            input_keyword = keyword_template.format(i_pca+50)
            print(input_keyword)
            sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword) 
            # change to sysMapi for healpy format
            frac_weight = fracHp[sysMap['PIXEL']]
            sysMap = sysMap[frac_weight != hp.UNSEEN]
        else:
            input_keyword = final_template.format(i_pca)
            print(input_keyword)
            sysMap = io.SearchAndCallFits(path = final_path, keyword = input_keyword)
            sysMap = cutPCA(sysMap)
            frac_weight = None
            #import pdb
            #pdb.set_trace()
        
        covariance = cov[i_pca]

        path = '/fs/scratch/PCON0008/warner785/bwarner/'
        linear_run = True
        sys_weights = False
    
        linear = True
        quadratic = False
        STOP = False
    
        if sys_weights == True:
            dmass_chron =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'dmass_sys_weight_spt_full.fits')
            h_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'h_ran_spt_full.fits')
            norm_number_density_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'norm_ran_spt_full.fits')
            fracerr_ran_norm = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'fracerr_ran_spt_full.fits')
            area = area_pixels(sysMap, frac_weight)
        
        else:
            index_mask = np.argsort(dmass_spt)
            #index_mask = np.argsort(dmass_val)
            dmass_chron = dmass_spt[index_mask] # ordered by hpix values
            #dmass_chron = dmass_val[index_mask]
            dmass_chron['HPIX_4096'] = hp.nest2ring(4096, dmass_chron['HPIX_4096']) 
            randoms4096 = random_pixel(random_spt_fracselected)
            #randoms4096 = random_pixel(random_val_fracselected)
            index_ran_mask = np.argsort(randoms4096)
            random_chron = randoms4096[index_ran_mask]
            print("sysMap signal: ",sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN])
            h_ran,_= number_gal(sysMap, random_chron, sys_weights = False)
            area = area_pixels(sysMap, frac_weight)
            pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)
        
        h, sysval_gal = number_gal(sysMap, dmass_chron, sys_weights = False) # change this to true if sys weights run
        pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)
        diag_cov = np.diagonal(covariance)
        error_cov = np.sqrt(diag_cov)
    

    #plotting:

        fig, ax = plt.subplots()
        #error_cov, fracerr_norm
        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
        ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms in spt")
        plt.legend()
        xlabel = input_keyword
        plt.xlabel(xlabel)
        plt.ylabel("n_gal/n_tot 4096")
        plt.axhline(y=1, color='grey', linestyle='--')
        if sys_weights == True:
            plt.ylim(top=1.2)
            plt.ylim(bottom=0.85)
            plt.title(xlabel+' sys weights applied')
            fig.savefig(xlabel+'sys_applied_spt.pdf')
        else:
            plt.title(xlabel+' systematics check')
            fig.savefig('/users/PCON0003/warner785/DMASSY3/custom2/'+xlabel+'sys_check_spt.pdf') 
        plt.close()

        ran_chi2, ran_chi2_reduced = chi2(norm_number_density_ran, np.ones(12), fracerr_ran_norm, 0 , 1)
        print('ran_chi2: ', ran_chi2_reduced)
        #chi2_randoms.append(ran_chi2_reduced)
    
        if sys_weights == True:
            trend_chi2, trend_chi2_reduced = chi2(norm_number_density, np.ones(12), None, 0, covariance)
            chi2_f =  trend_chi2_reduced
            print('applied_sys_chi2: ', chi2_f)
            chi2_dmassf.append(chi2_f)
    
        if sys_weights == False:    
            dmass_chi2, dmass_chi2_reduced = chi2(norm_number_density, np.ones(12), None, 0, covariance)
            chi2_i = dmass_chi2_reduced
            print('checking chi2 before correction: ', chi2_i)
            
            #if chi2_i < 2:
                #STOP=True
                #print("skip-- not sufficient")
            #chi2_dmassi.append(chi2_i)
        
        
 # -------------- only continue if chi2_dmassi is sufficient -------------------        
        #trendline:
        # fit to trend:
            if STOP!=True:
                fig,ax = plt.subplots(1,1)
        #linear trends first -- chi2 for higher order study --- check for threshold value (afterward)
                # sigma = error_cov, fracerr_norm
                params = scipy.optimize.curve_fit(func, pcenter, norm_number_density, sigma = error_cov)
                [m, b] = params[0]
                #z = np.polyfit(pcenter, norm_number_density, 1, w =1/error_cov)
                #p = np.poly1d(z)

                print("linear fit variables: ",m, b)

                ax.plot(pcenter,func(pcenter,m,b),"r--")
                #error_cov
                ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in spt")
                plt.title(xlabel+' systematic linear trendline')
         
                fig.savefig('/users/PCON0003/warner785/DMASSY3/custom2/' +xlabel+'linear_spt.pdf')

#                trend_chi2, trend_chi2_reduced = chi2(norm_number_density, func(pcenter, m, b), None, 2, covariance)

#                print('linear trend_chi2: ', trend_chi2_reduced)
#                chi2_trend1.append(trend_chi2_reduced)
                plt.close()
    
# difference between sum(chi2) between models (free parameters-- 1 new, want more than 1 better in sum(chi2))

        # second trendline:
        # fit to trend:
                if linear_run!=True:
                    fig,ax = plt.subplots(1,1)
        #linear trends first -- chi2 for higher order study --- check for threshold value (afterward)
                    z2 = np.polyfit(pcenter, norm_number_density, 2, w = 1/(fracerr_norm))
                    p2 = np.poly1d(z2)

                    print(p2)

                    ax.plot(pcenter,p2(pcenter),"r--")
                    ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in spt")
                    plt.title(xlabel+' systematic quadratic trendline')
                    fig.savefig(xlabel+'quadratic_spt.pdf')

                    trend2_chi2, trend2_chi2_reduced = chi2(norm_number_density, p2(pcenter), fracerr_norm, 3)
                    diff_chi2 = sum(trend_chi2)-sum(trend2_chi2)
                    print('quadratic trend_chi2: ', trend2_chi2_reduced)
                    chi2_trend2.append(trend2_chi2_reduced)

                    print("difference of chi2 between models: ", diff_chi2)
                    if diff_chi2 > 10000:
                        quadratic=True
                        print("Quadratic is better fit for ", xlabel)
                        trend.append(1)
                    else:
                        linear=True
                        print("Linear fit is suitable for ", xlabel)
                        trend.append(0)
        
        # work on applying the weights to dmass:

        #linear:
        #weight_pixel = (1/p(sysMap["PIXEL"]))
                if linear==True:
            
            #check chi2 first
            # plot chi2 of linears versus distribution as check for p-value
            # input parameters not hard-coded (change later)
#                    chi2_ = np.linspace(0,30,100)
#                    y = np.abs((100*(1.-scipy.stats.chi2(12).cdf(chi2_))-5.))  #for 5% p-value threshold
#                    index = np.where(y == y.min())[0][0]
#                    threshold = chi2_[index]
#                    if sum(trend_chi2)>threshold:
#                        print(xlabel, " NEEDS TO BE FLAGGED")
            
            #make sure object density stays the same
                    weight_object = (1/func(sysval_gal, m,b))
                    weight_object[sysval_gal == hp.UNSEEN] = 0 # check into this-- nothing should be zero
                    avg = np.average(weight_object[weight_object!=0])
#            print(avg)  # should be aprox. 1
            # normalize density
                    weight_object = weight_object/avg
        
        # quadratic:
        #weight_pixel = (1/p2(sysMap["PIXEL"]))
                if quadratic==True:
            
            #check chi2 first
                    chi2_ = np.linspace(0,30,100)
            # check p-value threshold ******
                    y = np.abs((100*(1.-scipy.stats.chi2(12).cdf(chi2_))-5.))  #for 5% p-value threshold****
                    index = np.where(y == y.min())[0][0]
                    threshold = chi2_[index]
                    if sum(trend2_chi2)>threshold:
                        print(xlabel, " NEEDS TO BE FLAGGED")
                    
            #make sure object density stays the same
                    weight_object = (1/p2(sysval_gal))
                    weight_object[sysval_gal == hp.UNSEEN] = 0
                    avg = np.average(weight_object[weight_object!=0])
#            print(avg)  # should be aprox. 1
            #normalize density
                    weight_object = weight_object/avg
        
                dmass_chron_sys_weight = weight_object
                print("weights being applied: ",dmass_chron_sys_weight, "weights that are zero: ", dmass_chron_sys_weight[dmass_chron_sys_weight==0].size)
    
                outdir = '/fs/scratch/PCON0008/warner785/bwarner/'
                os.makedirs(outdir, exist_ok=True)
                print("saving files...")
     # ONLY SAVE WEIGHTS COLUMN FOR FASTER RUN #
                esutil.io.write( outdir+xlabel+'dmass_sys_weight_spt_custom2.fits', dmass_chron_sys_weight, overwrite=True)
        
        
# save everything in text files

#if sys_weights == False:
#    np.savetxt('chi2_randoms_spt.txt', chi2_randoms)
#    np.savetxt('chi2_dmassi_spt.txt', chi2_dmassi)
##    np.savetxt('chi2_trend1_spt.txt', chi2_trend1)
##    np.savetxt('chi2_trend2_spt.txt', chi2_trend2)
##    np.savetxt('trend_spt.txt', trend)

#if sys_weights == True:
    #np.savetxt('chi2_dmassf_spt.txt', chi2_dmassf)
