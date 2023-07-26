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
from sys_functions import *

#debugging:
#import ipdb
#ipdb.set_trace()


# -----------------------

#variables to set:

run_name = "finalized_linear_v2"
SPT = True
SPmap = False
linear_run = True
sys_weights = False
STOP = False
custom = True
iterative = False # -- reading in weights from other runs
first_set = True # pca0-pca49 = first set; pca 50-106 = second set

# -----------------------

if SPT == True:
    dmass_spt = calling_catalog('/fs/scratch/PCON0008/warner785/bwarner/dmass_spt.fits')
    random_spt = uniform_random_on_sphere(dmass_spt, size = 10*int(np.sum(dmass_spt['CMASS_PROB']))) 
    random_spt = keepGoodRegion(random_spt)
    random_spt = appendColumn(random_spt, value=np.ones(random_spt.size), name='CMASS_PROB')
    index_mask = np.argsort(dmass_spt)
    dmass_chron = dmass_spt[index_mask] # ordered by hpix values
    
else:
    dmass_val = calling_catalog('/users/PCON0003/warner785/DMASSY3/output/test/train_cat/y3/dmass_part2.fits')
    random_val = uniform_random_on_sphere(dmass_val, size = 10*int(np.sum(dmass_val['CMASS_PROB'])))#larger size of randoms
    random_val = keepGoodRegion(random_val)
    random_val = appendColumn(random_val, value=np.ones(random_val.size), name='CMASS_PROB')
    index_mask = np.argsort(dmass_val)
    dmass_chron = dmass_val[index_mask]
    
dmass_chron['HPIX_4096'] = hp.nest2ring(4096, dmass_chron['HPIX_4096']) 

path = '/fs/scratch/PCON0008/warner785/bwarner/'
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')

if SPT == True:
    phi = random_spt['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_spt['DEC'] ) * np.pi/180.0

else:    
    phi = random_val['RA'] * np.pi / 180.0
    theta = ( 90.0 - random_val['DEC'] ) * np.pi/180.0
    
random_pix = hp.ang2pix(4096, theta, phi)

frac = np.zeros(hp.nside2npix(4096))
fracDet["PIXEL"] = hp.nest2ring(4096, fracDet['PIXEL'])
frac[fracDet['PIXEL']] = fracDet['SIGNAL']
fracHp = np.full(hp.nside2npix(4096), hp.UNSEEN)
fracHp[fracDet['PIXEL']] = fracDet['SIGNAL']

frac_obj = frac[random_pix]

u = np.random.rand(len(random_pix))
#select random points with the condition u < frac_obj
if SPT == True:
    random_spt_fracselected = random_spt[u < frac_obj]
    randoms4096 = random_pixel(random_spt_fracselected)
else:
    random_val_fracselected = random_val[u < frac_obj]
    randoms4096 = random_pixel(random_val_fracselected)

index_ran_mask = np.argsort(randoms4096)
random_chron = randoms4096[index_ran_mask]
    
#--------------------------different loaded files:----------------------#
if first_set == True:
    n_pca = 50
else:
    n_pca = 56
# PCA maps in RING by default

if SPT==True:
    input_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_SP107_SPT_v2_cformat/'
else:
    input_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_SP107_validationregion/'
final_path = '/fs/scratch/PCON0008/warner785/bwarner/pca_maps_jointmask_no_stars1623/'
#y3/band_z/
keyword_template = 'pc{0}_'
final_template = 'pca{0}_'

chi2_randoms = []
chi2_dmassi = []
chi2_trend1 = []
chi2_trend2 = []
chi2_dmassf = []
trend = []

mock_outdir = '/fs/scratch/PCON0008/warner785/bwarner/'

# OPENING COVARIANCE MATRICES: -----------------------
if first_set == True:
    cov = []
    m_pca = 50 #50
    cov_template = 'cov{0}'
    for i_pca in range(m_pca): #n_pca
        cov_input= cov_template.format(i_pca)
        cov.append(cov_input)

    cov_template = 'covariance{0}'
    for i_pca in range(m_pca): #n_pca
        cov_keyword = cov_template.format(i_pca)
        #print(cov_keyword)
        with open(mock_outdir + cov_keyword + '.txt') as mocks:
            array1 = [x.split() for x in mocks]
            array2 = np.array(array1)
            cov[i_pca] = array2.astype(float)
        mocks.close()

else:
    cov = []
    m_pca = 56 #56
    cov_template = 'cov{0}'
    for i_pca in range(m_pca): #n_pca
        cov_input= cov_template.format(i_pca)
        cov.append(cov_input)

    cov_template = 'covar107_{0}'
    for i_pca in range(m_pca): #n_pca
        cov_keyword = cov_template.format(i_pca+50)
        #print(cov_keyword)
        with open(mock_outdir + cov_keyword + '.txt') as mocks:
            array1 = [x.split() for x in mocks]
            array2 = np.array(array1)
            cov[i_pca] = array2.astype(float)
        mocks.close()
# ----------------------------------------------------

for i_pca in range(n_pca): #50, 56
    if i_pca > -1: # can check individual maps this way
        
        linear = True
        quadratic = False
        
        if custom == True:
            if first_set == True:
                input_keyword = keyword_template.format(i_pca)
            else:
                input_keyword = keyword_template.format(i_pca+50)
            print(input_keyword)
            sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword) 
            frac_weight = fracHp[sysMap['PIXEL']]
            sysMap = sysMap[frac_weight != hp.UNSEEN]
        else:
            if first_set == True:
                input_keyword = final_template.format(i_pca)
            else:
                input_keyword = final_template.format(i_pca+50)
            print(input_keyword)
            sysMap = io.SearchAndCallFits(path = final_path, keyword = input_keyword)
            sysMap = cutPCA(sysMap)
            frac_weight = None
            
        #fixing problem with chi2 -- getting rid of outer-most diagonal noise
        covariance_i = cov[i_pca]
        covariance = np.copy(covariance_i)
        covariance[0][-1]=0
        covariance[-1][0]=0

        path = '/fs/scratch/PCON0008/warner785/bwarner/'
    
        if sys_weights == True:
            dmass_chron_weights =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/dmass_validation_check107.fits') 
#        random_chron = fitsio.read('../output/test/train_cat/y3/'+input_keyword+'randoms.fits')
            h_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'h_ran_spt_full.fits')
            norm_number_density_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'norm_ran_spt_full.fits')
            fracerr_ran_norm = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/'+input_keyword+'fracerr_ran_spt_full.fits')
            area = area_pixels(sysMap, frac_weight, custom = custom)
        
        else:
            dmass_chron_weights = None
            print("sysMap signal: ",sysMap['SIGNAL'][sysMap['SIGNAL']!=hp.UNSEEN])
            h_ran,_= number_gal(sysMap, random_chron, None, sys_weights = False)
            area = area_pixels(sysMap, frac_weight, custom=custom)
            pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)
        if iterative == True:
            dmass_chron_weights =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'+'quad2_2.fits') 
        
        h, sysval_gal = number_gal(sysMap, dmass_chron, dmass_chron_weights, sys_weights = sys_weights, iterative = iterative) 
        pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)
        diag_cov = np.diagonal(covariance)
        error_cov = np.sqrt(diag_cov)
        if is_pos_def(covariance) == True:
            cov_matrix = covariance
        else:
            print("NOT POSITIVE DEFINITE") # try not zeroing out, or taking anout zeroing out //
            covariance[1][-1] = 0
            covariance[-1][1] = 0
            covariance[0][-2] = 0
            covariance[-2][0] = 0
            if is_pos_def(covariance) == True:
                cov_matrix = covariance
            else:
                print("STILL NOT POSITIVE DEFINITE")
                covariance[0][-3] = 0
                covariance[-3][0] = 0
                covariance[1][-2] = 0
                covariance[-2][1] = 0
                covariance[2][-1] = 0
                covariance[-1][2] = 0
                if is_pos_def(covariance) == True:
                    cov_matrix = covariance
                else:
                    print("STILL NOT POSITIVE DEFINITE x2")
                    cov_matrix = error_cov

    #plotting:
        xlabel = input_keyword
        fig, ax = plt.subplots()
        if SPT == True:
            ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
            ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms in spt")
        else:
            ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in val")
            ax.errorbar( pcenter, norm_number_density_ran, yerr=fracerr_ran_norm, label = "randoms in val")
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel("n_gal/n_tot 4096")
        plt.axhline(y=1, color='grey', linestyle='--')
        if sys_weights == True:
            plt.ylim(top=1.2)
            plt.ylim(bottom=0.85)
            plt.title(xlabel+' sys weights applied')
            if SPT == True:
                fig.savefig(xlabel+'sys_applied_spt.pdf')
            else:
                fig.savefig(xlabel+'sys_applied_val.pdf')
        else:
            plt.title(xlabel+' systematics check, no weights')
            if SPT == True:
                fig.savefig('/users/PCON0003/warner785/DMASSY3/june23_custom/'+xlabel+'sys_check_spt.pdf')
            else:
                fig.savefig('/users/PCON0003/warner785/DMASSY3/june23_validation/'+xlabel+'sys_check_val.pdf')
        plt.close()

        ran_chi2, ran_chi2_reduced = chi2(norm_number_density_ran, np.ones(12), fracerr_ran_norm, 0 , None, SPT = False)
        print('ran_chi2: ', ran_chi2_reduced)
        chi2_randoms.append(ran_chi2_reduced)
    
        if sys_weights == True:
            trend_chi2, trend_chi2_reduced = chi2(norm_number_density, np.ones(12), None, 0, covariance, SPT = SPT)
            chi2_f =  trend_chi2_reduced
            print('applied_sys_chi2: ', chi2_f)
            chi2_dmassf.append(chi2_f)
        
        if sys_weights == False: 
            #import pdb
            #pdb.set_trace()
            dmass_chi2, dmass_chi2_reduced = chi2(norm_number_density, np.ones(12), None, 0, covariance, SPT = SPT)
            chi2_i = dmass_chi2_reduced
            print('checking chi2 before correction: ', chi2_i)
            
            #if chi2_i < 2:  ------ for later runs
                #STOP=True
                #print("skip-- not sufficient")
            chi2_dmassi.append(chi2_i)
        
        
 # -------------- only continue if chi2_dmassi is sufficient -------------------        
        #trendline:
        # fit to trend:
            if STOP!=True:
                fig,ax = plt.subplots(1,1)
        #linear trends first -- chi2 for higher order study --- check for threshold value (afterward)
                # sigma = error_cov
                if SPT == True:
                    params = scipy.optimize.curve_fit(lin, pcenter, norm_number_density, sigma = cov_matrix)
                else:
                    params = scipy.optimize.curve_fit(lin, pcenter, norm_number_density, sigma = fracerr_norm)
                [m, b1] = params[0]
                #z = np.polyfit(pcenter, norm_number_density, 1, w =1/error_cov)
                #p = np.poly1d(z)

                print("linear fit variables: ",m, b1)

                ax.plot(pcenter,lin(pcenter,m,b1),"r--")
                plt.title(xlabel+' systematic linear trendline')
                
                if SPT == True:
                    ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
                    fig.savefig('/users/PCON0003/warner785/DMASSY3/june23_custom/' +xlabel+'linear_spt.pdf')

                else:
                    ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in val")
                    fig.savefig('/users/PCON0003/warner785/DMASSY3/june23_val/' +xlabel+'linear_val.pdf')

                trend_chi2, trend_chi2_reduced = chi2(norm_number_density, lin(pcenter, m, b1), fracerr_norm, 2, covariance, SPT = SPT)

                print('linear trend_chi2: ', trend_chi2_reduced)
                chi2_trend1.append(trend_chi2_reduced)
                plt.close()
    
# difference between sum(chi2) between models (free parameters-- 1 new, want more than 1 better in sum(chi2))

        # second trendline:
        # fit to trend:
                if linear_run!=True: 
                
                    fig,ax = plt.subplots(1,1)
        #linear trends first -- chi2 for higher order study --- check for threshold value (afterward)
                # sigma = error_cov
                    if SPT == True:
                        params = scipy.optimize.curve_fit(quad, pcenter, norm_number_density, sigma = cov_matrix)
                    else:
                        params = scipy.optimize.curve_fit(quad, pcenter, norm_number_density, sigma = fracerr_norm)
                    [a, b2, c] = params[0]


                    print("quadratic fit variables: ",a, b2, c)

                    ax.plot(pcenter,quad(pcenter,a,b2,c),"r--")
                    plt.title(xlabel+' systematic quadatric trendline')
                
                    if SPT == True:
                        ax.errorbar( pcenter, norm_number_density, yerr=error_cov, label = "dmass in spt")
                        fig.savefig('/users/PCON0003/warner785/DMASSY3/june23_custom/' +xlabel+'quad_spt.pdf')

                    else:
                        ax.errorbar( pcenter, norm_number_density, yerr=fracerr_norm, label = "dmass in val")
                        fig.savefig('/users/PCON0003/warner785/DMASSY3/june23_val/' +xlabel+'quad_val.pdf')

                    trend2_chi2, trend2_chi2_reduced = chi2(norm_number_density, quad(pcenter, a, b2, c), fracerr_norm, 3, covariance, SPT = SPT)

                    print('quadratic trend_chi2: ', trend2_chi2_reduced)
                    chi2_trend2.append(trend2_chi2_reduced)
                    plt.close()
       
                    diff_chi2 = trend_chi2-trend2_chi2

                    print("difference of chi2 between models: ", diff_chi2)
                    
                    chi2_ = np.linspace(0,30,100)
            # check p-value threshold ******
                    y = np.abs((100*(1.-scipy.stats.chi2(12).cdf(chi2_))-5.))  #for 5% p-value threshold****
                    index = np.where(y == y.min())[0][0]
                    threshold = chi2_[index]
                    if trend_chi2 > threshold: #diff_chi2
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
                    print("linear weights applied...")
            
            #check chi2 first
            # plot chi2 of linears versus distribution as check for p-value
            # input parameters not hard-coded (change later)
                    #if trend_chi2>threshold:
                        #print(xlabel, " NEEDS TO BE FLAGGED")
            
            #make sure object density stays the same
                    weight_object = (1/lin(sysval_gal, m,b1))
                    weight_object[sysval_gal == hp.UNSEEN] = 0 # check into this-- nothing should be zero
                    avg = np.average(weight_object[weight_object!=0])
#            print(avg)  # should be aprox. 1
            # normalize density
                    weight_object = weight_object/avg
        
        # quadratic:
                if quadratic==True:
                    print("quadratic weights applied...")
                    #if trend2_chi2>threshold:
                        #print(xlabel, " NEEDS TO BE FLAGGED")
                    
            #make sure object density stays the same
                    weight_object = (1/quad(sysval_gal, a,b2,c))
                    weight_object[sysval_gal == hp.UNSEEN] = 0
                    avg = np.average(weight_object[weight_object!=0])
#            print(avg)  # should be aprox. 1
            #normalize density
                    weight_object = weight_object/avg
        
                dmass_chron_sys_weight = weight_object
                print("weights being applied: ",dmass_chron_sys_weight, "weights that are zero: ", dmass_chron_sys_weight[dmass_chron_sys_weight==0].size)
    
                outdir = '/fs/scratch/PCON0008/warner785/bwarner/june23_tests/'
                os.makedirs(outdir, exist_ok=True)
                print("saving files...")
     # ONLY SAVE WEIGHTS COLUMN FOR FASTER RUN #
                esutil.io.write( outdir+run_name+xlabel+'dmass_weight_spt.fits', dmass_chron_sys_weight, overwrite=True)
                #esutil.io.write( outdir+xlabel+'h_ran_spt_full.fits', h_ran, overwrite=True)
                #esutil.io.write( outdir+xlabel+'norm_ran_spt_full.fits', norm_number_density_ran, overwrite=True)
                #esutil.io.write( outdir+xlabel+'fracerr_ran_spt_full.fits', fracerr_ran_norm, overwrite=True)
        
        
# save everything in text files

if sys_weights == False:
    np.savetxt(run_name+'chi2_randoms_spt.txt', chi2_randoms)
    np.savetxt(run_name+'chi2_dmassi_spt.txt', chi2_dmassi)
##    np.savetxt('chi2_trend1_spt.txt', chi2_trend1)
##    np.savetxt('chi2_trend2_spt.txt', chi2_trend2)
##    np.savetxt('trend_spt.txt', trend)

#if sys_weights == True:
    #np.savetxt('chi2_dmassf_spt.txt', chi2_dmassf)
    
# variable names:
# linear_run1
# linear_run2
# quad_run1
# quad_run2
# quad_updated1
# quad_updated2
