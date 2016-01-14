#import easyaccess as ea
import esutil
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy.lib.recfunctions as rf
#import seaborn as sns

import fitsio
from fitsio import FITS, FITSHDR



def getDEScatalogs(  file = '/n/des/lee.5922/data/stripe82_des_cut_000001.fits' ):
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


def getSDSScatalogs(  file = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit'):
    
    #file1 = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit'
    #file1 = '/Users/SJ/Dropbox/repositories/CMASS/data/test_emhuff.fit'
    #file3 = '../data/sdss_clean_galaxy_350_360_m05_0.fits'
    #file4 = '../data/sdss_clean_galaxy_350_351_m05_0.fits'
    data = fitsio.read(file)
    #data = esutil.io.read_header(file1,ext=1)
    
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