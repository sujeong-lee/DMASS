#import easyaccess as ea
import esutil
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import numpy.lib.recfunctions as rf
#import seaborn as sns

import fitsio
from fitsio import FITS, FITSHDR



def getDEScatalogs( file = '/n/des/huff.791/Projects/CMASS/Data/DES_Y1_S82.fits', bigSample = False):




    if bigSample is True:
        
        data = fitsio.read(file)

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

        #data = fitsio.read(filename, rows=[0,10], ext=2)
        #data,thing = esutil.io.read_header(cfile,ext=1,rows=rows)
        #file = '/n/des/huff.791/Projects/combined_i.fits'

    else:
    
        sample = np.arange(1743613)
        rows = np.random.choice( sample, size=500000 , replace = False)
        #rows = np.arange(500000)
        data = fitsio.read(file, rows=rows)
        data = fitsio.read(file)
    
    data.dtype.names = tuple([ data.dtype.names[i].upper() for i in range(len(data.dtype.names))])
    #data.sort(order = 'COADD_OBJECTS_ID')
    return data



def getSDSScatalogs(  file = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit', bigSample = False):
    
    
    if bigSample is True:
        filepath = '/n/des/lee.5922/data/'
        #file = filepath+'sdss_ra330_10.fit'
        
        sdss_files = [filepath+'sdss_ra340_345_dec1_0_sjlee_0.fit',
                      filepath+'sdss_ra340_345_decm1_0_sjlee_0.fit',
                      filepath+'sdss_ra345_350_decm1_0_sjlee_0.fit',
                      filepath+'sdss_ra345_350_dec1_0_sjlee_0.fit',
                      filepath+'sdss_ra355_360_decm1_0_sjlee.fit',
                      filepath+'sdss_ra355_360_dec1_0_sjlee.fit',
                      '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit',
                      '/n/des/huff.791/Projects/CMASS/Data/S82_SDSS_0_10.fit'
                      ]
        sdss_data = esutil.io.read(sdss_files,combine=True)
        
        #esutil.io.write('sdss_ra330_10.fit',sdss_data)
        #sdss_data = fitsio.read(file)
    
    else:
        #file1 = '/n/des/huff.791/Projects/CMASS/Data/s82_350_355_emhuff.fit'
        #file1 = '/Users/SJ/Dropbox/repositories/CMASS/data/test_emhuff.fit'
        #file3 = '../data/sdss_clean_galaxy_350_360_m05_0.fits'
        #file4 = '../data/sdss_clean_galaxy_350_351_m05_0.fits'
        sdss_data = fitsio.read(file)
        #data = esutil.io.read_header(file1,ext=1)
        
    sdss_data.dtype.names = tuple([ sdss_data.dtype.names[i].upper() for i in range(len(sdss_data.dtype.names))])
    
    return sdss_data

