from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *

from matplotlib import rc
import matplotlib.pylab as plt

from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import Corrfunc
from os.path import dirname, abspath, join as pjoin
from Corrfunc.io import read_catalog


def adding_dc(catalog, H0=67.77, Om0 = 0.307115):
    from astropy.cosmology import FlatLambdaCDM
    #cosmo = FlatLambdaCDM(H0=67.77, Om0=0.307115)
    cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
    print 'Calculate comoving distance with FlatLambdaCDM cosmology \nH0='+str(H0)+', Om0='+str(Om0)
    sys.stdout.flush()

    h = H0/100.
    r = cosmo.comoving_distance(catalog['Z']).value *h

    #import numpy.lib.recfuctions as rf
    from numpy.lib import recfunctions as rf
    print 'Adding Comoving distance column'
    sys.stdout.flush()
    catalog = rf.append_fields( catalog, 'DC', data = r )
    return catalog



def main( switch = None ):


	# calling catalogs
	mock_catalog_south = esutil.io.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_South_DC.fits.gz'
	                                  , ensure_native =True)
	mock_catalog_north = esutil.io.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR12v5_CMASS_North_DC.fits.gz'
	                                  , ensure_native =True)

	randoms_catalog_south = esutil.io.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz'
	                                    , columns=['RA', 'DEC', 'Z', 'WEIGHT_FKP'], ensure_native =True)
	randoms_catalog_north = esutil.io.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_North.fits.gz'
	                                    , columns=['RA', 'DEC', 'Z', 'WEIGHT_FKP'], ensure_native =True)
	"""
	Om0 = 0.307115
	H0 = 0.6777
	randoms_catalog_south = adding_dc(randoms_catalog_south, H0 = H0 , Om0 = Om0)
	fitsio.write('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South_DC.fits.gz', clobber=True)

	randoms_catalog_north = adding_dc(randoms_catalog_north, H0 = H0 , Om0 = Om0)
	fitsio.write('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_North_DC.fits.gz', clobber=True)

	stop
	"""

	mock_catalog = np.hstack([mock_catalog_south, mock_catalog_north])
	randoms_catalog = np.hstack([randoms_catalog_south, randoms_catalog_north])

	mock_catalog_south, mock_catalog_south, randoms_catalog_south, randoms_catalog_north \
	= None, None, None, None

	print 'random sampling'
	random_sam_ind = np.random.choice(np.arange(mock_catalog.size), size = mock_catalog.size)
	mock_catalog = mock_catalog[random_sam_ind]
	random_sam_ind2 = np.random.choice(np.arange(randoms_catalog.size), size = randoms_catalog.size/10)
	randoms_catalog = randoms_catalog[random_sam_ind2]
	print 'sample size (gal, random) :', mock_catalog.size, ' ', randoms_catalog.size


	Om0 = 0.307115
	H0 = 0.6777
	#mock_catalog = adding_dc(mock_catalog, H0 = H0 , Om0 = Om0)
	randoms_catalog = adding_dc(randoms_catalog, H0 = H0 , Om0 = Om0)

	# cutting out z tails
	mock_catalog_zcut = mock_catalog[(mock_catalog['Z'] > 0.43) & (mock_catalog['Z'] < 0.7)]
	randoms_catalog_zcut = randoms_catalog[(randoms_catalog['Z'] > 0.43) & (randoms_catalog['Z'] < 0.7)]


	from systematics_module.corr import correlation_function_multipoles, _cfz_multipoles
	print 'Initializing CorrFunc modules'

	#settings
	nthreads = 28
	cosmology = 2 # Specify cosmology (1->LasDamas, 2->Planck)
	njack = 10

	
	if switch == 0 : 

		suffix = 'cmass_ngc_sgc_weight_fkp_random10_njack'+str(njack)
		mockcat = mock_catalog
		randcat = randoms_catalog
		print 'switch', suffix

	elif switch == 1:

		suffix = 'cmass_ngc_sgc_zcut_weight_fkp_random10_njack'+str(njack)
		mockcat = mock_catalog_zcut
		randcat = randoms_catalog_zcut
		print 'switch', suffix

	
	weight_data = mockcat['WEIGHT_SYSTOT']*( mockcat['WEIGHT_CP'] + mockcat['WEIGHT_NOZ'] - 1.)
	weight_rand = randcat['WEIGHT_FKP']
	
	"""
	filename = 'data_txt/cfz_multipoles_'+suffix+'.txt'
	print 'file will be saved to ', filename
	scenter, multipoles02 = _cfz_multipoles( mockcat, randcat, nthreads = nthreads, weight=[weight_data, weight_rand] )
	print scenter
	print multipoles02
	DAT = np.column_stack((scenter, multipoles02[:scenter.size], multipoles02[scenter.size:]))	
	np.savetxt(filename, DAT)
	print 'file save to ', filename
	
	"""
	correlation_function_multipoles(data = mockcat, \
	rand = randcat, zlabel = 'Z', \
	njack = njack, \
	weight = [weight_data, weight_rand], \
	suffix = suffix, \
	out = None, \
	nthreads = nthreads)
	
############################
if __name__=='__main__':
	main(switch = 0)
	# switch 0 : no zcut
	# switch 1 : with zcut



