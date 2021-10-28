
from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
from cmass_modules import *

def main_ngc():


	path = '/n/des/lee.5922/data/cmass_cat/'
	cmass_ngc = esutil.io.read(path+'galaxy_DR12v5_CMASS_North.fits.gz', ensure_native=True)
	random_ngc = esutil.io.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_North.fits.gz', ensure_native=True)


	ind = np.random.choice(np.arange(random_ngc.size), size = random_ngc.size/10)

	#ind = np.random.choice(np.arange(cmass_ngc.size), size = cmass_ngc.size/10)
	from systematics_module.corr import correlation_function_multipoles, _cfz_multipoles
	correlation_function_multipoles(data = cmass_ngc, 
	                                rand = random_ngc[ind], zlabel = 'Z', njack = 20, 
	                                weight = True, suffix = 'cmass_ngc', nthreads = 20)


def main_sgc():

	path = '/n/des/lee.5922/data/cmass_cat/'
	cmass_sgc = esutil.io.read(path+'galaxy_DR12v5_CMASS_South.fits.gz', ensure_native=True)
	random_sgc = esutil.io.read('/n/des/lee.5922/data/cmass_cat/random0_DR12v5_CMASS_South.fits.gz', ensure_native=True)


	ind = np.random.choice(np.arange(random_sgc.size), size = random_sgc.size/10)

	#ind = np.random.choice(np.arange(cmass_ngc.size), size = cmass_ngc.size/10)
	from systematics_module.corr import correlation_function_multipoles, _cfz_multipoles
	correlation_function_multipoles(data = cmass_sgc, 
	                                rand = random_sgc[ind], zlabel = 'Z', njack = 20, 
	                                weight = True, suffix = 'cmass_sgc', nthreads = 20)


######################
#main_ngc()
main_sgc()