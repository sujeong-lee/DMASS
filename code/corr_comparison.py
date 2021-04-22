from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
from matplotlib.backends.backend_pdf import PdfPages
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

# sgc data
import esutil
import numpy as np

path = '/n/des/lee.5922/data/cmass_cat/'

cmass_sgc = esutil.io.read(path+'galaxy_DR12v5_CMASS_South.fits.gz')
random_sgc = esutil.io.read(path+'random0_DR12v5_CMASS_South.fits.gz')

#cmass_sgc = Cuts.SpatialCuts(cmass_sgc, ra=0, ra2=50, dec=-20, dec2 = 50)
#random_sgc = Cuts.SpatialCuts(random_sgc, ra=0, ra2=50, dec=-20, dec2 = 50)

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
print('Calculate comoving distance with FlatLambdaCDM cosmology \nH0=70, Om0=0.3')
sys.stdout.flush()

h = 0.7
r = cosmo.comoving_distance(cmass_sgc['Z']).value *h
r_rand = cosmo.comoving_distance(random_sgc['Z']).value *h

#import numpy.lib.recfuctions as rf
from numpy.lib import recfunctions as rf
print('Adding Comoving distance column')
sys.stdout.flush()
cmass_sgc = rf.append_fields( cmass_sgc, 'DC', data = r )
random_sgc = rf.append_fields( random_sgc, 'DC', data = r_rand )

print(cmass_sgc['DC'].data)

print('cutting redshift bin')
cmass_sgc_zcut = cmass_sgc[(cmass_sgc['Z'] > 0.43) & (cmass_sgc['Z'] < 0.7)]
random_sgc_zcut = random_sgc[(random_sgc['Z'] > 0.43) & (random_sgc['Z'] < 0.7)]

#tree corr
from systematics_module.corr import correlation_function_3d, _cfz

correlation_function_3d(data = cmass_sgc, rand = random_sgc, zlabel='Z', weight = None, njack = 20, suffix = 'cmass_sgc_20', out = None)
correlation_function_3d(data = cmass_sgc_zcut, rand = random_sgc_zcut, zlabel='Z', weight = None, njack = 20, suffix = 'cmass_sgc_zcut_20', out = None)
#r, xi, xijkerr = _cfz(cmass_sgc, random_sgc, zlabel = 'Z')
"""
print xi

suffix = 'cmass_sgc'
filename = 'data_txt/cfz_comparison_'+suffix+'.txt'
header = 'r (Mpc/h), xi(r), jkerr'
DAT = np.column_stack((r, xi, xijkerr ))
np.savetxt( filename, DAT, delimiter=' ', header=header )
print "saving data file to : ",filename

"""


