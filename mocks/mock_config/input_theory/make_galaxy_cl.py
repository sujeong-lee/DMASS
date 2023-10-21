# make galaxy C(l) for CMASS-like sample
# requires cosmological parameters and CMASS n(z) as an input
# uses the DESC CCL code https://github.com/LSSTDESC/CCL
import pyccl as ccl
import numpy as np
import os

######## config ########

cmass_nz_file = 'nz_density_CMASS_SGC_cosmosisformat.txt'
outdir = 'output/'

###########################################

if os.path.exists(outdir) == False:
	os.mkdir(outdir)
	os.mkdir(outdir+'/galaxy_cl/')
	os.mkdir(outdir+'/galaxy_xi/')

# Create new Cosmology object with a given set of parameters. This keeps track
# of previously-computed cosmological functions
cosmo = ccl.CosmologyVanillaLCDM()

# load the CMASS N(Z)
z_n, n = np.loadtxt(cmass_nz_file,unpack=True)

#set galaxy bias fixed at 2.0
bz = np.ones(len(z_n))*2.0

# Create object to represent tracer of the number density signal
cmass_density = ccl.NumberCountsTracer(cosmo, dndz=(z_n, n), has_rsd=True, bias=(z_n,bz), mag_bias=None)

# Calculate the angular power spectrum of the tracer as a function of ell
ell = np.logspace(0, 5, 100)
cl = cosmo.angular_cl(cmass_density, cmass_density, ell)

# Calculate the correlation function of the tracer as a function of theta
#theta=angular separation (in degrees)
theta = np.logspace(np.log10(2.5/60), np.log10(250/60), 20)
wtheta = cosmo.correlation(ell,cl,theta,type='NN')

#save the output in comsosis-like format 
np.savetxt(outdir+'/galaxy_cl/ell.txt', ell)
np.savetxt(outdir+'/galaxy_cl/bin_1_1.txt', cl)

theta_rad = theta*np.pi/180.
np.savetxt(outdir+'/galaxy_xi/theta.txt', theta_rad)
np.savetxt(outdir+'/galaxy_xi/bin_1_1.txt', wtheta)


