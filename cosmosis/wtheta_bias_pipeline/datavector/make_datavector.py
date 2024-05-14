"""
This script will make the "data vector" that cosmosis expects

This is a fits file that contains:
    - The computed correlation functions (e.g. wtheta)
    - The covariance matrix
    - The n(z) of the sample

The inputs in this example are fake (just to get the code running)
Anyone using this script should replace the below with their inputs

"""
import numpy as np 
import twopoint #https://github.com/joezuntz/2point
import scipy.stats

######## the n(z)
kernel_name='nz_lens'

zedges = np.linspace(0,1,100)
zlow = zedges[:-1]
zhigh = zedges[1:]
z = (zlow+zhigh)/2.
nz = scipy.stats.norm(0.6,0.1).pdf(z) ###fake n(z), replace this
nzs = [nz]

kernel1 = twopoint.NumberDensity(kernel_name, zlow, z, zhigh, nzs)


######## the w(theta)

n_theta_bins = 20
value = np.loadtxt('testdata/bin_1_1.txt') ###fake data vector, replace this
angle_edges = np.logspace(np.log10(2.5),np.log10(250.), n_theta_bins+1)
angle_min = angle_edges[:-1]
angle_max = angle_edges[1:]
angle = (angle_min+angle_max)/2.

name = "wtheta"
types = [twopoint.Types('GPR'), twopoint.Types('GPR')]
kernels = ["nz_lens", "nz_lens"]
windows = "SAMPLE"
bins = [np.ones(n_theta_bins),np.ones(n_theta_bins)]
angular_bin = np.arange(n_theta_bins)

spec = twopoint.SpectrumMeasurement(name, bins, types, kernels, 
            windows, angular_bin, value, 
            angle=angle, error=None, angle_unit="arcmin", 
            metadata=None, angle_min=angle_min, angle_max=angle_max)

######## the covariance matrix

covmat = np.identity(n_theta_bins)
np.fill_diagonal(covmat, 0.01**2)  ###fake data vector, replace this
covmat_info = twopoint.CovarianceMatrixInfo('COVMAT', ['wtheta'], [n_theta_bins], covmat)

tp = twopoint.TwoPointFile([spec], [kernel1], [], covmat_info)
tp.to_fits('TEST.fits', overwrite=True)
