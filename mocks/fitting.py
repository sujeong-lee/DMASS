from scipy.optimize import curve_fit
import numpy as np

def lognormal_func(x, k0, sig):
    return k0*(np.exp((x-((sig**2.)/(2*k0)))/k0) - 1)

def lognormal_pdf(x, k0, sig): ##eq 1 in 1405.2666
    return 1./(np.sqrt(2*np.pi)*(x / np.abs(k0) + 1)*sig) * np.exp(-((np.abs(k0)*np.log(x/np.abs(k0)+1) + sig**2/(2*np.abs(k0)))**2)/(2*sig**2))

def ngal_to_deltamap(Ngal, fracdet, mask=None):
    """compute overdensity (deltamap) from Ngal count map. Applies mask to Ngal and fracdet before computing overdensity , 
    but returns deltamap with shape Ngal, so will only be 0 centered when masked, i.e. (np.average(deltamap[mask], weights=fracdet[mask]) == 0.)"""
    if mask is None:
        mask = np.ones_like(Ngal, dtype=bool)
    nmean = Ngal[mask].sum() / fracdet[mask].sum()
    deltamap = (Ngal / fracdet) / nmean - 1
    return deltamap

def fit_lnshift_deltamap(deltamap, sig=None, bins=100, sigcount_floor=2):
    """fit for shift parameter. If sig=None, also fit for sig parameter. 
    Fit assumes poisson variance on number of pixels in each bin. Sigcount_floor is a minimum variance to apply to all pixels (e.g. we are not 100% certain that a delta bin with 0 pixels is correct). Effectively a regularization term.
    Helps to reduce impact of outliers a bit, but not huge impact.
    Return (popt, pcov, bins, binmids, counts), where popt = [shift_fit, sig_fit] and pcov is covariance."""
    counts, bins = np.histogram(deltamap[deltamap>-1], bins=bins, density=False)

    binmids = bins[0:-1] + np.diff(bins)/2.
    density = counts / np.trapz(counts, binmids)
    
    if sig is not None: ## fit shift assuming a fixed sig
        popt, pcov = curve_fit(lambda x, a: lognormal_pdf(x, a, sig), binmids, density, p0=[1], sigma=np.sqrt(sigcount_floor + counts))
        #eps = 0.0000001
        #bounds = ((None, None), (sig-eps, sig+eps))
        #popt, pcov = curve_fit(lognormal_pdf, binmids, density, p0=[1, sig], sigma=np.sqrt(sigcount_floor + counts), bounds=bounds )
    else: # fit both shift and sig
        popt, pcov = curve_fit(lognormal_pdf, binmids, density, p0=[1, deltamap.std()], sigma=np.sqrt(sigcount_floor + counts))
    return popt, pcov, bins, binmids, counts


##########################################
## Fit and plot maglim Y3 mocks

"""
def main():
    mocks_dir_maglim = '/project/projectdirs/des/monroy/lognormal_mocks/maglim_v2.2_new_zbinning_jointmask_lognormal_mocks/'
    imock = 1

    fn_mock = f'mag_lim_lens_sample_combined_jointmask_newzbinning_nside512_mock{imock}.fits.gz'
    f,axes = plt.subplots(2, 3, figsize=(10,8))
    mymock = fio.read(mocks_dir_maglim + fn_mock)

    for iz in range(6):

        ax=axes.ravel()[iz]
        mymask = mymock['FRACGOOD']>0.6 #avoid crazy outliers due to dividing by small fracdet
        mydelta = ngal_to_deltamap(mymock['bin{}'.format(iz+1)], mymock['FRACGOOD'], mymock['FRACGOOD']>0.6)

        popt, pcov, binedges, binmids, counts = fit_lnshift_deltamap(mydelta[mymask], bins=60)
        _ = ax.hist(mydelta[mymask], bins=binedges, alpha=0.8)
        ax.plot(binmids, lognormal_pdf(binmids, *popt) * np.trapz(counts, binmids), label='LN fit')
        print(iz)
        print(popt)
        print([pp**0.5 for pp in pcov.diagonal()])

        ax.legend(title=f'{popt[0]:.2f}, {popt[1]:.2f}')
    plt.grid()
"""