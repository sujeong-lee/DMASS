#gold_st82 = io.SearchAndCallFits(path = '/n/des/lee.5922/data/gold_cat/', keyword='Y1A1_GOLD_STRIPE82_v2_000')
#gold_st82 = clean_gold(gold_st82)
gold_st82 = io.SearchAndCallFits(path = '/n/des/lee.5922/data', keyword='gold_st82_prob.fits')

#cmass_spec = esutil.io.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR11v1_CMASS_South.fits.gz')
cmass_photo = esutil.io.read('/n/des/lee.5922/data/cmass_cat/galaxy_DR11v1_CMASS_South-photoObj.fits.gz')
cmass_stripe = Cuts.keepGoodRegion(cmass_photo)
#gold_train, cmass_train = matchCatalogsbyPosition( gold , cmass_stripe)
gold_train = train_sample.copy()

g_sdss = cmass_train['MODELMAG'][:,1]-cmass_train['EXTINCTION'][:,1]
r_sdss = cmass_train['MODELMAG'][:,2]-cmass_train['EXTINCTION'][:,2]
i_sdss = cmass_train['MODELMAG'][:,3]-cmass_train['EXTINCTION'][:,3]
gr_sdss = g_sdss - r_sdss
ri_sdss = r_sdss - i_sdss

gr_des = gold_train['MAG_DETMODEL_G']- gold_train['MAG_DETMODEL_R']
ri_des = gold_train['MAG_DETMODEL_R']- gold_train['MAG_DETMODEL_I']

gr_gold = gold_st82['MAG_DETMODEL_G']- gold_st82['MAG_DETMODEL_R']
ri_gold = gold_st82['MAG_DETMODEL_R']- gold_st82['MAG_DETMODEL_I']

x = np.linspace(-2,4,10)
dperp = 0.55 + x/8.

gr_sdss_err = np.sqrt(cmass_train['MODELMAGERR'][:,1]**2 + cmass_train['MODELMAGERR'][:,2]**2)
ri_sdss_err = np.sqrt(cmass_train['MODELMAGERR'][:,2]**2 + cmass_train['MODELMAGERR'][:,3]**2)

gr_des_err = np.sqrt(gold_train['MAGERR_DETMODEL_G']**2 +  gold_train['MAGERR_DETMODEL_R']**2 )
ri_des_err = np.sqrt(gold_train['MAGERR_DETMODEL_R']**2 +  gold_train['MAGERR_DETMODEL_I']**2 )

dperpcut = ((ri_sdss - gr_sdss/8. < 0.55) & 
((gr_sdss>0.5)&(gr_sdss<0.96)|
 ((gr_sdss>1.43)&(gr_sdss<1.435))|
 ((gr_sdss>1.85)&(gr_sdss<3.0)) ) )

dperpcut_des = (ri_des - gr_des/8. < 0.5505) & (ri_des - gr_des/8. > 0.55035)
print np.sum(dperpcut), np.sum(dperpcut_des)

from chainconsumer import ChainConsumer

#seed(0)
#cov = normal(size=(3, 3))
#data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)

data_gold = np.column_stack( [gr_gold, ri_gold] )
data_sdss = np.column_stack( [gr_sdss, ri_sdss] )
data_des = np.column_stack( [gr_des, ri_des] )

c = ChainConsumer()
c.add_chain(data_gold, parameters=["$g-r$", "$r-i$"])
#c.add_chain(data_des)
c.configure(plot_hists=False, cloud=True, sigmas=np.linspace(0, 3, 10), colors ='grey', 
            shade_alpha = 0.7, kde=True)


fig = c.plotter.plot()
ax = fig.axes[0]
ax.plot(gr_des, ri_des, 'g.', markersize = 1, alpha = 0.7)
#ax.plot(gr_sdss, ri_sdss, '.',markersize = 1, alpha = 0.7)
ax.plot(x, dperp, 'r-', linewidth=1)
ax.set_ylim(0,2)
ax.set_xlim(0, 3)

ax.text(2.0, 0.1, 'DES  Color')
plt.tight_layout()

#fig.savefig('../paper_figure/gri_des.pdf')

c = ChainConsumer()

dat = data_gold.copy()
dat[:,0] = dat[:,0]+10
dat[:,1] = dat[:,1]+10
c.add_chain(dat, parameters=["$g-r$", "$r-i$"])
#c.add_chain(data_des)
c.configure(plot_hists=False, cloud=False, sigmas=np.linspace(0, 3, 10), colors ='grey', 
            shade_alpha = 0.5, kde=True)


fig = c.plotter.plot()
ax = fig.axes[0]
#ax.plot(gr_des, ri_des, 'g.', markersize = 1, alpha = 0.7)
ax.plot(gr_sdss, ri_sdss, '.',markersize = 1, alpha = 0.7)
ax.plot(x, dperp, 'r-', linewidth=1)
ax.set_ylim(0,2)
ax.set_xlim(0,3)

ax.text(1.8, 0.1, 'SDSS  Color')

plt.tight_layout()
#fig.savefig('../paper_figure/gri_sdss.pdf')

from run_DMASS import *

cutmask = priorCut_test(gold_st82)
gold_cut_outlier = gold_st82[cutmask]

#dperp_des = ri_des - gr_des/8.

gr_gold = gold_cut_outlier['MAG_DETMODEL_G']- gold_cut_outlier['MAG_DETMODEL_R']
ri_gold = gold_cut_outlier['MAG_DETMODEL_R']- gold_cut_outlier['MAG_DETMODEL_I']
dperp_gold = ri_gold - gr_gold/8.

Ncmass = []
Nall = []
for dp in np.linspace(0.3, 0.6, 31):
    dpmask = dperp_gold > dp
    gold_train, cmass_train = matchCatalogsbyPosition(gold_cut_outlier[dpmask], cmass_stripe)
    Ncmass.append(gold_train.size)
    Nall.append(gold_cut_outlier[dpmask].size)
    print dp, gold_train.size, gold_cut_outlier[dpmask].size
    
ncmass = np.array(Ncmass) * 1./np.array(Nall)
Nno = np.array(Nall) - np.array(Ncmass)

fig, ax = plt.subplots()
ax.plot( np.linspace(0.3, 0.6, 31), ncmass*1./ncmass.max() )
ax.plot( np.linspace(0.3, 0.6, 31), Nno *1./np.array(Nall)  )
ax.axvline(x = 0.55, ls='--', color='k')

def mixing_SDSS_color(data, suffix = '', sdss = None, cmass = None ):
    
    #filter = ['G', 'R', 'I', 'Z']
    filter = [1,2,3,4]
    mag = ['CMODELMAG', 'MODELMAG']
    magtag = mag#[ m+'_'+f+suffix for m in mag for f in filter ]
    #del magtag[0], magtag[2]
    err = [ 'CMODELMAGERR','MODELMAGERR']
    errtag = err#[ e+'_'+f for e in err for f in filter ]
    #del errtag[0], errtag[2]
    
    
    print data['CMODELMAG'][:,0].size
    
    X = [ data[mt][:,i] for mt in magtag for i in filter ]
    del X[0], X[2]
    Xerr = [ data[mt][:,i] for mt in errtag for i in filter]
    del Xerr[0], Xerr[2]
    #reddeningtag = 'XCORR_SFD98'

    X = np.vstack(X).T
    Xerr = np.vstack(Xerr).T
    # mixing matrix
    W = np.array([
                  [1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],    # i cmag
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],   # r-i
                  [0, 0, 0, 0, 1, -1]])  # i-z

    X = np.dot(X, W.T)

    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))
    return X, Xcov

X_cmass_train, Xcov_cmass_train = mixing_SDSS_color(cmass_train)
X_cmass_train_new = [np.random.multivariate_normal( X_cmass_train[i], Xcov_cmass_train[i], 1 ) for i in range(cmass_train.size) ]
X_cmass_train_new = np.array(X_cmass_train_new)
new_gr_sdss = X_cmass_train_new[:,0][:,2]
new_ri_sdss = X_cmass_train_new[:,0][:,3]

xbins = np.linspace(0, 3.0, 150)
ybins = np.linspace(0.2, 1.8, 120)
Ngal_sdss, xedges, yedges = np.histogram2d(gr_sdss, ri_sdss, bins=(xbins, ybins), normed=False)
Ngal_des, xedges, yedges = np.histogram2d(gr_des, ri_des, bins=(xbins, ybins), normed=False)

Ngal_sdss[Ngal_sdss == 0] = np.nan
Ngal_des[Ngal_des == 0] = np.nan

from chainconsumer import ChainConsumer

c = ChainConsumer()

dat = data_gold.copy()
dat[:,0] = dat[:,0]+10
dat[:,1] = dat[:,1]+10
c.add_chain(dat, parameters=["$g-r$", "$r-i$"])
#c.add_chain(data_des)
c.configure(plot_hists=False, cloud=False, sigmas=np.linspace(0, 3, 10), colors ='grey', 
            shade_alpha = 0.5, kde=False)


fig = c.plotter.plot(figsize=(4,3.2))
ax = fig.axes[0]

im = ax.imshow(np.rot90(Ngal_sdss), extent=(np.amin(xbins), np.amax(xbins), np.amin(ybins), np.amax(ybins)),
        cmap=plt.cm.jet, aspect='auto', zorder = 2 )#, 
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$N_{\rm gal}$', fontsize=15)
ax.plot(x, dperp, 'r-', linewidth=1, zorder = 3)

#ax.errorbar( gr_sdss[dperpcut], ri_sdss[dperpcut], xerr = gr_sdss_err[dperpcut], 
#            yerr = ri_sdss_err[dperpcut], fmt = 'o', color='black', zorder=10)


ax.set_xlabel('g-r', fontsize=15)
ax.set_ylabel('r-i', fontsize=15)
ax.set_xlim(0,3.)
ax.set_ylim(0.2,1.8)

ax.text(1.9, 0.3, 'SDSS  Color' )
plt.tight_layout()

fig.savefig('../paper_figure/gri_sdss_color_cbar.pdf')

from chainconsumer import ChainConsumer

#seed(0)
#cov = normal(size=(3, 3))
#data = multivariate_normal(normal(size=3), np.dot(cov, cov.T), size=100000)

data_gold = np.column_stack( [gr_gold, ri_gold] )
data_sdss = np.column_stack( [gr_sdss, ri_sdss] )
data_des = np.column_stack( [gr_des, ri_des] )

c = ChainConsumer()
c.add_chain(data_gold, parameters=["$g-r$", "$r-i$"])
#c.add_chain(data_des)
c.configure(plot_hists=False, cloud=True, sigmas=np.linspace(0, 3, 10), colors ='grey', 
            shade_alpha = 0.7, kde=False)


fig = c.plotter.plot(figsize=(4,3.2))
ax = fig.axes[0]

ax.imshow(np.rot90(Ngal_des), extent=(np.amin(xbins), np.amax(xbins), np.amin(ybins), np.amax(ybins)),
        cmap=plt.cm.jet, aspect='auto', zorder = 2, alpha = 1.0 )#, 
cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r'$N_{\rm gal}$', fontsize=15)
ax.plot(x, dperp, 'r-', linewidth=1, zorder = 3)

#ax.errorbar( gr_sdss[dperpcut], ri_sdss[dperpcut], xerr = gr_sdss_err[dperpcut], 
#            yerr = ri_sdss_err[dperpcut], fmt = 'o', color='black', zorder=10)


ax.set_xlabel('g-r', fontsize=15)
ax.set_ylabel('r-i', fontsize=15)
ax.set_xlim(0,3.)
ax.set_ylim(0.2,1.8)

ax.text(2.0, 0.3, 'DES  Color')
plt.tight_layout()

fig.savefig('../paper_figure/gri_des_color_cbar.pdf')