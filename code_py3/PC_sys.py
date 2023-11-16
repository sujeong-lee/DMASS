input_path = '/fs/scratch/PCON0008/warner785/bwarner/band_' 
#/band_g,r,i,z/:
band = ['g', 'r', 'i', 'z']

#test weighted pca-dmass on the sp checks

#y3a2_g_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz
#y3a2_g_o.4096_t.32768_EXPTIME.SUM_EQU.fits.gz 
#y3a2_g_o.4096_t.32768_FWHM.WMEAN_EQU.fits.gz 
#y3a2_g_o.4096_t.32768_SKYBRITE.WMEAN_EQU.fits.gz
#redmine
#nersc pca location for sp

current_map = 'AIRMASS'

input_keyword_1 = 'y3a2_' +band[y] +'_o.4096_t.32768_AIRMASS.WMEAN_EQU.fits.gz'
sysMap = io.SearchAndCallFits(path = input_path, keyword = input_keyword)

path = '/fs/scratch/PCON0008/warner785/bwarner/'
    
sys_weights = False
    
linear = False
quadratic = False
    
sysMap = cutPCA(sysMap)
fracDet = fitsio.read(path+'y3a2_griz_o.4096_t.32768_coverfoot_EQU.fits.gz')
fracDet['PIXEL'] = hp.nest2ring(4096, fracDet['PIXEL'])

if sys_weights == True:
#    dmass_chron =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/dmass_sys_weight_'+current_map+'.fits')
    dmass_chron =fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/dmass_sys_weight_pca_FIRST_50.fits')
#        random_chron = fitsio.read('../output/test/train_cat/y3/'+input_keyword+'randoms.fits')
    h_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/h_ran_'+current_map+'.fits')
    norm_number_density_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/norm_ran_'+current_map+'.fits')
    fracerr_ran_norm = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/fracerr_ran_'+current_map+'.fits')
        
else:
    index_mask = np.argsort(dmass_spt)
    dmass_chron = dmass_spt[index_mask] # ordered by hpix values
    dmass_chron['HPIX_4096'] = hp.nest2ring(4096, dmass_chron['HPIX_4096']) 
#    randoms4096 = random_pixel(random_val_fracselected)
#    index_ran_mask = np.argsort(randoms4096)
#    random_chron = randoms4096[index_ran_mask]
#    h_ran,_= number_gal(sysMap, random_chron, sys_weights = False)
#    area = area_pixels(sysMap, fracDet)
#    pcenter, norm_number_density_ran, fracerr_ran_norm = number_density(sysMap, h_ran, area)

h_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/h_ran_'+current_map+'.fits')
norm_number_density_ran = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/norm_ran_'+current_map+'.fits')
fracerr_ran_norm = fitsio.read('/fs/scratch/PCON0008/warner785/bwarner/fracerr_ran_'+current_map+'.fits')

h, sysval_gal = number_gal(sysMap, dmass_chron, sys_weights = False) # change this to true if sys weights run
area = area_pixels(sysMap, fracDet)
pcenter, norm_number_density, fracerr_norm = number_density(sysMap, h, area)