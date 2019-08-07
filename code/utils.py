import numpy as np
import numpy.lib.recfunctions as rf
import healpy as hp
import esutil
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

# cordinate and healpix utils
def GetRa(phi):
    return phi*180.0/np.pi

def GetTheta(dec):
    return (90.0 - dec) * np.pi / 180.0

def GetDec(theta):
    return 90.0 - theta*180.0/np.pi

def GetRaDec(phi, theta):
    return [GetRa(phi), GetDec(theta)]

def convertThetaPhiToRaDec(theta, phi):
    ra = phi*180.0/np.pi
    dec = 90.0 - theta*180.0/np.pi
    return ra,dec

def convertRaDecToThetaPhi(ra, dec):
    theta = (90.0 - dec) * np.pi / 180.0
    phi =  ra * np.pi / 180.0
    return theta, phi


def separation(ra1, dec1, ra2, dec2):
    return np.sqrt( ((ra1 - ra2)*np.cos(dec1))**2 + (dec1-dec2)**2 )


def hpRaDecToHEALPixel(ra, dec, nside=  4096, nest= False):
    phi = ra * np.pi / 180.0
    theta = (90.0 - dec) * np.pi / 180.0
    hpInd = hp.ang2pix(nside, theta, phi, nest= nest)
    return hpInd


def hpHEALPixelToRaDec( hpPixel, nside = 256 ):
    theta, phi = hp.pix2ang(nside, hpPixel)
    ra, dec = GetRaDec(phi,theta)
    return [ra, dec]


def convertRaDecToGalactic( ra=None, dec=None ):

    import astropy
    import astropy.coordinates

    skycoords = astropy.coordinates.SkyCoord(
        ra=ra, dec=dec, unit="deg",frame="fk5")

    l = skycoords.galactic.l.deg
    b = skycoords.galactic.b.deg
    return l,b


def convertGalacticToRaDec( l=None, b=None ):

    import astropy
    import astropy.coordinates

    skycoords = astropy.coordinates.SkyCoord(
        l=l, b=b, unit="deg",frame="galactic")
    ra = skycoords.fk5.ra.deg
    dec = skycoords.fk5.dec.deg
    return ra, dec

def hpHEALPixelGalacticToRaDec( hpPixel, nside = 256 ):
    theta, phi = hp.pix2ang(nside, hpPixel)
    l,b = GetRaDec(phi,theta)
    ra,dec = convertGalacticToRaDec( l=l, b=b )
    return [ra, dec]


def HealPixifyCatalogs(catalog=None, healConfig=None, ratag='RA', dectag = 'DEC'):
    HealInds = hpRaDecToHEALPixel( catalog[ratag],catalog[dectag], nside= healConfig['out_nside'], nest= healConfig['nest'])
    if 'HEALIndex' in catalog.dtype.fields:
        healCat = catalog.copy()
        healCat['HEALIndex'] = HealInds
    else:
        healCat = rf.append_fields(catalog,'HEALIndex',HealInds,dtypes=HealInds.dtype)
    return healCat

                                  
def getHealConfig(map_nside = 4096, out_nside = 128,
                  depthfile = '../../Data/sva1_gold_1.0.2-4_nside4096_nest_i_auto_weights.fits'):
    HealConfig = {}
    HealConfig['map_nside'] = map_nside
    HealConfig['out_nside'] = out_nside
    HealConfig['finer_nside'] = map_nside
    HealConfig['depthfile'] = depthfile
    HealConfig['nest'] = True
    return HealConfig


# testing for same area of des_gold 
def hpRaDecToRotatedHEALPixel(ra, dec, nside = 8, nest = False):
    rmat = np.array([ 0.382192, 0.054546, 0.922472,\
                     -0.924079, 0.025571, 0.381346,\
                     -0.002787, -0.998184, 0.060177 ]).reshape(3,3)
    vec  = hp.ang2vec(-(dec - 90.) * np.pi / 180., ra * np.pi / 180.)
    rvec = np.dot(np.linalg.inv(rmat), vec.T)
    pix  = hp.vec2pix(nside, *rvec, nest=nest)
    return pix


def rotate_hp_map(hmap, coord = ['C', 'G']):
    """
    Take hmap (a healpix map array) and return another healpix map array 
    which is ordered such that it has been rotated in (theta, phi) by the 
    amounts given.
    """
    nside = hp.npix2nside(len(hmap))

    # Get theta, phi for non-rotated map
    t,p = hp.pix2ang(nside, np.arange(hp.nside2npix(nside))) #theta, phi

    # Define a rotator
    #r = hp.Rotator(deg=False, rot=[rot_phi,rot_theta])
    r = hp.Rotator(coord = coord)

    # Get theta, phi under rotated co-ordinates
    trot, prot = r(t,p)
    # Interpolate map onto these co-ordinates
    rot_map = hp.get_interp_val(hmap, trot, prot)

    return rot_map



def MatchHPArea(cat=None, sysMap=None, origin_cat=None, nside=512):

    origin_HealInds = hpRaDecToHEALPixel( origin_cat['RA'],origin_cat['DEC'], nside= nside, nest= False)

    if cat is not None:
        HealInds = hpRaDecToHEALPixel( cat['RA'],cat['DEC'], nside= nside, nest= False)
        HPmask = np.in1d( HealInds, origin_HealInds )
        maskedcat = cat[HPmask]
        return maskedcat

    elif sysMap is not None:

        HpPixnum_mask = np.in1d( sysMap['PIXEL'], origin_HealInds )
        return sysMap[HpPixnum_mask]
"""
def getHPArea(cat=None, nside=512):
    HealInds = hpRaDecToHEALPixel( cat['RA'],cat['DEC'], nside= nside, nest= False)
    pix = set(HealInds)
    area = len(pix) * hp.nside2pixarea( nside, degrees = True)
    return area  
"""
def getHPArea(pix = 158, nside2 = 8):
    import healpy as hp
    from systematics import callingEliGoldMask
    GoldMask = callingEliGoldMask()
    ra = GoldMask['RA']# np.linspace(ra, ra2, 2000)
    dec = GoldMask['DEC']# np.random.choice(np.linspace(dec, dec2, 10), size = 2000)
    #Buzzard rotational matrix
    rmat = np.array([ 0.382192, 0.054546, 0.922472,\
                     -0.924079, 0.025571, 0.381346,\
                     -0.002787, -0.998184, 0.060177 ]).reshape(3,3)
    vec  = hp.ang2vec(-(dec - 90.) * np.pi / 180., ra * np.pi / 180.)
    rvec = np.dot(np.linalg.inv(rmat), vec.T)
    Goldnside2 = hp.vec2pix(nside2, *rvec, nest=False)
    mask = (Goldnside2 == pix )
    n = np.sum(mask)
    area = n * hp.nside2pixarea( 4096, degrees = True)
    return area

def getLowerHPind(Map, nside = 4096, nside2 = 8):
    import healpy as hp

    ra, dec = hpHEALPixelToRaDec( Map['PIXEL'], nside = nside )
    #Buzzard rotational matrix
    rmat = np.array([ 0.382192, 0.054546, 0.922472,\
                     -0.924079, 0.025571, 0.381346,\
                     -0.002787, -0.998184, 0.060177 ]).reshape(3,3)
    vec  = hp.ang2vec(-(dec - 90.) * np.pi / 180., ra * np.pi / 180.)
    rvec = np.dot(np.linalg.inv(rmat), vec.T)
    Goldnside2 = hp.vec2pix(nside2, *rvec, nest=False)   
    return Goldnside2


def getGoodRegionIndices(catalog=None, badHPInds=None, nside=4096, band=None, raTag = 'ra', decTag = 'dec'):
    hpInd = hpRaDecToHEALPixel(catalog[raTag], catalog[decTag], nside=nside, nest= True)
    keep = ~np.in1d(hpInd, badHPInds)
    return keep



def spatialcheck(data, label = None, convert = None, ratag='RA',dectag='DEC', zaxis = None, zlabel = None, suffix = '', dir='./'):
    
    jj = 0
    #fig, ax = plt.subplots(1,1,figsize = (7,7))
    fig, ax = plt.subplots(figsize = (10,8))
    if len(data) > 1000 :
            print "Warning : Be careful when you put more than 20 data"
            data = [data]

    if label is None: label = [None for i in range(0,len(data))]
    for d, l in zip(data, label):
        
        #rows = np.random.choice(np.arange(d.size), size = 500 )
        #d = d[rows]
        
        try:
            ra = d[ratag]
            dec = d[dectag]
        
        except ValueError:

            ra = d['ALPHAWIN_J2000_DET']
            dec = d['DELTAWIN_J2000_DET']
        """
        if convert is True:
            ra = data['ALPHAWIN_J2000_DET'] + 360
            dec = data['DELTAWIN_J2000_DET']
            
            cut = ra < 180
            ra1 = ra[cut] + 360
            ra[cut] = ra1
        


        if convert is None :
            ra = data['RA']
            dec = data['DEC']
        """
        
        ra2 = ra.copy()
        ra2[ra > 180] = ra[ra > 180]-360
        if zaxis is None : 
            if jj == 0 : ax.plot(ra2, dec, '.', alpha = 0.2, label = l , color = 'grey', markersize= 5)
            else : ax.plot(ra2, dec, '.', alpha = 0.3, label = l, markersize = 5 )
        else : 
            sc = ax.scatter( ra2, dec, c=zaxis)
            fig.colorbar( sc, ax=ax, label=zlabel  )
        jj += 1

    #ax.axvline(x = 0, lw = 1, color = 'k')
    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')
    #ax.set_xlim(0,360)
    ax.legend(loc = 'best')
    fig.savefig(dir+'/spatialtest_'+suffix)
    print 'figsave : '+dir+'/spatialtest_'+suffix+'.png'




def doVisualization_z( cats = None, labels = None, suffix = 'test', zlabel = 'DESDM_ZP', outdir = './' ):
    
    import matplotlib.pyplot as plt
    z_bin = np.linspace(1e-5, 1.0, 200)
    if labels == None : labels = [None for i in range(len(cats)) ]
    if len(cats) > 10 : cats = [cats]


    fig, axes = plt.subplots( len(cats), 1, figsize = (6, 5*(len(cats)) ))
    for i in range(1, len(cats)):
        #axes[i].hist( photoz_des['DESDM_ZP'], bins = z_bin,facecolor = 'green', normed = True, label = 'cmass')
        N1, _,_ = axes[i-1].hist( cats[0][zlabel], bins = z_bin, alpha = 0.5, normed = 1, label = labels[0])
        N2, _,_ = axes[i-1].hist( cats[i][zlabel], bins = z_bin, alpha = 0.5, normed = 1, label = labels[i])
        axes[i-1].set_xlabel(zlabel)
        axes[i-1].set_ylabel('N(z)')
        #ax.set_yscale('log')
        axes[i-1].legend(loc='best')

    axes[0].set_title('\nredshift histogram')

    axes[-1].remove()
    figname =outdir+'hist_z_'+suffix+'.png'
    fig.savefig(figname)
    print 'saving fig to ',figname



def doVisualization_z_1( cats = None, labels = None, suffix = 'test', zlabel = 'DESDM_ZP' ):
    
    import matplotlib.pyplot as plt
    z_bin = np.linspace(1e-5, 1.0, 200)
    if labels == None : labels = [None for i in range(len(cats)) ]
    if len(cats) > 10 : cats = [cats]


    alpha = [0.8, 0.5, 0.8, 0.5]
    color = ['red', 'green', 'blue', 'yellow']
    fig, ax = plt.subplots( 1, 1, figsize = (6, 5))
    for i in range(len(cats)):
        #axes[i].hist( photoz_des['DESDM_ZP'], bins = z_bin,facecolor = 'green', normed = True, label = 'cmass')
        N1, _,_ = ax.hist( cats[i][zlabel], bins = z_bin, alpha = alpha[i], normed = 1, facecolor=color[i], label = labels[i], histtype='step')
        #N2, _,_ = axes[i-1].hist( cats[i][zlabel], bins = z_bin, facecolor = 'red', alpha = 0.35, normed = 1, label = labels[i])
    ax.set_xlabel(zlabel)
    ax.set_ylabel('N(z)')
    #ax.set_yscale('log')
    ax.legend(loc='best')
    ax.set_xlim(0.1,1)
    ax.set_title('\nredshift histogram')

    figname ='figure/hist_z_'+suffix+'.png'
    fig.savefig(figname)
    print 'saving fig to ',figname


def makeTagUppercase(cat):
    names = list(cat.dtype.names)
    names = [ n.upper() for n in names ]
    cat.dtype.names = names
    return cat



def divide_bins( cat, Tag = 'Z', min = 0.2, max = 1.2, bin_num = 5, TagIndex = None, log = False ):
    
    
    if log : values = np.logspace(np.log10(min), np.log10(max), num = bin_num+1)
    else : values, step = np.linspace(min, max, num = bin_num+1, retstep = True)


    binkeep = []
    binned_cat = []
    
    column = cat[Tag]
    if TagIndex is not None: column = cat[Tag][:,TagIndex]
    
    for i in range(len(values)-1) :
        bin = (column >= values[i]) & (column < values[i+1])
        binned_cat.append( cat[bin] )
        binkeep.append(bin)

    if log : bin_center = values[:-1] + (values[1:]-values[:-1])/2.
    else : bin_center = values[:-1] + step/2.
    
    return bin_center, binned_cat, binkeep


def chisquare_dof( bin, galaxy_density, err ):
    zeromask = (( np.isnan(galaxy_density) == False ) | (galaxy_density != 0.) | (err != 0.))
    galaxy_density = galaxy_density[zeromask]
    err = err[zeromask]
    bin = bin[zeromask]
    chisquare = np.sum( (galaxy_density - 1.0 * np.ones(galaxy_density.size))**2/err**2 )
    return chisquare, chisquare/np.sum(zeromask)


# magnitude utils
def getRawMag( cat, mags = None, reddening = 'SFD98', suffix='_corrected' ):
    
    filter = ['G', 'R', 'I', 'Z']
    #names = ['MAG_DETMODEL', 'MAG_MODEL']
    
    
    
    if reddening == 'SFD98' :
        fac = 1.0
        redtag = 'XCORR_SFD98'
    elif reddening == 'SLR' :
        fac = -1.0
        redtag = 'SLR_SHIFT'
        
    for name in mags:
        for f in filter:
            print 'raw color = ',name+'_'+f, fac, redtag+'_'+f
            changeColumnName( cat, name = name+'_'+f+suffix, rename = name+'_'+f )
            cat[name+'_'+f] = cat[name+'_'+f] + fac * cat[redtag+'_'+f]

    #cat['MAG_APER_4_I'] = cat['MAG_APER_4_I'] + fac* cat[redtag+'_I']
    return cat

# magnitude utils
def getCorrectedMag( cat, mags = None, reddening = 'SFD98', suffix='_corrected' ):
    
    filter = ['G', 'R', 'I', 'Z']
    magtag= [ m+'_'+f for m in mags for f in filter ]
    
    if reddening == 'SFD98' :
        fac = -1.0
        redtag = 'XCORR_SFD98'
    elif reddening == 'SLR' :
        fac = 1.0
        redtag = 'SLR_SHIFT'
    elif reddening == None:
        fac = 0.0

    for mag in mags:
        #try :
        for f in filter:
            catcolumn = list(cat.dtype.names)
            magname = mag+'_'+f
            ind = catcolumn.index(magname)
            catcolumn[ind] = magname + suffix
            cat.dtype.names = tuple(catcolumn)
            
            if reddening == None : pass
            else:
                magname = mag+'_'+f
                print 'corrected color = ',magname, fac, redtag+'_'+f
                cat[magname+suffix] = cat[magname+suffix] + fac * cat[redtag+'_'+f]

    #except ValueError : print "'MAG_APER_4' is not in list. Skip it"
    #cat['MAG_APER_4_I'] = cat['MAG_APER_4_I'] + fac* cat[redtag+'_I']

    return cat



def changeColumnName( cat, name = None, rename = None ):
    catcolumn = list(cat.dtype.names)
    ind = catcolumn.index(name)
    catcolumn[ind] = rename
    cat.dtype.names = list(catcolumn)
    return cat

def appendColumn(cat, name = None, value = None, dtypes=None):
    import numpy.lib.recfunctions as rf  

    if name in cat.dtype.names:
        cat[name] = value
    else : 
        cat = rf.append_fields(cat, name, value, dtypes=dtypes)
    return cat


def AddingReddening(cat):
    import numpy.lib.recfunctions as rf   
    #from suchyta_utils.y1a1_slr_shiftmap import SLRShift
    from y1a1_slr_shiftmap import SLRShift
    zpfile = '/n/des/lee.5922/data/systematic_maps/y1a1_wide_slr_wavg_zpshift2.fit'
    slrshift = SLRShift(zpfile, fill_periphery=True)
    offsets_g = slrshift.get_zeropoint_offset('g',cat['RA'],cat['DEC'],interpolate=True)
    offsets_r = slrshift.get_zeropoint_offset('r',cat['RA'],cat['DEC'],interpolate=True)
    offsets_i = slrshift.get_zeropoint_offset('i',cat['RA'],cat['DEC'],interpolate=True)
    offsets_z = slrshift.get_zeropoint_offset('z',cat['RA'],cat['DEC'],interpolate=True)

    offsets = [ offsets_g, offsets_r, offsets_i, offsets_z  ]
    from pandas import DataFrame, concat
    nametag = ['SLR_SHIFT_'+f for f in ['G', 'R', 'I', 'Z'] ]
    offsetsdf = DataFrame( offsets, index = nametag ).T
    cat = DataFrame(cat)
    #del cat['index']
    print 'concatenate two ndarrays'
    cat = concat([cat, offsetsdf], axis=1)
    print 'dataframe to recordarray'
    cat = cat.to_records()
    
    """
    cat = rf.append_fields(cat, 'SLR_SHIFT_G', offsets_g)
    cat = rf.append_fields(cat, 'SLR_SHIFT_R', offsets_r)
    cat = rf.append_fields(cat, 'SLR_SHIFT_I', offsets_i)
    cat = rf.append_fields(cat, 'SLR_SHIFT_Z', offsets_z)
    """
    return cat


def RemovingSLRReddening(cat):


    if 'SLR_SHIFT_G' not in cat.dtype.names : 
        import numpy.lib.recfunctions as rf   
        #from suchyta_utils.y1a1_slr_shiftmap import SLRShift
        from y1a1_slr_shiftmap import SLRShift
        zpfile = '/n/des/lee.5922/data/systematic_maps/y1a1_wide_slr_wavg_zpshift2.fit'
        slrshift = SLRShift(zpfile, fill_periphery=True)
        offsets_g = slrshift.get_zeropoint_offset('g',cat['RA'],cat['DEC'],interpolate=True)
        offsets_r = slrshift.get_zeropoint_offset('r',cat['RA'],cat['DEC'],interpolate=True)
        offsets_i = slrshift.get_zeropoint_offset('i',cat['RA'],cat['DEC'],interpolate=True)
        offsets_z = slrshift.get_zeropoint_offset('z',cat['RA'],cat['DEC'],interpolate=True)

        offsets = [ offsets_g, offsets_r, offsets_i, offsets_z  ]
        from pandas import DataFrame, concat
        nametag = ['SLR_SHIFT_'+f for f in ['G', 'R', 'I', 'Z'] ]
        catnametag = cat.dtype.names
        try : 
            offsetsdf = DataFrame( offsets, index = nametag ).T
            cat = DataFrame(cat, index = catnametag)
            #del cat['index']
            print 'concatenate two ndarrays'
            cat = concat([cat, offsetsdf], axis=1)
            print 'dataframe to recordarray'
            cat = cat.to_records()
        
        except ValueError :
            print "Big-endian buffer not supported on little-endian compiler"
            print "Doing byte swapping"
            
            #offsetsdf = np.array(offsetsdf).byteswap().newbyteorder()
            cat = np.array(cat).byteswap().newbyteorder()
            offsetsdf = DataFrame( offsets, index = nametag ).T
            cat = DataFrame(cat)

            print 'concatenate two ndarrays'
            cat = concat([cat, offsetsdf], axis=1)
            print 'dataframe to recordarray'
            cat = cat.to_records()
            cat.dtype.names = [str(x) for x in cat.dtype.names]
            
            #matched = pd.merge(desData, goldData, on=key, how=how, suffixes = suffixes, left_index=left_index)


    print 'Removing SLR Shift '
    for mag in ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_AUTO']:
        print '  removing SLR from ', mag
        for b in ['G', 'R', 'I', 'Z']:
            cat[mag + '_'+b] = cat[mag + '_'+b] - cat['SLR_SHIFT'+ '_'+b]

    """
    cat = rf.append_fields(cat, 'SLR_SHIFT_G', offsets_g)
    cat = rf.append_fields(cat, 'SLR_SHIFT_R', offsets_r)
    cat = rf.append_fields(cat, 'SLR_SHIFT_I', offsets_i)
    cat = rf.append_fields(cat, 'SLR_SHIFT_Z', offsets_z)
    """
    return cat



def AddingSLRReddening(cat):


    print 'Adding SLR Shift '
    for mag in ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_AUTO']:
        print '  Adding SLR from ', mag
        for b in ['G', 'R', 'I', 'Z']:
            cat[mag + '_'+b] = cat[mag + '_'+b] + cat['SLR_SHIFT'+ '_'+b]

    return cat


def AddingSFD98Reddening(cat, kind='SPT', coeff = [3.186,2.140,1.569,1.196 ] ):
    import numpy.lib.recfunctions as rf
    import pandas as pd

    band = ['G', 'R', 'I', 'Z']

    if 'EBV' not in cat.dtype.names :   
     
        print 'Using SFD98 nside 4096 healpix map'
        print 'Bands :',  band
        #print 'NSIDE = 4096'
        print 'coefficients = ', coeff
        nside = 4096

        #from suchyta_utils.y1a1_slr_shiftmap import SLRShift
        #sfdfile = '/n/des/lee.5922/data/systematic_maps/y1a1_wide_slr_wavg_zpshift2.fit'
        mapname = '/n/des/lee.5922/data/systematic_maps/ebv_sfd98_fullres_nside_4096_nest_equatorial.fits'
        #mapname = '/n/des/lee.5922/data/systematic_maps/ebv_lenz17_nside_4096_nest_equatorial.fits'
        reddening_ring = hp.read_map(mapname)
        hpIndices = np.arange(reddening_ring.size)
        #goodmask = hp.mask_good(reddening_ring)
        #goldmask = 

        goodIndices = hpIndices #hpIndices[goodmask]
        clean_map = reddening_ring #reddening_ring[goodmask]

        sysMap = np.zeros((clean_map.size, ), dtype=[('PIXEL', 'i4'), ('EBV', 'f8'), ('RA', 'f8'), ('DEC', 'f8')])
        sysMap['PIXEL'] = goodIndices
        sysMap['EBV'] = clean_map
        
        sys_ra, sys_dec = hpHEALPixelToRaDec(goodIndices, nside = nside)
        sysMap['RA'] = sys_ra
        sysMap['DEC'] = sys_dec

        from cmass_modules.Cuts import keepGoodRegion
        sysMap = keepGoodRegion(sysMap)
        if kind is 'STRIPE82': sysMap = sysMap[sysMap['DEC'] > -30]
        elif kind is 'SPT': sysMap = sysMap[sysMap['DEC'] < -30]


        cat_hp = cat
        hpind = hpRaDecToHEALPixel(cat_hp['RA'], cat_hp['DEC'], nside= 4096, nest= False)
        #cat_hp.dtype.names = [str(x) for x in cat_hp.dtype.names]
        cat_hp = changeColumnName(cat_hp, name = 'HPIX', rename = 'PIXEL')
        cat_hp['PIXEL'] = hpind
        
        #sfdmap = changeColumnName( sysMap_ge, name = 'SIGNAL', rename = 'SFD98' )


        try : 

            cat_Data = pd.DataFrame(cat_hp)
            sfdData = pd.DataFrame(sysMap)
            matched = pd.merge(cat_Data, sfdData, on='PIXEL', how='left', 
                               suffixes = ['','_sys'], left_index=False, right_index=False)
        except ValueError :
            print "Big-endian buffer not supported on little-endian compiler"
            print "Doing byte swapping ...."

            cat_hp = np.array(cat_hp).byteswap().newbyteorder()
            #sfdmap = np.array(sfdmap).byteswap().newbyteorder()
            cat_Data = pd.DataFrame(cat_hp)
            sfdData = pd.DataFrame(sysMap)
            

            #print cat_Data.keys()
            #print sfdData.keys()
            matched = pd.merge(cat_Data, sfdData, on='PIXEL', how='left', 
                               suffixes = ['','_sys'], left_index=False, right_index=False)
            
        matched_arr = matched.to_records(index=False)
        matched_arr.dtype.names = [str(x) for x in matched_arr.dtype.names]


    else : matched_arr = cat

    print 'Adding SFD98 Shift '
    print 'Bands :',  band
    print 'coefficients = ', coeff

    for mag in ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_AUTO']:
        print '  Adding SFD to ', mag
        for i,b in enumerate(band):
            matched_arr[mag + '_'+b] = matched_arr[mag + '_'+b] - matched_arr['EBV'] * coeff[i]    

    return matched_arr



def RemovingSFD98Reddening(cat, kind='SPT', coeff = [3.186,2.140,1.569,1.196 ] ):
    import numpy.lib.recfunctions as rf
    import pandas as pd

    band = ['G', 'R', 'I', 'Z']

    print 'Removing SFD98 Shift '
    print 'Bands :',  band
    print 'coefficients = ', coeff

    correction_tag = 'EBV'

    if 'EBV' not in cat.dtype.names : 
        correction_tag = 'SFD98'

    for mag in ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_AUTO']:
        print '  Removing SFD to ', mag
        for i,b in enumerate(band):
            cat[mag + '_'+b] = cat[mag + '_'+b] + cat[correction_tag] * coeff[i]    

    return cat



def _AddingSFD98Reddening(cat, kind='SPT', coeff = [3.186,2.140,1.569,1.196 ] ):
    import numpy.lib.recfunctions as rf
    import pandas as pd

    band = ['G', 'R', 'I', 'Z']

    print 'Using SFD98 nside 512 healpix map'
    print 'Bands :',  band
    print 'NSIDE = 512'
    print 'coefficients = ', coeff

    #from suchyta_utils.y1a1_slr_shiftmap import SLRShift
    #sfdfile = '/n/des/lee.5922/data/systematic_maps/y1a1_wide_slr_wavg_zpshift2.fit'
    sfdmap = '/n/des/lee.5922/data/systematic_maps/ebv_sfd98_fullres_nside_4096_nest_equatorial.fits'
    #sysMap_ge = calling_sysMap( properties=['GE'], kind=kind, nside=512 )
    gepath = '/n/des/lee.5922/data/2mass_cat/'
    from systematics import loadSystematicMaps
    sysMap_ge = loadSystematicMaps( property = 'GE', filter = 'g', nside = 512 , kind = kind, path = gepath)
    if kind is 'STRIPE82': sysMap_ge = sysMap_ge[sysMap_ge['DEC'] > -30]
    elif kind is 'SPT': sysMap_ge = sysMap_ge[sysMap_ge['DEC'] < -30]
    
    #import copy
    #cat_hp = copy.deepcopy(cat)

    cat_hp = cat



    hpind = hpRaDecToHEALPixel(cat_hp['RA'], cat_hp['DEC'], nside=  512, nest= False)

    #cat_hp.dtype.names = [str(x) for x in cat_hp.dtype.names]
    cat_hp = changeColumnName(cat_hp, name = 'HPIX', rename = 'PIXEL')
    cat_hp['PIXEL'] = hpind
    
    sfdmap = changeColumnName( sysMap_ge, name = 'SIGNAL', rename = 'SFD98' )


    try : 

        cat_Data = pd.DataFrame(cat_hp)
        sfdData = pd.DataFrame(sfdmap)
        matched = pd.merge(cat_Data, sfdData, on='PIXEL', how='left', 
                           suffixes = ['','_sys'], left_index=False, right_index=False)
    except ValueError :
        print "Big-endian buffer not supported on little-endian compiler"
        print "Doing byte swapping ...."

        cat_hp = np.array(cat_hp).byteswap().newbyteorder()
        #sfdmap = np.array(sfdmap).byteswap().newbyteorder()
        cat_Data = pd.DataFrame(cat_hp)
        sfdData = pd.DataFrame(sfdmap)
        

        #print cat_Data.keys()
        #print sfdData.keys()
        matched = pd.merge(cat_Data, sfdData, on='PIXEL', how='left', 
                           suffixes = ['','_sys'], left_index=False, right_index=False)
        
    matched_arr = matched.to_records(index=False)
    matched_arr.dtype.names = [str(x) for x in matched_arr.dtype.names]

    print 'Adding SFD98 Shift '
    for mag in ['MAG_MODEL', 'MAG_DETMODEL', 'MAG_AUTO']:
        print '  Adding SFD to ', mag
        for i,b in enumerate(band):
            matched_arr[mag + '_'+b] = matched_arr[mag + '_'+b] - matched_arr['SFD98'] * coeff[i]    

    return matched_arr



def addphotoz(des = None, im3shape = None):
    
    import esutil
    
    #h = esutil.htm.HTM(10)
    #m_re, m_im3,_ = h.match( des['RA'], des['DEC'], im3shape['RA'], im3shape['DEC'], 1./3600, maxmatch=1)

    m_re, m_im3 = esutil.numpy_util.match(des['COADD_OBJECTS_ID'], im3shape['COADD_OBJECTS_ID'])

    print des.size, im3shape.size, m_re.size, 'z_failure :', des.size - m_re.size
    
    desdm_zp = np.zeros(des.size, dtype=float)
    #desdm_zp.fill(1e+20)
    desdm_zp[m_re] = im3shape[m_im3]['DESDM_ZP']
    result = rf.append_fields(des, 'DESDM_ZP', desdm_zp)
    return result


def SDSSaddphotoz(cmass):
    # add photoz to cmass cat
    photoz = fitsio.read('/n/des/lee.5922/data/cmass_cat/cmass_photoz_radec.fits')
    sortIdx, sortIdx2 = cmass['OBJID'].sort(), photoz['OBJID'].sort()
    cmass, photoz = cmass[sortIdx].ravel(), photoz[sortIdx2].ravel()
    mask = np.in1d(cmass['OBJID'], photoz['OBJID'])
    mask2 = np.in1d(photoz['OBJID'], cmass['OBJID'][mask])
    photozline = np.zeros( cmass.size, dtype=float )
    photozline[mask] = photoz['Z'][mask2]
    cmass = rf.append_fields(cmass, 'PHOTOZ', photozline)
    return cmass



def matchCatalogs(cat1, cat2 ,tag = 'tagname'):
    import esutil
    ind1, ind2 = esutil.numpy_util.match(cat1[tag], cat2[tag])
    return cat1[ind1], cat2[ind2]

def matchCatalogsbyPosition(cat1, cat2):
    import esutil
    h = esutil.htm.HTM(10)
    m1, m2, _ = h.match( cat1['RA'], cat1['DEC'], cat2['RA'], cat2['DEC'], 2./3600, maxmatch=1)
    return cat1[m1], cat2[m2]


def mergeCatalogsUsingPandas(des=None, gold=None, how = 'right', key=None, left_key = None, right_key = None, suffixes = ['','_GOLD'], left_index=False, right_index = False):
    import pandas as pd

    try :
        desData = pd.DataFrame(des)
        goldData = pd.DataFrame(gold)
        matched = pd.merge(desData, goldData, on=key, how=how, suffixes = suffixes, left_index=left_index, right_index=right_index)

    except ValueError :
        print "Big-endian buffer not supported on little-endian compiler"
        print "Doing byte swapping"
        
        des = np.array(des).byteswap().newbyteorder()
        gold = np.array(gold).byteswap().newbyteorder()
        desData = pd.DataFrame(des)
        goldData = pd.DataFrame(gold)
        matched = pd.merge(desData, goldData, on=key, left_on = left_key, right_on = right_key, how=how, suffixes = suffixes, left_index=left_index)

    matched_arr = matched.to_records(index=False)
    # This last step is necessary because Pandas converts strings to Objects when eating structured arrays.
    # And np.recfunctions flips out when it has one.

    """
    oldDtype = matched_arr.dtype.descr
    newDtype = oldDtype
    for thisOldType,i in zip(oldDtype, xrange(len(oldDtype) )):
        if 'O' in thisOldType[1]:
            newDtype[i] = (thisOldType[0], 'S12')
    matched_arr = np.array(matched_arr,dtype=newDtype)
    """
    return matched_arr


def priorCut_buz(merged_buz):
    basicmask = (
                 ( merged_buz['MAG_G'] > 15.0 ) & ( merged_buz['MAG_G'] < 25.0 ) &
                 ( merged_buz['MAG_R'] > 15.0 ) & ( merged_buz['MAG_R'] < 25.0 ) &
                 ( merged_buz['MAG_I'] > 15.0 ) & ( merged_buz['MAG_I'] < 22.0 ) &
                 ( merged_buz['MAG_Z'] > 15.0 ) & ( merged_buz['MAG_Z'] < 25.0 ) &
                 ( merged_buz['MAG_G'] - merged_buz['MAG_R'] < 2.5 ) &
                 ( merged_buz['MAG_G'] - merged_buz['MAG_R'] > -0.5 ) &
                 #( merged_buz['MAG_G'] - merged_buz['MAG_R'] > 0.0 ) &
                 ( merged_buz['MAG_R'] - merged_buz['MAG_I'] < 2.0 ) &
                 ( merged_buz['MAG_R'] - merged_buz['MAG_I'] > -0.5 )
                 #( merged_buz['MAG_R'] - merged_buz['MAG_I'] > 0.0 )
                 )
    return merged_buz[basicmask]



# testing for same area of des_gold 
def getHealInd_buz(ra, ra2, dec, dec2):
    rmat = np.array([ 0.382192, 0.054546, 0.922472,\
                     -0.924079, 0.025571, 0.381346,\
                     -0.002787, -0.998184, 0.060177 ]).reshape(3,3)
    vec  = hp.ang2vec(-(dec - 90.) * np.pi / 180., ra * np.pi / 180.)
    rvec = np.dot(np.linalg.inv(rmat), vec.T)
    pix  = set(hp.vec2pix(8, *rvec, nest=False))
    print 'Healpix ind ', np.sort(list(pix))
    return pix


"""
def DownloadFileFromURL( url = None, path = None ):
    import urllib2
    #url = "http://download.thinkbroadband.com/10MB.zip"

    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(path+file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)

    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break

        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
    f.close()
"""



###### Creating randoms on sphere ------------------------------------------

def ra_dec_to_xyz(ra, dec):
    """Convert ra & dec to Euclidean points
    Parameters
    ----------
    ra, dec : ndarrays
    Returns
    x, y, z : ndarrays
    """
    sin_ra = np.sin(ra * np.pi / 180.)
    cos_ra = np.cos(ra * np.pi / 180.)

    sin_dec = np.sin(np.pi / 2 - dec * np.pi / 180.)
    cos_dec = np.cos(np.pi / 2 - dec * np.pi / 180.)

    return (cos_ra * sin_dec,
            sin_ra * sin_dec,
            cos_dec)

def uniform_sphere(RAlim, DEClim, size=1):
    """Draw a uniform sample on a sphere
    Parameters
    ----------
    RAlim : tuple
        select Right Ascension between RAlim[0] and RAlim[1]
        units are degrees
    DEClim : tuple
        select Declination between DEClim[0] and DEClim[1]
    size : int (optional)
        the size of the random arrays to return (default = 1)
    Returns
    -------
    RA, DEC : ndarray
        the random sample on the sphere within the given limits.
        arrays have shape equal to size.
    """
    zlim = np.sin(np.pi * np.asarray(DEClim) / 180.)

    z = zlim[0] + (zlim[1] - zlim[0]) * np.random.random(size)
    DEC = (180. / np.pi) * np.arcsin(z)
    RA = RAlim[0] + (RAlim[1] - RAlim[0]) * np.random.random(size)
    
    return RA, DEC

def uniform_random_on_sphere(data, size = None, z=False ):
    """Create random samples on the region of a given data
    Parameters
    ----------
    data : fits or record array data having 'RA' and 'DEC'
    size : sample size
    Returns
    -------
    data_R : created randoms
    """
    
    ra = data['RA']
    dec = data['DEC']
    
    n_features = ra.size
    #size = 100 * data.size
    
    # draw a random sample with N points
    ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                 (min(dec), max(dec)),
                                 size)
    #data = np.asarray(ra_dec_to_xyz(ra, dec), order='F').T
    #data_R = np.asarray(ra_dec_to_xyz(ra_R, dec_R), order='F').T
    
    data_R = np.zeros((ra_R.size,), dtype=[('RA', 'float'), ('DEC', 'float'), ('DESDM_ZP', 'float')])
    data_R['RA'] = ra_R
    data_R['DEC'] = dec_R
    
    #random redshift distribution
    if z is True:
        mu, sigma = np.mean(data['DESDM_ZP']), np.std(data['DESDM_ZP'])
        z_R = np.random.normal(mu, sigma, size)
        data_R['DESDM_ZP'] = z_R
                              
    return data_R

###### Creating randoms on sphere ------------------------------------------


##### JK SAMPLING -------------------------------------------------------------------------------

def construct_jk_catalog( cat, njack = 10, root='./', jtype = 'generate', jfile = 'jkregion.txt', suffix = '' , retind = False):

    import os
    os.system('mkdir '+root)
    km, jfile = GenerateRegions(cat, cat['RA'], cat['DEC'], root+jfile, njack, jtype)
    ind = AssignIndex(cat, cat['RA'], cat['DEC'], km)
    #ind_rand = AssignIndex(rand, rand['RA'], rand['DEC'], km)
    

    if retind : 
        if 'JKindex' in cat.dtype.names : cat['JKindex'] = ind
        else : cat = appendColumn(cat, name = 'JKindex', value = ind, dtypes=None)
        return cat

    else : 
        catlist = []
        for i in range(njack):
            mask = (ind == i)
            catlist.append(cat[mask])
            #mask_rand = (ind_rand == i)
            #_constructing_input_file(cat[mask], rand[mask_rand], root = root, suffix = suffix)
            
        #os.remove(jkfile)  

        return catlist

def GenerateRegions(jarrs, jras, jdecs, jfile, njack, jtype):

    import kmeans_radec
    
    if jtype=='generate':
        rdi = np.zeros( (len(jarrs),2) )
        rdi[:,0] = jras# jarrs[gindex][jras[gindex]]
        rdi[:,1] = jdecs #jarrs[gindex][jdecs[gindex]]

        if jfile is None:
            jfile = 'JK-{0}.txt'.format(njack)
        km = kmeans_radec.kmeans_sample(rdi, njack, maxiter=200, tol=1.0e-5)

        if not km.converged:
            raise RuntimeError("k means did not converge")
        np.savetxt(jfile, km.centers)

    elif jtype=='read':
	print 'read stored jfile :', jfile
        centers = np.loadtxt(jfile)
        km = kmeans_radec.KMeans(centers)
        njack = len(centers)

    return [km, jfile]


def AssignIndex(jarrs, jras, jdecs, km):
    
    ind = []
    for i in range(len(jarrs)):
        rdi = np.zeros( (len(jarrs[i]),2) )
        rdi[:,0] = jras[i]
        rdi[:,1] = jdecs[i]
        index = km.find_nearest(rdi)
        ind.append(index[0])
    return np.array(ind)


##### JK SAMPLING ------------------------------------------------------------------------------




#### survey mask ----------------------------------------------------
def catalog_masking(cat, nside_out = 128, area=None):
    print 'input cat ', cat.size
    #cat = reddening_mask(cat = cat, nside_out = nside_out)
    #print 'reddening ', cat.size
    if area is 'NGC' :
        cat = boss_mask(cat=cat, nside_out = nside_out, area=area)
    elif area is 'SGC' :
        cat = boss_mask(cat=cat, nside_out = nside_out, area=area)    
    elif area is 'SPT':
        cat = Cuts.keepGoodRegion(cat)
        cat = cat[cat['DEC']<-3.0]
        #cat = y1gold_mask(cat = cat, nside_out = nside_out)
    else : print 'area keyword input : either one of NGC SGC SPT'
        
    print 'output ', cat.size
    return cat

def boss_mask(cat=None, area='SGC', nside_out = 128):
    

    if area is 'SGC' : 
        boss_hpind = esutil.io.read('/n/des/lee.5922/data/cmass_cat/healpix_boss_footprint_SGC_1024.fits')
        
        #boss_hpind = hp.ud_grade(boss_hpind, pess=True, nside_out = nside)

    elif area is 'NGC':
        boss_hpind = esutil.io.read('/n/des/lee.5922/data/cmass_cat/healpix_boss_footprint_NGC_1024.fits')
        #boss_hpind = hp.ud_grade(boss_hpind, pess=True, nside_out = nside)
    else : 
        print 'area keyword input : SGC or NGC'
        return 0
     
    boss_hpfrac = np.zeros(hp.nside2npix(1024), dtype = 'float')
    boss_hpfrac[boss_hpind] = 1.0
    boss_hpind_up = np.arange(hp.nside2npix(nside_out))
    
    if nside_out == 1024 : boss_hpfrac_ud = boss_hpfrac
    else : boss_hpfrac_ud = hp.ud_grade(boss_hpfrac, pess=True, nside_out = nside_out)
    
    #print boss_hpfrac
    
    
    #boss_hpfrac_ud2 = boss_hpfrac_ud.copy()
    #boss_hpfrac_ud2[ boss_hpfrac_ud > 0.8] = 0
    #hp.mollview( boss_hpfrac_ud2, max = 1, nest = False)
    
    #if cat is None : 
    #    boss_hpmask = np.zeros(hp.nside2npix(4096), dtype = 'bool')
    #    boss_hpmask[boss_hpind] = 1
    #    return boss_hpmask
    
    #elif cat is not None : 
        
    hpind = hpRaDecToHEALPixel(cat['RA'], cat['DEC'], nside= nside_out, nest= False) 
    goodind = np.in1d(hpind, boss_hpind_up[[ boss_hpfrac_ud > 0.8]])
    return cat[goodind]





