import numpy as np

def mixing_color(data, suffix = '', sdss = None, cmass = None, 
    mag = ['MAG_MODEL', 'MAG_DETMODEL'], 
    err = [ 'MAGERR_MODEL','MAGERR_DETMODEL'], 
    no_zband = False  ):
    
    filter = ['G', 'R', 'I', 'Z']

    
    #mag = ['MAG_MODEL', 'MAG_DETMODEL']
    magtag = [ m+'_'+f+suffix for m in mag for f in filter ]
    del magtag[0], magtag[2]
    #err = [ 'MAGERR_MODEL','MAGERR_DETMODEL']
    errtag = [ e+'_'+f for e in err for f in filter ]
    del errtag[0], errtag[2]

    if no_zband : 
        print 'No z_band!'
        magtag = magtag[:-1]
        errtag = errtag[:-1]
        #print magtag
    #print magtag
    #print errtag
    X = [ data[mt] for mt in magtag[:2] ]
    Xerr = [ data[mt] for mt in errtag[:2] ]
    #reddeningtag = 'XCORR_SFD98'

    X = np.vstack(X).T
    Xerr = np.vstack(Xerr).T
    # mixing matrix
    W = np.array([
                  [1, 0, 0, 0, 0, 0],    # r mag
                  [0, 1, 0, 0, 0, 0],    # i mag
                  [0, 0, 1, -1, 0, 0],   # g-r
                  [0, 0, 0, 1, -1, 0],   # r-i
                  [0, 0, 0, 0, 1, -1]])  # i-z

    if no_zband : W = W[:-1,:-1]

    X = np.dot(X, W.T)


    Xcov = np.zeros(Xerr.shape + Xerr.shape[-1:])
    Xcov[:, range(Xerr.shape[1]), range(Xerr.shape[1])] = Xerr**2
    Xcov = np.tensordot(np.dot(Xcov, W.T), W, (-2, -1))
    return X, Xcov
    
tmp1=['MAG_MODEL_R', 'MAG_MODEL_I', 'MAG_DETMODEL_G', 'MAG_DETMODEL_R', 'MAG_DETMODEL_I', 'MAG_DETMODEL_Z']
tmp2=['MAGERR_MODEL_R', 'MAGERR_MODEL_I', 'MAGERR_DETMODEL_G', 'MAGERR_DETMODEL_R', 'MAGERR_DETMODEL_I', 'MAGERR_DETMODEL_Z']

# empty dictionary
data = {}
mag_r=np.array((23.,24.,22.5))
mag_i=np.array((18.,25.,28.))
magerr_r=np.array((0.1,0.2,0.3))
magerr_i=np.array((0.05,0.02,0.1))

# dictionary with integer keys
data = {'MAG_MODEL_R':mag_r, 'MAG_MODEL_I':mag_i, 'MAGERR_MODEL_R':magerr_r, 'MAGERR_MODEL_I':magerr_i}

dx, dxcov = mixing_color(data)
print(dx)
print(dxcov)
