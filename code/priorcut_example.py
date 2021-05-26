import fitsio
from fitsio import FITS

def priorCut_test():

    fitsfile= '/PCON0003/warner785/galaxy_DR11v1_CMASS_South-photoObj.fits.gz'
    hdu=fitsio.read(fitsfile)

    ## Should add MODEST_CLASS cut later.
    modelmag_g_des = hdu[1].data['SOF_CM_MAG_CORRECTED_G']
    modelmag_r_des = hdu[1].data['SOF_CM_MAG_CORRECTED_R']
    modelmag_i_des = hdu[1].data['SOF_CM_MAG_CORRECTED_I']
    cmodelmag_g_des = hdu[1].data['SOF_CM_MAG_CORRECTED_G']
    cmodelmag_r_des = hdu[1].data['SOF_CM_MAG_CORRECTED_R']
    cmodelmag_i_des = hdu[1].data['SOF_CM_MAG_CORRECTED_I']
    magauto_des = hdu[1].data['SOF_CM_MAG_CORRECTED_I']

    cut = (((cmodelmag_r_des > 16) & (cmodelmag_r_des <24)) &
           ((cmodelmag_i_des > 16) & (cmodelmag_i_des <24)) &
           ((cmodelmag_g_des > 16) & (cmodelmag_g_des <24)) &
           ((modelmag_r_des - modelmag_i_des ) < 1.5 ) & # 10122 (95%)
           ((modelmag_r_des - modelmag_i_des ) > 0. ) & # 10120 (95%)
           ((modelmag_g_des - modelmag_r_des ) > 0. ) & # 10118 (95%)
           ((modelmag_g_des - modelmag_r_des ) < 2.5 ) & # 10122 (95%)
           (magauto_des < 21.5 ) #&  10124 (95%)
        )
    return cut

priorCut_test()
