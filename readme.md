# Tutorial for DMASS selection code



### requirements

```
Python 2.7 or higher
astropy
astroML
healpy
matplotlib
esutil
argparse
yaml
fitsio 
scikit-learn
multiprocessing
```


### Install

```git clone https://github.com/sujeong-lee/CMASS.git```


### Configuration setting 

See `example.yaml`


### Run

1) Obtain a gaussian mixture model : 
- Set `Fitting : True` in yaml configuration file 
- type `python run_DMASS.py example.yaml`

2) Get catalogs containing the CMASS probability : 
- set `Fitting : False`
- type `python run_DMASS.py example.yaml`


### Train DMASS catalog should have the columns below
```
'FLAGS_GOLD', 'FLAGS_BADREGION', 'MAG_MODEL_G', 'MAG_MODEL_R', 'MAG_MODEL_I', 'MAG_MODEL_Z',
'MAG_DETMODEL_G', 'MAG_DETMODEL_R', 'MAG_DETMODEL_I', 'MAG_DETMODEL_Z', 'MAGERR_DETMODEL_G',
'MAGERR_DETMODEL_R', 'MAGERR_DETMODEL_I', 'MAGERR_DETMODEL_Z', 'MAGERR_MODEL_G', 'MAGERR_MODEL_R',
'MAGERR_MODEL_I', 'MAGERR_MODEL_Z', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'RA',
'DEC', 'COADD_OBJECTS_ID', 'MODEST_CLASS', 'HPIX', 'DESDM_ZP'
```

### Input train CMASS catalog can be downloaded from 
https://drive.google.com/open?id=1ZtlhFoaict_I4PDvI0rkPupYv6i8STLZ


