import os, sys
import esutil
import healpy as hp
import numpy as np
import fitsio

sys.path.append('code_py3/')
from cmass_modules import io
from utils import matchCatalogsbyPosition, hpHEALPixelToRaDec, HealPixifyCatalogs,
