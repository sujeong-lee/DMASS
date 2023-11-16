import sys
sys.path.append('../')
from xd import *
from utils import *
import esutil
import healpy as hp
from systematics import *
from systematics_module import *
import os
from numpy.lib.recfunctions import append_fields
import scipy.stats

import matplotlib.pyplot as plt
import numpy as np
from run_systematics import sys_iteration, weightmultiply, fitting_allSP, calling_sysMap

from sys_functions import *
from systematic_functions import *

contin = True
i = 0
keyword_template = 'vali{0}'
while contin == True:
    input_keyword = keyword_template.format(i)
    print("current run: ", input_keyword)
    run_name = input_keyword
    systematics(run_name)
    new_name = keyword_template.format(i+1)
    print(new_name)
    contin = weights_applied(run_name, new_name)
    i+=1
print("last weight: ", run_name)