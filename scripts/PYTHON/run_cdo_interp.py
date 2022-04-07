import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from pyproj import Transformer
import pandas as pd
import sys,os
from cdo import Cdo
import time
from tqdm.notebook import tqdm

cdo = Cdo()

#nemo_run = 'OPM027'
#yy_start = 1999 
#yy_end = 2038

nemo_run = 'OPM031'
yy_start = 1999 
yy_end = 2068

inputpath_mask = '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-'+nemo_run+'-MSH/'
inputpath_data = '/bettik/mathiotp/NEMO/DRAKKAR/eORCA025.L121/eORCA025.L121-'+nemo_run+'-S/'
outputpath = '/bettik/burgardc/DATA/BASAL_MELT_PARAM/interim/NEMO_eORCA025.L121_'+nemo_run+'_ANT_STEREO/'

for yy in range(yy_start,yy_end+1): 
    print('computing ',yy)
    time_start = time.time()
    cdo.remapbil(outputpath+'stereo_grid.nc', input = outputpath+'variables_of_interest_'+str(yy)+'_Ant.nc', output = outputpath+'variables_of_interest_'+str(yy)+'_Ant_stereo.nc')
    timelength = time.time() - time_start
    print(timelength)