"""
Created on Wed Mar 22 11:13 2023

This script is to prepare the normalising coefficients and input data for the cross-validation parallely

Author: Clara Burgard
"""
import numpy as np
import xarray as xr
#from tqdm.notebook import trange, tqdm
from tqdm import trange, tqdm
import glob
import matplotlib as mpl
import seaborn as sns
import datetime
import time

from basal_melt_neural_networks.constants import *
import basal_melt_neural_networks.diagnostic_functions as diag
import basal_melt_neural_networks.data_formatting as dfmt
import basal_melt_neural_networks.prep_input_data as indat

import sys

######### READ IN OPTIONS

tblock_out = int(sys.argv[1])
isf_out = int(sys.argv[2])
TS_opt = str(sys.argv[3]) # extrap, whole, thermocline

tblock_dim = range(1,14)
isf_dim = [10,11,12,13,18,22,23,24,25,30,31,33,38,39,40,42,43,44,45,47,48,51,52,53,54,55,58,61,65,66,69,70,71,73,75]

inputpath_data = '/bettik/burgardc/DATA/NN_PARAM/interim/INPUT_DATA/' 

if TS_opt == 'extrap':
    outputpath_CVinput = inputpath_data+'EXTRAPOLATED_ISFDRAFT_CHUNKS_CV/'
elif TS_opt == 'whole':
    outputpath_CVinput = inputpath_data+'WHOLE_PROF_CHUNKS_CV/'
elif TS_opt == 'thermocline':
    outputpath_CVinput = inputpath_data+'THERMOCLINE_CHUNKS_CV/'

metrics_ds, var_train_norm, var_val_norm = indat.prepare_input_data_CV(tblock_dim, isf_dim, tblock_out, isf_out, TS_opt, inputpath_data)

metrics_ds.to_netcdf(outputpath_CVinput + 'metrics_norm_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')

var_train_norm.to_netcdf(outputpath_CVinput + 'train_data_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')

var_val_norm.to_netcdf(outputpath_CVinput + 'val_data_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')  