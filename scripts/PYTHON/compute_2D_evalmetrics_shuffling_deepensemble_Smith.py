"""
Created on Mon Mar 27 17:37 2023

Compute evalmetrics after shuffling

Author: @claraburgard

"""

import numpy as np
import xarray as xr
import pandas as pd
import glob
import matplotlib as mpl
import seaborn as sns
import datetime
import time
import os,sys

import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout

from basal_melt_neural_networks.constants import *
import basal_melt_neural_networks.diagnostic_functions as diag
import basal_melt_neural_networks.data_formatting as dfmt
import basal_melt_neural_networks.postprocessing_functions as pp
from basal_melt_param.constants import *

#### INPUT STUFF

mod_size = str(sys.argv[1]) #'mini', 'xsmall96', 'small', 'medium', 'large', 'extra_large'
TS_opt = str(sys.argv[2]) # extrap, whole, thermocline
norm_method = str(sys.argv[3]) # std, interquart, minmax
exp_name = str(sys.argv[4])
nemo_run = str(sys.argv[5])
vv = str(sys.argv[6])

seedd = 1

##### READ IN DATA

inputpath_data_nn = '/bettik/burgardc/DATA/NN_PARAM/interim/INPUT_DATA/'

inputpath_CVinput = inputpath_data_nn+'EXTRAPOLATED_ISFDRAFT_CHUNKS/'
inputpath_csv = inputpath_data_nn+'SMITH_'+nemo_run+'_EXTRAPDRAFT_CHUNKS/'

if exp_name == 'onlyTSdraft':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in']
elif exp_name == 'TSdraftbotandiceddandwcd':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','water_col_depth','theta_bot','salinity_bot']
elif exp_name == 'TSdraftbotandiceddandwcdreldGL':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','water_col_depth','theta_bot','salinity_bot','rel_dGL']
elif exp_name == 'onlyTSdraftandslope':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','slope_ice_lon','slope_ice_lat']
elif exp_name == 'onlyTSdraft2':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in']
elif exp_name == 'TSTfdGLdIFwcd':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','dGL','dIF','slope_ice_lon','slope_ice_lat','water_col_depth']
elif exp_name == 'TSdraftslopereldGL':
    input_vars = ['corrected_isfdraft','theta_in','salinity_in','slope_ice_lon','slope_ice_lat','rel_dGL']
elif exp_name == 'allbutconstants':
    input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'isfdraft_conc','theta_in','salinity_in','u_tide']
elif exp_name == 'newbasic':
    input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'theta_in','salinity_in']
elif exp_name == 'newbasic2':
    input_vars = ['dGL','dIF','corrected_isfdraft','bathy_metry','slope_bed_lon','slope_bed_lat','slope_ice_lon','slope_ice_lat',
                'theta_in','salinity_in','T_mean', 'S_mean', 'T_std', 'S_std']

    
map_lim = [-3000000,3000000]
inputpath_data='/bettik/burgardc/DATA/NN_PARAM/interim/SMITH_'+nemo_run+'/'
file_other = xr.open_dataset(inputpath_data+'corrected_draft_bathy_isf.nc')#, chunks={'x': chunk_size, 'y': chunk_size})
file_other_cut = dfmt.cut_domain_stereo(file_other, map_lim, map_lim)
file_conc = xr.open_dataset(inputpath_data+'isfdraft_conc_Ant_stereo.nc')
file_conc_cut = dfmt.cut_domain_stereo(file_conc, map_lim, map_lim)

#continue with u_tide
outputpath_melt_nn = '/bettik/burgardc/DATA/NN_PARAM/processed/MELT_RATE/SMITH_'+nemo_run+'/'
path_model = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/experiments/WHOLE/'

startyy = 1980
endyy = 1980 + 60

if TS_opt == 'extrap_shuffboth':
    if nemo_run == 'bf663':
        df_shuffled = pd.read_csv(inputpath_data_nn+'SMITH_bi646_EXTRAPDRAFT_CHUNKS/dataframe_shuffledinput_allisf_'+str(startyy)+'-'+str(endyy)+'_bi646.csv')
    elif nemo_run == 'bi646':
        df_shuffled = pd.read_csv(inputpath_data_nn+'SMITH_bf663_EXTRAPDRAFT_CHUNKS/dataframe_shuffledinput_allisf_'+str(startyy)+'-'+str(endyy)+'_bf663.csv')
else:
    df_shuffled = pd.read_csv(inputpath_csv + 'dataframe_shuffledinput_allisf_'+str(startyy)+'-'+str(endyy)+'_'+nemo_run+'.csv')

res_1D_allyy_list = []
for tt in range(startyy,endyy):
    print(tt)
#for tt in tqdm([1970]):

    file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask = pp.read_input_evalmetrics_NN_yy(nemo_run, tt, file_conc_cut, file_other_cut)

    res_1D_list = []
    res_2D_all = None
    for kisf in file_isf.Nisf.values:  
    #for kisf in [66]:  


        norm_metrics_file = xr.open_dataset(inputpath_CVinput + 'metrics_norm_wholedataset.nc')
        #norm_metrics_file_addvar1 = xr.open_dataset(inputpath_CVinput + 'metrics_norm_addvar1_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
        #norm_metrics_file_addvar1 = norm_metrics_file_addvar1.drop('salinity_in')
        #norm_metrics_file = xr.merge([norm_metrics_file_orig,norm_metrics_file_addvar1])
        norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()

        df_nrun = pd.read_csv(inputpath_csv + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+str(tt)+'_'+nemo_run+'.csv',index_col=[0,1])
        #df_nrun_addvar1 = pd.read_csv(path_orig_data + 'dataframe_addvar1_isf'+str(kisf).zfill(3)+'_'+str(tblock_out).zfill(3)+'.csv',index_col=[0,1,2])
        #df_nrun_addvar1 = df_nrun_addvar1.drop(['salinity_in'], axis=1)
        #df_nrun = pd.concat([df_nrun_orig,df_nrun_addvar1],join = 'outer', axis = 1)

        nrows = len(df_nrun.index)
        if vv == 'watercolumn':
            shuffled_var = df_shuffled[['corrected_isfdraft', 'bathy_metry']].sample(n=nrows, random_state=kisf+seedd)
            df_nrun_in_shuffled = df_nrun.drop(['corrected_isfdraft', 'bathy_metry'], axis=1).copy()
            df_nrun_in_shuffled['corrected_isfdraft'] = shuffled_var['corrected_isfdraft'].values
            df_nrun_in_shuffled['bathy_metry'] = shuffled_var['bathy_metry'].values
        elif vv == 'position':
            shuffled_var = df_shuffled[['dGL', 'dIF']].sample(n=nrows, random_state=kisf+seedd)
            df_nrun_in_shuffled = df_nrun.drop(['dGL', 'dIF'], axis=1).copy()
            df_nrun_in_shuffled['dGL'] = shuffled_var['dGL'].values
            df_nrun_in_shuffled['dIF'] = shuffled_var['dIF'].values
        elif vv == 'slopesbed':
            shuffled_var = df_shuffled[['slope_bed_lon', 'slope_bed_lat']].sample(n=nrows, random_state=kisf+seedd)
            df_nrun_in_shuffled = df_nrun.drop(['slope_bed_lon', 'slope_bed_lat'], axis=1).copy()
            df_nrun_in_shuffled['slope_bed_lon'] = shuffled_var['slope_bed_lon'].values
            df_nrun_in_shuffled['slope_bed_lat'] = shuffled_var['slope_bed_lat'].values
        elif vv == 'slopesice':
            shuffled_var = df_shuffled[['slope_ice_lon', 'slope_ice_lat']].sample(n=nrows, random_state=kisf+seedd)
            df_nrun_in_shuffled = df_nrun.drop(['slope_ice_lon', 'slope_ice_lat'], axis=1).copy()
            df_nrun_in_shuffled['slope_ice_lon'] = shuffled_var['slope_ice_lon'].values
            df_nrun_in_shuffled['slope_ice_lat'] = shuffled_var['slope_ice_lat'].values
        elif vv == 'Tinfo':
            shuffled_var = df_shuffled[['theta_in','T_mean','T_std']].sample(n=nrows, random_state=kisf+seedd)
            df_nrun_in_shuffled = df_nrun.drop(['theta_in','T_mean','T_std'], axis=1).copy()
            df_nrun_in_shuffled['theta_in'] = shuffled_var['theta_in'].values
            df_nrun_in_shuffled['T_mean'] = shuffled_var['T_mean'].values
            df_nrun_in_shuffled['T_std'] = shuffled_var['T_std'].values
        elif vv == 'Sinfo':
            shuffled_var = df_shuffled[['salinity_in','S_mean','S_std']].sample(n=nrows, random_state=kisf+seedd)
            df_nrun_in_shuffled = df_nrun.drop(['salinity_in','S_mean','S_std'], axis=1).copy()
            df_nrun_in_shuffled['salinity_in'] = shuffled_var['salinity_in'].values
            df_nrun_in_shuffled['S_mean'] = shuffled_var['S_mean'].values
            df_nrun_in_shuffled['S_std'] = shuffled_var['S_std'].values
        else:
            shuffled_var = df_shuffled[vv].sample(n=nrows, random_state=kisf+seedd).values
            df_nrun_in_shuffled = df_nrun.drop(vv, axis=1).copy()
            df_nrun_in_shuffled[vv] = shuffled_var

        ens_res2D_list = []
        for seed_nb in range(1,11):
            if TS_opt == 'extrap_shuffboth':
                model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_'+exp_name+'_wholedataset_'+str(seed_nb).zfill(2)+'_TSextrap_norm'+norm_method+'.h5')
            else:
                model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_'+exp_name+'_wholedataset_'+str(seed_nb).zfill(2)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
                
            res_2D = pp.apply_NN_results_2D_1isf_1tblock(file_isf, norm_metrics, df_nrun_in_shuffled, model, input_vars)

            ens_res2D_list.append(res_2D['predicted_melt'].assign_coords({'seed_nb': seed_nb}))

        xr_ens_res2D = xr.concat(ens_res2D_list, dim='seed_nb')
        xr_ensmean_res2D = xr_ens_res2D.mean('seed_nb')

        if res_2D_all is None:
            res_2D_all = xr_ensmean_res2D
        else:
            res_2D_all = res_2D_all.combine_first(xr_ensmean_res2D)
        
        res_2D_all.to_netcdf(outputpath_melt_nn + 'evalmetrics_shuffled'+vv+'_2D_'+mod_size+'_'+exp_name+'_ensmean_'+TS_opt+'_norm'+norm_method+'_'+str(tt)+'_'+nemo_run+'.nc')
