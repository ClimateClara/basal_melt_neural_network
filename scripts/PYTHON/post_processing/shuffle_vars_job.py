"""
Created on Wed Apr 15 09:53 2022

Apply shuffling of input variables for uncertainty quantification

Author: @claraburgard

"""

import numpy as np
import xarray as xr
import pandas as pd
from tqdm import trange, tqdm
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

########### READ IN DATA

inputpath_data = '/bettik/burgardc/DATA/NN_PARAM/interim/INPUT_DATA/'
outputpath_melt_nn = '/bettik/burgardc/DATA/NN_PARAM/processed/MELT_RATE/'
outputpath_nn_models = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'
outputpath_doc = '/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/custom_doc/'

nemo_run0 = 'OPM021' #['OPM006', 'OPM016', 'OPM018', 'OPM021', 'OPM026', 'OPM027', 'OPM031-1', OPM031-2']
if nemo_run0 in ['OPM031-1','OPM031-2']:
    nemo_run = 'OPM031'
else:
    nemo_run = nemo_run0
    
inputpath_mask = '/bettik/burgardc/SCRIPTS/basal_melt_param/data/interim/ANTARCTICA_IS_MASKS/nemo_5km_'+nemo_run+'/'

file_isf = xr.open_dataset(inputpath_mask+'nemo_5km_isf_masks_and_info_and_distance_new.nc')

########### SET OPTIONS

isf_list = [10, 11, 12, 13, 18, 21, 22, 23, 24, 25, 30, 31, 33, 38, 39, 
            40, 42, 43, 44, 45, 47, 48, 51, 52, 53, 54, 55, 58, 61, 65, 
            66, 69, 70, 71, 73, 75]

#var_list = ['dGL', 'dIF', 'corrected_isfdraft', 'bathy_metry', 'slope_bed_lon',
#       'slope_bed_lat', 'slope_ice_lon', 'slope_ice_lat', 'isf_area',
#       'entry_depth_max', 'isfdraft_conc', 'u_tide', 'T_001', 'T_002',
#       'T_003', 'T_004', 'T_005', 'T_006', 'T_007', 'T_008', 'T_009',
#       'T_010', 'T_011', 'T_012', 'T_013', 'T_014', 'T_015', 'T_016',
#       'T_017', 'T_018', 'T_019', 'T_020', 'T_021', 'T_022', 'T_023',
#       'T_024', 'T_025', 'T_026', 'T_027', 'T_028', 'T_029', 'T_030',
#       'T_031', 'T_032', 'T_033', 'T_034', 'T_035', 'T_036', 'T_037',
#       'T_038', 'T_039', 'T_040', 'T_041', 'T_042', 'T_043', 'T_044',
#       'T_045', 'T_046', 'T_047', 'T_048', 'T_049', 'T_050', 'T_051',
#       'T_052', 'T_053', 'T_054', 'T_055', 'T_056', 'T_057', 'T_058',
#       'T_059', 'T_060', 'T_061', 'T_062', 'T_063', 'T_064', 'T_065',
#       'T_066', 'T_067', 'T_068', 'S_001', 'S_002', 'S_003', 'S_004',
#       'S_005', 'S_006', 'S_007', 'S_008', 'S_009', 'S_010', 'S_011',
#       'S_012', 'S_013', 'S_014', 'S_015', 'S_016', 'S_017', 'S_018',
#       'S_019', 'S_020', 'S_021', 'S_022', 'S_023', 'S_024', 'S_025',
#       'S_026', 'S_027', 'S_028', 'S_029', 'S_030', 'S_031', 'S_032',
#       'S_033', 'S_034', 'S_035', 'S_036', 'S_037', 'S_038', 'S_039',
#       'S_040', 'S_041', 'S_042', 'S_043', 'S_044', 'S_045', 'S_046',
#       'S_047', 'S_048', 'S_049', 'S_050', 'S_051', 'S_052', 'S_053',
#       'S_054', 'S_055', 'S_056', 'S_057', 'S_058', 'S_059', 'S_060',
#       'S_061', 'S_062', 'S_063', 'S_064', 'S_065', 'S_066', 'S_067',
#       'S_068', 'water_column']

var_list = ['corrected_isfdraft', 'bathy_metry', 'slope_bed_lon',
       'slope_bed_lat', 'slope_ice_lon', 'slope_ice_lat', 'isf_area',
       'entry_depth_max', 'isfdraft_conc', 'u_tide', 'T_001', 'T_002',
       'T_003', 'T_004', 'T_005', 'T_006', 'T_007', 'T_008', 'T_009',
       'T_010', 'T_011', 'T_012', 'T_013', 'T_014', 'T_015', 'T_016',
       'T_017', 'T_018', 'T_019', 'T_020', 'T_021', 'T_022', 'T_023',
       'T_024', 'T_025', 'T_026', 'T_027', 'T_028', 'T_029', 'T_030',
       'T_031', 'T_032', 'T_033', 'T_034', 'T_035', 'T_036', 'T_037',
       'T_038', 'T_039', 'T_040', 'T_041', 'T_042', 'T_043', 'T_044',
       'T_045', 'T_046', 'T_047', 'T_048', 'T_049', 'T_050', 'T_051',
       'T_052', 'T_053', 'T_054', 'T_055', 'T_056', 'T_057', 'T_058',
       'T_059', 'T_060', 'T_061', 'T_062', 'T_063', 'T_064', 'T_065',
       'T_066', 'T_067', 'T_068', 'S_001', 'S_002', 'S_003', 'S_004',
       'S_005', 'S_006', 'S_007', 'S_008', 'S_009', 'S_010', 'S_011',
       'S_012', 'S_013', 'S_014', 'S_015', 'S_016', 'S_017', 'S_018',
       'S_019', 'S_020', 'S_021', 'S_022', 'S_023', 'S_024', 'S_025',
       'S_026', 'S_027', 'S_028', 'S_029', 'S_030', 'S_031', 'S_032',
       'S_033', 'S_034', 'S_035', 'S_036', 'S_037', 'S_038', 'S_039',
       'S_040', 'S_041', 'S_042', 'S_043', 'S_044', 'S_045', 'S_046',
       'S_047', 'S_048', 'S_049', 'S_050', 'S_051', 'S_052', 'S_053',
       'S_054', 'S_055', 'S_056', 'S_057', 'S_058', 'S_059', 'S_060',
       'S_061', 'S_062', 'S_063', 'S_064', 'S_065', 'S_066', 'S_067',
       'S_068', 'water_column']

########### SHUFFLE VARIABLES

for shuff_var in var_list:
    print(shuff_var)

    for timetag in ['20220427-1051']:

        new_path_output = outputpath_melt_nn+timetag+'/'
        if not os.path.isdir(new_path_output):
            print("I did not find this folder "+timetag+") so I created a new one, I hope that's ok!")
            os.mkdir(new_path_output)
        else:
            print("This folder ("+timetag+") exists already!")

        new_path_model = outputpath_nn_models+timetag+'/'
        if not os.path.isdir(new_path_model):
            print("I did not find this folder ("+timetag+") in model folder so I created a new one, I hope that's ok!")

        print(timetag)
        if timetag in ['20220427-0957','20220427-1052','20220427-1058','20220427-1059']:
            timetag_data = '20220427-0957'
            path_data = inputpath_data+'EXTRAPOLATED_ISFDRAFT/'

        elif timetag in ['20220427-1002','20220427-1021','20220427-1042','20220427-1051']:
            timetag_data = '20220427-1002'
            path_data = inputpath_data+'WHOLE_PROF/'


        norm_data_path = outputpath_nn_models+timetag_data+'/'

        normalisation_coeff = xr.open_dataset(norm_data_path+ 'dataset_norm_training_factors_'+timetag_data+'.nc')
        model = keras.models.load_model(new_path_model + 'model_nn_'+timetag+'.h5')

        y_all_isf = None

        for kisf in isf_list: 

            ### READ DATA
            df_nrun = pd.read_csv(path_data + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+nemo_run0+'.csv',index_col=[0,1,2])
            clean_df_nrun_kisf = pd.read_csv(path_data + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+nemo_run0+'.csv',index_col=[0,1,2])
            clean_df_nrun_kisf.reset_index(drop=True, inplace=True)

            new_df = clean_df_nrun_kisf
            if shuff_var == 'water_column':
                shuffled_df = new_df[['corrected_isfdraft', 'bathy_metry']].copy().sample(frac=1, random_state=1)
                new_df['corrected_isfdraft'] = shuffled_df['corrected_isfdraft'].values
                new_df['bathy_metry'] = shuffled_df['bathy_metry'].values
            else:
                shuffled_df = new_df[shuff_var].copy().sample(frac=1, random_state=1)
                new_df[shuff_var] = shuffled_df.values

            new_df = new_df.to_xarray()

            normalisation_coeff_input = normalisation_coeff.drop_vars(['melt_m_ice_per_y'])
            normalised_vars = (new_df.drop_vars(['melt_m_ice_per_y']) - normalisation_coeff_input.sel(metric='mean_vars'))/normalisation_coeff_input.sel(metric='std_vars')

            input_var = normalised_vars.to_array().load()
            ref_melt = new_df['melt_m_ice_per_y'].load()

            ### RUN THE MODEL
            y_out_norm = model.predict(input_var.T.values)
            y_out_norm_xr = xr.DataArray(data=y_out_norm.squeeze()).rename({'dim_0': 'index'})
            y_out_norm_xr = y_out_norm_xr.assign_coords({'index': input_var.index})
            y_out = (y_out_norm_xr * normalisation_coeff['melt_m_ice_per_y'].sel(metric='std_vars')) + normalisation_coeff['melt_m_ice_per_y'].sel(metric='mean_vars')

            y_out_pd_s = pd.Series(y_out.values,index=df_nrun.index,name='predicted_melt') 
            y_target_pd_s = pd.Series(ref_melt.values,index=df_nrun.index,name='reference_melt') 

            ### PUT SOME ORDER IN THE FILE
            y_out_xr = y_out_pd_s.to_xarray()
            y_target_xr = y_target_pd_s.to_xarray()
            y_to_compare = xr.merge([y_out_xr.T, y_target_xr.T]).sortby('y')

            y_whole_grid = y_to_compare.reindex_like(file_isf['ISF_mask'])
            if y_all_isf is None:
                y_all_isf = y_whole_grid
            else:
                y_all_isf = y_all_isf.combine_first(y_whole_grid)

        y_all_isf.to_netcdf(new_path_output+'NN_melt_predicted_reference_m_ice_per_yr_'+nemo_run0+'_shuffled'+shuff_var+'.nc')    