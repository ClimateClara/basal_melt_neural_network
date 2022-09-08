"""
This functions are for the normalisation and preparation of the ready-to-use input data
"""

from tqdm.notebook import trange, tqdm
import pandas as pd
import xarray as xr
import numpy as np

def compute_norm_metrics(x_train, y_train, norm_method):
    
    """
    Computes the values to use for the normalisation of the input data.
        
    Parameters
    ----------
    x_train : pd.DataFrame or xr.DataArray or xr.Dataset
        Training values for predictors.
    y_train : pd.DataFrame or xr.DataArray or xr.Dataset
        Training values for target(s).
    norm_method : str
        The normalisation method. Can be 'std' (divide by standard deviation), 'interquart' (divide by range between 10th and 90th percentile), 'minmax' (divide by range between minimum and maximum)

    Returns
    -------
    summary_metrics: xr.Dataset
        Dataset containing mean and denominator of the normalisation.
    """
    
    
    x_mean = x_train.mean()
    y_mean = y_train.mean()
    
    if norm_method == 'std':
        x_range  = x_train.std()
        y_range  = y_train.std()
    elif norm_method == 'interquart':
        x_range  = x_train.quantile(0.9) - x_train.quantile(0.1)
        y_range  = y_train.quantile(0.9) - y_train.quantile(0.1)
    elif norm_method == 'minmax':
        x_range  = x_train.max() - x_train.min() 
        y_range  = y_train.max() - y_train.min() 
    
    norm_mean = xr.merge([x_mean,y_mean]).assign_coords({'metric': 'mean_vars', 'norm_method': norm_method})
    norm_range = xr.merge([x_range,y_range]).assign_coords({'metric': 'range_vars', 'norm_method': norm_method})
    
    summary_metrics = xr.concat([norm_mean, norm_range], dim='metric').assign_coords({'norm_method': norm_method})
    
    return summary_metrics

def prepare_input_data_CV(tblock_dim, isf_dim, tblock_out, isf_out, TS_opt, inputpath_data):

    """
    Computes normalisation metrics and normalised predictors and target for cross-validation.
    
    Parameters
    ----------
    tblock_dim : list
        List of all time blocks to conduct the cross-validation on.
    isf_dim : list
        List of all ice shelves to conduct the cross-validation on.
    tblock_out : int
        Time block to leave out in cross-validation.
    isf_out : list
        Ice shelf to leave out in cross-validation.
    TS_opt : str
        Type of input temperature and salinity profiles to use. Can be 'extrap', 'whole', 'thermocline'
    inputpath_data : str
        Path to folder where to find the preformatted csv files.

    Returns
    -------
    summary_ds_all: xr.Dataset
        Dataset containing mean and denominator of the normalisation.
    var_train_norm: xr.Dataset
        Dataset containing normalised training predictors and target.
    var_val_norm: xr.Dataset
        Dataset containing normalised validation predictors and target.
    """

    ## are we dealing with leave-one-out cross-validation over time blocks?
    tblock_list = list(tblock_dim)
    if tblock_out > 0:
        tblock_list.remove(tblock_out)
    #print(tblock_list)

    ## are we dealing with leave-one-out cross-validation over ice shelves?
    isf_list = list(isf_dim)
    if isf_out > 0:
        isf_list.remove(isf_out)
    
    ## which profile option are we using for temperature and salinity
    if TS_opt == 'extrap':
        inputpath_prof = inputpath_data+'EXTRAPOLATED_ISFDRAFT_CHUNKS/'
    elif TS_opt == 'whole':
        inputpath_prof = inputpath_data+'WHOLE_PROF_CHUNKS/'
    elif TS_opt == 'thermocline':
        inputpath_prof = inputpath_data+'THERMOCLINE_CHUNKS/'

    ### prepare training dataset

    train_input_df = None        

    for tt in tblock_list:

        for kisf in isf_list: 

            clean_df_nrun_kisf = pd.read_csv(inputpath_prof + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+str(tt).zfill(3)+'.csv',index_col=[0,1,2])
            clean_df_nrun_kisf.reset_index(drop=True, inplace=True)
            clean_ds_nrun_kisf = clean_df_nrun_kisf.to_xarray()

            if train_input_df is None:
                train_input_df = clean_ds_nrun_kisf.copy()
            else:
                new_index = clean_ds_nrun_kisf.index.values + train_input_df.index.max().values+1
                clean_ds_nrun_kisf = clean_ds_nrun_kisf.assign_coords({'index': new_index})
                train_input_df = xr.concat([train_input_df, clean_ds_nrun_kisf], dim='index') 

    ## prepare validation dataset

    if (tblock_out > 0) and (isf_out == 0):  
        tt_val = [tblock_out]
        isf_val = isf_list
    elif (isf_out > 0) and (tblock_out == 0):
        isf_val = [isf_out]
        tt_val = tblock_list
    else:
        print("I don't know how to handle leave ice shelves AND time blocks out, please teach me!")

    val_input_df = None        

    for tt in tt_val:

        for kisf in isf_val: 

            clean_df_nrun_kisf = pd.read_csv(inputpath_prof + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+str(tt).zfill(3)+'.csv',index_col=[0,1,2])
            clean_df_nrun_kisf.reset_index(drop=True, inplace=True)
            clean_ds_nrun_kisf = clean_df_nrun_kisf.to_xarray()

            if val_input_df is None:
                val_input_df = clean_ds_nrun_kisf.copy()
            else:
                new_index = clean_ds_nrun_kisf.index.values + val_input_df.index.max().values+1
                clean_ds_nrun_kisf = clean_ds_nrun_kisf.assign_coords({'index': new_index})
                val_input_df = xr.concat([val_input_df, clean_ds_nrun_kisf], dim='index') 

    ## prepare input and target

    y_train = train_input_df['melt_m_ice_per_y']
    x_train = train_input_df.drop_vars(['melt_m_ice_per_y'])

    y_val = val_input_df['melt_m_ice_per_y']
    x_val = val_input_df.drop_vars(['melt_m_ice_per_y'])

    #print('x_train : ',dfmt.print_shape_xr_ds(x_train), 'y_train : ',len(y_train))
    #print('x_val  : ',dfmt.print_shape_xr_ds(x_val),  'y_test  : ',len(y_val))

    ## normalise
    norm_summary_list = []

    for norm_method in ['std','interquart','minmax']:

        summary_ds = compute_norm_metrics(x_train, y_train, norm_method)
        norm_summary_list.append(summary_ds)

    summary_ds_all = xr.concat(norm_summary_list, dim='norm_method')
    
    var_mean = summary_ds_all.sel(metric='mean_vars')
    var_range = summary_ds_all.sel(metric='range_vars')

    var_train_norm = (train_input_df - var_mean)/var_range
    var_val_norm = (val_input_df - var_mean)/var_range
    
    return summary_ds_all, var_train_norm, var_val_norm