import xarray as xr
import numpy as np
import pandas as pd
import basal_melt_neural_networks.data_formatting as dfmt
from tqdm.notebook import tqdm
from tensorflow import keras
from basal_melt_neural_networks.constants import *


def identify_nemo_run_from_tblock(tblock):
    if tblock in [1,2,3,4]:
        nemo_run = 'OPM006'
    elif tblock in [5,6,7]:
        nemo_run = 'OPM016'
    elif tblock in [8,9,10]:
        nemo_run = 'OPM018'
    elif tblock in [11,12,13]:
        nemo_run = 'OPM021'
    return nemo_run

def denormalise_vars(var_norm, mean, std):
    var = (var_norm * std) + mean
    return var

def normalise_vars(var, mean, std):
    var_norm = (var - mean) / std
    return var_norm

def read_input_evalmetrics_NN(nemo_run):
    inputpath_boxes = '/bettik/burgardc/DATA/BASAL_MELT_PARAM/interim/BOXES/nemo_5km_'+nemo_run+'/'
    inputpath_data='/bettik/burgardc/DATA/BASAL_MELT_PARAM/interim/NEMO_eORCA025.L121_'+nemo_run+'_ANT_STEREO/'
    inputpath_mask = '/bettik/burgardc/SCRIPTS/basal_melt_param/data/interim/ANTARCTICA_IS_MASKS/nemo_5km_'+nemo_run+'/'

    file_isf_orig = xr.open_dataset(inputpath_mask+'nemo_5km_isf_masks_and_info_and_distance_new_oneFRIS.nc')
    nonnan_Nisf = file_isf_orig['Nisf'].where(np.isfinite(file_isf_orig['front_bot_depth_max']), drop=True).astype(int)
    file_isf_nonnan = file_isf_orig.sel(Nisf=nonnan_Nisf)
    large_isf = file_isf_nonnan['Nisf'].where(file_isf_nonnan['isf_area_here'] >= 2500, drop=True)
    file_isf = file_isf_nonnan.sel(Nisf=large_isf)
    
    map_lim = [-3000000,3000000]
    file_other = xr.open_dataset(inputpath_data+'corrected_draft_bathy_isf.nc')#, chunks={'x': chunk_size, 'y': chunk_size})
    file_other_cut = dfmt.cut_domain_stereo(file_other, map_lim, map_lim)
    file_conc = xr.open_dataset(inputpath_data+'isfdraft_conc_Ant_stereo.nc')
    file_conc_cut = dfmt.cut_domain_stereo(file_conc, map_lim, map_lim)
    
    ice_draft_pos = file_other_cut['corrected_isfdraft']
    ice_draft_neg = -ice_draft_pos
    
    box_charac_2D = xr.open_dataset(inputpath_boxes + 'nemo_5km_boxes_2D_oneFRIS.nc')
    box_charac_1D = xr.open_dataset(inputpath_boxes + 'nemo_5km_boxes_1D_oneFRIS.nc')
    
    isf_stack_mask = dfmt.create_stacked_mask(file_isf['ISF_mask'], file_isf.Nisf, ['y','x'], 'mask_coord')

    file_isf_conc = file_conc_cut['isfdraft_conc']

    xx = file_isf.x
    yy = file_isf.y
    dx = (xx[2] - xx[1]).values
    dy = (yy[2] - yy[1]).values
    grid_cell_area = abs(dx*dy)  
    grid_cell_area_weighted = file_isf_conc * grid_cell_area
    
    geometry_info_2D = xr.merge([ice_draft_pos.rename('ice_draft_pos'),
                            grid_cell_area_weighted.rename('grid_cell_area_weighted'),
                            file_isf_conc])
    
    return file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask

def apply_NN_results_2D_1isf_1tblock(file_isf, norm_metrics, df_nrun, model, input_vars=[]):
    """
    Compute 2D melt based on a given NN model
    
    """

    val_norm = normalise_vars(df_nrun,
                                norm_metrics.loc['mean_vars'],
                                norm_metrics.loc['range_vars'])

    x_val_norm = val_norm[input_vars]
    y_val_norm = val_norm['melt_m_ice_per_y']

    y_out_norm = model.predict(x_val_norm.values,verbose = 0)

    y_out_norm_xr = xr.DataArray(data=y_out_norm.squeeze()).rename({'dim_0': 'index'})
    y_out_norm_xr = y_out_norm_xr.assign_coords({'index': x_val_norm.index})

    # denormalise the output
    y_out = denormalise_vars(y_out_norm_xr, 
                             norm_metrics['melt_m_ice_per_y'].loc['mean_vars'],
                             norm_metrics['melt_m_ice_per_y'].loc['range_vars'])

    y_out_pd_s = pd.Series(y_out.values,index=df_nrun.index,name='predicted_melt') 
    y_target_pd_s = pd.Series(df_nrun['melt_m_ice_per_y'].values,index=df_nrun.index,name='reference_melt') 

    # put some order in the file
    y_out_xr = y_out_pd_s.to_xarray()
    y_target_xr = y_target_pd_s.to_xarray()
    y_to_compare = xr.merge([y_out_xr, y_target_xr]).sortby('y')

    y_whole_grid = y_to_compare.reindex_like(file_isf['ISF_mask'])
    return y_whole_grid

    
def evalmetrics_1D_NN(kisf, norm_metrics, df_nrun, model, file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask, input_vars=[]):
    
    """
    Compute 1D metrics based on a given NN model
    
    """
    
    melt2D = apply_NN_results_2D_1isf_1tblock(file_isf, norm_metrics, df_nrun, model, input_vars)

    if box_charac_2D and box_charac_1D:
        box_loc_config2 = box_charac_2D['box_location'].sel(box_nb_tot=box_charac_1D['nD_config'].sel(config=2))
        box1 = box_loc_config2.where(box_loc_config2==1).isel(Nisf=0).drop('Nisf')

    geometry_isf_2D = dfmt.choose_isf(geometry_info_2D,isf_stack_mask, kisf)
    melt_rate_2D_isf_m_per_y = dfmt.choose_isf(melt2D,isf_stack_mask, kisf)

    melt_rate_1D_isf_Gt_per_y = (melt_rate_2D_isf_m_per_y * geometry_isf_2D['grid_cell_area_weighted']).sum(dim=['mask_coord']) * rho_i / 10**12

    box_loc_config_stacked = dfmt.choose_isf(box1, isf_stack_mask, kisf)
    param_melt_2D_box1_isf = melt_rate_2D_isf_m_per_y.where(np.isfinite(box_loc_config_stacked))

    melt_rate_1D_isf_myr_box1_mean = dfmt.weighted_mean(param_melt_2D_box1_isf,['mask_coord'], geometry_isf_2D['isfdraft_conc'])     

    out_1D = xr.concat([melt_rate_1D_isf_Gt_per_y, melt_rate_1D_isf_myr_box1_mean], dim='metrics').assign_coords({'metrics': ['Gt','box1']})
    return out_1D

def compute_crossval_metric_1D_for_1CV(tblock_out,isf_out,tblock_dim,isf_dim,inputpath_CVinput,path_orig_data,norm_method,TS_opt,mod_size,path_model_end,input_vars=[],verbose=True):

    """
    Compute 1D metrics based on a given NN model for a given cross-validation axis
    
    """
    
    if ((isf_out > 0) and (tblock_out == 0)):
        path_model = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'+path_model_end+'CV_ISF/'
        tblock_list = tblock_dim
    elif ((tblock_out > 0) and (isf_out == 0)):     
        path_model = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'+path_model_end+'CV_TBLOCK/'
        isf_list = isf_dim
    else:
        raise ValueError("HELP, I DON'T KNOW HOW TO HANDLE AN ISF AND A TBLOCK OUT...")

    res_1D_list = []
    
    # CV over shelves
    if ((isf_out > 0) and (tblock_out == 0)):
        
        if verbose:
            loop_list = tqdm(tblock_list)
            print("Ok, we're doing cross-validation over ice shelves")
        else:
            loop_list = tblock_list
            
        for tblock in loop_list:
            
            nemo_run = identify_nemo_run_from_tblock(tblock)
            file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask = read_input_evalmetrics_NN(nemo_run)
            
            norm_metrics_file_orig = xr.open_dataset(inputpath_CVinput + 'metrics_norm_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = xr.open_dataset(inputpath_CVinput + 'metrics_norm_addvar1_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = norm_metrics_file_addvar1.drop('salinity_in')
            norm_metrics_file = xr.merge([norm_metrics_file_orig,norm_metrics_file_addvar1])
            norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()
            
            df_nrun_orig = pd.read_csv(path_orig_data + 'dataframe_input_isf'+str(isf_out).zfill(3)+'_'+str(tblock).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_orig = df_nrun_orig.reorder_levels(['y', 'x', 'time'])
            
            df_nrun_addvar1 = pd.read_csv(path_orig_data + 'dataframe_addvar1_isf'+str(isf_out).zfill(3)+'_'+str(tblock).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_addvar1 = df_nrun_addvar1.drop(['salinity_in'], axis=1)
            df_nrun_addvar1 = df_nrun_addvar1.reorder_levels(['y', 'x', 'time'])

            df_nrun = pd.concat([df_nrun_orig,df_nrun_addvar1],join = 'outer', axis = 1)

            model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
            
            res_1D = evalmetrics_1D_NN(isf_out, norm_metrics, df_nrun, model, file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask, input_vars)
            res_1D_list.append(res_1D)
        
        res_1D_all = xr.concat(res_1D_list, dim='time')
    
    # CV OVER TIME
    elif ((tblock_out > 0) and (isf_out == 0)):
        
        nemo_run = identify_nemo_run_from_tblock(tblock_out)
        file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask = read_input_evalmetrics_NN(nemo_run)

        if verbose:
            loop_list = tqdm(isf_list)
            print("Ok, we're doing cross-validation over time")
        else:
            loop_list = isf_list
        
        for kisf in loop_list: 
            
            norm_metrics_file_orig = xr.open_dataset(inputpath_CVinput + 'metrics_norm_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = xr.open_dataset(inputpath_CVinput + 'metrics_norm_addvar1_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = norm_metrics_file_addvar1.drop('salinity_in')
            norm_metrics_file = xr.merge([norm_metrics_file_orig,norm_metrics_file_addvar1])
            norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()
            
            df_nrun_orig = pd.read_csv(path_orig_data + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+str(tblock_out).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_orig = df_nrun_orig.reorder_levels(['y', 'x', 'time'])

            df_nrun_addvar1 = pd.read_csv(path_orig_data + 'dataframe_addvar1_isf'+str(kisf).zfill(3)+'_'+str(tblock_out).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_addvar1 = df_nrun_addvar1.drop(['salinity_in'], axis=1)
            df_nrun_addvar1 = df_nrun_addvar1.reorder_levels(['y', 'x', 'time'])
            
            df_nrun = pd.concat([df_nrun_orig,df_nrun_addvar1],join = 'outer', axis = 1)

            model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
            
            res_1D = evalmetrics_1D_NN(kisf, norm_metrics, df_nrun, model, file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask, input_vars)    
            res_1D_list.append(res_1D)
        
        res_1D_all = xr.concat(res_1D_list, dim='Nisf')
            
    return res_1D_all

def compute_crossval_metric_2D_for_1CV(tblock_out,isf_out,tblock_dim,isf_dim,inputpath_CVinput,path_orig_data,norm_method,TS_opt,mod_size,path_model_end,input_vars=[],verbose=True):

    """
    Compute 2D metrics based on a given NN model for a given cross-validation axis
    
    """
    
    if ((isf_out > 0) and (tblock_out == 0)):
        path_model = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'+path_model_end+'CV_ISF/'
        tblock_list = tblock_dim
    elif ((tblock_out > 0) and (isf_out == 0)):     
        path_model = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'+path_model_end+'CV_TBLOCK/'
        isf_list = isf_dim
    else:
        raise ValueError("HELP, I DON'T KNOW HOW TO HANDLE AN ISF AND A TBLOCK OUT...")

    res_2D_list = []
    
    # CV over shelves
    if ((isf_out > 0) and (tblock_out == 0)):
        
        if verbose:
            loop_list = tqdm(tblock_list)
            print("Ok, we're doing cross-validation over ice shelves")
        else:
            loop_list = tblock_list
            
        for tblock in loop_list:
            
            nemo_run = identify_nemo_run_from_tblock(tblock)
            file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask = read_input_evalmetrics_NN(nemo_run)
            
            norm_metrics_file_orig = xr.open_dataset(inputpath_CVinput + 'metrics_norm_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = xr.open_dataset(inputpath_CVinput + 'metrics_norm_addvar1_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = norm_metrics_file_addvar1.drop('salinity_in')
            norm_metrics_file = xr.merge([norm_metrics_file_orig,norm_metrics_file_addvar1])
            norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()
            
            df_nrun_orig = pd.read_csv(path_orig_data + 'dataframe_input_isf'+str(isf_out).zfill(3)+'_'+str(tblock).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_orig = df_nrun_orig.reorder_levels(['y', 'x', 'time'])

            df_nrun_addvar1 = pd.read_csv(path_orig_data + 'dataframe_addvar1_isf'+str(isf_out).zfill(3)+'_'+str(tblock).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_addvar1 = df_nrun_addvar1.reorder_levels(['y', 'x', 'time'])

            df_nrun_addvar1 = df_nrun_addvar1.drop(['salinity_in'], axis=1)
            df_nrun = pd.concat([df_nrun_orig,df_nrun_addvar1],join = 'outer', axis = 1)

            model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
            
            res_2D = apply_NN_results_2D_1isf_1tblock(file_isf, norm_metrics, df_nrun, model, input_vars)
            res_2D_list.append(res_2D)
            
        res_2D_all = xr.concat(res_2D_list, dim='time')
    
    # CV OVER TIME
    elif ((tblock_out > 0) and (isf_out == 0)):
        
        nemo_run = identify_nemo_run_from_tblock(tblock_out)
        file_isf, geometry_info_2D, box_charac_2D, box_charac_1D, isf_stack_mask = read_input_evalmetrics_NN(nemo_run)

        if verbose:
            loop_list = tqdm(isf_list)
            print("Ok, we're doing cross-validation over time")
        else:
            loop_list = isf_list
        
        res_2D_all = None
        for kisf in loop_list: 
            
            norm_metrics_file_orig = xr.open_dataset(inputpath_CVinput + 'metrics_norm_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = xr.open_dataset(inputpath_CVinput + 'metrics_norm_addvar1_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
            norm_metrics_file_addvar1 = norm_metrics_file_addvar1.drop('salinity_in')
            norm_metrics_file = xr.merge([norm_metrics_file_orig,norm_metrics_file_addvar1])
            norm_metrics = norm_metrics_file.sel(norm_method=norm_method).drop('norm_method').to_dataframe()
            
            df_nrun_orig = pd.read_csv(path_orig_data + 'dataframe_input_isf'+str(kisf).zfill(3)+'_'+str(tblock_out).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_orig = df_nrun_orig.reorder_levels(['y', 'x', 'time'])

            df_nrun_addvar1 = pd.read_csv(path_orig_data + 'dataframe_addvar1_isf'+str(kisf).zfill(3)+'_'+str(tblock_out).zfill(3)+'.csv',index_col=[0,1,2])
            df_nrun_addvar1 = df_nrun_addvar1.reorder_levels(['y', 'x', 'time'])

            df_nrun_addvar1 = df_nrun_addvar1.drop(['salinity_in'], axis=1)
            df_nrun = pd.concat([df_nrun_orig,df_nrun_addvar1],join = 'outer', axis = 1)

            model = keras.models.load_model(path_model + 'model_nn_'+mod_size+'_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'_TS'+TS_opt+'_norm'+norm_method+'.h5')
            
            res_2D = apply_NN_results_2D_1isf_1tblock(file_isf, norm_metrics, df_nrun, model, input_vars)
            
            if res_2D_all is None:
                res_2D_all = res_2D
            else:
                res_2D_all = res_2D_all.combine_first(res_2D)
            
    return res_2D_all