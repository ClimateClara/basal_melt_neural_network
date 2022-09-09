import xarray as xr
import numpy as np
import pandas as pd
import basal_melt_neural_networks.data_formatting as dfmt


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