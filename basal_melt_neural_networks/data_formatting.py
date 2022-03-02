import pandas as pd

# Formatting the data

# the domain stuff
def cut_domain_stereo(var_to_cut, map_lim_x, map_lim_y):
    var_cutted = var_to_cut.sel(x=var_to_cut.x.where(in_range(var_to_cut.x,map_lim_x),drop=True), y=var_to_cut.y.where(in_range(var_to_cut.y,map_lim_y),drop=True))
    return var_cutted

def in_range(in_xy,txy):
    return ((in_xy >= min(txy)) & (in_xy < max(txy)))

def prepare_input_df_1_year(TS_prof_isf, melt_rate_isf, yy, max_front_depth, geometry_2D_isf, only_1_yr_tot=False):

    # take profile of T and S for that ice shelf and only as deep as the deepest entrance point
    TS_isf_tt = TS_prof_isf.sel(time=yy).where(TS_prof_isf.depth < max_front_depth, drop=True).drop('profile_domain').drop('Nisf')
    if only_1_yr_tot:
        TS_isf_tt = TS_isf_tt.drop('time')
    # T and S profiles to dataframe
    TS_isf_df = TS_isf_tt.to_dataframe()
    
    # Geometry info to dataframe
    length_df = len(geometry_2D_isf.x)*len(geometry_2D_isf.y)
    geo_df = geometry_2D_isf.to_dataframe()
    
    # Write the T and S profiles in different columns for this year (all rows the same)
    
    T_list = [ ]
    S_list = [ ]
    depth_list = [ ]
    for ii in range(len(TS_isf_tt.depth)):
        T_list.append('T_'+str(ii).zfill(3))
        S_list.append('S_'+str(ii).zfill(3))
        depth_list.append('d_'+str(ii).zfill(3))
    
    for nn in range(length_df):
        for ii,icol in enumerate(T_list):
            geo_df[icol] = TS_isf_df['theta_ocean'].values[ii]
        for ii,icol in enumerate(S_list):    
            geo_df[icol] = TS_isf_df['salinity_ocean'].values[ii]
    
    # write out melt rate for that year (target)
    melt_rate_isf_tt = melt_rate_isf.sel(time=yy)
    # transform to dataframe
    melt_df = melt_rate_isf_tt.drop('longitude').drop('latitude').to_dataframe()#.drop(['mapping'],axis=1)#.reset_index()
    if only_1_yr_tot:
        melt_df = melt_df.drop(['time'],axis=1)
    
    # merge properties and melt
    merged_df = pd.merge(geo_df,melt_df,how='left',on=['x','y'])

    # remove rows where there are nans
    clean_df = merged_df.dropna()
    # remove the x and y axis
    # clean_df = clean_df.drop(['x'], axis=1).drop(['y'], axis=1)
    return clean_df, T_list, S_list