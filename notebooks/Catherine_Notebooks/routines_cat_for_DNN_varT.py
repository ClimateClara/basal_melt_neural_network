#!/usr/bin/env python
# coding: utf-8

# # <!-- TITLE --> [GRISLI temperature] 
# ## Routines for Regression with a Dense Neural Network (DNN)  
# 
# 
# <!-- AUTHOR : catritz -->
# <!-- date 3 january 2021 --> 
# 
# ## version  0.1

# ## Import Modules

import tensorflow as tf
from tensorflow import keras
import numpy as np

#import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
#from matplotlib import cm

import pandas as pd
import os,sys
import xarray as xr
import math
from datetime import date
import json

#import seaborn as sns

sys.path.append('..')
#import fidle.pwk as ooo

#ooo.init()
os.makedirs('./run/',   mode=0o750, exist_ok=True)


### Utilitary routines

# ### pretty print of a dictionnary


def pretty(d, indent = 1):   # pretty print d'un dictionnaire
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))
        print()
#____________________________________________________________________________

# ### subroutine nc2pd : read netcdf files and construct a dataframe

def nc2pd(file_in):     ### step 2. read a netcdf file from GRISLI obtained with : 
                                  ### read-GRISLI-xarray.ipynb

    DD = xr.open_dataset(file_in)

    ### 3D variablescharacteristics

    dft = DD.T3D.to_dataframe()

    dT3D = dft.unstack(level='z').swaplevel().sort_index()

    # change the column names to get rid of level
    dT3D.columns = (['T_00','T_01','T_02','T_03','T_04','T_05',
                     'T_06','T_07','T_08','T_09','T_10',
                    'T_11','T_12','T_13','T_14','T_15',
                     'T_16','T_17','T_18','T_19','T_20'])

    ### 2D variables

    Dss = DD.sel(z=0).drop("T3D")   # drop la variable 3D du xarray dataset

    pd_2D = Dss.drop("z").to_dataframe()  # drop z and convert to dataframe
    pd_2D['name'] = file_in
    ### add the name of the file in the dataframe
    


    pd_onefile = pd.merge(pd_2D,dT3D,how='left',on=['x','y'])
    pd_onefile.reset_index(level=[0,1], inplace=True)        
                # transform indexes x and y to columns

    # extract the name of the GRISLI run and fill a column with it to enable sorting a specific case later.

    posdeb = file_in.rfind('/')
    posend = file_in.rfind('.nc')
    subfile = file_in[posdeb+1:posend]    
        
    pd_onefile['file'] = subfile  
    
    return pd_onefile
#____________________________________________________________________________
def nc2pdtd(file_in):     ### step 2. read a netcdf file from GRISLI obtained with :
                                     ### read-GRISLI-xarray.ipynb
                                    ### time dependent version

    DD = xr.open_dataset(file_in)  #

    
### 2D variables at present time

    Dss = DD.sel(z=0).drop("T3D")   # drop la variable 3D du xarray dataset

    pd_2D = Dss.drop("z").to_dataframe()  # drop z and convert to dataframe
    

 ### time dependent surface temperature in x_array D_td

    # derive the name of the time dependent file from file_in
    numtime = 25   # numer of time slices taken into account
    s='DNN_'
    pos = file_in.rfind(s)
    file_td = file_in[0:pos+len(s)]+'time_dep_'+ file_in[pos+len(s):]

    # read the dataset
    D_Td = xr.open_dataset(file_td)
    print(D_Td.dims)

    # keep only Tann and select a time subset
#    Tann = D_Td.Tann.sel(time=slice(-numtime*1000,0)) # keep only Tann on numtime kyears
    timelist = [-x*1000 for x in range(numtime+1)] 

    Tann = D_Td.Tann.sel(time = timelist) # keep only Tann on numtime kyears

    # Convert in a dataframe in which each time slice will be a column
    Time_cols = ["Tann_"+ str(x).zfill(2)+"kyr" for x in range(numtime+1)]  # list columns
    print (Time_cols)

    dTann = Tann.to_dataframe()
    dTann = dTann.unstack(level='time').swaplevel().sort_index() # move the time in columns
                                                                 # at this index level "time" still 
                                                                 # appears as a sign of multiindex
    
    dTann.columns = Time_cols                                   # remove the index level "time"
    
    
    pd_2Dtd = pd.merge(pd_2D,dTann,how='left',on=['x','y'])   
    
### 3D variablescharacteristics

    dft = DD.T3D.to_dataframe()

    dT3D = dft.unstack(level='z').swaplevel().sort_index()

    # change the column names to get rid of level 'z' in the dataframe
    dT3D.columns = (['T_00','T_01','T_02','T_03','T_04','T_05',
                     'T_06','T_07','T_08','T_09','T_10',
                    'T_11','T_12','T_13','T_14','T_15',
                     'T_16','T_17','T_18','T_19','T_20'])
    
       

    pd_onefile = pd.merge(pd_2Dtd,dT3D,how='left',on=['x','y'])
    pd_onefile.reset_index(level=[0,1], inplace=True)      # transform indexes x and y to columns
    
    

# extract the name of the GRISLI run and fill a column with it to enable sorting a specific case later.
    posdeb = file_in.rfind('/')
    posend = file_in.rfind('.nc')
    subfile = file_in[posdeb+1:posend]    
    pd_onefile['GRISLI_run'] = subfile    
 
    
    return pd_onefile
#____________________________________________________________________________

# ### subroutine select_some : select points according to criteria in the dictionnary dic

def select_some(df_tot,dic):   # select points according to criteria in dictionnary. section 3.1
# return the dataframe data2    
   
    Hmin = dic['select_data']['thickmin'] 
    base_type = dic['select_data']['base']
    mk = dic['select_data']['mk']
    DTb = dic['select_data']['varTb'] 

    datagrounded = df_tot[df_tot.Mk < 0.5] 


    if DTb == 0:
        datagrounded = datagrounded[datagrounded.VarTb == 0] 


    data2 =  datagrounded[datagrounded.H > Hmin] 

    if base_type == 'cold':
        data2 =  data2[data2.Tb < 0] 

    elif base_type == 'wet':
        data2 =  data2[data2.Tb == 0] 

    else: 
        data2 =  data2

    return data2
#____________________________________________________________________________

# ### subroutine to make the difference with the surface 

def diff_surf(df_in, dic):  # make the difference with surface temperature step 3.2
    list_y = dic['y_columns']['list_y']
    
    df2 = df_in.copy(deep=True)  # if not, modify the original dataframe(warning)
    df2[list_y] = df_in[list_y].sub(df_in.Ts,axis=0)
    df_out = df2
    
    return df_out
#____________________________________________________________________________

# ###  subroutine diff_model_data(ydiff, df_in)

#  Make differences between predicted and target and merge with the initial dataframes step 7.4
# ydiff is obstained by :
# ypred = model.predict (x)  where x is normalized
# diff = ypred - y        where y is the  initial data once sorted and 
#  eventually in difference with surface temperature

def diff_model_data(ydiff, df_in):

    nbrow = ydiff.shape[1]   # number of elements in the vertical
    
    columns_diff = ['mse_row','sigma_row','mae_row','me_row','dT_01', 'dT_02', 'dT_03', 'dT_04', 'dT_05',
                'dT_06', 'dT_07', 'dT_08', 'dT_09', 'dT_10', 'dT_11', 'dT_12', 'dT_13', 'dT_14',
                'dT_15', 'dT_16', 'dT_17', 'dT_18', 'dT_19', 'dT_20']             # to name the diff columns
    
    # the bloc below calculate mse_row, mae_row, me_row and stack them with the ydiff 2D values
    # each row represent a physical vertical column
    
    yd2    = ydiff**2
    ydabs = np.abs(ydiff)
    mse_row   = yd2.sum(axis=1)/nbrow        # mse row
    sigma_row = np.sqrt(mse_row)             # sigma_row
    mae_row   = ydabs.sum(axis=1)/nbrow      # mae row
    me_row    = ydiff.sum(axis=1)/nbrow      # difference with sign
    
    all_diff = np.column_stack((mse_row,sigma_row,mae_row,me_row,ydiff))
    
    ### Merge differences with the initial dataframe df_in
    ddiff = pd.DataFrame(all_diff, columns=columns_diff,index=df_in.index)
    df_out = pd.concat([df_in,ddiff],axis=1,ignore_index=False)

    return df_out
#____________________________________________________________________________

# ### subroutine get_norm : normalize df according to parameters and norms from dictionnary

def get_norm(df,dic):    # calculate norms from df according to parameters
                         # in dic and return x_norm the dataframe with 
                         # the normalized columns of list_pred
    
    normalize = dic['normalization']['method']
    list_pred = dic['x_columns']['list_pred']   # to reconstruct the mean dataframe


    if normalize == 'mean':  # mean std normalisation

        # mean_list and std_list are list
        mean_list = dic['normalization']['xmean']
        std_list  = dic['normalization']['xstd']

        # to convert list to dataframe
        mean = pd.DataFrame(mean_list,index=list_pred)
        std = pd.DataFrame(std_list, index=list_pred)
        mean = mean.squeeze()
        std = std.squeeze()       

        #keep only only the columns 'list_pred' in df
        x_norm = (pd.Dataframe(df, columns=list_pred) - mean ) /std

 

    elif normalize == 'meanlog': # replace with the log10 columns from list_meanlog
                                 # limited to 1.e-7 to avoid log10 warning
                                 # then apply usual mean and std on all columns 

        list_meanlog = dic['normalization']['columns']
        mean_list = dic['normalization']['xmean']
        std_list  = dic['normalization']['xstd']

        #keep only only the columns 'list_pred' in df
        xx = pd.DataFrame(df, columns=list_pred)

        # take the log x_train
        x_log = xx.copy(deep=True)       # to avoid changing xx
        x_tolog   = x_log[list_meanlog]  # keep only the columns that must be log
        x_tolog   = np.maximum(x_tolog, 1.e-7)  # to avoid 0
        x_logged  = np.log10(x_tolog)

        
        x_log[list_meanlog] = x_logged
 
        # to convert list to dataframe
        mean = pd.DataFrame(mean_list,index=list_pred)
        std = pd.DataFrame(std_list, index=list_pred)
        mean = mean.squeeze()
        std = std.squeeze()    
 
        x_norm = (x_log - mean) / std  
 
    elif normalize == 'minmax': # min-max normalisation
        
        minx_list = dic['normalization']['minx']
        maxx_list = dic['normalization']['maxx']
      
        minx = pd.DataFrame(minx_list, index=list_pred)
        maxx = pd.DataFrame(maxx_list, index=list_pred)
        minx = minx.squeeze()
        maxx = maxx.squeeze()
        
        #keep only only the columns 'list_pred' in df
        xx = pd.Dataframe(df, columns=list_pred)
        
        x_norm = (xx - minx) / (maxx - minx)
        
    return x_norm
#____________________________________________________________________________


