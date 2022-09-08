"""
Created on Thu Sep 08 11:15 2022

This script is to automate a bit the cross-validation and avoiding having to do it in a notebook

Author: Clara Burgard
"""

import numpy as np
import xarray as xr
import pandas as pd
from tqdm.notebook import trange, tqdm
import glob
import datetime
import time
import sys

import tensorflow as tf
from tensorflow import keras
import basal_melt_neural_networks.model_functions as modf

######### READ IN OPTIONS

mod_size = str(sys.argv[1]) #'mini', 'small', 'medium', 'large', 'extra_large'
tblock_out = int(sys.argv[2])
isf_out = int(sys.argv[3])
TS_opt = str(sys.argv[4]) # extrap, whole, thermocline
norm_method = str(sys.argv[5]) # std, interquart, minmax

######### READ IN DATA

inputpath_data = '/bettik/burgardc/DATA/NN_PARAM/interim/INPUT_DATA/'
outputpath_nn_models = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'
outputpath_doc = '/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/custom_doc/'

if (tblock_out > 0) and (isf_out == 0):
    path_model = outputpath_nn_models+'CV_TBLOCK/'
    
elif (isf_out > 0) and (tblock_out == 0):
    path_model = outputpath_nn_models+'CV_ISF/'
    
else:
    print("I do not know what to do with both tblock and isf left out! ")

#new_path_doc = outputpath_doc+timetag+'/'
#if not os.path.isdir(new_path_doc):
#    print("I did not find this folder ("+timetag+") in doc folder! :( ")

if TS_opt == 'extrap':
    inputpath_CVinput = inputpath_data+'EXTRAPOLATED_ISFDRAFT_CHUNKS_CV/'
elif TS_opt == 'whole':
    inputpath_CVinput = inputpath_data+'WHOLE_PROF_CHUNKS_CV/'
elif TS_opt == 'thermocline':
    inputpath_CVinput = inputpath_data+'THERMOCLINE_CHUNKS_CV/'
    
data_train_norm = xr.open_dataset(inputpath_CVinput + 'train_data_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')
data_val_norm = xr.open_dataset(inputpath_CVinput + 'val_data_CV_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'.nc')   

## prepare input and target
            
y_train_norm = data_train_norm['melt_m_ice_per_y'].sel(norm_method=norm_method).load()
x_train_norm = data_train_norm.drop_vars(['melt_m_ice_per_y']).sel(norm_method=norm_method).to_array().load()

y_val_norm = data_val_norm['melt_m_ice_per_y'].sel(norm_method=norm_method).load()
x_val_norm = data_val_norm.drop_vars(['melt_m_ice_per_y']).sel(norm_method=norm_method).to_array().load()


######### TRAIN THE MODEL

input_size = x_train_norm.values.shape[0]
activ_fct = 'relu' #LeakyReLU
epoch_nb = 100
batch_siz = 512

model = modf.get_model(mod_size, input_size, activ_fct)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.0000001, min_delta=0.0005) #, min_delta=0.1
            
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    #min_delta=0.000001,
    patience=10,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

time_start = time.time()
history = model.fit(x_train_norm.T.values,
                    y_train_norm.values,
                    epochs          = epoch_nb,
                    batch_size      = batch_siz,
                    verbose         = 2,
                    validation_data = (x_val_norm.T.values, y_val_norm.values),
                   callbacks=[reduce_lr, early_stop])
time_end = time.time()
timelength = time_end - time_start
#with open(new_path_doc+'info_'+timetag+'.log','a') as file:
#    file.write('\n Reduce_lr: True')
#    file.write('\n Early_stop: True')
#    file.write('\n Training time (in s): '+str(timelength))
#model.save(new_path_model + 'model_nn_'+timetag+'.h5')

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

hist_csv_file = path_model + 'history_'+mod_size+'_noisf'+str(isf_out).zfill(3)+'_notblock'+str(tblock_out).zfill(3)+'_TS'+TS_opt+'_norm'+norm_method+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)