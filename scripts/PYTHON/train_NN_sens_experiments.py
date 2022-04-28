"""
Created on Wed Apr 27 15:28 2022

Run training in a script instead of notebook

"""

import numpy as np
import xarray as xr
from tqdm.notebook import trange, tqdm
import glob
import matplotlib as mpl
import seaborn as sns
import datetime
import time

from dask import delayed

import distributed

import tensorflow as tf
from tensorflow import keras
from contextlib import redirect_stdout

from basal_melt_neural_networks.constants import *
import basal_melt_neural_networks.diagnostic_functions as diag
import basal_melt_neural_networks.data_formatting as dfmt

#### READ IN DATA

inputpath_data = '/bettik/burgardc/DATA/NN_PARAM/interim/INPUT_DATA/'
outputpath_nn_models = '/bettik/burgardc/DATA/NN_PARAM/interim/NN_MODELS/'
outputpath_doc = '/bettik/burgardc/SCRIPTS/basal_melt_neural_networks/custom_doc/'

timetag = '20220427-1051'
mod_size = 'large' #'mini', 'small', 'medium', 'large'

new_path_model = outputpath_nn_models+timetag+'/'
if not os.path.isdir(new_path_model):
    print("I did not find this folder ("+timetag+") in model folder! :( ")

new_path_doc = outputpath_doc+timetag+'/'
if not os.path.isdir(new_path_doc):
    print("I did not find this folder ("+timetag+") in doc folder! :( ")
    
new_path_input = inputpath_data+timetag+'/'
if not os.path.isdir(new_path_input):
    print("I did not find this folder ("+timetag+") in input folder! :( ")
    
#### BUILD THE MODEL

if timetag in ['20220427-0957','20220427-1052','20220427-1058','20220427-1059']:
    timetag_data = '20220427-0957'
elif timetag in ['20220427-1002','20220427-1021','20220427-1042','20220427-1051']:
    timetag_data = '20220427-1002'
path_data = outputpath_nn_models+timetag_data+'/'
data_train_norm = xr.open_dataset(path_data + 'dataset_norm_training_data_'+timetag_data+'.nc')
data_test_norm = xr.open_dataset(path_data + 'dataset_norm_test_data_'+timetag_data+'.nc')

y_train_norm = data_train_norm['melt_m_ice_per_y'].load()
x_train_norm = data_train_norm.drop_vars(['melt_m_ice_per_y']).to_array().load()

y_test_norm = data_test_norm['melt_m_ice_per_y'].load()
x_test_norm = data_test_norm.drop_vars(['melt_m_ice_per_y']).to_array().load()

def get_model_mini(shape):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    model.add(keras.layers.Dense(1, name='Output'))
    
    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] )
    return model

def get_model_small(shape):
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    model.add(keras.layers.Dense(32, activation='relu', name='Dense_n1'))
    model.add(keras.layers.Dense(64, activation='relu', name='Dense_n2'))
    model.add(keras.layers.Dense(32, activation='relu', name='Dense_n3'))
    model.add(keras.layers.Dense(1, name='Output'))
    
    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] )
    return model

def get_model_medium(shape):
    
    activ = 'relu'
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    model.add(keras.layers.Dense(96, activation = activ, name='Dense_n1'))
    model.add(keras.layers.Dense(96, activation= activ, name='Dense_n2'))
    model.add(keras.layers.Dense(96, activation= activ, name='Dense_n3'))
    model.add(keras.layers.Dense(96, activation= activ, name='Dense_n4'))
    model.add(keras.layers.Dense(96, activation= activ, name='Dense_n5'))
    model.add(keras.layers.Dense(1, name='Output'))  
    
    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] )
    return model

def get_model_large(shape):
    
    activ = 'relu'
    nodes = 256
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    model.add(keras.layers.Dense(nodes, activation= activ, name='Dense_n1'))
    model.add(keras.layers.Dense(nodes, activation= activ, name='Dense_n2'))
    model.add(keras.layers.Dense(nodes, activation= activ, name='Dense_n3'))
    model.add(keras.layers.Dense(nodes, activation= activ, name='Dense_n4'))
    model.add(keras.layers.Dense(nodes, activation= activ, name='Dense_n5'))
    model.add(keras.layers.Dense(nodes, activation= activ, name='Dense_n6'))
    model.add(keras.layers.Dense(1, name='Output'))  
    
    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] )
    return model

#### TRAIN THE MODEL

input_size = x_train_norm.values.shape[0]

if mod_size == 'mini':
    model=get_model_mini( (input_size,) )
elif mod_size == 'small':
    model=get_model_small( (input_size,) )
elif mod_size == 'medium':
    model=get_model_medium( (input_size,) )
elif mod_size == 'large':
    model=get_model_large( (input_size,) )
    
model.summary()

epoch_nb = 100
batch_siz = 1024

#reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
#                              patience=5, min_lr=0.001)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=0.0000001, min_delta=0.0005) #, min_delta=0.1
            
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    #min_delta=0.000001,
    patience=6,
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
                    verbose         = 1,
                    validation_data = (x_test_norm.T.values, y_test_norm.values),
                   callbacks=[reduce_lr, early_stop])
time_end = time.time()
timelength = time_end - time_start
with open(new_path_doc+'info_'+timetag+'.log','a') as file:
    file.write('\n Reduce_lr: True')
    file.write('\n Early_stop: True')
    file.write('\n Training time (in s): '+str(timelength))
model.save(new_path_model + 'model_nn_'+timetag+'.h5')

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

hist_csv_file = new_path_model+'history_'+timetag+'.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)