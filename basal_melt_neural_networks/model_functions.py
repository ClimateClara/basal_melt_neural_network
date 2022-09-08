"""
These are the functions to setup and use the NN models
"""

from tensorflow import keras

def get_model(size,shape,activ_fct): #'mini', 'small', 'medium', 'large', 'extra_large'
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
        
    if size == 'small':

        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n3'))
    
    elif size == 'medium':

        model.add(keras.layers.Dense(96, activation = activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(96, activation= activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(96, activation= activ_fct, name='Dense_n3'))
        model.add(keras.layers.Dense(96, activation= activ_fct, name='Dense_n4'))
        model.add(keras.layers.Dense(96, activation= activ_fct, name='Dense_n5'))

    elif size == 'large':
        
        model.add(keras.layers.Dense(128, activation= activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(128, activation= activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(128, activation= activ_fct, name='Dense_n3'))
        model.add(keras.layers.Dense(128, activation= activ_fct, name='Dense_n4'))
        model.add(keras.layers.Dense(128, activation= activ_fct, name='Dense_n5'))
        
    elif size == 'extra_large':
        
        model.add(keras.layers.Dense(256, activation= activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(256, activation= activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(256, activation= activ_fct, name='Dense_n3'))
        model.add(keras.layers.Dense(256, activation= activ_fct, name='Dense_n4'))
        model.add(keras.layers.Dense(256, activation= activ_fct, name='Dense_n5'))
        model.add(keras.layers.Dense(256, activation= activ_fct, name='Dense_n6'))
    
    model.add(keras.layers.Dense(1, name='Output'))

    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] ) 
    
    return model




