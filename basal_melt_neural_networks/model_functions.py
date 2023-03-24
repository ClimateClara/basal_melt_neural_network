"""
These are the functions to setup and use the NN models
"""

from tensorflow import keras

def get_model(size,shape,activ_fct,output_shape): #'mini', 'small', 'medium', 'large', 'extra_large'
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
        
    if size == 'small':

        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(32, activation=activ_fct, name='Dense_n3'))
        
    elif size == 'small64':

        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n3'))

    elif size == 'small96':

        model.add(keras.layers.Dense(96, activation=activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(96, activation=activ_fct, name='Dense_n2'))
        model.add(keras.layers.Dense(96, activation=activ_fct, name='Dense_n3'))
        
    if size == 'xsmall96':

        model.add(keras.layers.Dense(96, activation=activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(96, activation=activ_fct, name='Dense_n2'))

    if size == 'xsmall64':

        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n1'))
        model.add(keras.layers.Dense(64, activation=activ_fct, name='Dense_n2'))
    
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
    
    model.add(keras.layers.Dense(output_shape, name='Output'))

    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] ) 
    
    return model

def get_custom_model(layer_nb,layer_size,shape,activ_fct,output_shape): #'mini', 'small', 'medium', 'large', 'extra_large'
    
    if activ_fct == 'LeakyReLU':
        lrelu = lambda x: keras.activations.relu(x, alpha=0.1)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape, name="InputLayer"))
    
    for ll in range(1,layer_nb+1):
        
        if activ_fct == 'LeakyReLU':
            model.add(keras.layers.Dense(layer_size, activation=lrelu, name='Dense_n'+str(ll)))
        else:
            model.add(keras.layers.Dense(layer_size, activation=activ_fct, name='Dense_n'+str(ll)))
         
    model.add(keras.layers.Dense(output_shape, name='Output'))

    model.compile(optimizer = 'adam',
                  loss      = 'mse',
                  metrics   = ['mae', 'mse'] ) 
    
    return model




