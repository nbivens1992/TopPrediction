from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from utilities.create_global_view import create_global_view
from utilities.create_local_view import create_local_view
from keras import layers, models
from keras.optimizers import Adam

# 48, 4, 24
def compile_model(learning_rate, nodes, dropout, dilation3x3 =1,dilation5x5=1, dilation7x7=1, global_bool = True,local_bool =True):
    if global_bool:
        global_view = create_global_view(initial_layer=nodes,dropout = dropout, dilation3x3=1, dilation5x5=1, dilation7x7=1)
        global_output = global_view.output
        # Apply tanh activation to both global and local views
        activated_global_view = layers.Activation('tanh')(global_output)
        attention_scores = tf.reduce_mean(activated_global_view, axis=-1)
        attention_scores = tf.expand_dims(attention_scores, axis=-1)
        output = layers.Conv1D(1, 1, activation="sigmoid")(global_output)
    if local_bool:
        local_view = create_local_view(initial_filters=nodes,dilation3x3=dilation3x3,dilation5x5= dilation5x5, dilation7x7=dilation7x7)
        local_output = local_view.output
        activated_local_output = layers.Conv1D(1, 1, activation="sigmoid")(local_output)
        output = layers.Conv1D(1, 1, activation="sigmoid")(local_output)
       
    if global_bool and local_bool:
        # Element-wise multiplication of the activated global and local views
        combined_features = layers.Multiply()([attention_scores, activated_local_output])
        # HYPERPARAMETER: Adjust the number of filters (1) and kernel size (1)
        output = layers.Conv1D(1, 1, activation="sigmoid")(combined_features)
    

    if global_bool and local_bool:
        model = models.Model([global_view.input, local_view.input], output)
    elif local_bool:
        model = models.Model([local_view.input], output)
    else:
        model = models.Model([global_view.input], output)
        
    optimizer = Adam(learning_rate=learning_rate)
    
    # optimizer = Nadam()
    
    # HYPERPARAMETER: Adjust the optimizer ('adam') and its parameters
    model.compile(optimizer=optimizer, loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
    
    
    return model

    