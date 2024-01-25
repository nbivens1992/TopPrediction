
from keras import layers
from keras import models
from utilities.inception_module import inception_module

def create_global_view(initial_layer = 4, dropout = 0.5, dilation3x3 =1,dilation5x5=1, dilation7x7=1):
    inputs = layers.Input(shape=(None, 4))  # Assuming the logs are 1D sequences
    x = layers.Masking(mask_value=-99)(inputs)  # Using -99 as the mask value
    
    
    # Encoding layer 1
    conv1 = inception_module(x, initial_layer, dilation3x3=dilation3x3, dilation5x5=dilation5x5, dilation7x7=dilation7x7)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(dropout)(conv1)
    pool1 = layers.AveragePooling1D(pool_size=8, strides=8)(conv1)

    # Encoding layer 2
    conv2 = inception_module(pool1, initial_layer*2, dilation3x3=dilation3x3, dilation5x5=dilation5x5,dilation7x7=dilation7x7)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(dropout)(conv2)
    pool2 = layers.AveragePooling1D(pool_size=4, strides=4)(conv2)
    
    # Encoding layer 3
    conv3 = inception_module(pool2, initial_layer*4, dilation3x3=dilation3x3, dilation5x5=dilation5x5,dilation7x7=dilation7x7)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(dropout)(conv3)
    pool3 = layers.AveragePooling1D(pool_size=2, strides=2)(conv3)

    # Middle layer
    conv_middle = inception_module(pool3,initial_layer*8, dilation3x3=dilation3x3, dilation5x5=dilation5x5,dilation7x7=dilation7x7)
    conv_middle = layers.BatchNormalization()(conv_middle)
    conv_middle = layers.Dropout(dropout)(conv_middle)
    
    # Decoding layer 3
    up0 = layers.UpSampling1D(size =2)(conv_middle)
    merge0 = layers.concatenate([conv3, up0]) 
    decode0 = inception_module(merge0,initial_layer*4, dilation3x3=dilation3x3, dilation5x5=dilation5x5,dilation7x7=dilation7x7)
    decode0 = layers.BatchNormalization()(decode0)
    decode0 = layers.Dropout(dropout)(decode0)
    

    # Decoding layer 1
    up1 = layers.UpSampling1D(size = 4)(decode0)
    merge1 = layers.concatenate([conv2, up1]) 
    decode1 = inception_module(merge1,initial_layer*2, dilation3x3=dilation3x3, dilation5x5=dilation5x5,dilation7x7=dilation7x7)
    decode1 = layers.BatchNormalization()(decode1)
    decode1 = layers.Dropout(dropout)(decode1)
    
    
    # Decoding layer 2
    up2 = layers.UpSampling1D(size= 8)(decode1)
    merge2 = layers.concatenate([conv1, up2])
    decode2 = inception_module(merge2, initial_layer, dilation3x3=dilation3x3, dilation5x5=dilation5x5,dilation7x7=dilation7x7)
    decode2 = layers.BatchNormalization()(decode2)
    decode2 = layers.Dropout(dropout)(decode2)

    return models.Model(inputs, decode2)