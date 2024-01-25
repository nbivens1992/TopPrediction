from keras import layers, models
from utilities.inception_module import inception_module


def create_local_view(initial_filters=2, input_shape=(None, 4), mask_value=-99, dilation3x3 =48,dilation5x5=4, dilation7x7=24):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Masking layer
    x = layers.Masking(mask_value=mask_value)(inputs)
    
    conv1 = inception_module(x,initial_filters, dilation3x3=dilation3x3, dilation5x5=dilation5x5, dilation7x7=dilation7x7)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(0.5)(conv1)
    
    conv2 = inception_module(conv1,initial_filters, dilation3x3=dilation3x3, dilation5x5=dilation5x5, dilation7x7=dilation7x7)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(0.5)(conv2)
    
    conv3 = inception_module(conv2,initial_filters, dilation3x3=dilation3x3, dilation5x5=dilation5x5, dilation7x7=dilation7x7)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Dropout(0.5)(conv3)
    
    
    return models.Model(inputs, conv3)