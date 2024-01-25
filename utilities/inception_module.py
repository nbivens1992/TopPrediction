from keras import layers

def inception_module(input_tensor, filter_channels, dilation3x3=1, dilation5x5=1, dilation7x7=1):
    # Branch 1: 1x1 conv
    branch1x1 = layers.Conv1D(filter_channels, 1, padding='same', activation='relu')(input_tensor)
    
    # Branch 2: 1x1 conv followed by 3x3 conv
    branch3x3 = layers.Conv1D(filter_channels, 1, padding='same', activation='relu')(input_tensor)
    branch3x3 = layers.Conv1D(filter_channels, 3, padding='same', activation='relu', dilation_rate=dilation3x3)(branch3x3)
    
    # Branch 2: 1x1 conv followed by 3x3 conv
    branch5x5 = layers.Conv1D(filter_channels//2, 1, padding='same', activation='relu')(input_tensor)
    branch5x5 = layers.Conv1D(filter_channels//4, 10, padding='same', activation='relu', dilation_rate=dilation5x5)(branch5x5)
    
    # Branch 2: 1x1 conv followed by 3x3 conv
    branch7x7 = layers.Conv1D(filter_channels//2, 1, padding='same', activation='relu')(input_tensor)
    branch7x7 = layers.Conv1D(filter_channels//8, 24, padding='same', activation='relu', dilation_rate=dilation7x7)(branch7x7)
    
    pool3x3 = layers.AveragePooling1D(pool_size=3, strides=1, padding='same')(input_tensor)
    pool3x3 = layers.Conv1D(filter_channels, 1, padding='same', activation='relu')(pool3x3)
        
    # Concatenate all the branches
    concatenated = layers.Concatenate(axis=-1)([branch3x3, branch5x5,branch1x1,pool3x3, branch7x7])
    
    return concatenated