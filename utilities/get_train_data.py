import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from utilities.check_for_nans import check_for_nans, check_for_nans_in_dataframe
from utilities.ensure_power_of_2_length import ensure_power_of_2_length
from utilities.normalize_data import normalize_data
from utilities.padding_functions import pad_dataframe, pad_to_max_length
from utilities.smooth_labels import smooth_labels

def get_train_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

    max_length = 0
    dfs = []
    for file in files:
        df = pd.read_csv(folder_path + file)
        if len(df.index)>max_length:
            max_length = len(df.index)
        check_for_nans_in_dataframe(df,name = file)
        dfs.append(df)
    
    
    train_dfs, val_dfs = train_test_split(dfs, test_size=0.2, random_state=42) # Here 20% of the data is kept for validation

    features = ["VSH", "RESD", "DPHI", "NPHI"]
    
    # Normalize data 
    train_dfs = [normalize_data(df,features) for df in train_dfs]
    val_dfs = [normalize_data(df,features) for df in val_dfs]
    
    # The CNN model requires each well log to be of the same shape and the shape needs to be a multiple of 8 or it will cause an error
    
    # Pad to max_length with -99 as none of the logs will have this as a value
    train_dfs = [pad_dataframe(df, max_length) for df in train_dfs]
    val_dfs = [pad_dataframe(df, max_length) for df in val_dfs]
    

    # Extract features and labels for training and validation sets
    X_train = np.array([df[features].values for df in train_dfs])
    y_train = np.array([df["Pick"].values for df in train_dfs])

    X_val = np.array([df[features].values for df in val_dfs])
    y_val = np.array([df["Pick"].values for df in val_dfs])
    
    # Ensure the input is not only even but a multiple of 8 so that the CNN Model can handle it. NOTE that that afterwards there might be a new max length
    X_train = [ensure_power_of_2_length(x) for x in X_train]
    X_val = [ensure_power_of_2_length(x) for x in X_val]

    # # Compute max_length again after ensuring a power of 2 length 
    # max_length2 = max(max(x.shape[0] for x in X_train), max(x.shape[0] for x in X_val))

    # # Pad to new max_length
    # X_train = [pad_to_max_length(x, max_length2) for x in X_train]
    # X_val = [pad_to_max_length(x, max_length2) for x in X_val]

    # apply gaussian kernal to y values
    y_train = [smooth_labels(y) for y in y_train]
    y_val = [smooth_labels(y) for y in y_val]

    # Ensure even length for labels and then pad
    y_train = [ensure_power_of_2_length(y.reshape(-1, 1),padding_value=0) for y in y_train]
    y_val = [ensure_power_of_2_length(y.reshape(-1, 1),padding_value=0) for y in y_val]
    
    
    # # Now we need to pad the y values. The y values will be padded with 0 as they are not the picks
    # y_train = [pad_to_max_length(y, max_length2,padding_value=0) for y in y_train]
    # y_val = [pad_to_max_length(y, max_length2,padding_value=0) for y in y_val]
    
    [check_for_nans(x, "X_train array "+ str(i)) for i, x in enumerate(X_train)]
    [check_for_nans(x, "X_val array " + str(i)) for i, x in enumerate(X_val)]
    [check_for_nans(x, "y_train array " + str(i)) for i, x in enumerate(y_train)]
    [check_for_nans(x, "y_val array " + str(i)) for i, x in enumerate(y_val)]
    
    return X_train, y_train, X_val, y_val