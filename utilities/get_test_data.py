import os
import numpy as np
import pandas as pd

from utilities.check_for_nans import check_for_nans
from utilities.ensure_power_of_2_length import ensure_power_of_2_length
from utilities.normalize_data import normalize_data
from utilities.padding_functions import pad_dataframe, pad_to_max_length
from utilities.smooth_labels import smooth_labels

def get_test_data(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    test_dfs = []
    max_length = 0
    for file in files:
        df = pd.read_csv(folder_path + file)
        if len(df.index)>max_length:
            max_length = len(df.index)
        test_dfs.append(df)

    features = ["VSH", "RESD", "DPHI", "NPHI"]
    
    # Normalize data
    
    test_dfs = [normalize_data(df,features) for df in test_dfs]
    
    # The CNN model requires each well log to be of the same shape and the shape needs to be a multiple of 8 or it will cause an error
    
    # Pad INPUT to max_length with -99 as none of the logs will have this as a value
    
    test_dfs = [pad_dataframe(df, max_length) for df in test_dfs]
    
    # Extract features and labels for training and validation sets
    X_test = np.array([df[features].values for df in test_dfs])
    y_test = np.array([df["Pick"].values for df in test_dfs])

    # Ensure multiple of 8 for UNET
    X_test = [ensure_power_of_2_length(x) for x in X_test]
    
    # Compute NEW max_length after ensuring lengths OF MULTIPLE OF 8
    max_length2 = max(x.shape[0] for x in X_test)
    
    # Pad to max_length
    X_test = [pad_to_max_length(x, max_length2) for x in X_test]
    
    # apply gaussian kernal to y values
    y_test = [smooth_labels(y) for y in y_test]
    
    # Ensure even length for labels and then pad
    y_test = [ensure_power_of_2_length(y.reshape(-1, 1),padding_value=0) for y in y_test]
    
    y_test = [pad_to_max_length(y, max_length2,padding_value=0) for y in y_test]
    
    test_dfs = [pad_dataframe(df, max_length2,padding_value=-99) for df in test_dfs]
    
    # # Update the 'Pick' column in test_dfs with y_test values
    # for i, df in enumerate(test_dfs):
    #     df['Pick'] = y_test[i][:len(df)]  # Assuming y_test[i] is already the correct length
    
    [check_for_nans(x, "X_test " + str(i)) for i, x in enumerate(X_test)]
    [check_for_nans(x, "y_test " + str(i)) for i, x in enumerate(y_test)]
    
    return X_test, y_test, test_dfs