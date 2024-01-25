import pandas as pd
import numpy as np


def pad_to_max_length(sequence, max_length, padding_value=-99):
    padding = [(0, max_length - sequence.shape[0]), (0, 0)]
    return np.pad(sequence, padding, mode='constant', constant_values=padding_value)


def adjust_max_length(max_length):
    last_digit = max_length % 10
    
    if last_digit in [2, 6]:
        return max_length + 2
    elif last_digit == 0:
        return max_length + 4
    else:
        return max_length

def pad_np_array(np_array, desired_length, padding_value=-99):
    # Calculate the number of rows to add
    num_rows_to_add = desired_length - np_array.shape[0]

    # If the array is already longer or equal to the desired length, return the original array
    if num_rows_to_add <= 0:
        return np_array

    # Create a new array with the required number of rows filled with the padding value
    padding_array = np.full((num_rows_to_add, np_array.shape[1]), padding_value)

    # Concatenate the original array with the padding array
    padded_array = np.concatenate([np_array, padding_array], axis=0)

    return padded_array

def pad_dataframe(df, desired_length, padding_value=-99):
    # Calculate the number of rows to add
    num_rows_to_add = desired_length - len(df)
    
    # If the dataframe is already longer or equal to the desired length, return the original dataframe
    if num_rows_to_add <= 0:
        return df
    
    # Create a new dataframe with the required number of rows filled with the padding value
    padding_df = pd.DataFrame(padding_value, index=range(num_rows_to_add), columns=df.columns)
    
    # Concatenate the original dataframe with the padding dataframe
    padded_df = pd.concat([df, padding_df], axis=0).reset_index(drop=True)
    
    return padded_df