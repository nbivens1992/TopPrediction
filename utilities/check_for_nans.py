import numpy as np


def check_for_nans(array, name="Array"):
    if isinstance(array, list):
        # Handle case where array is a list of arrays
        for i, arr in enumerate(array):
            if np.isnan(arr).any():
                nan_rows = np.any(np.isnan(arr), axis=1)
                print(f"NaNs found in {name} at array index {i}. Rows with NaNs:")
                print(arr[nan_rows])
    else:
        # Handle case where array is a single NumPy array
        if np.isnan(array).any():
            nan_rows = np.any(np.isnan(array), axis=1)
            print(f"NaNs found in {name}. Rows with NaNs:")
            print(array[nan_rows])
            
            
def check_for_nans_in_dataframe(df, name="DataFrame"):
    # Check if the DataFrame contains any NaN values
    if df.isna().any().any():
        # Get rows which contain NaNs
        rows_with_nans = df[df.isna().any(axis=1)]
        print(f"NaNs found in {name}. Rows with NaNs:")
        print(rows_with_nans)