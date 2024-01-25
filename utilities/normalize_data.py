def normalize_data(df, features):
    # Compute mean and std only for specified features
    mean = df[features].mean()
    std = df[features].std()

    # Apply normalization only to the specified features
    df[features] = (df[features] - mean) / std

    return df