import tensorflow as tf
from keras import layers, models
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Load and preprocess your data here


# Global View (U-Net)
def create_global_view():
    inputs = layers.Input(shape=(None, 1))  # Assuming the logs are 1D sequences
    x = inputs

    # Encoding layer 1
    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    conv1 = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    conv1 = layers.BatchNormalization()(conv1)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    conv1 = layers.Dropout(0.5)(conv1)
    pool1 = layers.MaxPooling1D()(conv1)

    # Encoding layer 2 (and so on, if you want deeper architecture)
    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    conv2 = layers.Conv1D(128, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    conv2 = layers.Dropout(0.5)(conv2)
    pool2 = layers.MaxPooling1D()(conv2)

    # Middle layer (You can continue with more encoding layers before this)
    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    conv_middle = layers.Conv1D(256, 3, activation="relu", padding="same")(pool2)
    conv_middle = layers.BatchNormalization()(conv_middle)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    conv_middle = layers.Dropout(0.5)(conv_middle)

    # Decoding layer 1
    up1 = layers.UpSampling1D()(conv_middle)
    merge1 = layers.concatenate([conv2, up1])
    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    decode1 = layers.Conv1D(128, 3, activation="relu", padding="same")(merge1)
    decode1 = layers.BatchNormalization()(decode1)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    decode1 = layers.Dropout(0.5)(decode1)

    # Decoding layer 2 (and so on, mirroring the encoding architecture)
    up2 = layers.UpSampling1D()(decode1)
    merge2 = layers.concatenate([conv1, up2])
    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    decode2 = layers.Conv1D(64, 3, activation="relu", padding="same")(merge2)
    decode2 = layers.BatchNormalization()(decode2)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    decode2 = layers.Dropout(0.5)(decode2)

    return models.Model(inputs, decode2)


# Local View
def create_local_view():
    inputs = layers.Input(shape=(None, 1))
    x = inputs

    # Inception layers
    # HYPERPARAMETER: Adjust the number of filters (32) and kernel sizes (1 and 3)
    conv1 = layers.Conv1D(32, 1, padding="same")(x)
    conv3 = layers.Conv1D(32, 3, padding="same")(x)
    concat = layers.Concatenate()([conv1, conv3])
    return models.Model(inputs, concat)


# Train your model here
folder_path = "data/"
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dfs = []
for file in files:
    df = pd.read_csv(folder_path + file)
    dfs.append(df)


# Concatenate all data for normalization
all_data = pd.concat(dfs, ignore_index=True)


def normalize(df, feature_names):
    for feature in feature_names:
        mean = all_data[feature].mean()
        std = all_data[feature].std()
        df[feature] = (df[feature] - mean) / std
    return df


features = ["VSH", "NPHI", "SW", "PHIF", "RHOB"]
for i in range(len(dfs)):
    dfs[i] = normalize(dfs[i], features)

# For simplicity, extracting features and labels
Xs = [df[features].values for df in dfs]
ys = [df["top"].values for df in dfs]


# Combining all but the first well for training
X_train = np.concatenate(Xs[1:])
y_train = np.concatenate(ys[1:])

X_val = Xs[0]
y_val = ys[0]

# Reshape for the model
X_train = X_train.reshape((-1, len(features), 1))
y_train = y_train.reshape((-1, 1))

X_val = X_val.reshape((-1, len(features), 1))
y_val = y_val.reshape((-1, 1))

# Create Model

global_view = create_global_view()
local_view = create_local_view()

combined = layers.Multiply()([global_view.output, local_view.output])

# Soft Attention
attention_output = layers.Activation("tanh")(combined)

# HYPERPARAMETER: Adjust the number of filters (1) and kernel size (1)
output = layers.Conv1D(1, 1, activation="sigmoid")(attention_output)

model = models.Model([global_view.input, local_view.input], output)

# HYPERPARAMETER: Adjust the optimizer ('adam') and its parameters
model.compile(optimizer="adam", loss="binary_crossentropy")


# Train the model
history = model.fit(
    [X_train, X_train], y_train, epochs=50, validation_data=([X_val, X_val], y_val)
)
