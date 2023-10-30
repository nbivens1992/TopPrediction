import tensorflow as tf
from keras import layers, models
from keras.layers import Normalization, Lambda
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


# Global View (U-Net)
def create_global_view():
    inputs = layers.Input(shape=(None, 3))  # Assuming the logs are 1D sequences
    
    # Adding a normalization layer
    x = Normalization(axis=-1)(inputs) 

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
    
    # Ensure sizes match before concatenating
    # cropped_conv2 = layers.Cropping1D(cropping=((0, 1)))(conv2)  # Adjust as needed
    merge1 = layers.concatenate([conv2, up1])
    
    
    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    decode1 = layers.Conv1D(128, 3, activation="relu", padding="same")(merge1)
    decode1 = layers.BatchNormalization()(decode1)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    decode1 = layers.Dropout(0.5)(decode1)

    # Decoding layer 2 (and so on, mirroring the encoding architecture)
    up2 = layers.UpSampling1D()(decode1)
    
    # Ensure sizes match before concatenating for the second merge
    # cropped_conv1 = layers.Cropping1D(cropping=((0, 1)))(conv1)  # Adjust as needed
    merge2 = layers.concatenate([conv1, up2])

    # HYPERPARAMETER: Adjust the number of filters (64) and kernel size (3)
    decode2 = layers.Conv1D(64, 3, activation="relu", padding="same")(merge2)
    decode2 = layers.BatchNormalization()(decode2)
    # HYPERPARAMETER: Adjust dropout rate (0.5)
    decode2 = layers.Dropout(0.5)(decode2)

    return models.Model(inputs, decode2)

# Local View
def create_local_view():
    inputs = layers.Input(shape=(None, 3))
    x = Normalization(axis=-1)(inputs) 
    # Inception layers
    # HYPERPARAMETER: Adjust the number of filters (32) and kernel sizes (1 and 3)
    conv1 = layers.Conv1D(32, 1, padding="same")(x)
    conv3 = layers.Conv1D(32, 3, padding="same")(x)
    concat = layers.Concatenate()([conv1, conv3])
    return models.Model(inputs, concat)

current_dir = os.getcwd()
# Train your model here
folder_path = current_dir + "/Data/Well Log Data/"
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dfs = []
for file in files:
    df = pd.read_csv(folder_path + file)
    dfs.append(df)

train_dfs, val_dfs = train_test_split(dfs, test_size=0.2, random_state=42) # Here 20% of the data is kept for validation


def ensure_even_length(sequence):
    # If the sequence length is odd, append zeros to make it even
    if sequence.shape[0] % 2 != 0:
        padding_shape = list(sequence.shape)
        padding_shape[0] = 1  # we only need to add one row
        
        # Create a padding with the same number of columns as the sequence
        padding = np.zeros(padding_shape)
        
        # Append padding to the sequence
        sequence = np.vstack([sequence, padding])
        
    return sequence

features = ["GR", "ILD", "DPHI"]


# Extract features and labels for training and validation sets
# Convert to numpy arrays
X_train = np.array([df[features].values for df in train_dfs])
y_train = np.array([df["Pick"].values for df in train_dfs])

X_val = np.array([df[features].values for df in val_dfs])
y_val = np.array([df["Pick"].values for df in val_dfs])

X_train = [ensure_even_length(x) for x in X_train]
X_val = [ensure_even_length(x) for x in X_val]

# Find the maximum sequence length
max_length = max(max(len(x) for x in X_train), max(len(x) for x in X_val))

# Pad sequences to have the same length
X_train = [np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), mode='constant') for x in X_train]
X_val = [np.pad(x, ((0, max_length - x.shape[0]), (0, 0)), mode='constant') for x in X_val]


# Pad label sequences to have the same length as the input sequences
y_train = [np.pad(y, (0, max_length - len(y)), mode='constant') for y in y_train]
y_val = [np.pad(y, (0, max_length - len(y)), mode='constant') for y in y_val]


def create_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    all_data = np.vstack(X_train)

    global_view = create_global_view()
    local_view = create_local_view()

    # Adapt the normalization layer in global_view
    global_view.layers[1].adapt(all_data)

    # Adapt the normalization layer in local_view
    local_view.layers[1].adapt(all_data)

    combined = layers.Multiply()([global_view.output, local_view.output])

    # Soft Attention
    attention_output = layers.Activation("tanh")(combined)

    # HYPERPARAMETER: Adjust the number of filters (1) and kernel size (1)
    output = layers.Conv1D(1, 1, activation="sigmoid")(attention_output)

    model = models.Model([global_view.input, local_view.input], output)
    optimizer = Adam(learning_rate=learning_rate)

    # HYPERPARAMETER: Adjust the optimizer ('adam') and its parameters
    model.compile(optimizer=optimizer, loss="binary_crossentropy")


    train_losses = []
    val_losses = []
    # We'll only use this training loop and remove the redundant one.
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        epoch_train_losses = []
        for i in range(len(X_train)):
            X_train_well = X_train[i].reshape(1, X_train[i].shape[0], X_train[i].shape[1])
            y_train_well = y_train[i].reshape(1, y_train[i].shape[0], 1)
            loss = model.train_on_batch([X_train_well, X_train_well], y_train_well)  # Capturing the loss
            epoch_train_losses.append(loss)
        train_losses.append(np.mean(epoch_train_losses))

        # Validation (optional)
        epoch_val_losses = []
        for i in range(len(X_val)):
            X_val_well = X_val[i].reshape(1, X_val[i].shape[0], X_val[i].shape[1])
            y_val_well = y_val[i].reshape(1, y_val[i].shape[0], 1)
            loss = model.test_on_batch([X_val_well, X_val_well], y_val_well)
            epoch_val_losses.append(loss)
        mean_val_loss = np.mean(epoch_val_losses)
        val_losses.append(mean_val_loss)
        print(f"Training Loss: {train_losses[-1]:.4f}, Validation Loss: {mean_val_loss:.4f}")



    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()
    
    
create_model(X_train, y_train,X_val, y_val, learning_rate = 0.0001,epochs=50)