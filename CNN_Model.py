import tensorflow as tf
from keras import layers, models
from keras.layers import Normalization, Lambda
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split


def create_global_view(initial_layer = 8, dropout = 0.2):
    inputs = layers.Input(shape=(None, 3))  # Assuming the logs are 1D sequences
    
    x = layers.Masking(mask_value=-99)(inputs)  # Using -99 as the mask value

    # Encoding layer 1
    conv1 = layers.Conv1D(initial_layer, 3, activation="relu", padding="same")(x)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Dropout(dropout)(conv1)
    pool1 = layers.MaxPooling1D()(conv1)

    # Encoding layer 2
    conv2 = layers.Conv1D(2*initial_layer, 3, activation="relu", padding="same")(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Dropout(dropout)(conv2)
    pool2 = layers.MaxPooling1D()(conv2)

    # Middle layer
    conv_middle = layers.Conv1D(4*initial_layer, 3, activation="relu", padding="same")(pool2)
    conv_middle = layers.BatchNormalization()(conv_middle)
    
    conv_middle = layers.Dropout(dropout)(conv_middle)

    # Decoding layer 1
    up1 = layers.UpSampling1D()(conv_middle)
    
    # # Ensure sizes match before concatenating
    # cropped_conv2 = layers.Cropping1D(cropping=((1,0)))(conv2)
    merge1 = layers.concatenate([conv2, up1])
    
    decode1 = layers.Conv1D(2*initial_layer, 3, activation="relu", padding="same")(merge1)
    decode1 = layers.BatchNormalization()(decode1)
    decode1 = layers.Dropout(dropout)(decode1)

    # Decoding layer 2
    up2 = layers.UpSampling1D()(decode1)
    
    # Ensure sizes match before concatenating for the second merge
    # cropped_conv1 = layers.Cropping1D(cropping=((2,0)))(conv1)  # Adjusted cropping
    merge2 = layers.concatenate([conv1, up2])

    decode2 = layers.Conv1D(initial_layer, 3, activation="relu", padding="same")(merge2)
    decode2 = layers.BatchNormalization()(decode2)
    decode2 = layers.Dropout(dropout)(decode2)

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

# Normalize function
def normalize_data(data, mean, std):
    return (data - mean) / std


def ensure_even_length(sequence, padding_value=-99):
    # If the sequence length is odd, append padding to make it even
    if sequence.shape[0] % 2 != 0:
        padding_shape = list(sequence.shape)
        padding_shape[0] = 1  # we only need to add one row
        
        # Create a padding with the same number of columns as the sequence
        padding = np.full(padding_shape, padding_value)
        
        # Append padding to the sequence
        sequence = np.vstack([sequence, padding])
        
    return sequence

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
    
def smooth_labels(y, sigma=1.5):
    # Ensure y is a 1D array
    y = y.flatten()
    smoothed_y = np.zeros_like(y, dtype=float)
    for idx in np.where(y == 1)[0]:
        gaussian = np.exp(-0.5 * ((np.arange(len(y)) - idx) / sigma) ** 2)
        gaussian /= gaussian.sum()
        smoothed_y += 4* gaussian  # Both are 1D arrays, so this should work
    return smoothed_y


current_dir = os.getcwd()
# Train your model here
folder_path = current_dir + "/Data/Formation_DATA/"
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dfs = []
for file in files:
    df = pd.read_csv(folder_path + file)
    dfs.append(df)

train_dfs, val_dfs = train_test_split(dfs, test_size=0.2, random_state=42) # Here 20% of the data is kept for validation

features = ["GR", "ILD", "DPHI"]

# Compute the mean and standard deviation of the training data
all_train_data = np.vstack([df[features].values for df in train_dfs])
mean = np.mean(all_train_data, axis=0)
std = np.std(all_train_data, axis=0)

# Extract features and labels for training and validation sets
X_train = np.array([df[features].values for df in train_dfs])
y_train = np.array([df["Pick"].values for df in train_dfs])

X_val = np.array([df[features].values for df in val_dfs])
y_val = np.array([df["Pick"].values for df in val_dfs])


# Normalize data
X_train = [normalize_data(x, mean, std) for x in X_train]
X_val = [normalize_data(x, mean, std) for x in X_val]

# Ensure even length
X_train = [ensure_even_length(x) for x in X_train]
X_val = [ensure_even_length(x) for x in X_val]

# Compute max_length after ensuring even lengths
max_length = max(max(x.shape[0] for x in X_train), max(x.shape[0] for x in X_val))+2

# The UNET structure requires the total size of the input when divided by 4  or divided by 2 be even. 
max_length = adjust_max_length(max_length)

# Pad to max_length
X_train = [pad_to_max_length(x, max_length) for x in X_train]
X_val = [pad_to_max_length(x, max_length) for x in X_val]

# # smooth y values
y_train = [smooth_labels(y) for y in y_train]
y_val = [smooth_labels(y) for y in y_val]

# Ensure even length for labels and then pad
y_train = [ensure_even_length(y.reshape(-1, 1),padding_value=0) for y in y_train]
y_val = [ensure_even_length(y.reshape(-1, 1),padding_value=0) for y in y_val]

y_train = [pad_to_max_length(y, max_length,padding_value=0) for y in y_train]
y_val = [pad_to_max_length(y, max_length,padding_value=0) for y in y_val]




# def focal_loss(gamma=2., alpha=0.8):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed

def train_model(X_train, y_train, X_val, y_val, learning_rate, epochs):
    global_view = create_global_view()
    local_view = create_local_view()

    combined = layers.Multiply()([global_view.output, local_view.output])

    # Soft Attention
    attention_output = layers.Activation("tanh")(combined)

    # HYPERPARAMETER: Adjust the number of filters (1) and kernel size (1)
    output = layers.Conv1D(1, 1, activation="sigmoid")(attention_output)

    model = models.Model([global_view.input, local_view.input], output)
    optimizer = Adam(learning_rate=learning_rate)

    # # HYPERPARAMETER: Adjust the optimizer ('adam') and its parameters
    # model.compile(optimizer=optimizer, loss= focal_loss())
    
    # HYPERPARAMETER: Adjust the optimizer ('adam') and its parameters
    model.compile(optimizer=optimizer, loss= tf.losses.BinaryCrossentropy())


    # Early stopping parameters
    patience = 5
    min_delta = 0.001
    best_val_loss = float('inf')
    wait = 0
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
        mean_train_loss = np.mean(epoch_train_losses)
        train_losses.append(np.mean(mean_train_loss))

        # Validation (optional)
        epoch_val_losses = []
        for i in range(len(X_val)):
            X_val_well = X_val[i].reshape(1, X_val[i].shape[0], X_val[i].shape[1])
            y_val_well = y_val[i].reshape(1, y_val[i].shape[0], 1)
            loss = model.test_on_batch([X_val_well, X_val_well], y_val_well)
            epoch_val_losses.append(loss)
        mean_val_loss = np.mean(epoch_val_losses)
        val_losses.append(mean_val_loss)
        print(f"Training Loss: {train_losses[-1]:.5f}, Validation Loss: {mean_val_loss:.5f}")

        # Early stopping check
        if (best_val_loss - mean_val_loss) > min_delta:
            best_val_loss = mean_val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}. Best validation loss: {best_val_loss:.5f}")
                break

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()
    
    return model
    
    
model = train_model(X_train, y_train,X_val, y_val, learning_rate = 0.001,epochs=200)


current_dir = os.getcwd()
# Train your model here
model_path = current_dir + "/Model/"

model.save(model_path)

current_dir = os.getcwd()
# Train your model here
model_path = current_dir + "/Model/"

# Load the model
loaded_model = load_model(model_path, )


current_dir = os.getcwd()
# Train your model here
folder_path = current_dir + "/Data/Formation_DATA/"
files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

test_dfs = []
for file in files:
    df = pd.read_csv(folder_path + file)
    test_dfs.append(df)

features = ["GR", "ILD", "DPHI"]


# Extract features and labels for training and validation sets
X_test = np.array([df[features].values for df in test_dfs])
y_test = np.array([df["Pick"].values for df in test_dfs])

# Normalize data
X_test = [normalize_data(x, mean, std) for x in X_test]

# Ensure even length
X_test = [ensure_even_length(x) for x in X_test]

# Compute max_length after ensuring even lengths
max_length = max(max(x.shape[0] for x in X_train), max(x.shape[0] for x in X_test))
max_length = adjust_max_length(max_length)



# Pad to max_length
X_test = [pad_to_max_length(x, max_length) for x in X_test]

# # Ensure even length for labels and then pad
y_test = [smooth_labels(y) for y in y_test]
y_test = [ensure_even_length(y.reshape(-1, 1),padding_value=0) for y in y_test]
y_test = [pad_to_max_length(y, max_length,padding_value=0) for y in y_test]


folder_path = current_dir + "/Data/Formation_DATA/"

# Ensure the directory exists
output_dir = os.path.join(current_dir, folder_path, "Predictions")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Loop through X_test and y_test
for df in test_dfs:
    name=df['SitID'][0]
    # Save the DataFrame as a CSV file
    output_path = os.path.join(output_dir, f"predictions_{name}.csv")
    df.to_csv(output_path, index=False)
