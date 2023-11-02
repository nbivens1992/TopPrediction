import tensorflow as tf
from keras import layers, models
from keras.layers import Normalization, Lambda
from keras.models import load_model
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.regularizers import l2
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split

def inception_module(input_tensor, filter_channels):
    # 1x1 conv
    conv1 = layers.Conv1D(filter_channels, 1, padding='same', activation='relu')(input_tensor)
    # 3x3 conv
    conv3 = layers.Conv1D(filter_channels, 3, padding='same', activation='relu')(input_tensor)
    # 5x5 conv
    conv5 = layers.Conv1D(filter_channels, 5, padding='same', activation='relu')(input_tensor)
    # Concatenate filters, assumes filters/channels last
    concat = layers.Concatenate(axis=-1)([conv1, conv3, conv5])
    # Reduce number of channels from 12 to 8 using 1x1 convolution
    reduced_channels = layers.Conv1D(filter_channels*2, 1, padding='same', activation='relu')(concat)
    
    return reduced_channels


def create_global_view(initial_layer = 4, dropout = 0.2):
    inputs = layers.Input(shape=(None, 3))  # Assuming the logs are 1D sequences
    
    # x = layers.Masking(mask_value=-99)(inputs)  # Using -99 as the mask value
    x = inception_module(inputs, initial_layer)
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
def create_local_view(initial_layer = 2):
    inputs = layers.Input(shape=(None, 3))
    x = inception_module(inputs, initial_layer)
    return models.Model(inputs, x)


# from keras import backend as K

# def focal_loss(gamma=2., alpha=0.8):
#     def focal_loss_fixed(y_true, y_pred):
#         pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
#         pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
#         return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
#     return focal_loss_fixed

def train_model(X_train, y_train, X_val, y_val, learning_rate, epochs, nodes, dropout):
    global_view = create_global_view(initial_layer=nodes,dropout = dropout)
    local_view = create_local_view(initial_layer=nodes/2)
    
    # Apply tanh activation to both global and local views
    activated_global_view = layers.Activation('tanh')(global_view.output)
    activated_local_view = layers.Activation('tanh')(local_view.output)

    # Element-wise multiplication of the activated global and local views
    attention_output = layers.Multiply()([activated_global_view, activated_local_view])

    # HYPERPARAMETER: Adjust the number of filters (1) and kernel size (1)
    output = layers.Conv1D(1, 1, activation="sigmoid")(attention_output)

    model = models.Model([global_view.input, local_view.input], output)
    optimizer = Adam(learning_rate=learning_rate)

    # # HYPERPARAMETER: Adjust the optimizer ('adam') and its parameters
    # model.compile(optimizer=optimizer, loss= focal_loss(alpha=alpha))
    
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
    
def smooth_labels(y, sigma=7):
    # Ensure y is a 1D array
    y = y.flatten()
    smoothed_y = np.zeros_like(y, dtype=float)
    for idx in np.where(y == 1)[0]:
        gaussian = np.exp(-0.5 * ((np.arange(len(y)) - idx) / sigma) ** 2)
        gaussian /= gaussian.sum()
        smoothed_y += 17.75* gaussian  # Both are 1D arrays, so this should work
    return smoothed_y

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



def get_train_data(folder_path):
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
    max_length = max(max(x.shape[0] for x in X_train), max(x.shape[0] for x in X_val))

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
    
    return X_train, y_train, X_val, y_val , mean, std


def get_test_data(folder_path, train_mean, train_std):
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
    X_test = [normalize_data(x, train_mean, train_std) for x in X_test]

    # Ensure even length
    X_test = [ensure_even_length(x) for x in X_test]

    # Compute max_length after ensuring even lengths
    max_length = max(x.shape[0] for x in X_test)
    max_length = adjust_max_length(max_length)

    # Pad to max_length
    X_test = [pad_to_max_length(x, max_length) for x in X_test]

    # # Ensure even length for labels and then pad
    y_test = [smooth_labels(y) for y in y_test]
    
    y_test = [ensure_even_length(y.reshape(-1, 1)) for y in y_test]
    y_test = [pad_to_max_length(y, max_length,padding_value=0) for y in y_test]
    
    test_dfs = [pad_dataframe(df, max_length) for df in test_dfs]
    
    # # Update the 'Pick' column in test_dfs with y_test values
    # for i, df in enumerate(test_dfs):
    #     df['Pick'] = y_test[i][:len(df)]  # Assuming y_test[i] is already the correct length
    
    
    return X_test, y_test, test_dfs


def make_predicitons(test_df, X_test,y_test, model, formation_test_data, acceptance):
    x_reshape = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    predictions = model.predict([x_reshape, x_reshape])
    binary_predictions = (predictions >= acceptance).astype(int)
    test_df['Binary_Predict'] = binary_predictions.flatten()
    test_df['Predict'] = predictions.flatten()
    test_df['Acutal Pick']=y_test.flatten()
    name=test_df['SitID'][0]
    
    os.makedirs(formation_test_data+"/Predictions", exist_ok=True)
    # Save the DataFrame as a CSV file
    output_path = os.path.join(formation_test_data+"/Predictions", f"predictions_{name}.csv")
    test_df.to_csv(output_path, index=False)
    

def evaluate_model(test_dfs):
    metrics_at_delta_T = {}
    # Initialize lists to store metrics for each well log
    all_precisions = []
    all_recalls = []
    all_F1_scores = []

    # Define the different delta_T values in terms of depth (e.g., meters)
    delta_T_depth_values = [1, 2, 5, 10, 20]  # Example values in meters

    # Loop over each delta_T value
    for delta_T_depth in delta_T_depth_values:
        # Initialize lists to store metrics for each well log
        all_precisions = []
        all_recalls = []
        all_F1_scores = []

        # Loop over each well log
        for i in range(len(test_dfs)):
            ground_truth_depths = test_dfs[i].loc[test_dfs[i]['Pick'] == 1, 'DEPT']  # Depths of actual formation tops
            prediction_depths = test_dfs[i].loc[test_dfs[i]['Binary_Predict'] == 1, 'DEPT']  # Depths of predicted formation tops

            # 2. Precision
            true_positives = sum(any(abs(gt_depth - prediction_depths) <= delta_T_depth) for gt_depth in ground_truth_depths)
            prec = true_positives / len(prediction_depths) if len(prediction_depths) > 0 else 0
            all_precisions.append(prec)

            # 3. Recall
            detected_tops = sum(any(abs(gt_depth - prediction_depths) <= delta_T_depth) for gt_depth in ground_truth_depths)
            rec = detected_tops / len(ground_truth_depths) if len(ground_truth_depths) > 0 else 0
            all_recalls.append(rec)

            # 4. F1 Score
            if prec + rec > 0:  # Avoid division by zero
                F1 = 2 * (prec * rec) / (prec + rec)
            else:
                F1 = 0
            all_F1_scores.append(F1)

        # Compute average metrics across all well logs
        average_precision = np.mean(all_precisions)
        average_recall = np.mean(all_recalls)
        average_F1_score = np.mean(all_F1_scores)

        # Print the average metrics for the current delta_T_depth
        print(f"Metrics for delta_T_depth = {delta_T_depth}m:")
        print("Average Precision:", average_precision)
        print("Average Recall:", average_recall)
        print("Average F1 Score:", average_F1_score)
        print("----------")
        
         # Store the metrics for the current delta_T_depth in the dictionary
        metrics_at_delta_T[delta_T_depth] = {
            'average_precision': average_precision,
            'average_recall': average_recall,
            'average_f1': average_F1_score
        }
    return metrics_at_delta_T

current_dir = os.getcwd()
# Train your model here
folder_path = current_dir + "/Data/Formation_DATA/"

formations = os.listdir(folder_path)
formations = sorted(formations, key=lambda x: int(x))
print(formations)


# Initialize dictionaries to store metrics for each formation and delta_T
recall_metrics = {1: [], 2: [], 5: [], 10: [], 20: []}
precision_metrics = {1: [], 2: [], 5: [], 10: [], 20: []}
f1_metrics = {1: [], 2: [], 5: [], 10: [], 20: []}

models_list=[]
model_metrics = {}

for formation in formations:
    print(f"starting trainging for formation: {formation}")
    # get training data
    formation_train_data = folder_path+f'{formation}/train/'
    X_train, y_train,X_val, y_val, mean, std = get_train_data(formation_train_data) 
    
    # train model at depth
    model = train_model(X_train, y_train,X_val, y_val, learning_rate = 0.0001,epochs=200, nodes=8, dropout = .2)
    models_list.append(model)
    current_dir = os.getcwd()
    model_path = current_dir + f"/Model/{formation}/"
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path)
    
    # get test data
    formation_test_data = folder_path+f'/{formation}/test/'
    X_test, y_test, test_dfs = get_test_data(formation_test_data, mean, std)
    
    # make predictions
    for i, df in enumerate(test_dfs):
        make_predicitons(df,X_test[i],y_test[i], model, formation_test_data, acceptance = 0.1)
    
    model_deltaT_metrics= evaluate_model(test_dfs)
    # Store metrics for each delta_T
    for delta_T, metrics in model_deltaT_metrics.items():
        recall_metrics[delta_T].append(metrics['average_recall'])
        precision_metrics[delta_T].append(metrics['average_precision'])
        f1_metrics[delta_T].append(metrics['average_f1'])
        
# Convert dictionaries to pandas DataFrames
recall_df = pd.DataFrame(recall_metrics, index=formations)
precision_df = pd.DataFrame(precision_metrics, index=formations)
f1_df = pd.DataFrame(f1_metrics, index=formations)