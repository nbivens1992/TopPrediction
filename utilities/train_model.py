import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np


def train_model(model, X_train, y_train, X_val, y_val, epochs, local_bool= True, global_bool =True):    
    # Early stopping parameters
    patience = 5
    min_delta = 0.001
    best_val_loss = float('inf')
    wait = 0
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # We'll only use this training loop and remove the redundant one.
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        epoch_train_losses = []
        epoch_train_accuracies = []
        for i in range(len(X_train)):
            X_train_well = X_train[i].reshape(1, X_train[i].shape[0], X_train[i].shape[1])
            y_train_well = y_train[i].reshape(1, y_train[i].shape[0], 1)
            if global_bool and local_bool:
                loss = model.train_on_batch([X_train_well, X_train_well], y_train_well)  # Capturing the loss
                train_accuracy = model.evaluate([X_train_well, X_train_well], y_train_well, verbose=0)[1]
            else:
                loss = model.train_on_batch(X_train_well, y_train_well)  # Capturing the loss
                train_accuracy = model.evaluate(X_train_well, y_train_well, verbose=0)[1]
            # loss = model.train_on_batch([X_train_well], y_train_well)  # Capturing the loss
            epoch_train_losses.append(loss)
            epoch_train_accuracies.append(train_accuracy)
        mean_train_loss = np.mean(epoch_train_losses)
        train_losses.append(mean_train_loss)
        
        mean_train_accuracy = np.mean(epoch_train_accuracies)
        train_accuracies.append(mean_train_accuracy)

        # Validation (optional)
        epoch_val_losses = []
        epoch_test_accuracies = []
        for i in range(len(X_val)):
            X_val_well = X_val[i].reshape(1, X_val[i].shape[0], X_val[i].shape[1])
            y_val_well = y_val[i].reshape(1, y_val[i].shape[0], 1)
            if global_bool and local_bool:
                loss = model.test_on_batch([X_val_well, X_val_well], y_val_well)
                test_accuracy = model.evaluate([X_val_well, X_val_well], y_val_well, verbose=0)[1]
            else:
                loss = model.test_on_batch(X_val_well, y_val_well)   
                test_accuracy = model.evaluate(X_val_well, y_val_well, verbose=0)[1]
            epoch_val_losses.append(loss)
            epoch_test_accuracies.append(test_accuracy)
        mean_val_loss = np.mean(epoch_val_losses)
        val_losses.append(mean_val_loss)
        
        mean_val_accuracy = np.mean(epoch_test_accuracies)
        val_accuracies.append(mean_val_accuracy)

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
    
    plt.ioff()
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.show()
    
    # Plotting accuracies
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curves")
    plt.legend()
    plt.show()
    
    return model