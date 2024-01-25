import os
import numpy as np
from keras import Model


def make_predictions(test_df, X_test,y_test, model: Model, formation_test_data,single_model=False):
    x_reshape = X_test.reshape(1, X_test.shape[0], X_test.shape[1])
    if single_model:
        predictions = model.predict(x_reshape)
    else:
        predictions = model.predict([x_reshape, x_reshape])
    # predictions = model.predict([x_reshape])
    
    # Flatten the predictions to 1D array for easier indexing
    predictions_flat = predictions.flatten()
    
    # Initialize binary_predictions to all zeros
    binary_predictions = np.zeros_like(predictions_flat, dtype=int)
    
    # Find the index of the maximum prediction probability
    max_prob_index = np.argmax(predictions_flat)
    
    # Set only the max index to 1
    binary_predictions[max_prob_index] = 1
    
    test_df['Binary_Predict'] = binary_predictions.flatten()
    test_df['Predict'] = predictions.flatten()
    test_df['Acutal Pick']=y_test.flatten()
    name=test_df['SitID'][0]
    
    os.makedirs(formation_test_data+"/Predictions", exist_ok=True)
    # Save the DataFrame as a CSV file
    output_path = os.path.join(formation_test_data+"/Predictions", f"predictions_{name}.csv")
    test_df.to_csv(output_path, index=False)