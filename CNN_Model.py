import os
import pandas as pd
from utilities.compile_model import compile_model
from utilities.evaluate_model import evaluate_model
from utilities.get_test_data import get_test_data
from utilities.get_train_data import get_train_data
from utilities.make_predictions import make_predictions
from utilities.train_model import train_model


current_dir = os.getcwd()
# Train your model here
folder_path = current_dir + "/Data/Formation_DATA/"

formations = os.listdir(folder_path)
formations = sorted(formations, key=lambda x: int(x))
print(formations)

# Initialize dictionaries to store metrics for each formation and delta_T
recall_metrics = {0.5: [], 1: [], 2: [], 3: [], 6: []}
precision_metrics = {0.5: [], 1: [], 2: [], 3: [], 6: []}
f1_metrics = {0.5: [], 1: [], 2: [], 3: [], 6: []}

models_list=[]
model_metrics = {}

formations = ['1000','7000','9500','10000']

for formation in formations:
    print(f"starting training for formation: {formation}")
    # get training data
    formation_train_data = folder_path+f'{formation}/train/'
    X_train, y_train,X_val, y_val = get_train_data(formation_train_data) 
    # train model at depth
    
    model = compile_model(learning_rate = 0.001, nodes=32, dropout = 0.5, dilation3x3=48, dilation5x5=4,dilation7x7=24,local_bool=True, global_bool=False)
    model = train_model(model,X_train, y_train,X_val, y_val,epochs=200, local_bool=True, global_bool=False)
    models_list.append(model)
    current_dir = os.getcwd()
    model_path = current_dir + f"/Model/{formation}/"
    os.makedirs(model_path, exist_ok=True)
    model.save(model_path) 
    
    # get test data
    formation_test_data = folder_path+f'/{formation}/test/'
    X_test, y_test, test_dfs = get_test_data(formation_test_data)
    
    # make predictions
    for i, df in enumerate(test_dfs):
        make_predictions(df,X_test[i],y_test[i], model, formation_test_data,single_model= True)
    
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