import numpy as np


def evaluate_model(test_dfs):
    metrics_at_delta_T = {}
    # Initialize lists to store metrics for each well log
    all_precisions = []
    all_recalls = []
    all_F1_scores = []

    # Define the different delta_T values in terms of depth (e.g., meters)
    delta_T_depth_values = [0.5, 1, 2, 3, 6]  # Example values in meters

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
            true_positives = sum(np.any(np.abs(gt_depth - prediction_depths) <= delta_T_depth) for gt_depth in ground_truth_depths)
            prec = true_positives / len(prediction_depths) if len(prediction_depths) > 0 else 0
            all_precisions.append(prec)

            # 3. Recall
            detected_tops = sum(np.any(np.abs(gt_depth - prediction_depths) <= delta_T_depth) for gt_depth in ground_truth_depths)
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