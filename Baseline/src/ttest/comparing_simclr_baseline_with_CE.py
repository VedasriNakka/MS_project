import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import os, json
import logging
from datetime import datetime


os.makedirs('./src/aug_ttest_result', exist_ok=True)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Setup logging configuration
logging.basicConfig(filename=f'./src/aug_ttest_result/simclr_vs_baseline_CE_test_{current_time}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the result files
simclr_path_1 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/SimCLR_finetine_with_ce_loss_unfreeze_backbone_93_agumentations_2024-07-07_21-08-24"
simclr_path_2 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/SimCLR_finetine_with_ce_loss_unfreeze_backbone_93_agumentations_seed2_2024-08-06_06-07-59"

ce_path_1 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_2024-06-26_23-13-46"
ce_path_2 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_seed2_2024-08-01_06-11-53"

# Load data from JSON files
with open(os.path.join(simclr_path_1, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    simclr_data_1 = [json.loads(line) for line in f]

with open(os.path.join(simclr_path_2, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    simclr_data_2 = [json.loads(line) for line in f]

# Function to load data from JSON files
def load_data(path):
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return data

# Load data for SimCLR and Baseline CE
simclr_data_1 = load_data(os.path.join(simclr_path_1, "best_epoch_in_each_trial_by_valid_acc.json"))
simclr_data_2 = load_data(os.path.join(simclr_path_2, "best_epoch_in_each_trial_by_valid_acc.json"))
ce_data_1 = load_data(os.path.join(ce_path_1, "best_epoch_in_each_trial_by_valid_acc.json"))
ce_data_2 = load_data(os.path.join(ce_path_2, "best_epoch_in_each_trial_by_valid_acc.json"))

# Combine the data
simclr_data = simclr_data_1 + simclr_data_2
baseline_ce_data = ce_data_1 + ce_data_2

logging.info(f"SimCLR data points: {len(simclr_data)} \n")
logging.info(f"Baseline CE data points: {len(baseline_ce_data)} \n")

# Function to aggregate test accuracy for each transformation type
def aggregate_accuracy(data):
    results = {}
    for entry in data:
        transform_type = entry["config/transform_type"]
        test_acc = entry["test_acc"]
        if transform_type not in results:
            results[transform_type] = []
        results[transform_type].append(test_acc)
    return results

# Aggregate test accuracy for SimCLR and Baseline CE
simclr_results = aggregate_accuracy(simclr_data)
baseline_ce_results = aggregate_accuracy(baseline_ce_data)


# Assuming the same transformation types for both methods, compare their performance
transform_types = set(simclr_results.keys()).intersection(set(baseline_ce_results.keys()))

# Perform paired t-test and store results
test_results = []

for transform_type in transform_types:
    simclr_acc = simclr_results[transform_type]
    baseline_ce_acc = baseline_ce_results[transform_type]
    
    # Check if both lists have the same length and contain valid data
    if len(simclr_acc) == len(baseline_ce_acc) and len(simclr_acc) > 0:
        if np.isnan(simclr_acc).any() or np.isnan(baseline_ce_acc).any():
            logging.warning(f"Transform {transform_type} has NaN values in SimCLR or Baseline CE.")
            continue
        if np.std(simclr_acc) == 0 and np.std(baseline_ce_acc) == 0:
            logging.warning(f"Transform {transform_type} has identical values in both SimCLR and Baseline CE.")
            continue
        stat, p_value = ttest_rel(simclr_acc, baseline_ce_acc, alternative='greater')
        avg_acc_diff = np.mean(simclr_acc) - np.mean(baseline_ce_acc)
        test_results.append((transform_type, stat, p_value, avg_acc_diff))
        #logging.info(f"Transform: {transform_type}, SimCLR Acc: {simclr_acc}, Baseline CE Acc: {baseline_ce_acc}, Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")

# Log results
logging.info("\n ----Results of the one-sided paired t-tests (SimCLR > Baseline CE): \n")
for transform_type, stat, p_value, avg_acc_diff in test_results:
    logging.info(f"Transform: {transform_type}, Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")

# Check if SimCLR performs better than the baseline CE
better_than_baseline_ce = [result for result in test_results if result[2] < 0.05]

logging.info("-------------------------------------------------------------------------------------")
if better_than_baseline_ce:
    logging.info("\nSome SimCLR configurations perform significantly better than the baseline CE:")
    for transform_type, stat, p_value, avg_acc_diff in better_than_baseline_ce:
        logging.info(f"Transform: {transform_type}, Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")
else:
    logging.info("\nNo SimCLR configuration performs significantly better than the baseline CE.")