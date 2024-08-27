import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import os, json
import logging
from datetime import datetime


os.makedirs('./src/aug_ttest_result', exist_ok=True)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Setup logging configuration
logging.basicConfig(filename=f'./src/aug_ttest_result/baseline_CE_augmentation_test_{current_time}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the result files
results_path_1 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_2024-06-26_23-13-46"
results_path_2 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_seed2_2024-08-01_06-11-53"

# Load data from JSON files
with open(os.path.join(results_path_1, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    data_1 = [json.loads(line) for line in f]

with open(os.path.join(results_path_2, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    data_2 = [json.loads(line) for line in f]

# Combine the data
all_data = data_1 + data_2
logging.info("Length of all data:", len(all_data))

# Aggregate test accuracy for each augmentation
augmentation_results = {}
for entry in all_data:
    transform_type = entry["config/transform_type"]
    test_acc = entry["test_acc"]
    if transform_type not in augmentation_results:
        augmentation_results[transform_type] = []
    augmentation_results[transform_type].append(test_acc)

# Separate base augmentation and other augmentations
base_augmentation = "randomcrop224"
base_acc = augmentation_results.get(base_augmentation, [])
logging.info(f"Base agumentation accuracies : \n {base_acc}")

other_augmentations = {k: v for k, v in augmentation_results.items() if k != base_augmentation}

# Perform one-sided t-test and store results
test_results = []

for i, (aug, acc) in enumerate(other_augmentations.items()):
    logging.info(f"Augumentation combination: {i + 1}")
    logging.info(f"Other augmentation '{aug}' length: {acc}")
    if len(base_acc) == len(acc):
        stat, p_value = ttest_rel(base_acc, acc, alternative='greater')
        avg_acc_diff = np.mean(base_acc) - np.mean(acc)
        test_results.append((aug, stat, p_value, avg_acc_diff))

# Check if any augmentation performs better than the base augmentation
better_than_base = [result for result in test_results if result[2] < 0.05]

# Print results
logging.info("Results of the one-sided t-tests (base > other):")
for aug, stat, p_value, avg_acc_diff in test_results:
    logging.info(f"Augmentation (Transform: {aug}): Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")

# Sort results by p-value and average accuracy difference
test_results.sort(key=lambda x: (x[2], -x[3]))

# Select the top 4 augmentations
top_4_augmentations = test_results[:4]
logging.info("\n----------------------------------------------------------------\n")

logging.info("Top-4 Augmentations: ")
# Print results
for aug, stat, p_value, avg_acc_diff in top_4_augmentations:
    logging.info(f"Augmentation (Transform: {aug}): Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")


if better_than_base:
    logging.info("\nSome augmentations perform significantly better than the base augmentation:")
    for aug, stat, p_value, avg_acc_diff in better_than_base:
        logging.info(f"Augmentation (Transform: {aug}): Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")
else:
    logging.info("\nNo augmentation performs significantly better than the base augmentation.")
