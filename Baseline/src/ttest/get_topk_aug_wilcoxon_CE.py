import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Simulated performance data (accuracy scores) for a single run
np.random.seed(0)
import os, json


# Load the data from both folders
# folder1_path = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_2024-06-26_23-13-46"
# folder2_path = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_seed2_2024-08-01_06-11-53"


# def load_data(folder_path):
#     data = []
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".json"):
#             with open(os.path.join(folder_path, file_name), "r") as f:
#                 for line in f:
#                     data.append(json.loads(line))
#     return data

# data_folder1 = load_data(folder1_path)
# data_folder2 = load_data(folder2_path)
# all_data = data_folder1 + data_folder2


results_path_1 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_2024-06-26_23-13-46"
with open(os.path.join(results_path_1, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    data_1 = [json.loads(line) for line in f]


results_path_2 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/CE_loss_93_augmentations_seed2_2024-08-01_06-11-53"
with open(os.path.join(results_path_2, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    data_2 = [json.loads(line) for line in f]


all_data = data_1 + data_2

print("Length of all data: \n",len(all_data))


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
print("Base augmentation length: \n" , base_acc)

other_augmentations = {k: v for k, v in augmentation_results.items() if k != base_augmentation}


# Perform Wilcoxon signed-rank test and store results
test_results = []

for i, (aug, acc) in enumerate(other_augmentations.items()):
    print(f"Loop number: {i + 1}")
    print(f"Other augmentation: '{aug}' length: {acc}")
    if len(base_acc) == len(acc):
        stat, p_value = wilcoxon(base_acc, acc)
        avg_acc_diff = np.mean(acc) - np.mean(base_acc)
        test_results.append((aug, stat, p_value, avg_acc_diff))


# Sort results by p-value and average accuracy difference
test_results.sort(key=lambda x: (x[2], -x[3]))

# Select the top 4 augmentations
top_4_augmentations = test_results[:4]

# Print results
for aug, stat, p_value, avg_acc_diff in top_4_augmentations:
    print(f"Augmentation (Transform: {aug}): Statistic={stat}, p-value={p_value}, Avg Acc Diff={avg_acc_diff}")

