import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import os, json
import logging
from datetime import datetime

# Create directory for results if it doesn't exist
os.makedirs('./src/aug_ttest_result', exist_ok=True)

# Get current time for logging
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Setup logging configuration
logging.basicConfig(
    filename=f'./src/aug_ttest_result/simclr_avegare_sorted_result_{current_time}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Paths to the result files
results_path_1 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/SimCLR_finetine_with_ce_loss_unfreeze_backbone_93_agumentations_2024-07-07_21-08-24"
results_path_2 = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/SimCLR_finetine_with_ce_loss_unfreeze_backbone_93_agumentations_seed2_2024-08-06_06-07-59"


# Load data from JSON files
with open(os.path.join(results_path_1, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    data_1 = [json.loads(line) for line in f]

with open(os.path.join(results_path_2, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
    data_2 = [json.loads(line) for line in f]

# Combine the data
all_data = data_1 + data_2
logging.info(f"Length of all data: {len(all_data)}")

# Aggregate test accuracy for each augmentation
augmentation_results = {}
for entry in all_data:
    transform_type = entry["config/transform_type"]
    test_acc = entry["test_acc"]
    if transform_type not in augmentation_results:
        augmentation_results[transform_type] = []
    augmentation_results[transform_type].append(test_acc)

# Calculate the average test accuracy for each augmentation
average_acc = {transform: np.mean(accs) for transform, accs in augmentation_results.items()}


for i, (aug, acc) in  enumerate(average_acc.items()):
    logging.info(f"Augumentation combination: {i + 1}")
    logging.info(f"Other augmentation '{aug}' length: {acc}")


# Sort augmentations by their average test accuracy
sorted_augmentations = sorted(average_acc.items(), key=lambda item: item[1], reverse=True)


logging.info("\n\n -------------------------------- \n\n")
# Log the sorted results
logging.info("Sorted Augmentations by Average Test Accuracy:")
for transform, avg_acc in sorted_augmentations:
    logging.info(f"Augmentation: {transform}, Average Test Accuracy: {avg_acc:.4f}")

# Optionally, save the sorted results to a CSV file
sorted_df = pd.DataFrame(sorted_augmentations, columns=['Augmentation', 'Average Test Accuracy'])
sorted_df.to_csv('./src/aug_ttest_result/sorted_augmentations.csv', index=False)