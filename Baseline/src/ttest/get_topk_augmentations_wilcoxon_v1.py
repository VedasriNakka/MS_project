import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

# Simulated performance data (accuracy scores) for a single run
np.random.seed(0)

if False:
    performance_data = {
        "randomcrop198": np.random.normal(0.85, 0.01, 1)[0],
        "randomcrop198,morpho_erosion": np.random.normal(0.86, 0.01, 1)[0],
        "randomcrop198,morpho_dilation": np.random.normal(0.87, 0.01, 1)[0],
        "randomcrop198,affine": np.random.normal(0.88, 0.01, 1)[0],
        "randomcrop198,colorjitter": np.random.normal(0.89, 0.01, 1)[0],
        "randomcrop198,hflip": np.random.normal(0.90, 0.01, 1)[0],
        "randomcrop198,invert": np.random.normal(0.91, 0.01, 1)[0],
        "randomcrop198,gaussianblur": np.random.normal(0.92, 0.01, 1)[0],
        "randomcrop198,gray": np.random.normal(0.93, 0.01, 1)[0],
        # Additional augmentations
        "randomcrop198,morpho_erosion,morpho_dilation": np.random.normal(0.87, 0.01, 1)[0],
        "randomcrop198,morpho_erosion,affine": np.random.normal(0.86, 0.01, 1)[0],
        "randomcrop198,morpho_erosion,colorjitter": np.random.normal(0.88, 0.01, 1)[0],
        "randomcrop198,morpho_erosion,hflip": np.random.normal(0.87, 0.01, 1)[0],
        "randomcrop198,morpho_erosion,invert": np.random.normal(0.86, 0.01, 1)[0],
        "randomcrop198,morpho_erosion,gaussianblur": np.random.normal(0.88, 0.01, 1)[0],
        "randomcrop198,morpho_erosion,gray": np.random.normal(0.87, 0.01, 1)[0],
        "randomcrop198,morpho_dilation,affine": np.random.normal(0.89, 0.01, 1)[0],
        "randomcrop198,morpho_dilation,colorjitter": np.random.normal(0.90, 0.01, 1)[0],
        "randomcrop198,morpho_dilation,hflip": np.random.normal(0.91, 0.01, 1)[0],
        "randomcrop198,morpho_dilation,invert": np.random.normal(0.92, 0.01, 1)[0],
        "randomcrop198,morpho_dilation,gaussianblur": np.random.normal(0.93, 0.01, 1)[0],
        "randomcrop198,morpho_dilation,gray": np.random.normal(0.94, 0.01, 1)[0],
    }

else:
    import os, json
    results_path = "/home/vedasri/Baseline_V2/results_hpo/final_experiments/SimCLR_finetine_with_ce_loss_unfreeze_backbone_93_agumentations_2024-07-07_21-08-24"

    # Load the data
    with open(os.path.join(results_path, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
        data = [json.loads(line) for line in f]

# Extract base augmentation accuracy
base_augmentation = data[0]
base_acc = base_augmentation["test_acc"]

# Extract and combine accuracy values for other augmentations
all_acc = []
augmentation_types = []
for entry in data[1:]:
    combined_acc = entry["test_acc"]
    augmentation_types.append(entry["config/transform_type"])
    all_acc.append(combined_acc)

# Convert to numpy arrays
base_acc = np.array(base_acc)
all_acc = np.array(all_acc)

p_values = []
differences = []

# Perform the Wilcoxon signed-rank test
for i in range(len(all_acc)):
    diff = all_acc[i] - base_acc
    differences.append(diff)
    # Perform the test with a single value; no need to use a list
    stat, p_value = wilcoxon([diff])
    p_values.append(p_value)

# Convert to numpy arrays for easier manipulation
differences = np.array(differences)
p_values = np.array(p_values)

# Sort indices based on performance differences
sorted_indices = np.argsort(differences)[-4:]
top_4_indices = sorted_indices[-4:]

# Get top 4 performances, their types, and p-values
top_4_performances = all_acc[top_4_indices]
top_4_augmentation_types = [augmentation_types[i] for i in top_4_indices]
top_4_p_values = p_values[top_4_indices]

# Output the results
print("Top 4 Augmentations' Performances:", top_4_performances)
print("Top 4 Augmentation Types:", top_4_augmentation_types)
print("Corresponding p-values:", top_4_p_values)