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

    with open(os.path.join(results_path, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
        data = [json.loads(line) for line in f]

base_augmentation = data[0]
base_acc = [base_augmentation["train_acc"], base_augmentation["valid_acc"], base_augmentation["test_acc"]]

# Extract and combine accuracy values for other augmentations
all_acc = []
for entry in data[1:]:
    combined_acc = [entry["train_acc"], entry["valid_acc"], entry["test_acc"]]
    all_acc.append((entry["config/transform_type"], combined_acc))

base_acc = np.array(base_acc)

# Perform the Wilcoxon signed-rank test
wilcoxon_results = []
for transform_type, acc in all_acc:
    acc = np.array(acc)
    differences = base_acc - acc
    stat, p_value = wilcoxon(base_acc, acc)
    wilcoxon_results.append((transform_type, stat, p_value, differences))

# Sort results by p-value in ascending order
wilcoxon_results.sort(key=lambda x: x[2])

# Select the top 4 augmentations with the lowest p-values
top4_augmentations = wilcoxon_results[:10]

# Display the results
for transform_type, stat, p_value, differences in top4_augmentations:
    print(f"Augmentation (Transform: {transform_type}): Statistic={stat}, p-value={p_value}, Differences={differences}")