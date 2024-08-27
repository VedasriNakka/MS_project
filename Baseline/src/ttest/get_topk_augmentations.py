import pandas as pd
import numpy as np

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
    performance_data = { data_line["config/transform_type"] :data_line["test_acc"] for data_line in data}




# Convert to DataFrame
df = pd.DataFrame(list(performance_data.items()), columns=["Augmentation", "Accuracy"])

print(df)

# Calculate the difference from the baseline (randomcrop198)
baseline_accuracy = df[df["Augmentation"] == "randomcrop224"]["Accuracy"].values[0]

print(f"baseline_accuracy: {baseline_accuracy}")
df["Difference"] = df["Accuracy"] - baseline_accuracy

# Sort the augmentations based on their performance difference
df_sorted = df.sort_values(by="Difference", ascending=False)

# Select the top 10 augmentations
top_10_augmentations = df_sorted[df_sorted["Augmentation"] != "randomcrop224"].head(10)
print(f"top_10_augmentations:----> {top_10_augmentations}")

# Return the names of the top 10 augmentations
top_10_augmentations_list = top_10_augmentations["Augmentation"].tolist()
print("Top 10 Augmentations compared to 'randomcrop224':")
for augmentation in top_10_augmentations_list:
    print(augmentation)
