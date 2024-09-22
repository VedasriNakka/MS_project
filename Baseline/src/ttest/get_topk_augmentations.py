import pandas as pd
import numpy as np

# Simulated performance data (accuracy scores) for a single run
np.random.seed(0)
import os, json
import logging
from datetime import datetime


os.makedirs('./src/aug_ttest_result/wi', exist_ok=True)

current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Setup logging configuration
logging.basicConfig(filename=f'./src/aug_ttest_result/wi/simclr_18_93aug_difference_{current_time}.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    results_path = "/home/vedasri/Baseline_V2/results_hpo/final_experiments_v2/SimCLR_18_finetine_with_ce_loss_unfreeze_backbone_93_aug_wi_2024-09-12_13-43-34"

    with open(os.path.join(results_path, "best_epoch_in_each_trial_by_valid_acc.json"), "r") as f:
        data = [json.loads(line) for line in f]
    performance_data = { data_line["config/transform_type"] :data_line["test_acc"] for data_line in data}




# Convert to DataFrame
df = pd.DataFrame(list(performance_data.items()), columns=["Augmentation", "Accuracy"])

logging.info(df)

# Calculate the difference from the baseline (randomcrop198)
baseline_accuracy = df[df["Augmentation"] == "randomcrop224"]["Accuracy"].values[0]

logging.info(f"baseline_accuracy: {baseline_accuracy}")
df["Difference"] = df["Accuracy"] - baseline_accuracy

# Sort the augmentations based on their performance difference
df_sorted = df.sort_values(by="Difference", ascending=False)

# Select the top 10 augmentations
top_10_augmentations = df_sorted[df_sorted["Augmentation"] != "randomcrop224"].head(10)
logging.info(f"top_10_augmentations:----> {top_10_augmentations}")

# Return the names of the top 10 augmentations
top_10_augmentations_list = top_10_augmentations["Augmentation"].tolist()
logging.info("Top 10 Augmentations compared to 'randomcrop224':")
for augmentation in top_10_augmentations_list:
    logging.info(augmentation)
