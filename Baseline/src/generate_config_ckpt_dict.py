
from ray import tune
import os, json
from os.path import basename, normpath

#exp_path = "/home/vedasri/SimCLR/results_hpo/InfoNCE_93_combinations_v1_2024-06-10_09-39-25"
#exp_path = "/home/vedasri/SimCLR/results_hpo/InfoNCE_93_combinations_v1_2024-06-10_09-39-25"
#exp_path ="/home/vedasri/Baseline_V2/results_hpo/final_experiments/Triplet_resnet50_loss_alpub_4_agumentations_2024-07-26_14-06-11"
#exp_path = "/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_loss_alpub_4_agumentations_resnet18_seed2_2024-08-09_16-47-28"
#exp_path = "/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_resnet50_loss_alpub_4_agumentations_seed2_2024-08-10_07-34-50"
#exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_loss_93_agumentations_seed2_2024-08-11_22-43-06'
#exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/CE_loss_resnet18_4_augmentations_alpub_seed2_2024-08-11_11-50-13'
#exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/CE_loss_resnet50_4_augmentations_alpub_seed2_2024-08-11_14-47-45'
#exp_path ='/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_loss_alpub_resnet18_top4_1_agumentations_seed2_1aug_2024-09-04_19-49-59'
#exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_resnet50_loss_alpub_4_1_agumentations_seed_1aug_2024-09-04_19-58-26'
#exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/CE_loss_resnet18_4_aug_alpub_seed2_sorted_4augs_2024-09-05_15-21-49'
#exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/CE_loss_resnet50_4_aug_alpub_seed2_new_4aug_2024-09-05_21-59-36'
exp_path = '/home/vedasri/Baseline_V2/results_hpo/final_experiments_v1/Triplet_resnet50_loss_alpub_4_1_agumentations_seed_1aug_new_2024-09-05_17-45-05'

def runner(config, args):
    pass

trainable = tune.with_resources(
        tune.with_parameters(runner, args=None),
        resources={"cpu": 1, "gpu": 1},
    )

tuner = tune.Tuner.restore(
    path=exp_path,
    trainable=trainable,
    resume_errored=True,

    )   
results = tuner.fit()
trial_config_2_path = {}
 
for result in results:
      
    trial_path = basename(normpath(result.path))
    trial_config = result.config["transform_type"]

    trial_config_2_path[trial_config] =  os.path.join(result.path, 'model_checkpoint_epoch=19.pth')
    #trial_config_2_path[trial_config] =  os.path.join(result.path, 'best_model_checkpoint.pth')

    print(trial_config)

with open(os.path.join(exp_path, "config2ckpt_path.json"), "w") as f:
    json.dump(trial_config_2_path, f, indent=4)