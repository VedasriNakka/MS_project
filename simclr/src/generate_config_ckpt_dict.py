
from ray import tune
import os, json
from os.path import basename, normpath

#exp_path = "/home/vedasri/SimCLR/results_hpo/InfoNCE_93_combinations_v1_2024-06-10_09-39-25"
#exp_path = "/home/vedasri/SimCLR/results_hpo/InfoNCE_93_combinations_v1_2024-06-10_09-39-25"
exp_path ="/home/vedasri/SimCLR_V2/results_hpo/final_experiments/icdar_resize198_93_combinations_100epochs_resnet50_2024-07-20_06-16-11"


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

    trial_config_2_path[trial_config] =  os.path.join(result.path, 'best_checkpoint.pth.tar')

    print(trial_config)

with open(os.path.join(exp_path, "config2ckpt_path.json"), "w") as f:
    json.dump(trial_config_2_path, f, indent=4)