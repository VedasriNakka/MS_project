import yaml
from ray import tune


def parse_search_space(config_path):
    """
    Parse yaml format configuration to generate search space

    Arguments:
        config_path (str): the path of the yaml file.
    Return:
        ConfigSpace object: the search space.

    """

    with open(config_path, "r") as ips:
        raw_ss_config = yaml.load(ips, Loader=yaml.FullLoader)

    search_space_dict = {}

    # Add hyperparameters
    for name in raw_ss_config.keys():

        hyper_param = raw_ss_config[name]

        # print(hyper_param)

        hyper_type = hyper_param["type"]

        if hyper_type == "grid":
            search_space_dict[name] = tune.grid_search(hyper_param["choices"])

        else:
            raise NotImplementedError

    return search_space_dict
