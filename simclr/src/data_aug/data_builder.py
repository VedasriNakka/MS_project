def get_dataset(args):
    if args.dataset_name == "icdar":
        train_directory = (
            "/home/vedasri/datasets/HomerCompTraining_Cropped"  #'imds_small/train'
        )

    elif args.dataset_name == "alpub":
        train_directory = "/home/vedasri/datasets/alpub_v2/images"  #'imds_small/train'
    else:
        raise NotImplementedError

    return train_directory
