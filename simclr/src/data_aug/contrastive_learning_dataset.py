from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection
from data_aug.datatransform_builder import get_transform, compose_transform
import argparse
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from src.data_builder import get_dataset_name


class ContrastiveLearningDataset:
    def __init__(self, root_folder, args):
        self.root_folder = root_folder
        self.args = args

    # Construct argument parser
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--transform_type", help="transform type", default="", type=str)
    # ap.add_argument("--dataset", help="dataset", default="icdar", type=str)
    # ap.add_argument("--batch_size", help="batch size", default=64, type=int)

    # # args = vars(ap.parse_args())
    # args = ap.parse_args()

    @staticmethod
    def get_simclr_pipeline_transform(size, transform_type=None):
        # """Return a set of data augmentation transformations as described in the SimCLR paper."""
        # if transform_type == "default":
        #     color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        #     data_transforms = transforms.Compose(
        #         [
        #             transforms.RandomResizedCrop(size=size),
        #             transforms.RandomHorizontalFlip(),
        #             transforms.RandomApply([color_jitter], p=0.8),
        #             transforms.RandomGrayscale(p=0.2),
        #             GaussianBlur(kernel_size=int(0.1 * size)),
        #             transforms.ToTensor(),
        #         ]
        #     )
        # elif transform_type == "randomcrop":
        #     data_transforms = transforms.Compose(
        #         [
        #             transforms.RandomResizedCrop(size=size),
        #             transforms.CenterCrop(size=size),
        #             transforms.ToTensor(),
        #         ]
        #     )
        # else:
        #     raise ValueError("Invalid transform_type. Use 'full' or 'center'.")
        # return data_transforms

        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # output_dir = session.get_trial_dir()

        transforms_dict = compose_transform(transform_type, size)
        logger.info(f"transform type used is: {transforms_dict}")

        # exit()

        return transforms_dict

    def get_dataset(self, name, n_views, transform_type):

        print(name, "YOYOYOYO")

        dataset_root = get_dataset_name(self.args)

        valid_datasets = {
            # "cifar10": lambda: datasets.CIFAR10(
            #     self.root_folder,
            #     train=True,
            #     transform=get_transform(args.transform_type)
            #     download=True,
            # ),
            "stl10": lambda: datasets.STL10(
                self.root_folder,
                split="unlabeled",
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        96, transform_type=transform_type
                    ),
                    n_views,
                ),
                download=True,
            ),
            "icdar": lambda: datasets.ImageFolder(
                root=dataset_root,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        size=96, transform_type=transform_type
                    ),
                    n_views,
                ),
            ),
            "alpub": lambda: datasets.ImageFolder(
                root=dataset_root,
                transform=ContrastiveLearningViewGenerator(
                    self.get_simclr_pipeline_transform(
                        size=96, transform_type=transform_type
                    ),
                    n_views,
                ),
            ),
        }
        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
