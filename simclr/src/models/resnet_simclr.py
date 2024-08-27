import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch.nn as nn
import torchvision.models as models
import torch

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, use_pretrained=True):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(
                pretrained=use_pretrained,
                # num_classes=1000 out_dim
            ),
            "resnet34": models.resnet34(
                pretrained=use_pretrained,
                # num_classes=out_dim
            ),
            "resnet50": models.resnet50(
                pretrained=use_pretrained,
                # num_classes=out_dim
            ),
        }

        self.backbone = self._get_basemodel(base_model, out_dim)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc
        )

        logger.info(f"Model {base_model} is instantiated!")

        logger.info(f"using Imagenet pretrained weights: {use_pretrained}")

        # exit()

    def _get_basemodel(self, model_name, num_classes):
        try:
            model = self.resnet_dict[model_name]
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50"
            )
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
