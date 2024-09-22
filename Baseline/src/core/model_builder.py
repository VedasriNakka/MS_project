import torch.nn as nn
from nets import *
from torchvision import datasets, models, transforms
import json


import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
import torch.nn as nn
import torchvision.models as models
import torch

class BaseSimCLRException(Exception):
    """Base exception"""

class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""




class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim, num_classes, use_pretrained=True):
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

        self.classification_layer = nn.Linear(out_dim, num_classes)
        logger.info(f"Model {base_model} is instantiated!")

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

    def forward(self, x,):
        features = self.backbone(x)
        logits = self.classification_layer(features)

        return {"logits":logits, "features": features}





class ResNet(nn.Module):

    def __init__(self, arch, num_classes, use_pretrained, args=None):
        super(ResNet, self).__init__()
        self.resnet_dict = {
            "resnet18": models.resnet18(
                pretrained=use_pretrained,
            ),
            "resnet34": models.resnet34(
                pretrained=use_pretrained,
            ),
            "resnet50": models.resnet50(
                pretrained=use_pretrained,
            ),
        }

    

        self.args = args
        self.backbone = self.resnet_dict[arch]
        infeatures = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()


        if self.args.loss_fn =="ce":
            self.classification_layer = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
           nn.Linear(infeatures, num_classes)
        )
        elif self.args.loss_fn == "triplet":
            self.projection_layer = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(infeatures, self.args.triplet_embedding_size)
            )
        elif self.args.loss_fn == "triplet_finetune_with_ce":
            self.projection_layer = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(infeatures, self.args.triplet_embedding_size)
            )

            self.classification_layer = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(self.args.triplet_embedding_size, num_classes)
            )
        else:
            raise NotImplementedError()
    


        

    def forward(self, x,):
        features = self.backbone(x)

        if self.args.loss_fn == "ce":
            logits = self.classification_layer(features)
            return {"logits":logits, "features": features}
        
        elif self.args.loss_fn == "triplet":
            embeddings = self.projection_layer(features)
            return {"features": embeddings}
        elif self.args.loss_fn == "triplet_finetune_with_ce":
            embeddings = self.projection_layer(features)
            logits = self.classification_layer(embeddings)

            return {"features": embeddings, "logits":logits}
        

        else:
            raise NotImplementedError()




def get_model(args, num_classes, logger):

    train_mode = args.mode


    logger.info(args)

    if train_mode == "finetune":
        model_ft = ResNet(args.arch, num_classes=num_classes, use_pretrained=args.use_imagenet_pretrained_weights, args=args )
        logger.info(f"Model {model_ft} is instantiated!")
        logger.info(f"Loading imagenet pretrained weights: {args.use_imagenet_pretrained_weights}")



    elif train_mode == "triplet_finetune_with_ce" or train_mode == "finetune_with_ce":
        model_ft = ResNet(args.arch, num_classes=num_classes, use_pretrained=args.use_imagenet_pretrained_weights, args=args )
        logger.info(f"Model {model_ft} is instantiated!")
        logger.info(f"Loading imagenet pretrained weights: {args.use_imagenet_pretrained_weights}")
       

        with open(args.ckpt_path_dict, 'r') as f:
            transform2ckpt_dict = json.load(f)

        ckpt_path = transform2ckpt_dict[args.transform_type]
            
        weights = torch.load(ckpt_path)['model_state_dict']

        
        if train_mode == "finetune_with_ce":
            # Get the state dictionary of the model
            model_dict = model_ft.state_dict()

            # Filter out keys from the pretrained weights that don't match the current model's layers
            pretrained_dict = {k: v for k, v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}


            logger.info(f"loading the weights of keys: {pretrained_dict.keys()}")

            # Update the model's state dictionary with the filtered pretrained weights
            model_dict.update(pretrained_dict)

            # Load the updated state dictionary into the model
            model_ft.load_state_dict(model_dict)

        else:
            
            model_ft.load_state_dict(weights, strict=False)

            missing_keys, unexpected_keys = model_ft.load_state_dict(weights, strict=False) # changing strict to True
            logger.info(f"loading ckpt from {ckpt_path}")

            # Print missing and unexpected keys
            if missing_keys:
                logger.info("Missing keys:")
                for key in missing_keys:
                    logger.info(key)
            else:
                logger.info("All keys matched successfully.")

            if unexpected_keys:
                logger.info("\nUnexpected keys:")
                for key in unexpected_keys:
                    logger.info(key)

                assert False, "Unexpected keys found in the model"
                
            else:
                logger.info("No unexpected keys found.")


        if args.freeze_backbone:
            logger.info("Freezing the weights of the backbone layers")
            for name, param in model_ft.backbone.named_parameters():
                param.requires_grad = False    
                logger.info(f"Freezing the weights of the  layer: {name}")

            if train_mode == 'triplet_finetune_with_ce':
            
                for name, param in model_ft.projection_layer.named_parameters():
                    param.requires_grad = False    
                    logger.info(f"Freezing the weights of the  layer: {name}")




    elif train_mode == "simclr_finetune_with_ce":
        # Load a pretrained model - Resnet18

        model_ft = ResNetSimCLR(args.arch, out_dim=128, num_classes=num_classes, )

        if True:

            with open(args.ckpt_path_dict, 'r') as f:
                transform2ckpt_dict = json.load(f)

            if True:
                transform_type_for_full_res = args.transform_type.replace("randomcrop224", "randomcrop198")
                #transform_type_for_full_res = args.transform_type.replace("randomcrop224", "randomcrop224")
                logger.info("Replacing the transform type randomcrop224 with randcrop198 to load the simclr ckpt")
                #logger.info("Replacing the transform type randomcrop224 with randcrop224 to load the simclr ckpt")

            ckpt_path = transform2ckpt_dict[transform_type_for_full_res]
            
            if True:
                logger.info(f"Loading the model from the path: {ckpt_path}")
                weights = torch.load(ckpt_path)['state_dict']
                missing_keys, unexpected_keys =  model_ft.load_state_dict(weights, strict=False)


                # Print missing and unexpected keys
                if missing_keys:
                    logger.info("Missing keys:")
                    for key in missing_keys:
                        logger.info(key)
                else:
                    logger.info("All keys matched successfully.")

                if unexpected_keys:
                    logger.info("\nUnexpected keys:")
                    for key in unexpected_keys:
                        logger.info(key)

                    assert False, "Unexpected keys found in the model"
                    
                else:
                    logger.info("No unexpected keys found.")


            if args.freeze_backbone:
                logger.info("Freezing the weights of the backbone layers")
                for name, param in model_ft.backbone.named_parameters():
                    param.requires_grad = False    
                    logger.info(f"Freezing the weights of the  layer: {name}")



    elif train_mode == "scratch":
        # Load a custom model - VGG11
        print("\nLoading VGG11 for training from scratch ...\n")
        model_ft = MyVGG11(in_ch=3, num_classes=11)

        # Set number of epochs to a higher value
        num_epochs = 100

    elif train_mode == "transfer":
        # Load a pretrained model - MobilenetV2
        print("\nLoading mobilenetv2 as feature extractor ...\n")
        model_ft = models.mobilenet_v2(pretrained=True)

        # Freeze all the required layers (i.e except last conv block and fc layers)
        for params in list(model_ft.parameters())[0:-5]:
            params.requires_grad = False

        # Modify fc layers to match num_classes
        num_ftrs = model_ft.classifier[-1].in_features
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=False),
            nn.Linear(in_features=num_ftrs, out_features=num_classes, bias=True),
        )

    else:
        raise NotImplementedError()

    return model_ft
