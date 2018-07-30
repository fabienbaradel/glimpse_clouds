from model.backbone.imagenet_pretraining import load_pretrained_2D_weights
from model.backbone.resnet.bottleneck import Bottleneck3D
from model.backbone.resnet.resnet import ResNet
import ipdb

def inflated_resnet(**kwargs):
    list_block = [Bottleneck3D, Bottleneck3D, Bottleneck3D, Bottleneck3D]
    list_layers = [3, 4, 6, 3]

    # Create the model
    model = ResNet(list_block,
                   list_layers,
                   **kwargs)

    # Pretrained from imagenet weights
    load_pretrained_2D_weights('resnet50', model, inflation='center')

    return model

