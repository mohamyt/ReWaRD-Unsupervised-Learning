import torch
import torch.nn as nn
import torch.nn.functional as F

from args import conf
from alex import bn_alexnet
from vgg import vgg16_bn, vgg19_bn
from resnet import *
from resnext import *
from densenet  import * 

args = conf()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Selecting the autoencoder
def model_select(args):
    if args.usenet == "bn_alexnet":
        model = bn_alexnet(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "vgg16":
        model = vgg16_bn(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "vgg19":
        model = vgg19_bn(pretrained=False,num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet18":
        model = resnet18(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet34":
        model = resnet34(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet50":
        model = resnet50(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet101":
        model = resnet101(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet152":
        model = resnet152(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnet200":
        model = resnet200(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "resnext101":
        model = resnext101(pretrained=False, num_classes=args.numof_classes).to(device)
        return model
    elif args.usenet == "densenet161":
        model = densenet161(pretrained=False, num_classes=args.numof_classes).to(device)
        return model


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        model = model_select(args)
        self.encoder = nn.Sequential(*list(model.children())[:-1])  # Remove the final classifier layer
        
        # Determine the output feature size
        self.feature_dim = model.fc.in_features if hasattr(model, 'fc') else model.classifier[6].in_features

    def forward(self, x):
        x = self.encoder(x)
        # Flatten the output
        x = x.view(x.size(0), -1)  # Flatten the output while keeping the batch dimension
        return x

    def get_feature_dim(self):
        return self.feature_dim


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()
        self.encoder = Encoder(args)
        
        # Projection head (2-layer MLP)
        self.fc = nn.Sequential(
            nn.Linear(self.encoder.get_feature_dim(), 2048),
            nn.ReLU(),
            nn.Linear(2048, args.out_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return F.normalize(x, dim=-1)

