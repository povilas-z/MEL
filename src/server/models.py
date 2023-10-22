import torchvision.models as models
import torch.nn as nn

def createModel():
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 7) 

    return model