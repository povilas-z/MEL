import torchvision.models as models
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class Model:
    def __init__(self, outputClasses):
        self. outputClasses = outputClasses
        self.model = self.createModel()
    
    def createModel(self):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.outputClasses) 
        return model
    
    def getModel(self):
        return self.model