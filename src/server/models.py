import torchvision.models as models
import torch.nn as nn

class Model:
    def __init__(self, outputClasses):
        self. outputClasses = outputClasses
        self.model = self.createModel()
    
    def createModel(self):
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.outputClasses) 
        return model
    
    def getModel(self):
        return self.model