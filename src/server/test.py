import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import Model  # Make sure this path is correct
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Load the model
output_classes = 7  # set this to the number of your output classes
model = Model(output_classes).getModel()
model_path = os.getcwd() + "/src/server/saved_models/model1.pth"
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()  # set model to evaluation mode

# Define the transformations for the test data
test_transforms = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.229,0.224,0.225])
])

# Load the test dataset
test_data = datasets.ImageFolder(root= os.getcwd() + "/HAM10000/test_reorganized/", transform=test_transforms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test data: {100 * correct / total:.2f}%")
