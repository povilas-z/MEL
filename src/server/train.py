import torch
import torch.optim as optim
import torch.nn as nn
from dataset import DataSet
from models import Model
import os
from torch.utils.tensorboard import SummaryWriter

# Constants
DATA_DIR = os.getcwd() + "/HAM10000/"
TARGET_SAMPLES = 500 
NUM_CLASSES = 7 
BATCH_SIZE = 24
NUM_EPOCHS = 25
LEARNING_RATE = 0.002
STEP_SIZE = 7
GAMMA = 0.1
BEST_LOSS = float("inf")
BEST_MODEL_WEIGHTS = 0

# Instantiate dataset and model
data = DataSet(DATA_DIR)
train_dataset = data.prepareDataSet(TARGET_SAMPLES)
train_loader = DataSet.getDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Model(NUM_CLASSES).getModel()

writer = SummaryWriter('runs/experiment_1')

#Loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

#Device for macbook
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
model = model.to(device)

#CONSTANTS FOR LAYER FREEZING
UNFREEZE_TRIGGER = 0.02
LAST_IMPROVED = 0
LAYERS_TO_UNFREEZE = [
    model.layer4,
    model.layer3,
    model.layer2,
    model.layer1
]

#Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct_predictions = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    scheduler.step(epoch_loss)
    epoch_acc = correct_predictions.float() / len(train_loader.dataset)

    if epoch_loss < BEST_LOSS - UNFREEZE_TRIGGER:
        BEST_LOSS = epoch_loss
        LAST_IMPROVED = epoch
        BEST_MODEL_WEIGHTS = model.state_dict().copy()
    elif epoch - LAST_IMPROVED >= 2 and LAYERS_TO_UNFREEZE:
        to_unfreeze = LAYERS_TO_UNFREEZE.pop(0)
        for param in to_unfreeze.parameters():
            param.requires_grad = True
        LAST_IMPROVED = epoch
        print(f"Unfreezing layers: {to_unfreeze}")

        optimizer = optim.Adam(model.parameters(), lr=(LEARNING_RATE * 0.1))


    print('Epoch {}/{} - Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, epoch_loss))
    
    writer.add_scalar('Training Loss', epoch_loss, epoch)
    writer.add_scalar('Training Accuracy', epoch_acc, epoch)

writer.close()
print("Finished training")

#Save model
model.load_state_dict(BEST_MODEL_WEIGHTS)
torch.save(model.state_dict(), os.getcwd() + "/src/server/saved_models/model7.pth")
print("model saved")