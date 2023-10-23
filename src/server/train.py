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
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.001
STEP_SIZE = 7
GAMMA = 0.1

# Instantiate dataset and model
data = DataSet(DATA_DIR)
train_dataset = data.prepareDataSet(TARGET_SAMPLES)
train_loader = DataSet.getDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Model(NUM_CLASSES).getModel()

writer = SummaryWriter('runs/experiment_1')

#Loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = STEP_SIZE, gamma = GAMMA)

#Device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
model = model.to(device)

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
    
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct_predictions.double() / len(train_loader.dataset)

    print('Epoch {}/{} - Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, epoch_loss))
    
    writer.add_scalar('Training Loss', epoch_loss, epoch)
    writer.add_scalar('Training Accuracy', epoch_acc, epoch)

writer.close()
print("Finished training")

#Save model
torch.save(model.state_dict(), os.getcwd() + "/src/server/saved_models")
print("model saved")