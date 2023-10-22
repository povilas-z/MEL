import torch
import torch.optim as optim
import torch.nn as nn
import dataset
import models
import os

#Load dataset
train_transforms = dataset.getTransforms()
train_dataset = dataset.getDataSet(os.getcwd() + "/HAM10000/reorganized/", train_transforms)
train_loader = dataset.getDataLoader(train_dataset, batch_size=32, shuffle=True)

#Load model
model = models.createModel()

#Loss function, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)


#Device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Training loop
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
    
    scheduler.step()
    epoch_loss = running_loss / len(train_loader.dataset)
    print('Epoch {}/{} - Loss: {:.4f}'.format(epoch+1, num_epochs, epoch_loss))

print("Finished training")