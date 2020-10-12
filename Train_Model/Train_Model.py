import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

from PIL import Image
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

base_dir = os.path.dirname(__file__)

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'test':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

#loading the data

data = {
    'train':
    datasets.ImageFolder(root=base_dir+'/data/train', transform=image_transforms['train']),
    'test':
    datasets.ImageFolder(root=base_dir+'/data/test', transform=image_transforms['test']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=100, shuffle=True),
    'test': DataLoader(data['test'], batch_size=100, shuffle=True)
}

#loading MobileNetv2
model = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier[1] = nn.Sequential(
    nn.Linear(1280, 256),
    nn.ReLU(), 
    nn.Linear(256, 128),
    nn.ReLU(), 
    nn.Dropout(0.4), 
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.4), 
    nn.Linear(32, 2), 
    nn.LogSoftmax(dim=1))

#checking GPU Avaibility
if torch.cuda.is_available():
    print('training on GPU')
    device = torch.device("cuda:0")
else:
    print('training on CPU')
    device = torch.device("cpu")

#training data

def train(model, optimizer, loss_fn, train_loader, val_loader, epochs,device):
    
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        print("TRAINING MODEL")
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
        training_loss /= len(train_loader.dataset)
        
        model.eval()
        num_correct = 0 
        num_examples = 0
        print("EVALUATING MODEL")
        for batch in tqdm(val_loader):
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fn(output,targets) 
            valid_loss += loss.data.item() * inputs.size(0)
                        
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss,valid_loss, num_correct / num_examples))

#testing data
def test_model(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))

model.to(device)
#choosing optimizer and loss Function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss = torch.nn.CrossEntropyLoss()

while True:
    epochs = int(input("Enter the Number of Epochs: "))
    lr = int(input("Enter the Learning Rate: "))
    train(model, optimizer,loss,dataloaders['train'],dataloaders['train'],epochs,device)
    optimizer = optim.Adam(model.parameters(), lr)
    loss = torch.nn.CrossEntropyLoss()
    test_model(model)
    choice = input("Do you want to save the trained model? (y/n): ")
    if(choice == 'y'):
        name = input("Enter the name of the model: ")
        torch.save(model, "./"+name+".pth")
        break
    else:
        choice = input("Want to Run More Epochs? (y/n): ")
        if(choice=='y'):
            continue
        else:
            choice = input("Press Enter to Close ")
            break
