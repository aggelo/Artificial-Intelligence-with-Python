import argparse
import sys
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description="Trains a network on a dataset of images and saves the model to a checkpoint")
    parser.add_argument('data_dir', type = str, default = '/home/workspace/aipnd-project/flowers',
                   help = 'path to the folder for images')
    parser.add_argument('--data_dir', help = 'Data directory')
    parser.add_argument('--arch', type = str, default='densenet121', help='choose model')
    parser.add_argument('--save_dir', type = str, help='Directory to save checkpoint')
    parser.add_argument('--learning_rate', type = float, default=0.001, help='model learning_rate')
    parser.add_argument('--hidden_layers', type = int, default=512, help='Number of hidden layers')
    parser.add_argument('--epochs', type = int, default=5, help='Numbers of epochs train')
    args = parser.parse_args()
    return args


data_dir = args.data_dir
train_dir = args.data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
save_dir = args.save_dir


train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = data_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = data_transforms)

train_dataloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
valid_dataloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
test_dataloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64)

def choose_architecture(name):
    if name == 'densenet121' and input== 1024:
        return models.densenet121(pretrained = True)
    elif name == 'densenet169' and input== 1664:
        return models.densenet169(pretrained = True)
    elif name == 'densenet161' and input== 2208:
        return models.densenet161(pretrained = True)
    elif name == 'densenet201' and input== 1920:
        return models.densenet201(pretrained = True)
    else: # Error message
        print("This project supports only DenseNet models. You can choose one of the following: 'densenet121 (input:1024)', 'densenet169 (input:1664)', 'densenet201 (input:1920)', 'densenet161 input:2208'")
        sys.exit()
model = choose_architecture(args.arch)

learning_rate= args.learning_rate
hidden_layers= args.hidden_layer
epochs= args.epochs

print('Hidden layers: {}'.format(hidden_layers))
print('Learning rate: {}'.format(learning_rate))
print('Epochs:        {}'.format(epochs))

if gpu_mode and torch.cuda.is_available():
        device = torch.device("cuda:0")
else:
        device = torch.device("cpu")
print('Current device: {}'.format(device))
            
for param in model.parameters():
    param.requires_grad = False
print('grad false')
              
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 512)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(p = 0.5)),
    ('fc2', nn.Linear(512, 256)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(256, 102)),
    ('output', nn.LogSoftmax(dim = 1))
]))


model.classifier = classifier

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learn_rate)
input_size = 1024
output_size = 102
batch_size= 64
pr_evry = 10

def do_deep_learning(model, trainloader, testloader, validloader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    steps = 0
    model = model.to(device)
    model.train()
    print("Training model...")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_dataloaders):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if steps % print_every == 0:
                model.eval()
                accuracy = 0
                validation_loss  = 0
                for ii, (inputs, labels) in enumerate(valid_dataloaders):
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    validation_loss  += criterion(output, labels)
                    probabilities = torch.exp(output).data
                    equality = (labels.data == probabilities.max(1)[1])
                    accuracy += equality.type_as(torch.FloatTensor()).mean()
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "| Training Loss: {:.4f}".format(running_loss / print_every),
                      "| Validation Loss: {:.3f}.. ".format(validation_loss  / len(valid_dataloaders)),
                      "| Validation Accuracy: {:.3f}%".format(accuracy / len(valid_dataloaders) * 100))
                running_loss = 0
                model.train()
    
    print("Done!")

do_deep_learning(model, train_dataloaders, test_dataloaders, valid_dataloaders, epochs, pr_evry, criterion, optimizer, device)
              
              
checkpoint = {
    'epochs': 10,
    'learn_rate': learn_rate,
    'input_size': input_size,
    'output_size': output_size,
    'batch_size': batch_size,
    'arch': 'densenet121',
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'class_to_idx': train_datasets.class_to_idx,
    'classifier': model.classifier,
}
torch.save(checkpoint, save_dir + '/checkpoint.pth')
        
if __name__ == '__main__':
    print("Checkpoint saved in: {}".format(save_dir))
    
