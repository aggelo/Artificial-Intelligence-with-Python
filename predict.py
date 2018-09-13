from PIL import Image
import argparse
import torch
import numpy as np
from torch.autograd import Variable
import json
from torchvision import models

parser = argparse.ArgumentParser(description='Predict')
parser.add_argument('filepath', help = 'Filepath to image')
parser.add_argument('--checkpoint_path', help = 'Filepath to load checkpoint')
parser.add_argument('--gpu', action='store_true', help = 'Enable GPU')
parser.add_argument('--category_names', help = 'Select JSON file')
parser.add_argument('--top_k', type = int, help = 'Define Top-K')
args = parser.parse_args()


with open('args category_names', 'r') as f:
    cat_to_name = json.load(f)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer_dict']
    arch = checkpoint['arch']
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    for param in model.parameters():
        param.requires_grad = False
    return model

model = load_checkpoint('checkpoint.pth')
model
return model



def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    
    im = Image.open(image)
    width, height = im.size
    ratio = width / height
    if width <= height: 
        im = im.resize((256, int(ratio * 256)))
    else:
        im = im.resize((int(ratio * 256), 256))

   
    left = (im.size[0] - 224)/2
    top = (im.size[1] - 224)/2
    right = (im.size[0] + 224)/2
    bottom = (im.size[1] + 224)/2
    im = im.crop((left, top, right, bottom))
    
    
    np_image = np.array(im) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2, 1, 0))
    return np_image

device = torch.device("cuda:0" if torch.cuda.is_available() and gpu else "cpu")

if topk_enabled:
    print("Top {} probabilities: ".format(args.top_k) + "%")
    print(probs.numpy()[0] * 100)
    print("Top {} classes: ".format(args.top_k))
    print(name_list)
    probability = round(probs.numpy()[0].tolist()[0] * 100, 2)
    flower_name = name_list[0].title()
    print("Flower name is: {} with {}% probability".format(flower_name, probability)
    
def predict (image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    image = torch.from_numpy(process_image(image_path))
    image = image.unsqueeze(0).float()
    model, image = model.to(device), image.to(device)
    model.eval()
    model.requires_grad = False
    outputs = torch.exp(model.forward(image)).topk(topk)
    probs, classes = outputs[0].data.cpu().numpy()[0], outputs[1].data.cpu().numpy()[0]
    idx_to_class = {key: value for value, key in model.class_to_idx.items()}
    classes = [idx_to_class[classes[i]] for i in range(classes.size)]
    return probs, classes
    

image_path = 'flowers/test/3/image_06641.jpg'
probs, classes = predict(image_path, model, device)
print(probs)
print(classes)