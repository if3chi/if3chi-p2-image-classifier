import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from torchvision import models
import argparse
from train import prep_data, setup_model

def arg_parser():
    parser = argparse.ArgumentParser(description='Predict.py')
    parser.add_argument('input', default='flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
    parser.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store",type = str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default="False", action="store_true", dest="gpu")

    
    return parser.parse_args()

def load_checkpoint(path, gpu):
    checkpoint = torch.load(path)
    
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr = checkpoint['lr']

    _,_,model, device = setup_model(arch = structure, gpu = gpu, hidden_layer1 = hidden_layer1, dropout= dropout, lr = lr)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
        
    return model, device

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    img = PIL.Image.open(image)

    # Get original dimensions
    original_width, original_height = img.size

    # Find shorter size and create settings to crop shortest side to 256
    if original_width < original_height:
        size=[256, 256**600]
    else: 
        size=[256**600, 256]
        
    img.thumbnail(size)
   

    center = original_width/4, original_height/4
    left, top, right, bottom = center[0]-(244/2), center[1]-(244/2), center[0]+(244/2), center[1]+(244/2)
    img = img.crop((left, top, right, bottom))

    numpy_img = np.array(img)/255 

    # Normalize each color channel
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    numpy_img = (numpy_img-mean)/std
        
    # Set the color to the first channel
    numpy_img = numpy_img.transpose(2, 0, 1)
    
    return numpy_img

def predict(image, model, cat_to_name, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    model.eval();

    torch_image = torch.from_numpy(np.expand_dims(process_image(image), 
                                                  axis=0)).type(torch.FloatTensor).to(device)

    log_probs = model.forward(torch_image)

    linear_probs = torch.exp(log_probs)

    top_probs, top_classes = linear_probs.topk(topk)
    
    top_probs = np.array(top_probs.detach())[0] 
    top_classes = np.array(top_classes.detach())[0]
    
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    top_classes = [idx_to_class[lab] for lab in top_classes]
    top_flowers = [cat_to_name[lab] for lab in top_classes]
    
    return top_flowers, top_classes, top_probs

def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
    checkpoint, device = load_checkpoint(args.checkpoint, args.gpu)
    
    flowers, classes, probs = predict(args.input,
                                      checkpoint, 
                                      cat_to_name, 
                                      args.top_k, 
                                      device)
    
    for i, j in enumerate(zip(flowers, classes, probs)):
        print ("Rank {}:".format(i+1),
               "Flower: {}, \t|| Class: {}, || liklihood: {}%".format(j[0], j[1], ceil(j[2]*100)))
    
if __name__ == '__main__': main()