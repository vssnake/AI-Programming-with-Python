from get_input_args import get_testing_args
import os
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import json


alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet121(pretrained=True)

models = {'alexnet': alexnet, 'vgg': vgg16,'densenet': densenet}


def main():
    
    in_arg = get_testing_args()
    
    print(in_arg)
    
    model_trained = load_checkpoint(in_arg.load_file)
    
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
   
    
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    top_p,top_labels,top_flowers = predict(in_arg.image_patch,model_trained,in_arg.top_k,cat_to_name,device)
    
    print_prediction(top_p,top_labels,top_flowers,cat_to_name,in_arg.image_patch)


def print_prediction(top_p,top_labels,top_flowers,cat_to_name,image_path):
    flower_num = image_path.split('/')[2]
    title = cat_to_name[flower_num]# Plot flower
    print("")
    print("The flower  is {}".format(title))
    out_str = ''
    print("----------------------------")
    for flower, probability in zip(top_flowers, top_p):
        out_str += "Flower name {} and probability {} \n".format(flower,probability)
        
    print(out_str)

    
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models[checkpoint['model_name']]
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

    
def process_image(image):
    
    image = Image.open(image)
    # Resize
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))    # Crop 
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224    
    image = image.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    image = (image - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))
    
    return image

def predict(image_path, model, topk,cat_to_name, device):
    
    image_array = process_image(image_path)
    image_array = torch.from_numpy(image_array).type(torch.FloatTensor)
    image_array.unsqueeze_(0)
    model.to(device)
    with torch.no_grad():
        model.eval()
        ps = model.forward(image_array.to(device))

        ps = torch.exp(ps)

        top_p, top_class =  ps.topk(topk, dim=1)
        
        top_p = top_p.detach().cpu().numpy().tolist()[0]
        top_class = top_class.detach().cpu().numpy().tolist()[0]
        
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        
        top_labels = [idx_to_class[one_top_class] for one_top_class in top_class]
        
        top_flowers = [cat_to_name[idx_to_class[one_top_class]] for one_top_class in top_class]
    
        
        return top_p,top_labels,top_flowers
   


main()    