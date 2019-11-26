from get_input_args import get_trainig_args
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
models_nodes = {'alexnet': 9216, 'vgg': 25088,'densenet': 1024}




def main():
    
    in_arg = get_trainig_args()
    
    print(in_arg)
    
    data_dir = in_arg.dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    image_datasets = datasets.ImageFolder(train_dir, transform=data_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() and in_arg.gpu else "cpu")
    
    model = models[in_arg.arch]
    
    incomming_nodes = models_nodes[in_arg.arch]
    
   
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = nn.Sequential(nn.Linear(incomming_nodes, in_arg.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(in_arg.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))
    criterion = nn.NLLLoss()


    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.learning_rate)

    model.to(device);        
            
    epochs = in_arg.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(testloader):.3f}.. "
                      f"Test accuracy: {accuracy/len(testloader):.3f}")
                running_loss = 0
                model.train()   
                
                
    dir = in_arg.save_dir
    if not os.path.exists(dir):
        os.mkdir(dir)        
         
    checkpoint = {'classifier': model.classifier,
              'class_to_idx': image_datasets.class_to_idx,
              'state_dict': model.state_dict(),
              'model_name' : in_arg.arch}

    torch.save(checkpoint, in_arg.save_dir + "/" + in_arg.save_file)
    

    

main()