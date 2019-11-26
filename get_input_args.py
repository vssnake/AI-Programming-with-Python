
import argparse

def get_trainig_args():
     # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
    
    
   
    parser.add_argument("dir", type = str, default = 'flowers/',
                        help = 'path to the folter of flowers')
    
    parser.add_argument('--arch', type = str, default = 'densenet',
                        help = 'CNN Model Architecture')
    
    parser.add_argument('--learning_rate', type = float, default = 0.003,
                        help = 'The learning rate of the algorithm')
    
    parser.add_argument('--hidden_units', type = int, default = 512,
                        help = 'Number of hidden units')
                        
    parser.add_argument('--gpu', action='store_true',
                            help = 'Use GPU to train')
    
    parser.add_argument('--epochs', type = int, default = 3,
                        help = 'Number of epocs of the training')
    
    parser.add_argument('--save_dir', type = str, default = 'trained',
                        help = 'Ouput directory of save models')
    
    parser.add_argument('--save_file', type = str, default = 'checkpoint.pth',
                        help = 'File name of training model')
    
    return parser.parse_args()

def get_testing_args():
   
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()
  
    parser.add_argument("image_patch", type = str,
                        help = 'the image patch to test')
                        
    parser.add_argument('--gpu', action='store_true',
                        help = 'Use GPU to train')
    
    parser.add_argument('--load_file', type = str, default = 'trained/checkpoint.pth',
                        help = 'Ouput directory of save models')
    
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json',
                        help = 'Ouput directory of save models')
    
    parser.add_argument('--top_k', type = int, default = 5,
                        help = 'The most K result classes')
    
  
    
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()
