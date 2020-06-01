import torch.nn as nn
from models.models_config import get_model_config
from models.pre_train_test_split import trainer
import torch
from torch.utils.data import DataLoader
from utils import *
device = torch.device('cpu')

# Load data, depending on the file path
my_data= torch.load("/content/drive/My Drive/data/1s_2c_os.pt") 

train_dl = DataLoader(MyDataset(my_data['train_data'], my_data['train_labels']), batch_size=256, shuffle=True, drop_last=True)

test_dl = DataLoader(MyDataset(my_data['test_data'], my_data['test_labels']), batch_size=5, shuffle=False, drop_last=False)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

## 1D CNN model
class CNN_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_1D, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        
        ## Feature Extraction or Selection
        self.encoder = nn.Sequential(
           
            ### First Convolutional Layer    
            nn.Conv1d(self.input_dim, 8, kernel_size=11, stride=1, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            ## Second Convolutional Layer
            nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            
            ## Third Convolutional Layer
            #nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            #nn.BatchNorm1d(8),
            #nn.ReLU(),
            
            ## Forth Convolutional Layer
            #nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            #nn.BatchNorm1d(8),
            #nn.ReLU(),
            
            ## Fifth Convolutional Layer
            #nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            #nn.BatchNorm1d(8),
            #nn.ReLU(),

            ## Sixth Convolutional Layer
            #nn.Conv1d(8, 8, kernel_size=3, stride=2, padding=1, dilation=1),
            
            Flatten(),
            nn.Linear(4064, self.hidden_dim))
        
        ## Feature Classification
        self.Classifier= nn.Sequential(
            
            ## First MLP Layer    
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            
            ## Second MLP Layer
            nn.Linear(self.hidden_dim, self.hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            ## Third MLP Layer
            nn.Linear(self.hidden_dim//2, 2))
    
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.Classifier(features)
        return predictions, features

## Calling the model 
model=CNN_1D(1,256,0.5).to(device) 

params = {'pretrain_epoch': 1000, 'lr': 1e-3} 

# load model
config = get_model_config('CNN')

# load data
trained_model=trainer(model, train_dl, test_dl,'SHM_C' ,config,params)
