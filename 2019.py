import torch.nn as nn
from models.models_config import get_model_config
from models.pre_train_test_split import trainer
import torch
from torch.utils.data import DataLoader
from utils import *
device = torch.device('cpu')

# Load data according to your directory
my_data= torch.load("data/1s_2c_os.pt")
train_dl = DataLoader(MyDataset(my_data['train_data'], my_data['train_labels']), batch_size=256, shuffle=True, drop_last=True)
test_dl = DataLoader(MyDataset(my_data['test_data'], my_data['test_labels']), batch_size=5, shuffle=False, drop_last=False)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

## 1D CNN class
class CNN_1D(nn.Module):
    def __init__(self, input_dim, hidden_dim,dropout):
        super(CNN_1D, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.dropout=dropout
        
        ## Feature Extraction or Selection 
        self.encoder = nn.Sequential(
            
            nn.Conv1d(self.input_dim, 16, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),
            
            nn.Conv1d(16, 64, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3,stride=2),
           
            #nn.Conv1d(64, 256, kernel_size=3),
            #nn.ReLU(inplace=True),
            #nn.MaxPool1d(kernel_size=3,stride=2),
            
            Flatten(),
            nn.Linear(16064, self.hidden_dim))
        
        ## Feature Classification
        self.Classifier= nn.Sequential(
            
            nn.Dropout(p=self.dropout), 
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim//2) ,
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim//2, 2))
    
    def forward(self, src):
        features = self.encoder(src)
        predictions = self.Classifier(features)
        return predictions, features

## Calling the model
model=CNN_1D(1,256,0.5).to(device)              # Model Paramaters, 1: one signal, 1000: Sampling Frequency, 0.5 Dropout Rate
params = {'pretrain_epoch': 1000, 'lr': 1e-4}    # 1000 epcohs, lr = 0.0001

## Load model
config = get_model_config('CNN')

## Load data
trained_model=trainer(model, train_dl, test_dl,'SHM_C', config, params)