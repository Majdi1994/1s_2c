import warnings
import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from models.train_eval import train, evaluate
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
device = torch.device('cpu')
def trainer(model, train_dl, test_dl, data_id, config, params):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    target_names = ['Healthy','Damaged'] 
    for epoch in range(params['pretrain_epoch']):
        start_time = time.time()
        train_loss, train_pred, train_labels = train(model, train_dl, optimizer, criterion, config)
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        # Evaluate on the test set
        test_loss,_, _= evaluate(model, test_dl, criterion, config)
        print('=' * 50)
       # print(f'\t  Performance on test set::: Loss: {test_loss:.3f} ')#| Score: {test_score:7.3f}')
        train_labels = torch.stack(train_labels).view(-1)
        train_pred = torch.stack(train_pred).view(-1)
        print(classification_report(train_labels, train_pred, target_names=target_names))
        test_loss, y_pred, y_true = evaluate(model, test_dl, criterion, config)
        y_true = torch.stack(y_true).view(-1)
        y_pred = torch.stack(y_pred).view(-1)
        print('=' * 50)
        print(f'\tTest Loss: {test_loss:.3f}')
        print('=' * 50)
        print(classification_report(y_true, y_pred, target_names=target_names))
    # Evaluate on the test set
    test_loss, y_pred, y_true = evaluate(model, test_dl, criterion, config)
    y_true = torch.stack(y_true).view(-1)
    y_pred = torch.stack(y_pred).view(-1)
    print('=' * 50)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} ')#| Score: {test_score:7.3f}')
    print('=' * 50)
    print(classification_report(y_true, y_pred, target_names=target_names))
    print('The classification accuracy is =', accuracy_score(y_true, y_pred , normalize=True))
    print('The f1 score is = ', f1_score(y_true, y_pred, average='weighted'))
    print('The MCC Coefficient is =', matthews_corrcoef(y_true, y_pred))
    norm_soft=nn.Softmax(dim=1)
    norm_y_prob=norm_soft(y_pred)
    roc=roc_auc_score(y_true.cpu(), norm_y_prob.cpu(), multi_class='ovr')
    print('The ROC_AUC scrore is', roc)
    print('| End of Pre-training  |')
    print('=' * 50)
    return model
