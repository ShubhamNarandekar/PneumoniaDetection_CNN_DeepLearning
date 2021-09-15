import os
import numpy as np
import pandas as pd
import torch
import random
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
import matplotlib.pyplot as plt
from sklearn import metrics



#To keep reproduce the same results
def seed():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed()

#Basic simple model
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.relu1 = nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.relu3 = nn.ReLU()
        
        self.fc = nn.Linear(in_features=(128*56*56), out_features= 2)
        
    def forward(self,input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.pool1(output)
        
        
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool2(output)
        
        output = self.conv3(output)
        output = self.relu3(output)
        
        output = output.view(-1,128*56*56)
        output = self.fc(output)
        
        return output

#Model after improvement    
class cnn2(nn.Module):
    def __init__(self):
        super(cnn2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.relu1 = nn.ReLU()
        self.pool0=nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.relu2 = nn.ReLU()
        self.pool1=nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool2=nn.MaxPool2d(kernel_size=2)
        
        self.conv4 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool3=nn.MaxPool2d(kernel_size=2)
                                     
        self.dropout1 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(in_features=(256*14*14), out_features= 256)
        self.relu5 = nn.ReLU()
        
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(in_features=256, out_features= 2)
        
        
    def forward(self,input):
        output = self.conv1(input)
        output = self.relu1(output)
        output = self.pool0(output)
        
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.pool1(output)
        
        output = self.conv3(output)
        output = self.batchnorm1(output)
        output = self.relu3(output)
        output = self.pool2(output)
        
        output = self.conv4(output)
        output = self.batchnorm2(output)
        output = self.relu4(output)
        output = self.pool3(output)
        
        output = output.view(-1,256*14*14)
        
        output = self.dropout1(output)
        output = self.fc1(output)
        output = self.relu5(output)
        
        output = self.dropout2(output)
        output = self.fc2(output)
                                                
        return output
    
#For training the model.    
def fit(model, device, train_loader, train_count, optimizer, loss_function):
        model.train()
        train_accuracy=0.0
        train_loss=0.0
        t_preds = []
        t_targets = []
        
        for i, (images,labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)           
            optimizer.zero_grad()        
            outputs=model(images)
            loss=loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            train_loss+= loss.cpu().data*images.size(0)
            _,prediction=torch.max(outputs.data,1)
            t_preds.append(prediction.cpu().numpy())
            t_targets.append(labels.cpu())
            train_accuracy+=int(torch.sum(prediction==labels.data)) 
            del images, labels
            torch.cuda.empty_cache()
        #Calculating AUC and Accuracy per epoch    
        train_accuracy=(train_accuracy/train_count)
        train_loss=(train_loss/train_count)  
        t_preds = np.concatenate(t_preds)
        t_targets  = np.concatenate(t_targets)
        t_fpr, t_tpr, thresholds = metrics.roc_curve(t_targets, t_preds)
        t_roc = metrics.auc(t_fpr, t_tpr)
        
        return t_roc, train_loss, train_accuracy

#For evaluating the validation set
def evaluate(model, device, val_loader, val_count, loss_function, checkpoint=None):
    if checkpoint is not None:
        state = torch.load(checkpoint)
        model.load_state_dict(state)
    model.eval()
    test_accuracy=0.0
    val_loss=0.0
    v_preds = []
    v_targets = []
    with torch.no_grad():
        for i, (images,labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs=model(images)
            loss=loss_function(outputs,labels)          
            val_loss+= loss.cpu().data*images.size(0)
            _,prediction=torch.max(outputs.data,1)            
            v_preds.append(prediction.cpu().numpy())
            v_targets.append(labels.cpu())
            test_accuracy+=int(torch.sum(prediction==labels.data))
    #Calculating AUC and Accuracy per epoch
    test_accuracy=(test_accuracy/val_count)
    val_loss=(val_loss/val_count)
    v_preds = np.concatenate(v_preds)
    v_targets  = np.concatenate(v_targets)
    v_fpr, v_tpr, thresholds = metrics.roc_curve(v_targets, v_preds)
    v_roc = metrics.auc(v_fpr, v_tpr)

    return v_roc, val_loss, test_accuracy

#I have defined separate function for testing because here I am calculating all the evaluation metrics.
def test(model, device, test_loader, test_count, checkpoint=None):
    if checkpoint is not None:
        state = torch.load(checkpoint)
        model.load_state_dict(state)
    model.eval()
    test_accuracy=0.0
    val_loss=0.0
    v_preds = []
    v_targets = []
    with torch.no_grad():
        for i, (images,labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs=model(images)
            _,prediction=torch.max(outputs.data,1)            
            v_preds.append(prediction.cpu().numpy())
            v_targets.append(labels.cpu())
            test_accuracy+=int(torch.sum(prediction==labels.data))
    #Calculating AUC and Accuracy per epoch
    test_accuracy=(test_accuracy/test_count)
    v_preds = np.concatenate(v_preds)
    v_targets  = np.concatenate(v_targets)
    v_fpr, v_tpr, thresholds = metrics.roc_curve(v_targets, v_preds)
    v_p, v_r, t = metrics.precision_recall_curve(v_targets, v_preds)
    v_roc = metrics.auc(v_fpr, v_tpr)
    v_pr = metrics.auc(v_r, v_p)
    v_cm = metrics.confusion_matrix(v_targets, v_preds)
    v_cr = metrics.classification_report(v_targets, v_preds, target_names=['Normal', 'Pneumonia'])
    return v_roc, test_accuracy, v_pr, v_cm, v_cr
