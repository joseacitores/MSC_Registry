# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 17:41:32 2022

@author: acito
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:25:32 2022

@author: josemiguelacitores
"""

"Import section"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader

import json


import cv2

import csv

import torchvision
from torchvision import models, transforms

import numpy as np

from glob import glob
import os
#from google.colab import drive
import time 


from PIL import Image
import matplotlib.pyplot as plt


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print(device)

#%% Train model


def train_model(model,  train_loader, val_loader, epochs, device):
    print('training')
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50 ], gamma=0.5)
    loss_fct  = nn.CrossEntropyLoss(label_smoothing=0.1)
    loss_epoch = {'train':[], 'valid':[]}
    for epoch in range(epochs):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        for i, data_ in enumerate(train_loader, 0): #  batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = data_
            inputs = inputs.to(device)
            targets = targets.to(device)
            output = model(inputs)
            loss = loss_fct(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * inputs.size(0)
        
        

        training_loss /= len(train_loader.dataset)
        model.eval()
        num_correct = 0 
        num_examples = 0
        for batch in val_loader:
            inputs, targets = batch
            inputs = inputs.to(device)
            output = model(inputs)
            targets = targets.to(device)
            loss = loss_fct(output,targets) 
            valid_loss += loss.item() * inputs.size(0)
                        
            correct = torch.eq(torch.max(F.softmax(output, dim=1), dim=1)[1], targets).view(-1)
            num_correct += torch.sum(correct).item()
            num_examples += correct.shape[0]
        valid_loss /= len(val_loader.dataset)
        scheduler.step()
        
        loss_epoch['train'].append(training_loss)
        loss_epoch['valid'].append(valid_loss)
        
        if epoch >5:
            if loss_epoch['train'][-1]>loss_epoch['train'][-2] and loss_epoch['valid'][-1]>loss_epoch['valid'][-2]:
                break
        
        print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, accuracy = {:.4f}'.format(epoch, training_loss, valid_loss, num_correct / num_examples))
        
    return loss_epoch


#%% Test model
from sklearn import metrics


def test_model(model, test_data_loader, filename):
    correct = 0
    total = 0
    conf_matrix = np.zeros(shape=(9,9))
    list_prob=[]
    pred_list = []
    true_list = []
    with torch.no_grad():
        for data_point in test_data_loader:
            images, labels = data_point #data[0].to(device), data[1].to(device)
                        
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #print(predicted)
            out=F.softmax(outputs, dim=-1)
            
            # print labels
            #print(predicted)
            #print(out[0])
            out=F.softmax(outputs, dim=-1)
            #out=[t.numpy() for t in out.cpu()]
            out=out[0]

            list_prob.append([labels.cpu().detach().numpy()[0],
                              predicted.cpu().detach().numpy()[0],
                              out.cpu().detach().numpy()])
            true_list.append(labels.cpu().detach().numpy()[0])
            pred_list.append(predicted.cpu().detach().numpy()[0])
            
            conf_matrix[labels,predicted]+=1

    with open(path_dir + filename, 'w') as f:
      
    # using csv.writer method from CSV package
        write = csv.writer(f)
        write.writerows(list_prob)
    # saving the dataframe 
    
    # sensi= c11/( c11+c01)
    # specif = c00/(c00+c10)
    # youden_index = sensi + specif -1
    
    print('correct: {:d}  total: {:d}'.format(correct, total))
    print('accuracy = {:f}'.format(correct / total))
    # print('C00: {:d}, C01:{:d}, C10: {:d}, C11: {:d}'.format(c00, c01, c10, c11))
    # print('Specificity:  {:f}, Sensitivity:  {:f}'.format(specif, sensi))
    # print('Youden Index: {:f}'.format(youden_index) )
    for row in conf_matrix:
        print(row)
    
    # fpr, tpr, thresholds = metrics.roc_curve(true_list, pred_list, pos_label=1)
    # roc_auc = metrics.roc_auc_score(true_list,pred_list)
    
    # print('ROC AUC: {:f}'.format(roc_auc))
    
    return pred_list, true_list, conf_matrix



#%% Initializing models

"""Load model (pretrained resnet50)"""

model_resnet   = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V2)
model_efficient  = models.efficientnet_v2_l(weights=models.EfficientNet_V2_L_Weights.IMAGENET1K_V1)
model_vit   = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)


num_classes = 2

model_resnet.fc = nn.Sequential(nn.Linear(model_resnet.fc.in_features,512), nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
model_efficient.classifier  = nn.Sequential(nn.Linear(model_efficient.classifier[1].in_features,512), nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
model_vit.heads = nn.Sequential(nn.Linear(model_vit.heads[0].in_features,512), nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))
# model_vit.heads = nn.Sequential(nn.Linear(model_vit.hidden_dim,512), nn.ReLU(), nn.Dropout(), nn.Linear(512, num_classes))

    
#%% Data loading

# divide into 3 train test val

"""# Training / Validation / Test set"""

transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor()])
# transforms.Normalize(mean = [0.4073, 0.4073, 0.4073],
                     # std= [0.2752, 0.2752, 0.2752])])
                     
                     
                    # transforms.Normalize(mean =[0.4157, 0.4157, 0.4157],
                                          # std= [0.2708, 0.2708, 0.2708])]

def check_image(path):  
    try:
        im = Image.open(path)
        return True
    except:
        return False
    
transforms_train = transforms.Compose([
        transforms.RandomRotation(90),
        transforms.RandomVerticalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor()])
    
    # ,
    # transforms.Normalize(mean =[0.4157, 0.4157, 0.4157],
    #                      std= [0.2708, 0.2708, 0.2708]

# path_dir = main folder of the project with train valdtaion and test set
path_dir = ""

train_dataset = torchvision.datasets.ImageFolder(root=path_dir+'/Train', transform=transforms_train)
train_datset = torch.utils.data.ConcatDataset( [train_dataset,torchvision.datasets.ImageFolder(root=path_dir+'/Train',transform=transforms_train)])

valid_dataset = torchvision.datasets.ImageFolder(root=path_dir+'/Validation', transform=transforms_train)

test_dataset = torchvision.datasets.ImageFolder(root=path_dir+'/Test', transform=transform)




train_data_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

val_data_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=8, shuffle=True, num_workers=0)

test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


#%% Creating model training framework

# model_resnet_epic = train_model(model_resnet_epic.cuda(), train_epic_loader, val_epic_loader, epochs=10, device=device)

#test_model(model_resnet_epic, test_phone_loader, '/probs/problist_e-p.csv')
# torch.save(model_resnet.state_dict(),path_dir+'/models/model_resnet.pt')
# 
times = {}


#%% ResNet Model train

start_time = time.time()

epoch_loss_res = train_model(model_resnet.cuda(), train_data_loader, val_data_loader, epochs=100, device=device)

end_time = time.time()

times['res_train_'] = end_time-start_time
json_object = json.dumps(times)
with open(path_dir+'/epoch_losses/times_.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(epoch_loss_res)

with open(path_dir+'/epoch_losses/model_resnet_loss_2b.json', "w") as outfile:
    outfile.write(json_object)

torch.save(model_resnet.state_dict(),path_dir+'/models/model_resnet_2b.pt')

pred_list0, true_list0, conf_mat0 = test_model(model_resnet, test_data_loader, '/probs/problist_res_test_2b.csv' )


#%% EfficientNet Model train
torch.cuda.empty_cache()

start_time = time.time()

epoch_loss_eff = train_model(model_efficient.cuda(), train_data_loader, val_data_loader, epochs=100, device=device)

end_time = time.time()

times['eff_train_'] = end_time-start_time
json_object = json.dumps(times)
with open(path_dir+'/epoch_losses/times_.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(epoch_loss_eff)

with open(path_dir+'/epoch_losses/model_eff_loss_b_2.json', "w") as outfile:
    outfile.write(json_object)
torch.save(model_efficient.state_dict(),path_dir+'/models/model_efficient_b_2.pt')

pred_list1, true_list1, conf_mat1 = test_model(model_efficient, test_data_loader, '/probs/problist_eff_test_b_2.csv' )

#%% Vision Transformer Model train
torch.cuda.empty_cache()

start_time = time.time()

epoch_loss_vit = train_model(model_vit.cuda(), train_data_loader, val_data_loader, epochs=100, device=device)

end_time = time.time()

times['vit_train_'] = end_time-start_time

json_object = json.dumps(times)
with open(path_dir+'/epoch_losses/times_.json', "w") as outfile:
    outfile.write(json_object)

json_object = json.dumps(epoch_loss_vit)

with open(path_dir+'/epoch_losses/model_vit_loss_b.json', "w") as outfile:
    outfile.write(json_object)

torch.save(model_vit.state_dict(),path_dir+'/models/model_vit_b.pt')

pred_list2, true_list2, conf_mat2 = test_model(model_vit, test_data_loader, '/probs/problist_vit_test_b.csv' )


#%%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

classes = ('ankle','elbow','fott','hand','hip','knee','shoulder','spine','wrist')

cf_matrix = confusion_matrix(true_list0, pred_list0)
print(cf_matrix)


#%%




# display1 = metrics.RocCurveDisplay(fpr=fpr1, tpr=tpr1, roc_auc=roc_auc1,
#                                        estimator_name='e-e')

# display2 = metrics.RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=roc_auc2,
#                                        estimator_name='e-p')

#gold, royalblue, silver

plt.plot(fpr3,tpr3,label="Model 1, tested on phone, auc="+str(roc_auc3)[:4])
plt.plot(fpr1,tpr1,label="Model 2 tested on EPIC, auc="+str(roc_auc1)[:4])
plt.plot(fpr2,tpr2,label="Model 2, tested on phone, auc="+str(roc_auc2)[:4])
plt.plot(fpr4,tpr4,label="Model 3, tested on phone, auc="+str(roc_auc4)[:4])
plt.plot(fpr5,tpr5,label="Model 3 tested on EPIC, auc="+str(roc_auc5)[:4])
plt.plot(fpr6,tpr6,label="Model 3 tested on EPIC+phone, auc="+str(roc_auc6)[:4])
plt.xlabel('specificity')
plt.ylabel('sensitivity')
plt.legend(loc=0)
plt.show()

























