# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:47:43 2022

@author: acito
"""

# python code to calculate mean and std 
import torchvision
from torchvision import transforms
import torch
from torch.utils.data import DataLoader


path_dir = r'C:\Users\acito\projects\FxHip'

transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])

dataset = torchvision.datasets.ImageFolder(root=path_dir+'/Train', transform=transform)

loader  = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)


def batch_mean_and_sd(loader):
    
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data_point in loader:
        images, labels = data_point
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2,
                                  dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (
                      cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (
                            cnt + nb_pixels)
        cnt += nb_pixels

    mean, std = fst_moment, torch.sqrt(
      snd_moment - fst_moment ** 2)        
    return mean,std
  
mean, std = batch_mean_and_sd(loader)
print("mean and std: \n", mean, std)