#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:06:35 2023

@author: josemiguelacitores
"""
from sklearn import metrics
import numpy as np

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import torch
import matplotlib.pyplot as plt

def calc_auc_ci_bootstrap(y_actual, y_pred_prob):
    auc = round(metrics.roc_auc_score(y_actual, y_pred_prob),2)
    n_bootstraps = 5000
    rng_seed = 42  # control reproducibility
    bootstrapped_scores = []
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred_prob), len(y_pred_prob))
        if len(np.unique(y_actual[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue
        score = metrics.roc_auc_score(y_actual[indices], y_pred_prob[indices])
        bootstrapped_scores.append(score)
    # 124 for lower 5% and 4874 for upper 5%
    bootstrapped_scores = np.sort( bootstrapped_scores)
    lower_ci = round((bootstrapped_scores[124]),2)
    upper_ci = round((bootstrapped_scores[4874]),2)
    return auc, lower_ci, upper_ci



def ci_conf_mat(y_actual, y_pred_prob):
    
    auc = round(metrics.roc_auc_score(y_actual, y_pred_prob),2)
    
    y_pred_prob = np.round(y_pred_prob)
    n_bootstraps = 5000
    rng_seed = 42  # control reproducibility
    sensitivities = []
    specificities = []
    you_scores = []
    f1_scores = []
    accuracies = []
    PPVs = []
    NPVs = []
    bootstrapped_auc = []
    tn, fp, fn, tp = metrics.confusion_matrix(y_actual, y_pred_prob).ravel()
    
    sensi= tp/( tp+fn)
    specif = tn/(tn+fp)
    youden_index = sensi + specif -1
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    
    accuracy = (tn+tp)/(tn+tp+fn+fp)
    
    f1score = tp/(tp+0.5*(fp+fn))
    
    print('AUC: '+str(auc))
    print('sensibility: ' + str(sensi))
    print('specificity: ' + str(specif))
    print('youden index: ' + str(youden_index))
    print('f1 score: ' + str(f1score))
    print('PPV: ' + str(PPV))
    print('NPV: ' + str(NPV))
    print('accuracy: '+str(accuracy))
    
    rng = np.random.RandomState(rng_seed)
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.randint(0, len(y_pred_prob), len(y_pred_prob))
        if len(np.unique(y_actual[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        tn, fp, fn, tp = metrics.confusion_matrix(y_actual[indices], y_pred_prob[indices]).ravel()
        
        sensi= tp/( tp+fn)
        specif = tn/(tn+fp)
        youden_index = sensi + specif -1
        
        PPV = tp/(tp+fp)
        NPV = tn/(tn+fn)
        
        accuracy = (tn+tp)/(tn+tp+fn+fp)
        
        f1score = tp/(tp+0.5*(fp+fn))
        
        sensitivities.append(sensi)
        specificities.append(specif)
        you_scores.append(youden_index)
        f1_scores.append(f1score)
        accuracies.append(accuracy)
        PPVs.append(PPV)
        NPVs.append(NPV)
        bootstrapped_auc.append(metrics.roc_auc_score(y_actual[indices], y_pred_prob[indices]))
        
        
    # 124 for lower 5% and 4874 for upper 5%
    
    sensitivities = np.sort(sensitivities)
    specificities = np.sort(specificities)
    you_scores = np.sort(you_scores)
    f1_scores = np.sort(f1_scores)
    accuracies = np.sort(accuracies)
    PPVs = np.sort(PPVs)
    NPVs = np.sort(NPVs)
    bootstrapped_auc = np.sort(bootstrapped_auc)
    
    lower_ci = {'sensi':round((sensitivities[124]),2),
                'specif':round((specificities[124]),2),
                'you_score':round((you_scores[124]),2),
                'f1':round((f1_scores[124]),2),
                'acc':round((accuracies[124]),2),
                'PPV':round((PPVs[124]),2),
                'NPV':round((NPVs[124]),2),
                'AUC':round((bootstrapped_auc[124]),2)
                } 
    upper_ci = {'sensi':round((sensitivities[4874]),2),
                'specif':round((specificities[4874]),2),
                'you_score':round((you_scores[4874]),2),
                'f1':round((f1_scores[4874]),2),
                'acc':round((accuracies[4874]),2),
                'PPV':round((PPVs[4874]),2),
                'NPV':round((NPVs[4874]),2),
                'AUC':round((bootstrapped_auc[4874]),2)
                } 

    return lower_ci, upper_ci


def print_grads(model, data_loader, device):

    saliency = Saliency(model)
    with torch.no_grad():
        for data_point in data_loader:
            print(data_point)
            images, labels = data_point #data[0].to(device), data[1].to(device)
            
            print(images.size(),labels.size())
            
            images= images.to(device)
            labels = labels.to(device)
            
            grads = saliency.attribute(images, target=labels.item())
            grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))
    
            plt.imshow(grads, interpolation='nearest')
            plt.show()
            
            _ = viz.visualize_image_attr(grads, images.cpu().detach(), method="blended_heat_map", sign="absolute_value",show_colorbar=True, title="Overlayed Gradient Magnitudes")
            break



