# -*- coding: utf-8 -*-
"""
Demo for histogram layer networks (HistRes_B)
Current script is only for experiments on
single cpu/gpu. If you want to run the demo
on multiple gpus (two were used in paper), 
please contact me at jpeeples@ufl.edu 
for the parallel version of 
demo.py
@author: jpeeples
"""

## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import os

## PyTorch dependencies
import torch
import torch.nn as nn
import torch.optim as optim
from sedinet_utils import *
## Local external libraries
from Utils.Network_functions import initialize_model_single, train_model,test_model
from Utils.vgg16 import VGG16
from Utils.RBFHistogramPooling import HistogramLayer1 as RBFHist
from Utils.LinearHistogramPooling import HistogramLayer as LinearHist
from Utils.Save_Results import save_results
from Parameters import Network_parameters
from Prepare_Data import Prepare_DataLoaders


#Name of dataset
Dataset = Network_parameters['Dataset']

#Model(s) to be used
model_name = Network_parameters['Model_names'][Dataset]

#Number of classes in dataset
num_classes = Network_parameters['num_classes'][Dataset]
                                 
#Number of runs and/or splits for dataset
numRuns = Network_parameters['Splits'][Dataset]

#Number of bins and input convolution feature maps after channel-wise pooling
numBins = Network_parameters['numBins']
num_feature_maps = Network_parameters['out_channels'][model_name]

#Local area of feature map after histogram layer
feat_map_size = Network_parameters['feat_map_size']

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Location to store trained models
current_directory = os.getcwd()
final_directory = os.path.join(current_directory, Network_parameters['folder'])

print('Starting Experiments...')
# train_csvfile = 'labelDX_train.csv'
# test_csvfile = 'labelDX_test.csv'
# train_idx, train_df = get_df(train_csvfile)
# test_idx, test_df = get_df(test_csvfile)
for split in range(0, numRuns):
    
    #Keep track of the bins and widths as these values are updated each
    #epoch
    saved_bins = np.zeros((Network_parameters['num_epochs']+1,
                           numBins*int(num_feature_maps/(feat_map_size*numBins))))
    saved_widths = np.zeros((Network_parameters['num_epochs']+1,
                             numBins*int(num_feature_maps/(feat_map_size*numBins))))
    
    #Initialize histogram layer based on type
    if Network_parameters['histogram_type'] == 'RBF':
        histogram_layer = RBFHist(128,
                                  Network_parameters['kernel_size'][model_name],
                                  num_bins=numBins,stride=Network_parameters['stride'],
                                  normalize_count=Network_parameters['normalize_count'],
                                  normalize_bins=Network_parameters['normalize_bins'])


    elif Network_parameters['histogram_type'] == 'Linear': 
        histogram_layer = LinearHist(128,
                                  Network_parameters['kernel_size'][model_name],
                                  num_bins=numBins,stride=Network_parameters['stride'],
                                  normalize_count=Network_parameters['normalize_count'],
                                  normalize_bins=Network_parameters['normalize_bins'])
    else:
        raise RuntimeError('Invalid type for histogram layer')
    in_channels = Network_parameters['in_channels'][model_name]
    # Initialize the histogram model for this run
    model_ft, input_size = initialize_model_single(model_name, num_classes,
                                            Network_parameters['in_channels'][model_name],
                                            num_feature_maps,
                                            feature_extract = Network_parameters['feature_extraction'],
                                            histogram= Network_parameters['histogram'],
                                            histogram_layer=histogram_layer,
                                            parallel=Network_parameters['parallel'],
                                            use_pretrained=Network_parameters['use_pretrained'],
                                            add_bn = Network_parameters['add_bn'],
                                            scale = Network_parameters['scale'],
                                            feat_map_size=feat_map_size)

    model_ft = model_ft.to(device)
    
    #Print number of trainable parameters
    num_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
    print("Number of parameters: %d" % (num_params))    
    print("Initializing Datasets and Dataloaders...")
    
    # Create training, validation, and test dataloaders
    dataloaders_dict = Prepare_DataLoaders(Network_parameters,split,input_size=input_size)
    # for idx, (inputs, labels, index) in enumerate(dataloaders_dict['train']):
    #     dd = inputs.to(device)
    #     labels = labels.to(device)
    #Save the initial values for bins and widths of histogram layer
    #Set optimizer for model
    if(Network_parameters['histogram']):
        reduced_dim = int((num_feature_maps/feat_map_size)/(numBins))
        if (Network_parameters['in_channels'][model_name]==reduced_dim):
            dim_reduced = False
            saved_bins[0,:] = model_ft.histogram_layer.centers.detach().cpu().numpy()
            saved_widths[0,:] = model_ft.histogram_layer.widths.reshape(-1).detach().cpu().numpy()
        else:
            dim_reduced = True
            # saved_bins[0,:] = model_ft.histogram_layer[-1].centers.detach().cpu().numpy()
            # saved_widths[0,:] = model_ft.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
        if(Network_parameters['add_bn']):
            optimizer_ft = optim.SGD([
                    {'params': model_ft.backbone.conv1.parameters()},
                    {'params': model_ft.backbone.bn1.parameters()},
                    {'params': model_ft.backbone.layer1.parameters()},
                    {'params': model_ft.backbone.layer2.parameters()},
                    {'params': model_ft.backbone.layer3.parameters()},
                    {'params': model_ft.backbone.layer4.parameters()},                
                    {'params': model_ft.histogram_layer.parameters(), 'lr': Network_parameters['new_lr']},
                    {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
                    {'params': model_ft.bn_norm.parameters(), 'lr': Network_parameters['new_lr']}
                ], lr=Network_parameters['pt_lr'], momentum=Network_parameters['momentum'])
        else:

            optimizer_ft = optim.SGD([
                {'params': model_ft.backbone.conv1.parameters()},
                {'params': model_ft.backbone.bn1.parameters()},
                {'params': model_ft.backbone.layer1.parameters()},
                {'params': model_ft.backbone.layer2.parameters()},
                {'params': model_ft.backbone.layer3.parameters()},
                {'params': model_ft.backbone.layer4.parameters()},
                {'params': model_ft.histogram_layer.parameters()},
                # {'params': model_ft.histogram_layer[0].parameters(), 'lr': Network_parameters['new_lr']},
                {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
            ], lr=Network_parameters['pt_lr'], momentum=Network_parameters['momentum'])
    else:
        saved_bins = None
        saved_widths = None
        optimizer_ft = optim.SGD([
                {'params': model_ft.conv1.parameters()},
                {'params': model_ft.bn1.parameters()},
                {'params': model_ft.layer1.parameters()},
                {'params': model_ft.layer2.parameters()},
                {'params': model_ft.layer3.parameters()},
                {'params': model_ft.layer4.parameters()},
                {'params': model_ft.fc.parameters(), 'lr': Network_parameters['new_lr']},
            ], lr=Network_parameters['pt_lr'], momentum = Network_parameters['momentum'])
   
    # Setup the loss fxn
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.SmoothL1Loss(reduction='mean') #'none','mean','sum' nn.MSELoss(reduce=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer_ft,
                                          step_size=Network_parameters['step_size'],
                                          gamma= Network_parameters['gamma'])
    
    # Train and evaluate
    train_dict = train_model(
            model_ft, dataloaders_dict, criterion, optimizer_ft, device,
            saved_bins=saved_bins,saved_widths=saved_widths,histogram=Network_parameters['histogram'],
            num_epochs=Network_parameters['num_epochs'],scheduler=scheduler,
            dim_reduced=dim_reduced)
