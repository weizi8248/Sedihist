# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 10:26:08 2019

@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy
from sklearn.metrics import accuracy_score,r2_score
from torchvision import transforms
## PyTorch dependencies
import torch
import torch.nn as nn
from torchvision import models

## Local external libraries
from Utils.Histogram_Model import HistRes
from Utils.Histogram_Model_single1 import HistRes_single,Res_single
from barbar import Bar
import pandas as pd
from PIL import Image

def train_model(model, dataloaders, criterion, optimizer, device, 
                          saved_bins=None, saved_widths=None,histogram=True,
                          num_epochs=25,scheduler=None,dim_reduced=True):
    since = time.time()
    best_epoch = 1
    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = -100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode 
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0
            running_corrects = 0

            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                index = index.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
    
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)

                    # _, preds = torch.max(outputs, 1)
                    # running_corrects += acc
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # l = labels[:,0]
                        # g = outputs[0].reshape(-1)
                        loss1 = criterion(outputs[0].reshape(-1).float(), labels[:,0].float())
                        loss2 = criterion(outputs[1].reshape(-1).float(), labels[:,1].float())
                        loss3 = criterion(outputs[2].reshape(-1).float(), labels[:,2].float())
                        loss4 = criterion(outputs[3].reshape(-1).float(), labels[:,3].float())
                        loss5 = criterion(outputs[4].reshape(-1).float(), labels[:,4].float())
                        loss6 = criterion(outputs[5].reshape(-1).float(), labels[:,5].float())
                        loss7= criterion(outputs[6].reshape(-1).float(), labels[:,6].float())
                        loss8 = criterion(outputs[7].reshape(-1).float(), labels[:,7].float())
                        loss9 = criterion(outputs[8].reshape(-1).float(), labels[:,8].float())
                        loss = (1.0/9)*(loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9)
                        cc = outputs[0].detach().numpy()
                        for i in range(8):
                            aa =outputs[i+1].detach().numpy()
                            cc = np.concatenate([cc,aa],axis=1)
                        acc = r2_score(labels.detach().numpy(), cc)
                        print(acc)


                        # print(labels[0], outputs[0])
                        # loss1.backward(retain_graph=True)
                        # loss2.backward(retain_graph=True)
                        # loss3.backward(retain_graph=True)
                        # loss4.backward(retain_graph=True)
                        # loss5.backward(retain_graph=True)
                        # loss6.backward(retain_graph=True)
                        # loss7.backward(retain_graph=True)
                        # loss8.backward(retain_graph=True)
                        # loss9.backward(retain_graph=True)
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss1.item() +loss2.item()+loss3.item()+loss4.item()+loss5.item()+loss6.item()+loss7.item()+loss8.item()+loss9.item()
                # acc = r2_score(outputs, labels.data)
                running_corrects += acc
                # running_corrects += torch.sum(preds == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            epoch_acc = running_corrects / (len(dataloaders[phase].sampler))
            
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                train_error_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                # if(histogram):
                #     if dim_reduced:
                #         #save bins and widths
                #         # saved_bins[epoch+1,:] = model.histogram_layer[-1].centers.detach().cpu().numpy()
                #         # saved_widths[epoch+1,:] = model.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
                #     else:
                        #save bins and widths
                        # saved_bins[epoch+1,:] = model.histogram_layer.centers.detach().cpu().numpy()
                        # saved_widths[epoch+1,:] = model.histogram_layer.widths.reshape(-1).detach().cpu().numpy()
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))               
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'model_weights.pth')
    #Returning error (unhashable), need to fix
    print(best_epoch)
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_error_history,'train_acc_track': train_acc_history, 
                  'train_error_track': train_error_history,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
          
def test_model(dataloader,model,device):

    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    #Initialize and accumalate ground truth, predictions, and image indices

    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    model.eval()
    
    # Iterate over data
    print('Testing Model...')
    df = pd.read_csv('labelDX_test.csv')
    sample_num = len(df)
    result=[]
    label=[]
    label1=[]
    label2 = []
    label3 = []
    label4 = []
    label5 = []
    label6 = []
    label7 = []
    label8 = []
    label9 = []
    result1 = []
    result2 = []
    result3 = []
    result4 = []
    result5 = []
    result6 = []
    result7 = []
    result8= []
    result9 = []

    for i in range(sample_num):
        sample = df.iloc[i]
        image_path = sample['files']
        label1.append(sample['P5'])
        label2.append(sample['P10'])
        label3.append(sample['P16'])
        label4.append(sample['P25'])
        label5.append(sample['P50'])
        label6.append(sample['P75'])
        label7.append(sample['P84'])
        label8.append(sample['P90'])
        label9.append(sample['P95'])
        label0 = [sample['P5'],sample['P10'],sample['P16'],sample['P25'],sample['P50'],sample['P75'],sample['P84'],sample['P90'],sample['P95']]
        img = Image.open(image_path).convert('RGB')
        img = data_transforms['val'](img)
        img = img.unsqueeze(0)
        inputs = img.to(device)

        outputs = model(inputs)
        print(outputs)
        result.append(outputs.detach().numpy())
        label.append(label0)
        result1.append(outputs.detach().numpy()[0][0])
        result2.append(outputs.detach().numpy()[0][1])
        result3.append(outputs.detach().numpy()[0][2])
        result4.append(outputs.detach().numpy()[0][3])
        result5.append(outputs.detach().numpy()[0][4])
        result6.append(outputs.detach().numpy()[0][5])
        result7.append(outputs.detach().numpy()[0][6])
        result8.append(outputs.detach().numpy()[0][7])
        result9.append(outputs.detach().numpy()[0][8])

    # 转换成DataFrame对象
    result_df = pd.DataFrame(np.reshape(result,(sample_num,9)))
    # 保存为CSV文件
    result_df.to_csv("output-single5.csv", index=False)
    acc1 = r2_score(label1, result1)
    acc2 = r2_score(label2, result2)
    acc3 = r2_score(label3, result3)
    acc4 = r2_score(label4, result4)
    acc5 = r2_score(label5, result5)
    acc6 = r2_score(label6, result6)
    acc7 = r2_score(label7, result7)
    acc8 = r2_score(label8, result8)
    acc9 = r2_score(label9, result9)
    ll  =  np.array(result)
    acc0 = r2_score(np.array(label), np.array(result).squeeze(1))


    print(acc1,acc2,acc3,acc4,acc5,acc6,acc7,acc8,acc9,acc0)
    with torch.no_grad():

        for idx, (inputs, labels,index) in enumerate(Bar(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
    
            # forward
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            # GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            # Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)

            acc = r2_score(labels.detach().numpy(), outputs.detach().numpy())
            print(acc)
            running_corrects += acc

    test_acc = running_corrects / (len(dataloader.sampler))
    # print(test_acc)
    
    # test_dict = {'test_acc': np.round(test_acc.cpu().numpy()*100,2)}
    
    return test_acc
       
def initialize_model(model_name, num_classes,in_channels,out_channels, 
                     feature_extract=False, histogram=True,histogram_layer=None,
                     parallel=True, use_pretrained=True,add_bn=True,scale=1,
                     feat_map_size=4):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if(histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = HistRes(histogram_layer,parallel=parallel,
                           model_name=model_name,add_bn=add_bn,scale=scale,
                           use_pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft.backbone, feature_extract)
        
        #Reduce number of conv channels from input channels to input channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels/feat_map_size)/(histogram_layer[0].numBins))
        if (in_channels==reduced_dim): #If input channels equals reduced/increase, don't apply 1x1 convolution
            model_ft.histogram_layer = histogram_layer
        else:
            # conv_reduce1 = nn.Conv2d(in_channels[0],reduced_dim,(1,1))
            # model_ft.histogram_layer[0] = nn.Sequential(conv_reduce1,histogram_layer[0])
            # conv_reduce2 = nn.Conv2d(in_channels[1], reduced_dim, (1, 1))
            # model_ft.histogram_layer[1] = nn.Sequential(conv_reduce2, histogram_layer[1])
            # conv_reduce3 = nn.Conv2d(in_channels[2], reduced_dim, (1, 1))
            # model_ft.histogram_layer[2] = nn.Sequential(conv_reduce3, histogram_layer[2])
            conv_reduce4 = nn.Conv2d(in_channels[3], reduced_dim, (1, 1))
            model_ft.histogram_layer[0] = nn.Sequential(conv_reduce4, histogram_layer[0])
            conv_reduce5 = nn.Conv2d(in_channels[4], reduced_dim, (1, 1))
            model_ft.histogram_layer[1] = nn.Sequential(conv_reduce5, histogram_layer[1])
        if(parallel):
            # num_ftrs = model_ft.fc1.in_features*2
            num_ftrs = model_ft.fc.in_features

        else:
            num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)


        input_size = 224

    # Baseline model
    else:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
    
        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224
        
    return model_ft, input_size


def initialize_model_single(model_name, num_classes, in_channels, out_channels,
                     feature_extract=False, histogram=True, histogram_layer=None,
                     parallel=True, use_pretrained=True, add_bn=True, scale=1,
                     feat_map_size=4):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if (histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = HistRes_single(histogram_layer, parallel=parallel,
                           model_name=model_name, add_bn=add_bn, scale=scale,
                           use_pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft.backbone, feature_extract)

        # Reduce number of conv channels from input channels to input channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels / feat_map_size) / (histogram_layer.numBins))

        if (in_channels == reduced_dim):  # If input channels equals reduced/increase, don't apply 1x1 convolution
            model_ft.histogram_layer = histogram_layer
        else:
            conv_reduce = nn.Conv2d(in_channels, reduced_dim, (1, 1))
            model_ft.histogram_layer = nn.Sequential(conv_reduce, histogram_layer)
        # if (parallel):
        #     # num_ftrs = model_ft.fc1.in_features*2
        #     num_ftrs = model_ft.fc.in_features
        #
        # else:
        #     num_ftrs = model_ft.fc.in_features
        #
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    # Baseline model
    else:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

    return model_ft, input_size
def initialize_model_res(model_name, num_classes, in_channels, out_channels,
                     feature_extract=False, histogram=True, histogram_layer=None,
                     parallel=True, use_pretrained=True, add_bn=True, scale=1,
                     feat_map_size=4):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if (histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = Res_single(histogram_layer, parallel=parallel,
                           model_name=model_name, add_bn=add_bn, scale=scale,
                           use_pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft.backbone, feature_extract)

        # Reduce number of conv channels from input channels to input channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels / feat_map_size) / (histogram_layer.numBins))

        if (in_channels == reduced_dim):  # If input channels equals reduced/increase, don't apply 1x1 convolution
            model_ft.histogram_layer = histogram_layer
        else:
            conv_reduce = nn.Conv2d(in_channels, reduced_dim, (1, 1))
            model_ft.histogram_layer = nn.Sequential(conv_reduce, histogram_layer)
        if (parallel):
            # num_ftrs = model_ft.fc1.in_features*2
            num_ftrs = model_ft.fc.in_features

        else:
            num_ftrs = model_ft.fc.in_features

        model_ft.fc = nn.Linear(num_ftrs, num_classes)

        input_size = 224

    # Baseline model
    else:
        if model_name == "resnet18":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "resnet50":
            """ Resnet50
            """
            model_ft = models.resnet50(pretrained=use_pretrained)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

    return model_ft, input_size