## PyTorch dependencies
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch


class HistRes(nn.Module):
    
    def __init__(self,histogram_layer,parallel=True,model_name ='resnet18',
                 add_bn=True,scale=1,use_pretrained=True):
        
        #inherit nn.module
        super(HistRes,self).__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        #Default to use resnet18, otherwise use Resnet50
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=use_pretrained)
            if self.add_bn:
                self.bn_norm = nn.BatchNorm2d(512)
            
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=use_pretrained)
            if self.add_bn:
                self.bn_norm = nn.BatchNorm2d(2048)
            
        else: 
            print('Model not defined')
            
        
        #Define histogram layer and fc
        self.histogram_layer = histogram_layer
        self.combineLayer = nn.Conv2d(2048 * 2, 2048, (1, 1))
        # self.fc = self.backbone.fc
        self.fc = nn.Linear(512,9)
        self.fc1 = nn.Linear(2048, 9)
        self.fc_hebing  = nn.Linear(2048+512,9)


        # self.backbone.fc = torch.nn.Sequential()
        
        
    def forward(self,x):

        #All scales except for scale 5 default to parallel
        #Will add series implementation later
        # if self.scale == 1:
        #     output1 = self.forward_scale_1(x)
        # elif self.scale == 2:
        #     output2 = self.forward_scale_2(x)
        # elif self.scale == 3:
        #     output3 = self.forward_scale_3(x)
        # elif self.scale == 4:
        #     output4 = self.forward_scale_4(x)
        # else: #Default to have histogram layer at end
        # output1 = self.forward_scale_1(x)
        # output2 = self.forward_scale_2(x)
        # output3 = self.forward_scale_3(x)
        # output4 = self.forward_scale_4(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        #
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)

        # x_1 = torch.flatten(self.histogram_layer[0](x), start_dim=1)
        x = self.backbone.layer4(x)

        #Pass through histogram layer and pooling layer
        if(self.parallel):
            if self.add_bn:
                x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
            else:
                x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
            x2 = self.histogram_layer[1](x)
            x_hist = torch.flatten(self.histogram_layer[1](x),start_dim=1)
            xx = self.histogram_layer[0](x)
            x_1 = torch.flatten(self.histogram_layer[0](x), start_dim=1)
            # s1 = self.fc(x_1)
            s2 = self.fc(x_hist)
            x_combine = torch.cat((x_1,x_hist),dim=1)
            # x_combine = self.combineLayer(x_combine)
            # x_hist = torch.flatten(x_combine, start_dim=1)
            output = self.fc_hebing(x_combine)
            # output = self.fc(output)

        else:
            x = torch.flatten(self.histogram_layer[0](x),start_dim=1)
            output = self.fc(x)

     
        return s2
    
    def forward_scale_1(self,x):

        x = self.backbone.conv1(x)

        x_hist = torch.flatten(self.histogram_layer[0](x),start_dim=1)
        # x = self.backbone.bn1(x)
        # x = self.backbone.relu(x)
        # x = self.backbone.maxpool(x)
        #
        # x = self.backbone.layer1(x)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        # output = self.fc(x_combine)
        output = self.fc(x_hist)
        return x
        
    def forward_scale_2(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x_hist = torch.flatten(self.histogram_layer[1](x),start_dim=1)
        # x = self.backbone.layer2(x)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_hist)
        
        return x
        
    def forward_scale_3(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x_hist = torch.flatten(self.histogram_layer[2](x),start_dim=1)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_hist)
        
        return x
        
    def forward_scale_4(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x_hist = torch.flatten(self.histogram_layer[0](x),start_dim=1)
        # x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_hist)
        
        return x_hist
        
        
        
        
        
        
        
        