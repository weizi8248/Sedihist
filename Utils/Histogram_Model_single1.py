## PyTorch dependencies
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch


class HistRes_single(nn.Module):
    
    def __init__(self,histogram_layer,parallel=True,model_name ='resnet18',
                 add_bn=False,scale=4,use_pretrained=True):
        
        #inherit nn.module
        super(HistRes_single,self).__init__()
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
        # self.fc = self.backbone.fc
        self.fc1= nn.Linear(2048, 1)
        self.fc2 = nn.Linear(2048, 1)
        self.fc3 = nn.Linear(2048, 1)
        self.fc4= nn.Linear(2048, 1)
        self.fc5 = nn.Linear(2048, 1)
        self.fc6 = nn.Linear(2048, 1)
        self.fc7 = nn.Linear(2048, 1)
        self.fc8 = nn.Linear(2048, 1)
        self.fc9 = nn.Linear(2048, 1)

        self.fc_hebing  = nn.Linear(2048*2,9)


        self.backbone.fc = torch.nn.Sequential()
        
        
    def forward(self,x):

        #All scales except for scale 5 default to parallel
        #Will add series implementation later
        if self.scale == 1:
            output = self.forward_scale_1(x)
        elif self.scale == 2:
            output = self.forward_scale_2(x)
        elif self.scale == 3:
            output = self.forward_scale_3(x)
        elif self.scale == 4:
            output1,output2,output3,output4,output5,output6,output7,output8,output9 = self.forward_scale_4(x)
        else: #Default to have histogram layer at end

            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            #
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

        #Pass through histogram layer and pooling layer
            if(self.parallel):
                if self.add_bn:
                    x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
                else:
                    x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)

                x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
                x_combine = torch.cat((x_pool,x_hist),dim=1)
                # output = self.fc_hebing(x_combine)
                output1 = self.fc1(x_hist)
                output2 = self.fc2(x_hist)
                output3 = self.fc3(x_hist)
                output4 = self.fc4(x_hist)
                output5 = self.fc5(x_hist)
                output6 = self.fc6(x_hist)
                output7 = self.fc7(x_hist)
                output8 = self.fc8(x_hist)
                output9 = self.fc9(x_hist)




            else:
                x = torch.flatten(self.histogram_layer(x),start_dim=1)
                output = self.fc(x)

     
        return output1,output2,output3,output4,output5,output6,output7,output8,output9
    
    def forward_scale_1(self,x):

        x = self.backbone.conv1(x)

        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)
        x_combine = torch.cat((x_pool, x_hist), dim=1)
        output = self.fc(x_hist)
        return x_hist
        
    def forward_scale_2(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_hist)
        
        return x_hist
        
    def forward_scale_3(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        # x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_hist)
        
        return output
        
    def forward_scale_4(self,x):
        
        x = self.backbone.conv1(x)    
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x_hist = torch.flatten(self.histogram_layer(x),start_dim=1)
        x = self.backbone.layer4(x)
    
        #Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)),start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x),start_dim=1)
        x_combine = torch.cat((x_pool,x_hist),dim=1)
        output1 = self.fc1(x_hist)
        output2 = self.fc2(x_hist)
        output3 = self.fc3(x_hist)
        output4 = self.fc4(x_hist)
        output5 = self.fc5(x_hist)
        output6 = self.fc6(x_hist)
        output7 = self.fc7(x_hist)
        output8 = self.fc8(x_hist)
        output9 = self.fc9(x_hist)
        
        return output1,output2,output3,output4,output5,output6,output7,output8,output9


class Res_single(nn.Module):

    def __init__(self, histogram_layer, parallel=True, model_name='resnet18',
                 add_bn=False, scale=4, use_pretrained=True):

        # inherit nn.module
        super(Res_single, self).__init__()
        self.parallel = parallel
        self.add_bn = add_bn
        self.scale = scale
        # Default to use resnet18, otherwise use Resnet50
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

        # Define histogram layer and fc
        self.histogram_layer = histogram_layer
        self.fc = self.backbone.fc


        self.backbone.fc = torch.nn.Sequential()

    def forward(self, x):

        # All scales except for scale 5 default to parallel
        # Will add series implementation later
        if self.scale == 1:
            output = self.forward_scale_1(x)
        elif self.scale == 2:
            output = self.forward_scale_2(x)
        elif self.scale == 3:
            output = self.forward_scale_3(x)
        elif self.scale == 4:
            output = self.forward_scale_4(x)
        else:  # Default to have histogram layer at end

            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            #
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            # Pass through histogram layer and pooling layer
            if (self.parallel):
                if self.add_bn:
                    x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
                else:
                    x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)


                output = self.fc(x_pool)

            else:
                x = torch.flatten(self.histogram_layer(x), start_dim=1)
                output = self.fc(x)

        return output

    def forward_scale_1(self, x):

        x = self.backbone.conv1(x)

        x_hist = torch.flatten(self.histogram_layer(x), start_dim=1)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)
        x_combine = torch.cat((x_pool, x_hist), dim=1)
        output = self.fc(x_hist)
        return x_hist

    def forward_scale_2(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x_hist = torch.flatten(self.histogram_layer(x), start_dim=1)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)
        x_combine = torch.cat((x_pool, x_hist), dim=1)
        output = self.fc(x_hist)

        return x_hist

    def forward_scale_3(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x_hist = torch.flatten(self.histogram_layer(x), start_dim=1)
        # x = self.backbone.layer3(x)
        # x = self.backbone.layer4(x)

        # Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)
        # x_combine = torch.cat((x_pool,x_hist),dim=1)
        output = self.fc(x_hist)

        return output

    def forward_scale_4(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x_hist = torch.flatten(self.histogram_layer(x), start_dim=1)
        x = self.backbone.layer4(x)

        # Pass through histogram layer and pooling layer
        if self.add_bn:
            x_pool = torch.flatten(self.bn_norm(self.backbone.avgpool(x)), start_dim=1)
        else:
            x_pool = torch.flatten(self.backbone.avgpool(x), start_dim=1)
        x_combine = torch.cat((x_pool, x_hist), dim=1)
        output = self.fc(x_hist)

        return output
        
        
        
        
        
        