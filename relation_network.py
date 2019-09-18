import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    def __init__(self, input_channels, output_channels, padding=0, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=padding)
        self.batch_norm = nn.BatchNorm2d(output_channels, momentum=1, affine=True)
        if pool:
            self.last = nn.Sequential(nn.ReLU(), nn.MaxPool2d(2))
        else:
            self.last = nn.ReLU()
        
    def forward(self, x):
        out = self.conv(x) 
        out = self.batch_norm(out)
        out = self.last(out)
        return out


class CNNEncoder(nn.Module):
    def __init__(self, input_channels, conv_depth):
        super().__init__()
        self.layer1 = ConvLayer(input_channels, conv_depth)
        self.layer2 = ConvLayer(conv_depth, conv_depth)
        self.layer3 = ConvLayer(conv_depth, conv_depth, padding=1, pool=False)
        self.layer4 = ConvLayer(conv_depth, conv_depth, padding=1, pool=False)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

    
class RelationModule(nn.Module):
    def __init__(self, hidden_size, conv_depth):
        super().__init__()
        self.layer1 = ConvLayer(2*conv_depth, conv_depth)
        self.layer2 = ConvLayer(conv_depth, conv_depth)
        self.fc1 = nn.Linear(conv_depth*3*3, hidden_size) # this will break if image size is different from 84x84
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out


class RelationNetwork(nn.Module):
    def __init__(self, encoder, relation_module, class_num, sample_num_per_class, query_num_per_class):
        super().__init__()
        self.encoder = encoder
        self.relation_module = relation_module
        self.class_num = class_num
        self.sample_num_per_class = sample_num_per_class
        self.query_num_per_class = query_num_per_class
        
    def forward(self, sample, query):
        sample_features = self.encoder(sample)
        sample_features = sample_features.view(self.class_num, self.sample_num_per_class, *sample_features.shape[1:])
        sample_features = sample_features.sum(1)
        query_features = self.encoder(query)
        
        sample_features_ext = sample_features.unsqueeze(0).repeat(query.size()[0], 1, 1, 1, 1)
        query_features_ext = query_features.unsqueeze(1).repeat(1, self.class_num, 1, 1, 1)
        relation_pairs = torch.cat((sample_features_ext, query_features_ext), 2).view(-1, 2*sample_features_ext.shape[-3], *sample_features_ext.shape[-2:])
        relations = self.relation_module(relation_pairs).view(-1, self.class_num)
        return relations