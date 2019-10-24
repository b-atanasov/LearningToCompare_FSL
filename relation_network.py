import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


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


def initialize_weights(layer):
    if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)
    if isinstance(layer, nn.BatchNorm2d):
        nn.init.constant_(layer.weight, 1)
        nn.init.constant_(layer.bias, 0)


class CNNEncoder(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        conv_depth = 64
        self.layer1 = ConvLayer(input_channels, conv_depth)
        self.layer2 = ConvLayer(conv_depth, conv_depth)
        self.layer3 = ConvLayer(conv_depth, conv_depth, padding=1, pool=False)
        self.layer4 = ConvLayer(conv_depth, conv_depth, padding=1, pool=False)
        for m in self.modules():
            initialize_weights(m)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out


class RelationModuleOriginal(nn.Module):
    def __init__(self, conv_depth, args):
        super().__init__()
        hidden_size = 8
        self.layer1 = ConvLayer(2*conv_depth, conv_depth)
        self.layer2 = ConvLayer(conv_depth, conv_depth)
        if args.img_size == 224:
            fc1_in = conv_depth * 12 * 12
        elif args.img_size == 84:
            fc1_in = conv_depth * 3 * 3
        self.fc1 = nn.Linear(fc1_in, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        for m in self.modules():
            initialize_weights(m)
        self.loss_type = args.loss_type

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        if self.loss_type == 'mse':
            out = torch.sigmoid(out)
        return out


class RelationModule(nn.Module):
    def __init__(self, conv_depth, args):
        super().__init__()
        resnet = models.ResNet(models.resnet.BasicBlock, [2, 2, 2, 2])
        resnet.inplanes = 2 * conv_depth
        self.layer1 = resnet._make_layer(models.resnet.BasicBlock, planes=128, blocks=2, stride=2)
        self.layer2 = resnet._make_layer(models.resnet.BasicBlock, planes=64, blocks=2, stride=1)
        if args.img_size == 224:
            fc1_in = 64 * 7 * 7
        elif args.img_size == 84:
            fc1_in = 64 * 3 * 3
        fc1_out = fc1_in // 2
        self.fc1 = nn.Linear(fc1_in, fc1_out)
        # self.batch_norm = nn.BatchNorm1d(fc1_out)
        self.fc2 = nn.Linear(fc1_out, 1)
        for m in self.modules():
            initialize_weights(m)
        self.loss_type = args.loss_type

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        # out = self.batch_norm(out)
        out = F.relu(out)
        out = self.fc2(out)
        if self.loss_type == 'mse':
            out = torch.sigmoid(out)
        return out


class RelationNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.class_num = args.class_num
        self.sample_num_per_class = args.sample_num_per_class
        self.query_num_per_class = args.batch_num_per_class
        self.encoder = self.__get_encoder(args)
        self.relation_module = self.__get_relation_module(args)

    def __get_encoder(self, args):
        if args.backbone == 'Conv4':
            return CNNEncoder(args.input_channels)
        elif args.backbone == 'ResNet18':
            resnet18 = models.resnet18()
            return nn.Sequential(*list(resnet18.children())[:-3])

    def __get_relation_module(self, args):
        encoder_output_channels = [m.out_channels for m in self.encoder.modules()
                                   if isinstance(m, nn.Conv2d)][-1]
        if args.backbone == 'ResNet18':
            return RelationModule(encoder_output_channels, args)
        elif args.backbone == 'Conv4':
            return RelationModuleOriginal(encoder_output_channels, args)

    def forward(self, sample, query):
        sample_features = self.encoder(sample)
        sample_features = sample_features.view(self.class_num, self.sample_num_per_class, *sample_features.shape[1:])
        sample_features = sample_features.mean(1)
        query_features = self.encoder(query)

        sample_features_ext = sample_features.unsqueeze(0).repeat(query.size()[0], 1, 1, 1, 1)
        query_features_ext = query_features.unsqueeze(1).repeat(1, self.class_num, 1, 1, 1)
        relation_pairs = torch.cat((sample_features_ext, query_features_ext), 2).view(-1, 2*sample_features_ext.shape[-3], *sample_features_ext.shape[-2:])
        relations = self.relation_module(relation_pairs).view(-1, self.class_num)
        return relations
