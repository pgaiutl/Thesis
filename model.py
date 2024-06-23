# model.py
import torch
import torch.nn as nn
import torchvision

class CustomResNet(nn.Module):
    def __init__(self, num_classes, custom_weights_path):
        super(CustomResNet, self).__init__()
        self.num_classes = num_classes
        self.custom_weights_path = custom_weights_path
        self.init_resnet()

    def init_resnet(self):
        self.resnet = torchvision.models.resnet34(pretrained=False)
        self.load_custom_weights()
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.fc1 = nn.Linear(self.resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.resnet.fc = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def load_custom_weights(self):
        state_dict = torch.load(self.custom_weights_path)
        state_dict['fc2.weight'] = state_dict.pop('fc.weight')[:self.num_classes, :]
        state_dict['fc2.bias'] = state_dict.pop('fc.bias')[:self.num_classes]
        self.resnet.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        return self.resnet(x)


class CustomResNetForEvaluation(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNetForEvaluation, self).__init__()
        self.num_classes = num_classes
        self.init_resnet()

    def init_resnet(self):
        self.resnet = torchvision.models.resnet34(pretrained=False)
        self.fc1 = nn.Linear(self.resnet.fc.in_features, 512)
        self.fc2 = nn.Linear(512, self.num_classes)
        self.resnet.fc = nn.Sequential(self.fc1, nn.ReLU(), self.fc2)

    def forward(self, x):
        return self.resnet(x)

