import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn

from layers import *

class Cifar100Net(nn.Module):
    '''
    The base model for few-shot learning on Omniglot
    '''

    def __init__(self, num_classes, loss_fn, num_in_channels=3):
        super(Cifar100Net, self).__init__()
        # Define the network
        self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(num_in_channels, 6, 5)),
                ('relu1', nn.ReLU(inplace=True)),
                ('pool1', nn.MaxPool2d(2,2)),
                ('conv2', nn.Conv2d(6,16,5)),
                ('relu2', nn.ReLU(inplace=True)),
                ('pool2', nn.MaxPool2d(2,2)),
        ]))
        self.add_module('fc1', nn.Linear(400, 20))
        self.add_module('fc2', nn.Linear(20, num_classes))

        # Define loss function
        self.loss_fn = loss_fn

        # Initialize weights
        self._init_weights()

    def forward(self, x, weights=None):
        x = x.float()
        ''' Define what happens to data in the net '''
        if weights is None:
            x = self.features(x)
            x = x.view(x.size(0), 400)
            x = self.fc1(x)
            x = self.fc2(x)
        else:
            x = conv2d(x, weights['features.conv1.weight'], weights['features.conv1.bias'])
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2)
            x = conv2d(x, weights['features.conv2.weight'], weights['features.conv2.bias'])
            x = relu(x)
            x = maxpool(x, kernel_size=2, stride=2)
            x = x.view(x.size(0), 400)
            x = linear(x, weights['fc1.weight'], weights['fc1.bias'])
            x = linear(x, weights['fc2.weight'], weights['fc2.bias'])
        return x

    def net_forward(self, x, weights=None):
        return self.forward(x, weights)

    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        print 'init weights'
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_() + 1
                m.bias.data = torch.ones(m.bias.data.size())

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        # TODO: breaks if nets are not identical
        # TODO: won't copy buffers, e.g. for batch norm
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()
