# cnn model to partition the space for anns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import argparse
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_loader import MyDataset
from loss_fn import MyLoss
import matplotlib.pyplot as plt
from torchsummary import summary

# -should also pass args parser arguments
# -should set input ' n_input ' accordingly for cnn (different than simple fcn) ,it takes batches of
# -should set up training in case i need to modify training in the trees
# -should set up criterion loss if we need different loss ,optimizer



class ConvNetAnn(nn.Module):
    #initialize model
    # n_input : we need the x_train/with knn concatenated batch
    # note : ALSO must take into account the feature dimensions (784 for mnist)
    # n_classes : should be same as n_bins (number of partitions to partition the space)
    def __init__(self,n_input,num_class,opt,toplevel=False,n_hidden=84):
        super(ConvNetAnn,self).__init__()
        #define layers
        # all values could be customized
        self.device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.main_device="cuda:0" if torch.cuda.is_available() else "cpu"
        self.secondary_device="cuda:0"
        # reshape dimensions for input
        #self.n_input =
        #define id of model
        self.id = None
        self.n_classes = num_class
        self.n_hidden = n_hidden
        # bias = False for now beacause we have type mismatch error
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3, stride=1,bias=False)

        self.batch_norm_2d = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=3, stride=1,bias=False )
        self.conv3 = nn.Conv2d(16, 120, kernel_size=3, stride=1,bias=False )

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        ############ define the linear layers ###########
        self.fc1 = nn.Linear(120*3*3, self.n_hidden)
        self.fc2 = nn.Linear(self.n_hidden,self.n_classes)
        ############# relu and softmax ################
        self.relu = nn.ReLU()
        # dim = -1 following their implementation , not sure if dim should be -1
        self.softmax=nn.Softmax(dim=-1)
        self.dropout_2d = nn.Dropout2d(0.22)
        self.batch_norm_1d = nn.BatchNorm1d(self.n_hidden)
        # also add batch norm
        # also add dropout


    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.batch_norm_2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.relu(x)

        # reshape output
        x = x.view(x.size(0),-1)

        # first linear layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.batch_norm_1d(x)
        x = self.fc2(x)

        #print('\nOUTPUT OF CONV3 HAS SHAPE {x.shape}\n')
        # flatten and pass through fcn
        #x = x.view(-1,32*3*3) # 32 x 5 x 5
        #x=x.view(x.shape[0], -1)
        #x=self.relu(self.fc1(x))
        #x=self.relu(self.fc2(x))
        #x=self.fc3(x) # probably no relu / softmax since we will use cross entropy loss (we output partitions / as classification classes)
        # define bins for prediction and confidence scores (following their implementation)
        bins_prediction = self.softmax(x)
        confidence = torch.max(bins_prediction, dim=1)[0]

        return (bins_prediction,confidence.flatten())


    # IF CALLED WITH FOLLOWING , softmax should be dropped , in the case of CrossEntropyLoss #
    #model = ConvNetAnn().to('cuda')
    #criterion = nn.CrossEntropyLoss()
    #optimizer = torch.optim.SGD(model.parameters,lr= opt.lr)
