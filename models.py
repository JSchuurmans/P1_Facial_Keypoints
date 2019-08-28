## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (224-5)/1 +1 = 220
        # (32, 220, 220)
        # after pool; (32, 110, 110)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2,2)
        
        # output size = (110-5)/1+1 = 106
        # (64, 106, 106)
        # after pool: (64, 53, 53)
        self.conv2 = nn.Conv2d(32, 64, 5)
        
        # output size = (53-3)/1+1 = 51
        # (128, 51, 51)
        # after pool: (128, 25, 25)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
#         self.conv3 = nn.Conv2d(128, 128, 3)
        
    
        self.fc1 = nn.Linear(128*25*25, 1000)
        self.fc1_drop = nn.Dropout(p=0.4)
        
        self.fc2 = nn.Linear(1000, 136)
        
#         self.fc2 = nn.Linear(272, )

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    
class NaimishNet(nn.Module):
    def __init__(self):
        super(NaimishNet, self).__init__()
        
        # (224-4)/1+1 = 221
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.drop1 = nn.Dropout(.1)
        # 221/2 = 110
        
        # (110-3)/1 +1 = 108
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.drop2 = nn.Dropout(.2)
        # 108/2 = 504
        
        # (54-2)/1 +1 = 53
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.drop3 = nn.Dropout(.3)
        # 53/2 = 26
        
        # (26-1)/1+1 = 26
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.drop4 = nn.Dropout(.4)
        # 26/2 = 13
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256*13*13, 1000)
        self.drop5 = nn.Dropout(.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(.6)
        
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = self.fc3(x)
        
        return x
    
    
    
class NaimNet(nn.Module):
    def __init__(self):
        super(NaimNet, self).__init__()
        
        # (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.drop1 = nn.Dropout(.1)
        # 220/2 = 110
        
        # (110-4)/1 +1 = 107
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.drop2 = nn.Dropout(.2)
        # 107/2 = 53
        
        # (53-3)/1 +1 = 51
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.drop3 = nn.Dropout(.3)
        # 51/2 = 25
        
        # (25-2)/1+1 = 24
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.drop4 = nn.Dropout(.4)
        # 24/2 = 12
        
#         self.conv4 = nn.Conv2d(256, 512 1)
#         self.drop4 = nn.Dropout(.4)
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(256*12*12, 1000)
        self.drop5 = nn.Dropout(.5)
        
        self.fc2a = nn.Linear(1000, 1000)
        self.drop6 = nn.Dropout(.6)
        
        self.fc2b = nn.Linear(1000, 1000)
#         self.drop6 = nn.Dropout(.6)
        
        self.fc3 = nn.Linear(1000, 136)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2a(x))
        x = self.drop6(x)
        
        x = F.relu(self.fc2b(x))
        x = self.drop6(x)
        
        x = self.fc3(x)
        
        return x
    
    
class NaNet(nn.Module):
    def __init__(self):
        super(NaNet, self).__init__()
        
        # (224-5)/1+1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.drop1 = nn.Dropout(.1)
        # 220/2 = 110
        
        # (110-4)/1 +1 = 107
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.drop2 = nn.Dropout(.2)
        # 107/2 = 53
        
        # (53-3)/1 +1 = 51
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.drop3 = nn.Dropout(.3)
        # 51/2 = 25
        
        # (25-2)/1+1 = 24
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.drop4 = nn.Dropout(.4)
        # 24/2 = 12
        
        # (12-1)/1 +1 = 12
        self.conv5 = nn.Conv2d(256, 512, 1)
#         self.drop4 = nn.Dropout(.4)
        # 12/2 = 6
        
        self.pool = nn.MaxPool2d(2,2)
        
        self.fc1 = nn.Linear(512*6*6, 3000)
        self.drop5 = nn.Dropout(.5)
        
        self.fc2 = nn.Linear(3000, 2000)
        self.drop6 = nn.Dropout(.6)
        
        self.fc3 = nn.Linear(2000, 1000)
#         self.drop6 = nn.Dropout(.6)
        
        self.fc4 = nn.Linear(1000, 136)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        
        x = F.relu(self.fc3(x))
        x = self.drop6(x)
        
        x = self.fc4(x)
        
        return x