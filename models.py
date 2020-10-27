## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/4
def init_weights(m):
    # Kaiming He initializer is a suitable weight initializer for relu based networks as it realizes its non-linearity
    if isinstance(m, nn.Conv2d):
        I.kaiming_uniform_(m.weight, nonlinearity='relu')
        

class Net(nn.Module):
        # No zero padding
        # maxpooling layers have (3,3) kernel with stride 3
        # O = floor((I - k)/s + 1)
        # (1, 224, 224) -> conv1,k=5-> (32, 220, 220)
        #               ->   relu   -> (32, 220, 220)
        #               ->  maxpool -> (32, 73, 73)
        #               ->  dropout1-> (32, 73, 73)
        #               -> conv2,k=3-> (64, 71, 71)
        #               ->batch_norm-> (64, 71, 71)
        #               ->   relu   -> (64, 71, 71)
        #               ->  maxpool -> (64, 23, 23)
        #               ->  dropout2-> (64, 23, 23)
        #               -> conv3,k=3-> (128, 21, 21)
        #               ->   relu   -> (128, 21, 21)
        #               ->  maxpool -> (128, 7, 7)
        #               ->  dropout3-> (128, 7, 7)
        #               ->  flatten -> (6272)
        #               ->    fc1   -> (512)
        #               ->batch_norm-> (512)
        #               ->   relu   -> (512)
        #               ->    fc2   -> (136)
    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        

        
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
      
       
        self.max_pool = nn.MaxPool2d(3,3)
        self.conv_bn = nn.BatchNorm2d(64)
        
        self.drop_out1 = nn.Dropout2d(0.1)
        self.drop_out2 = nn.Dropout2d(0.2)
        self.drop_out3 = nn.Dropout2d(0.3)
       
        
        self.fc_bn = nn.BatchNorm1d(512)
        
        self.fc1 = nn.Linear(6272,512)
        self.fc2 = nn.Linear(512,136)
       
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.drop_out1(self.max_pool(F.relu(self.conv1(x))))
        x = self.drop_out2(self.max_pool(F.relu(self.conv_bn(self.conv2(x)))))
        x = self.drop_out3(self.max_pool(F.relu(self.conv3(x))))
        
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc_bn(self.fc1(x)))
        x = self.fc2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
