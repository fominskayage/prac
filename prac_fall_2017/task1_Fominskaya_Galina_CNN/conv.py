import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class ConvAutoEncoder(nn.Module):
    def __init__(self,
                 conv_out_channels=6,
                 conv_kernel_size=2,
                 conv_stride=2,
                 pool_kernel_size=2,
                 pool_stride=1,
                 conv_padding=2):

        super(ConvAutoEncoder, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, \
                                kernel_size=conv_kernel_size, \
                              stride=conv_stride, \
                              padding=conv_padding, \
                              out_channels=conv_out_channels)
        self.transpose = nn.ConvTranspose2d(in_channels=conv_out_channels, \
                                            kernel_size=conv_kernel_size, \
                                            stride=conv_stride, \
                                            padding=conv_padding, \
                                            out_channels=3)
        self.pool = nn.MaxPool2d(stride=pool_stride, \
                                    return_indices=True, \
                                 kernel_size=pool_kernel_size)
        self.unpool = nn.MaxUnpool2d(stride=pool_stride, \
                                     kernel_size=pool_kernel_size)

    def forward(self, x):
        code, indices = self.pool(F.relu(self.conv(x)))
        x = F.tanh(self.transpose(self.unpool(code, indices, output_size=self.conv(x).size())))
        return x, code



class ConvNet(nn.Module):
    def __init__(self,
                 input_size=(4, 3, 32, 32),
                 layers_num=1,
                 conv1_out_channels=6,
                 conv1_kernel_size=2,
                 conv1_stride=2,
                 conv2_out_channels=6,
                 conv2_kernel_size=2,
                 conv2_stride=2,
                 pool_kernel_size=2,
                 pool_stride=1,
                 conv1_padding=0,
                 weight=None):
        super(ConvNet, self).__init__()
        self.layers_num = layers_num
        self.conv1 = nn.Conv2d(in_channels=input_size[1], \
                                  out_channels=conv1_out_channels, \
                                  kernel_size=conv1_kernel_size, \
                                  padding=conv1_padding,\
                                  stride=conv1_stride)
        if not (weight is None):
            self.conv1.weight = weight
        if layers_num == 2:
            self.conv2 = nn.Conv2d(in_channels=conv1_out_channels, \
                                  out_channels=conv2_out_channels, \
                                  kernel_size=conv2_kernel_size, \
                                  stride=conv2_stride)
        
        self.pool = nn.MaxPool2d(stride=pool_stride, kernel_size=pool_kernel_size)
        out1_size = (((input_size[2] - conv1_kernel_size + conv1_stride + 2 * conv1_padding) // conv1_stride) \
                                                        - pool_kernel_size + pool_stride) // pool_stride
        if layers_num == 1:
            self.fc1 = nn.Linear(conv1_out_channels * out1_size ** 2, 120)
        else:
            self.fc1 = nn.Linear(conv2_out_channels * ((((out1_size - conv2_kernel_size \
                                                          + conv2_stride) // conv2_stride) \
                                                        - pool_kernel_size + pool_stride) // pool_stride) ** 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        if self.layers_num == 2:
            x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
