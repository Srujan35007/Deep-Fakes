import torch 
import torch.nn as nn 
from torchinfo import summary

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
        self.encoder_block(3, 16, 8, 2, 3),
        nn.AvgPool2d(2),
        self.encoder_block(16, 32, 6, 2, 2),
        nn.AvgPool2d(2),
        self.encoder_block(32, 64, 4, 2, 1),
        nn.AvgPool2d(2),
        self.encoder_block(64, 128, 3, 2, 1)
        ) 

    def encoder_block(self, ins, outs, k, s, p):
        '''
        ins: input channels
        outs: output channels
        k: kernel size/filter size
        s: stride
        p: padding
        '''
        return nn.Sequential(
        nn.Conv2d(ins, outs, k, s, p),
        nn.BatchNorm2d(outs),
        nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x).view(-1,2,16,16)

net = Encoder()
a = torch.rand(1,3,256,256)
print(net(a).shape)
