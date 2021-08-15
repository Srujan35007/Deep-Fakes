import torch
import torch.nn as nn 
from torchsummary import summary

class Encoder(nn.Module):
    def __init__(self, img_channels=3):
        self.model_name = 'Face_Encoder_Conv'
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
        # in: image_channels x dim x dim
        self.encoder_block(img_channels,16,3,1,1),
        nn.AvgPool2d(2),
        # out: 16 x dim/2 x dim/2
        self.encoder_block(16,32,3,1,1),
        nn.AvgPool2d(2),
        # out: 32 x dim/4 x dim/4
        self.encoder_block(32,64,3,1,1),
        nn.AvgPool2d(2),
        # out: 64 x dim/8 x dim/8
        self.encoder_block(64,128,3,1,1),
        nn.AvgPool2d(2),
        # out: 128 x dim/16 x dim/16
        self.encoder_block(128,256,3,1,1),
        # out: 256 x dim/16 x dim/16
        )
        self.init_weigths()

    def encoder_block(self, ins_,outs_,kernel_,stride_,padding_):
        return nn.Sequential(
        nn.Conv2d(ins_,ins_*2,kernel_,stride_,padding_,bias=False),
        nn.BatchNorm2d(ins_*2),
        nn.ReLU(inplace=True),
        nn.Conv2d(ins_*2,outs_,kernel_,stride_,padding_,bias=False),
        nn.BatchNorm2d(outs_),
        nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.layers(x)

    def init_weigths(self):
        count_, total_ = 0, 0
        for module in self.parameters():
            if len(module.shape) > 1:
                nn.init.kaiming_normal_(module)
                count_ += 1
                total_ += 1
            else:
                total_ += 1
        print(f'{self.model_name} weights initialized ({count_}/{total_})')

class Decoder(nn.Module):
    def __init__(self, in_channels=256, img_channels=3):
        self.model_name = 'Face_Decoder_Conv'
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
        # in: in_channels x dim x dim
        self.decoder_block(in_channels,256,3,1,1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        # out: 256 x dim*2 x dim*2
        self.decoder_block(256,128,3,1,1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        # out: 128 x dim*4 x dim*4
        self.decoder_block(128,64,3,1,1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        # out: 64 x dim*8 x dim*8
        self.decoder_block(64,32,3,1,1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        # out: 32 x dim*16 x dim*16
        self.decoder_block(32,img_channels,3,1,1),
        # out: img_channels x dim*16 x dim*16
        )
        self.init_weigths()

    def decoder_block(self, ins_,outs_,kernel_,stride_,padding_):
        return nn.Sequential(
        nn.Conv2d(ins_,outs_,kernel_,stride_,padding_,bias=False),
        nn.BatchNorm2d(outs_),
        nn.ReLU(inplace=True),
        nn.Conv2d(outs_,outs_,kernel_,stride_,padding_,bias=False),
        nn.BatchNorm2d(outs_),
        nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.layers(x)

    def init_weigths(self):
        count_, total_ = 0, 0
        for module in self.parameters():
            if len(module.shape) > 1:
                nn.init.kaiming_normal_(module)
                count_ += 1
                total_ += 1
            else:
                total_ += 1
        print(f'{self.model_name} weights initialized ({count_}/{total_})')

enc = Encoder(3)
dec = Decoder(256,3)
print(summary(dec,(256,8,8)))
