import torch
import torch.nn as nn 
#from torchsummary import summary

class AutoEncoderConv(nn.Module):
    def __init__(self, input_channels):
        super(AutoEncoderConv, self).__init__()
        self.model_name = 'AutoEncoderConv_Face'
        self.Encoder = nn.Sequential(
        self.ConvBlock(input_channels,64,3,1,1),
        nn.AvgPool2d(2),
        self.ConvBlock(64,128,3,1,1),
        nn.AvgPool2d(2),
        self.ConvBlock(128,256,3,1,1)
        )
        self.Decoder = nn.Sequential(
        self.ConvBlock(256,128,3,1,1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        self.ConvBlock(128,64,3,1,1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        self.ConvBlock(64,input_channels,3,1,1),
        )
        self.init_weights()

    def ConvBlock(self, ins_, outs_, kernel_, stride_, padding_):
        return nn.Sequential(
        nn.Conv2d(ins_, ins_*2, kernel_, stride_, padding_, bias=False),
        nn.BatchNorm2d(ins_*2),
        nn.ReLU(inplace=True),
        nn.Conv2d(ins_*2, outs_, kernel_, stride_, padding_, bias=False),
        nn.BatchNorm2d(outs_),
        nn.ReLU(inplace=True)
        )

    def forward(self,x):
        return self.Decoder(self.Encoder(x))

    def init_weights(self):
        _count_module, _total_module = 0, 0
        for module in self.parameters():
            if len(module.shape) >= 2:
                nn.init.kaiming_normal_(module)
                _count_module += 1
                _total_module += 1
            #elif len(module.shape) == 1:
            #    nn.init.normal_(module, mean=0., std=0.414)
            #    _count_module += 1
            #    _total_module += 1
            else:
                _total_module += 1
        print(f"Model weights initialized ({_count_module}/{_total_module})")
