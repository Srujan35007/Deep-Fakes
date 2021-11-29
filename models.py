import torch 
import torch.nn as nn 


class Encoder(nn.Module):
    def __init__(self):
        # input shape: -1, 3, 256, 256
        # output shape: -1, 2, 16, 16
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
        self.init_weights()

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
    
    def init_weights(self):
        init, total = 0, 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 0, 0.5)
                init += 1
                total += 1
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                init += 1
                total += 1
            else:
                total += 1
        print(f"({self.__class__}) Model weigths initialized. ({init}/{total})")

    def forward(self, x):
        return self.layers(x).view(-1,2,16,16)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # input shape: -1, 2, 16, 16
        # output shape: -1, 3, 256, 256
        self.layers = nn.Sequential(
        self.decoder_block(2, 16, 3, 1, 1),
        nn.UpsamplingBilinear2d(scale_factor=2),
        self.decoder_block(16, 32, 5, 1, 2),
        nn.UpsamplingBilinear2d(scale_factor=2),
        self.decoder_block(32, 64, 7, 1, 3),
        nn.UpsamplingBilinear2d(scale_factor=2),
        self.decoder_block(64, 128, 5, 1, 2),
        nn.UpsamplingBilinear2d(scale_factor=2),
        self.decoder_block(128, 3, 3, 1, 1)
        )
        self.init_weights()

    def decoder_block(self, ins, outs, k, s, p):
        '''
        ins: input channels
        outs: output channels
        k: kernel/filter size
        s: stride
        p: padding
        '''
        return nn.Sequential(
        nn.Conv2d(ins, outs, k, s, p),
        nn.BatchNorm2d(outs),
        nn.ReLU(),
        )
    
    def init_weights(self):
        init, total = 0, 0
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, 0, 0.5)
                init += 1
                total += 1
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                init += 1
                total += 1
            else:
                total += 1
        print(f"({self.__class__}) Model weigths initialized. ({init}/{total})")

    def forward(self, x):
        return self.layers(x)
