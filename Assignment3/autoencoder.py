# %% imports
import torch
import torch.nn as nn

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 x 16
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8 x 8
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4 x 4
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2 x 2
            
            nn.Conv2d(16, 1, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((1,2)), # 2 x 1
            )
        
    def forward(self, x):
        h = self.layers(x)
        return h
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=(1,2)),
            nn.ConvTranspose2d(1, 16, 3, padding=1),
            nn.ReLU(), # 2 x 2
            
            nn.Upsample(scale_factor=(2)),
            nn.ConvTranspose2d(16, 16, 3, padding=1),
            nn.ReLU(), # 4 x 4
            
            nn.Upsample(scale_factor=(2)),
            nn.ConvTranspose2d(16, 16, 3, padding=1),
            nn.ReLU(), # 8 x 8
            
            nn.Upsample(scale_factor=(2)),
            nn.ConvTranspose2d(16, 16, 3, padding=1),
            nn.ReLU(), # 16 x 16
            
            nn.Upsample(scale_factor=(2)),
            nn.ConvTranspose2d(16, 1, 3, padding=1),# 32 x 32
            )
        
    def forward(self, h):
        r = self.layers(h)
        return r
    
# %%  Autoencoder
class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        
    def forward(self, x):
        h = self.encoder(x)
        r = self.decoder(h)
        return r, h
    
    
# %% Classifier
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 16 x 16
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 8 x 8
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 4 x 4
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 2 x 2
            
            nn.Flatten(),
            nn.Linear(64,10)
            )
        
    def forward(self, x):
        y = self.layers(x)
        return y
