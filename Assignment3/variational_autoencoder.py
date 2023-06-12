# %% imports
import torch
import torch.nn as nn

# %%  Encoder
class Encoder(nn.Module):
    def __init__(self,latent_size):
        super(Encoder, self).__init__()
        
        self.shared_layers = nn.Sequential(
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
            )
        
        self.mu_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,latent_size),
            )
        
        self.log_var_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,latent_size),
            )
        
    def forward(self, x):
        shared = self.shared_layers(x)
        mu = self.mu_layers(shared)
        log_var = self.log_var_layers(shared)
        return mu, log_var
    
# %%  Decoder
class Decoder(nn.Module):
    def __init__(self,latent_size):
        super(Decoder, self).__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(latent_size,64),
            nn.ReLU(),
            )
        
        self.conv_layers = nn.Sequential(            
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
        h = self.linear_layer(h)
        h = h.reshape(h.size(0),16,2,2)
        rec = self.conv_layers(h)
        return rec
    
# %%  Autoencoder
class VAE(nn.Module):
    def __init__(self,latent_size):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_size)
        self.decoder = Decoder(latent_size)
        
    def forward(self, x):
        mu,log_var = self.encoder(x)
        
        h = self.reparameterize(mu, log_var)
        
        r = self.decoder(h)
        return r, mu, log_var
    
    def reparameterize(self,mu,log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        h = mu + eps*std
        return h
