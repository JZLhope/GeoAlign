import torch
import torch.nn as nn
import torch.nn.functional as F

class ISMS(nn.Module):
    def __init__(self, input_dim=3840, latent_dim=1280, drop_rate=0.0):
        super(ISMS, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim, bias=False)
        self.bn_whiten = nn.BatchNorm1d(latent_dim, affine=True)
        self.decoder = nn.Linear(latent_dim, input_dim, bias=True)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.encoder.weight)
        nn.init.ones_(self.bn_whiten.weight) 
        nn.init.zeros_(self.bn_whiten.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        z_raw = self.encoder(x)
        z = self.bn_whiten(z_raw)
        recon = self.decoder(z)
        return z, recon