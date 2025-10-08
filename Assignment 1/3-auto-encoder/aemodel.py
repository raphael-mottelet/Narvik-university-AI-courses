import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.ReLU()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Autoencoder(nn.Module):
    def __init__(self, in_dim: int, hidden: list[int], z_dim: int, health_head_hidden=None):
        super().__init__()
        health_head_hidden = health_head_hidden or []

        # Encoder
        if hidden:
            self.encoder = MLP([in_dim, *hidden])
            enc_out = hidden[-1]
        else:
            self.encoder = nn.Identity()
            enc_out = in_dim
        self.enc_head = nn.Linear(enc_out, z_dim)

        # Decoder
        rev = list(reversed(hidden))
        if rev:
            self.decoder = MLP([z_dim, *rev])
            dec_in = rev[-1]
        else:
            self.decoder = nn.Identity()
            dec_in = z_dim
        self.dec_out = nn.Linear(dec_in, in_dim)
        self.out_act = nn.Sigmoid()

        # Auxiliary health classifier head (z -> 5 logits)
        h_layers = []
        in_h = z_dim
        for h in health_head_hidden:
            h_layers += [nn.Linear(in_h, h), nn.ReLU()]
            in_h = h
        h_layers += [nn.Linear(in_h, 5)]
        self.health_head = nn.Sequential(*h_layers)

    def forward(self, x):
        h = self.encoder(x)
        z = self.enc_head(h)
        d = self.decoder(z)
        xhat = self.out_act(self.dec_out(d))
        health_logits = self.health_head(z)
        return xhat, z, health_logits
