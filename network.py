import torch.nn as nn
from torch.nn.functional import normalize
import torch
from torch.nn.parameter import Parameter

class Encoder_imvc(nn.Module):
    def __init__(self, input_dim, feature_dim, n_clusters):
        super(Encoder_imvc, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim),
        )
    def forward(self, x):
        return self.encoder(x)

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Autoencoder(nn.Module):
    def __init__(self,encoder_dim):
        super(Autoencoder, self).__init__()
        self._dim = len(encoder_dim) - 1
        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                encoder_layers.append(nn.ReLU())
        encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            decoder_layers.append(nn.ReLU())
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return latent, -1, x_hat, -1


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view
        self.ae_list = []
        for vi in range(self.view):
            auencoder = Autoencoder([input_size[vi], 1024, 1024, 1024, high_feature_dim]).to(device)
            self.ae_list.append(auencoder)

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_module(z), dim=1) # Note 不同view共享了h
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return hs, qs, xrs, zs

    def forward_mmi(self, xs):
        hs = []
        xrs = []
        for v in range(self.view):
            h, _, xr, _ = self.ae_list[v](xs[v])
            hs.append(h)
            xrs.append(xr)
        return hs, -1, xrs, -1


class GCFAggMVC(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(GCFAggMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim*view, high_feature_dim),
        )
        self.view = view
        self.TransformerEncoderLayer = nn.TransformerEncoderLayer(d_model=low_feature_dim*view, nhead=1, dim_feedforward=256)
        self.TransformerEncoder = nn.TransformerEncoder(self.TransformerEncoderLayer, num_layers=1)

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return  hs, -1, xrs, zs

    def GCFAgg(self, xs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
        commonz = torch.cat(zs, 1)
        commonz, S = self.TransformerEncoderLayer(commonz)
        commonz = normalize(self.Common_view(commonz), dim=1)
        return commonz, S