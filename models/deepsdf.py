# Code modified from original version:
# https://github.com/facebookresearch/DeepSDF/blob/master/networks/deep_sdf_decoder.py

import torch.nn as nn
import torch.nn.functional as F


class DeepSDFDecoder(nn.Module):
    def __init__(self, latent_size, output_size=3, **kwargs):
        super(DeepSDFDecoder, self).__init__()
        self.latent_size = latent_size
        self.output_size = output_size

        self.decoder = Decoder(latent_size=latent_size, output_size=output_size, **kwargs)

    def forward(self, latent, vertices):
        b, f = latent.shape
        assert f == self.latent_size

        if len(vertices.shape) == 3:
            # vertices already have the batch dimension (assuming it's been already expanded)
            b_v, v, xyz = vertices.shape
            assert b == b_v
        else:
            # assuming vertices without the batch dimension
            v, xyz = vertices.shape
            vertices = vertices[None, :, :].expand(b, -1, -1)

        inp = torch.empty((b, v, self.latent_size + 3), device=latent.device)
        inp[..., -3:] = vertices
        inp[..., :-3] = latent[:, None, :].expand(-1, v, -1)

        inp = inp.reshape(b * v, self.latent_size + 3)

        out = self.decoder(inp)

        out = out.reshape(b, v, self.output_size)

        return out


class Decoder(nn.Module):
    def __init__(
        self,
        latent_size=256,
        output_size=3,
        last_activation='tanh',
        dims=[512, 512, 512, 512, 512, 512, 512, 512],
        dropout=[0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob=0.2,
        norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
        latent_in=[4],
        weight_norm=True,
        xyz_in_all=False,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        dims = [latent_size + 3] + dims + [output_size]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                # CHANGE (w.r.t. the original DeepSDF network)
                # without this change the model works only if (input_features + 3) < 512
                # out_dim = dims[layer + 1] - dims[0]
                out_dim = dims[layer + 1]
                dims[layer + 1] = dims[layer + 1] + dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 3

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        if last_activation == 'tanh':
            self.last_activation = nn.Tanh()
        elif last_activation is None:
            self.last_activation = None
        else:
            raise ValueError

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -3:]

        if input.shape[1] > 3 and self.latent_dropout:
            latent_vecs = input[:, :-3]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if self.last_activation is not None:
            x = self.last_activation(x)

        return x


if __name__ == '__main__':
    # test
    import time
    import torch
    from torch.optim import SGD

    t = time.time()

    print('test model init and forward')

    verts = torch.tensor([[1, 3, 4], [2, 5, 5], [6, 4, 2], [1, 1, 1], [5, 6, 7]],
                            dtype=torch.float, requires_grad=True)
    gt = torch.tensor([
        [[1, 1, 1], [3, 5, 4], [7, 2, 4], [0, 0, 2], [4, 8, 9]],
        [[7, 2, 4], [0, 0, 2], [4, 8, 9], [1, 1, 1], [3, 5, 4]],
    ], dtype=torch.float)

    model = DeepSDFDecoder(latent_size=256, weight_norm=False, last_activation=None)

    latent_code = torch.rand((2, 256))

    out = model(latent_code, verts)
    print(latent_code.shape, out.shape)
    print(out.min(), out.max())

    print('test training')

    print(gt)
    print(verts)

    optim = SGD([verts] + list(model.parameters()), lr=0.001, momentum=0.9)

    for step in range(1000):
        out = model(latent_code, verts)

        optim.zero_grad()
        loss = torch.nn.functional.mse_loss(verts + out, gt)
        loss.backward()
        optim.step()

        if step % 10 == 0:
            print(step, loss)

    print(gt)
    print(verts)
    print(out)
    print(verts + out)

    print(time.time() - t)
