# Generative Guided Discriminator Regularization(GGDR)
# Copyright (c) 2022-present NAVER Corp.
# Under NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from training.networks import Conv2dLayer, MappingNetwork, DiscriminatorBlock, DiscriminatorEpilogue
from training.networks import SynthesisNetwork as OrigSynthesisNetwork

#----------------------------------------------------------------------------

@persistence.persistent_class
class SynthesisNetwork(OrigSynthesisNetwork):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        super().__init__(w_dim, img_resolution, img_channels, channel_base, channel_max, num_fp16_res, **block_kwargs)

    def forward(self, ws, get_feat=False, **block_kwargs):
        block_ws = []
        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        x = img = None

        feats = {}
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, cur_ws, **block_kwargs)

            if get_feat:
                feats[res] = x.float()

        if get_feat:
            return img, feats
        else:
            return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        img = self.synthesis(ws, **synthesis_kwargs)
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Discriminator(torch.nn.Module):
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 0,        # Use FP16 for the N highest resolutions.
        conv_clamp          = None,     # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
        w_dim               = 512,
        decoder_res         = 64,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)
        self.fp16_resolution = fp16_resolution

        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers

        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)

        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, **epilogue_kwargs, **common_kwargs)

        # *************************************************
        # Decoder part for GGDR loss
        # *************************************************
        dec_kernel_size = 1
        self.dec_resolutions = [2 ** i for i in range(3, int(np.log2(decoder_res)) + 1)]

        for res in self.dec_resolutions:
            out_channels = channels_dict[res]
            in_channels = channels_dict[res // 2]
            if res != self.dec_resolutions[0]:
                in_channels *= 2

            if res != self.dec_resolutions[-1]:
                act = 'lrelu'
            else:
                act = 'linear'

            block = Conv2dLayer(in_channels, out_channels, kernel_size=dec_kernel_size,
                                activation=act, up=2)
            setattr(self, f'b{res}_dec', block)

    def forward(self, img, c, **block_kwargs):
        x = None
        feats = {}
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)
            feats[res // 2] = x  # keep feature maps for unet decoder

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)

        logits = self.b4(x, img, cmap)  # original real/fake logits

        # Run decoder part
        fmaps = {}
        for idx, res in enumerate(self.dec_resolutions):
            block = getattr(self, f'b{res}_dec')
            if idx == 0:
                y = feats[res // 2]
            else:
                y = torch.cat([y, feats[res // 2]], dim=1)
            y = block(y)
            fmaps[res] = y

        return logits, fmaps

#----------------------------------------------------------------------------
