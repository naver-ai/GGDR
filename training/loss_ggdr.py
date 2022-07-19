# Generative Guided Discriminator Regularization(GGDR)
# Copyright (c) 2022-present NAVER Corp.
# Under NVIDIA Source Code License for StyleGAN2 with Adaptive Discriminator
# Augmentation (ADA)

import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from training.loss import Loss


class StyleGAN2GGDRLoss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2, ggdr_res=64):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.ggdr_res = [ggdr_res]

        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def run_G(self, z, c, ws=None, sync=True):
        with misc.ddp_sync(self.G_mapping, sync):
            if ws is None:
                ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img, output_feat = self.G_synthesis(ws, get_feat=True)
        return img, ws, output_feat

    def run_aug_if_needed(self, img, gfeats):
        """
        Augment image and feature map consistently
        """
        if self.augment_pipe is not None:
            aug_img, gfeats = self.augment_pipe(img, gfeats)
        else:
            aug_img = img
        return aug_img, gfeats

    def run_D(self, img, c, gfeats=None, sync=None):
        aug_img, gfeats = self.run_aug_if_needed(img, gfeats)
        with misc.ddp_sync(self.D, sync):
            logits, out = self.D(aug_img, c)

        return logits, out, aug_img, gfeats

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl   = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, _gen_feat = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits, _recon_gen_fmaps, _, _ = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())

                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws, gen_fmaps = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # recon fake features and w
                gen_img, _gen_ws, gen_fmaps = self.run_G(gen_z, gen_c, sync=sync)

                aug_gen_logits, aug_recon_gen_fmaps, aug_gen_img, aug_fmaps = \
                    self.run_D(gen_img, gen_c, gen_fmaps, sync=sync)

                loss_gan_gen = torch.nn.functional.softplus(aug_gen_logits) + \
                    aug_recon_gen_fmaps[max(aug_recon_gen_fmaps.keys())][:, 0, 0, 0] * 0

                loss_gen_reg = self.get_ggdr_reg(self.ggdr_res, aug_recon_gen_fmaps, aug_fmaps)

                loss_Dmain = loss_gan_gen + loss_gen_reg

                training_stats.report('Loss/D/loss_gan_gen', loss_gan_gen)
                training_stats.report('Loss/D/loss_gen_reg', loss_gen_reg)

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dmain.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits, aug_recon_real_fmaps, _, _ = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report(f'Loss/D/loss', loss_Dreal + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

                # collect not used branch for DDP training
                loss_not_used = aug_recon_real_fmaps[max(aug_recon_real_fmaps.keys())][:, 0, 0, 0] * 0

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1 + real_logits * 0 + loss_not_used * 0).mean().mul(gain).backward()

    def cosine_distance(self, x, y):
        return 1. - F.cosine_similarity(x, y).mean()

    def get_ggdr_reg(self, ggdr_resolutions, source, target):
        loss_gen_recon = 0

        for res in ggdr_resolutions:
            loss_gen_recon += 10 * self.cosine_distance(source[res], target[res]) / len(ggdr_resolutions)

        return loss_gen_recon
