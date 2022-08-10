import os
import random

import click
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from kmeans_pytorch import kmeans

import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import legacy
import dnnlib


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num_iters', help='Number of iteration for visualization', type=int, default=1)
@click.option('--batch_size', help='Batch size for clustering', type=int, default=64)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    outdir: str,
    num_iters: int,
    batch_size: int,
):
    """K-means visualization of generator feature maps. Cluster the images in the same batch(So the batch size matters here)

    Usage:
        python tools/visualize_gfeat.py --outdir=out --network=your_network_path.pkl
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(f'{outdir}', exist_ok=True)

    for iter_idx in range(num_iters):

        z = torch.from_numpy(np.random.randn(batch_size, G.z_dim)).to(device)
        ws = G.mapping(z, c=None, truncation_psi=truncation_psi)

        fake_imgs, fake_feat = G.synthesis(ws, get_feat=True)

        vis_img = []

        # the feature maps are saved in the dictionary whose keys are their
        # resolutions.
        target_layers = [16, 32, 64]
        num_clusters = 6

        for res in target_layers:
            img = get_cluster_vis(fake_feat[res], num_clusters=num_clusters, target_res=res)  # bnum, 256, 256
            vis_img.append(img)

        for idx, val in enumerate(vis_img):
            vis_img[idx] = F.interpolate(val, size=(256, 256))

        vis_img = torch.cat(vis_img, dim=0)  # bnum * res_num, 256, 256
        vis_img = (vis_img + 1) * 127.5 / 255.0
        fake_imgs = (fake_imgs + 1) * 127.5 / 255.0
        fake_imgs = F.interpolate(fake_imgs, size=(256, 256))

        vis_img = torch.cat([fake_imgs, vis_img], dim=0)
        vis_img = torchvision.utils.make_grid(vis_img, normalize=False, nrow=batch_size)
        torchvision.utils.save_image(vis_img, f'{outdir}/{iter_idx}.png')


def get_colors():
    dummy_color = np.array([
        [178, 34, 34],  # firebrick
        [0, 139, 139],  # dark cyan
        [245, 222, 179],  # wheat
        [25, 25, 112],  # midnight blue
        [255, 140, 0],  # dark orange
        [128, 128, 0],  # olive
        [50, 50, 50],  # dark grey
        [34, 139, 34],  # forest green
        [100, 149, 237],  # corn flower blue
        [153, 50, 204],  # dark orchid
        [240, 128, 128],  # light coral
    ])

    for t in (0.6, 0.3):  # just increase the number of colors for big K
        dummy_color = np.concatenate((dummy_color, dummy_color * t))

    dummy_color = (np.array(dummy_color) - 128.0) / 128.0
    dummy_color = torch.from_numpy(dummy_color)

    return dummy_color


def get_cluster_vis(feat, num_clusters=10, target_res=16):
    # feat : NCHW
    print(feat.size())
    img_num, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(img_num * H * W, -1)
    feat = feat.to(torch.float32).cuda()
    cluster_ids_x, cluster_centers = kmeans(
        X=feat, num_clusters=num_clusters, distance='cosine',
        tol=1e-4,
        device=torch.device("cuda:0"))

    cluster_ids_x = cluster_ids_x.cuda()
    cluster_centers = cluster_centers.cuda()
    color_rgb = get_colors().cuda()
    vis_img = []
    for idx in range(img_num):
        num_pixel = target_res * target_res
        current_res = cluster_ids_x[num_pixel * idx:num_pixel * (idx + 1)].cuda()
        color_ids = torch.index_select(color_rgb, 0, current_res)
        color_map = color_ids.permute(1, 0).view(1, 3, target_res, target_res)
        color_map = F.interpolate(color_map, size=(256, 256))
        vis_img.append(color_map.cuda())

    vis_img = torch.cat(vis_img, dim=0)

    return vis_img


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter
