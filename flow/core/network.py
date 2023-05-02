import torch
import torch.nn as nn
import torch.nn.functional as F

from update import GMAUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from utils.utils import *
from gma import Attention, Aggregate

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

import matplotlib.pyplot as plt


class RAFTGMA(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4
        args.corr_radius = 4

        # if 'dropout' not in self.args:
        #     self.args.dropout = 0
        self.args.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
        self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0# normalized to [-1,1]
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.args.mixed_precision):# convert to float16
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()# convert back to float32
        fmap2 = fmap2.float()
        corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)
            # attention, att_c, att_p = self.att(inp)
            attention = self.att(inp)# spatio attention matrix

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)  # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow, attention)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions


if __name__=='__main__':
    import argparse
    import numpy as np
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='bla', help="name your experiment")
    # parser.add_argument('--stage', help="determines which dataset to use for training")
    # parser.add_argument('--validation', type=str, nargs='+')
    # parser.add_argument('--restore_ckpt', help="restore checkpoint")
    # parser.add_argument('--output', type=str, default='checkpoints',
    #                     help='output directory to save checkpoints and plots')
    #
    # parser.add_argument('--lr', type=float, default=0.00002)
    # parser.add_argument('--num_steps', type=int, default=100000)
    # parser.add_argument('--batch_size', type=int, default=6)
    # parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    # parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    #
    # parser.add_argument('--wdecay', type=float, default=.00005)
    # parser.add_argument('--epsilon', type=float, default=1e-8)
    # parser.add_argument('--clip', type=float, default=1.0)
    # parser.add_argument('--dropout', type=float, default=0.0)
    # parser.add_argument('--upsample-learn', action='store_true', default=False,
    #                     help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    # parser.add_argument('--gamma', type=float, default=0.8, help='exponential weighting')
    # parser.add_argument('--iters', type=int, default=12)
    # parser.add_argument('--val_freq', type=int, default=10000,
    #                     help='validation frequency')
    # parser.add_argument('--print_freq', type=int, default=100,
    #                     help='printing frequency')
    #
    # parser.add_argument('--mixed_precision', default=False, action='store_true',
    #                     help='use mixed precision')
    # parser.add_argument('--model_name', default='', help='specify model name')
    #
    # parser.add_argument('--position_only', default=False, action='store_true',
    #                     help='only use position-wise attention')
    # parser.add_argument('--position_and_content', default=False, action='store_true',
    #                     help='use position and content-wise attention')
    # parser.add_argument('--num_heads', default=1, type=int,
    #                     help='number of heads in attention and aggregation')
    #
    # args = parser.parse_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/gma-sintel.pth', help="restore checkpoint")
    parser.add_argument('--dataset', default='sintel', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')

    args = parser.parse_args()
    model = RAFTGMA(args)
    image1 = torch.Tensor(np.ones((3, 3, 368, 768)).astype(np.float32))
    image2 = torch.Tensor(np.ones((3, 3, 368, 768)).astype(np.float32))
    param_dict = torch.load('./checkpoints/gma-sintel.pth',map_location='cpu')
    from collections import OrderedDict
    def mod(param_dict):
        new_dict=OrderedDict()
        for key, value in param_dict.items():
            if key.startswith('module.'):
                new_dict[key[7:]]=value
        return new_dict
    param_dict=mod(param_dict)
    model.eval()
    model.load_state_dict(param_dict)
    output = model(image1,image2)

    print('input:')
    print(image1.shape, image2.shape)
    print('output:')
    for el in output:
        print(el.shape)

    # save_output = np.array(torch.cat(output,1).detach())
    # save_path = './output1.npy'
    # np.save(save_path,save_output)
    #
    # save_output_ = np.load(save_path)
    # print(np.sum(np.abs(save_output_-save_output)))
    # print(save_output_.shape)

    save_output1_ = np.load('./output1.npy')
    save_output2_ = np.load('./output2.npy')

    print(np.sum(np.abs(save_output1_ - save_output2_)))
    print(save_output1_.shape)