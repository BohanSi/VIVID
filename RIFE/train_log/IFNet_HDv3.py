import torch
import torch.nn as nn
import torch.nn.functional as F
from model.warplayer import warp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Correspondence perturbation utilities
# ============================================================

def _build_nested_perturbation_indices(
    num_positions,
    fraction,
    seed,
    device,
):
    """
    Build nested perturbation locations.

    For a fixed seed:

        S_0.25 ⊂ S_0.50 ⊂ S_0.75 ⊂ S_1.00

    The donor correspondence is also fixed across fractions.
    """

    num_positions = int(num_positions)
    fraction = float(fraction)
    seed = int(seed)

    if num_positions <= 1:
        raise ValueError(
            f"num_positions must be > 1, got {num_positions}"
        )

    if not (0.0 <= fraction <= 1.0):
        raise ValueError(
            f"perturb fraction must be in [0,1], got {fraction}"
        )

    k = int(np.floor(fraction * num_positions))

    if k <= 0:
        return None, None

    k = min(k, num_positions)

    # --------------------------------------------------------
    # IMPORTANT:
    # local RNG only
    #
    # This does NOT alter:
    #   torch.manual_seed(...)
    #   config["seed"]
    #   Random-RIFE initialization seed
    # --------------------------------------------------------

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    # One fixed random ordering
    order = torch.randperm(
        num_positions,
        generator=generator,
        device="cpu",
    )

    # donor for each target pixel
    #
    # Using a cyclic shift guarantees:
    #   target != donor
    # when num_positions > 1
    # --------------------------------------------------------

    target_index = order[:k].to(device)
    
    # A permutation requires at least two selected locations.
    if k < 2:
        return None, None
    
    # --------------------------------------------------------
    # Shuffle ONLY within the selected locations
    #
    # Example:
    #
    # selected:
    #     A, B, C, D
    #
    # donor:
    #     B, C, D, A
    #
    # Therefore:
    #     A <- flow(B)
    #     B <- flow(C)
    #     C <- flow(D)
    #     D <- flow(A)
    #
    # No new flow values are generated.
    # No flow vector from an unselected location is introduced.
    # --------------------------------------------------------
    
    donor_index = torch.roll(
        target_index,
        shifts=-1,
        dims=0,
    )
    
    return target_index, donor_index


def _perturb_correspondence_flow(
    flow,
    fraction,
    seed,
):
    """
    Randomly shuffle the spatial organization of flow vectors.

    flow:
        B x 4 x H x W

    IMPORTANT:
        all four channels at one spatial location are moved together:

        [left_dx, left_dy, right_dx, right_dy]

    Therefore the left/right correspondence pair remains internally
    consistent, while its spatial organization is perturbed.
    """

    fraction = float(fraction)

    if fraction <= 0.0:
        return flow

    if seed is None:
        raise ValueError(
            "perturb_seed must be explicitly provided "
            "when perturb_fraction > 0."
        )

    if flow.ndim != 4 or flow.shape[1] != 4:
        raise ValueError(
            f"Expected flow shape Bx4xHxW, got {tuple(flow.shape)}"
        )

    B, C, H, W = flow.shape
    N = H * W

    target_index, donor_index = (
        _build_nested_perturbation_indices(
            num_positions=N,
            fraction=fraction,
            seed=int(seed),
            device=flow.device,
        )
    )

    if target_index is None:
        return flow

    # B x 4 x N
    flat = flow.reshape(B, C, N)

    # IMPORTANT:
    # Read donor values from an unchanged copy.
    source = flat.clone()
    perturbed = flat.clone()

    # Shuffle complete 4-channel correspondence vectors.
    perturbed[:, :, target_index] = (
        source[:, :, donor_index]
    )

    k = int(target_index.numel())

    print(
        f"[Correspondence perturbation] "
        f"fraction={fraction:.4f} | "
        f"seed={int(seed)} | "
        f"selected={k}/{N} "
        f"({100.0 * k / N:.2f}%)"
    )

    return perturbed.reshape(B, C, H, W)



def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),        
        nn.PReLU(out_planes)
    )

def conv_bn(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.PReLU(out_planes)
    )

class IFBlock(nn.Module):
    def __init__(self, in_planes, c=64):
        super(IFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c//2, 3, 2, 1),
            conv(c//2, c, 3, 2, 1),
            )
        self.convblock0 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock1 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock2 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.convblock3 = nn.Sequential(
            conv(c, c),
            conv(c, c)
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(c, c//2, 4, 2, 1),
            nn.PReLU(c//2),
            nn.ConvTranspose2d(c//2, 1, 4, 2, 1),
        )

    def forward(self, x, flow, scale=1):
        x = F.interpolate(x, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        flow = F.interpolate(flow, scale_factor= 1. / scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * 1. / scale
        feat = self.conv0(torch.cat((x, flow), 1))
        feat = self.convblock0(feat) + feat
        feat = self.convblock1(feat) + feat
        feat = self.convblock2(feat) + feat
        feat = self.convblock3(feat) + feat        
        flow = self.conv1(feat)
        mask = self.conv2(feat)
        flow = F.interpolate(flow, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False) * scale
        mask = F.interpolate(mask, scale_factor=scale, mode="bilinear", align_corners=False, recompute_scale_factor=False)
        return flow, mask
        
class IFNet(nn.Module):
    def __init__(self):
        super(IFNet, self).__init__()
        self.block0 = IFBlock(7+4, c=90)
        self.block1 = IFBlock(7+4, c=90)
        self.block2 = IFBlock(7+4, c=90)
        self.block_tea = IFBlock(10+4, c=90)
        # self.contextnet = Contextnet()
        # self.unet = Unet()

    def forward(self, x, scale_list=[4, 2, 1], training=False, zero_correspondence=False, perturb_fraction=0.0, perturb_seed=None, uniform_mask=False,):
        
        perturb_fraction = float(perturb_fraction)
        # --------------------------------------------------------
        # Safety checks
        # --------------------------------------------------------
        if zero_correspondence and perturb_fraction > 0.0:
            raise ValueError(
                "zero_correspondence and perturb_fraction > 0 "
                "cannot be enabled simultaneously."
            )
    
        if not (0.0 <= perturb_fraction <= 1.0):
            raise ValueError(
                f"perturb_fraction must be in [0,1], "
                f"got {perturb_fraction}"
            )
    
        if perturb_fraction > 0.0 and perturb_seed is None:
            raise ValueError(
                "perturb_seed must be provided when "
                "perturb_fraction > 0."
            )
        
        
        if training == False:
            channel = x.shape[1] // 2
            img0 = x[:, :channel]
            img1 = x[:, channel:]
        flow_list = []
        merged = []
        mask_list = []
        warped_img0 = img0
        warped_img1 = img1
        flow = (x[:, :4]).detach() * 0
        mask = (x[:, :1]).detach() * 0
        loss_cons = 0
        block = [self.block0, self.block1, self.block2]
        for i in range(3):
            f0, m0 = block[i](torch.cat((warped_img0[:, :3], warped_img1[:, :3], mask), 1), flow, scale=scale_list[i])
            f1, m1 = block[i](torch.cat((warped_img1[:, :3], warped_img0[:, :3], -mask), 1), torch.cat((flow[:, 2:4], flow[:, :2]), 1), scale=scale_list[i])
            predicted_flow = flow + (f0 + torch.cat((f1[:, 2:4], f1[:, :2]), 1)) / 2
            mask = mask + (m0 + (-m1)) / 2

            # ----------------------------------------------------
            # Mode A:
            # w/o correspondence
            # C(u) = u
            # ----------------------------------------------------
            if zero_correspondence and i == 2:
                flow = torch.zeros_like(predicted_flow)
                warped_img0 = warp(img0, flow[:, :2],)
                warped_img1 = warp(img1, flow[:, 2:4],)
            # ----------------------------------------------------
            # Mode B:
            # correspondence perturbation
            # IMPORTANT:
            # Only perturb the FINAL RIFE correspondence field.
            # i = 0,1:
            #     completely normal pretrained RIFE refinement
            # i = 2:
            #     perturb final flow immediately before warp
            # ----------------------------------------------------
            elif (perturb_fraction > 0.0 and i == 2):
                flow = _perturb_correspondence_flow(
                    predicted_flow,
                    fraction=perturb_fraction,
                    seed=int(perturb_seed),
                )
                warped_img0 = warp(img0, flow[:, :2],)
                warped_img1 = warp(img1, flow[:, 2:4],)

            else:
                flow = predicted_flow
                warped_img0 = warp(
                    img0,
                    flow[:, :2],
                )
                warped_img1 = warp(
                    img1,
                    flow[:, 2:4],
                )
        
            mask_list.append(mask)
            flow_list.append(flow)
            warped_img0 = warp(img0, flow[:, :2])
            warped_img1 = warp(img1, flow[:, 2:4])
            merged.append((warped_img0, warped_img1))
        '''
        c0 = self.contextnet(img0, flow[:, :2])
        c1 = self.contextnet(img1, flow[:, 2:4])
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask, flow, c0, c1)
        res = tmp[:, 1:4] * 2 - 1
        '''
        for i in range(3):

            # RIFE learned spatial blending mask
            learned_mask = torch.sigmoid(mask_list[i])
        
            if uniform_mask:
                # ----------------------------------------------------
                # Mask ablation:
                # every local RIFE interpolation is a midpoint,
                # therefore use direct 0.5 / 0.5 blending.
                # ----------------------------------------------------
                blend_mask = torch.full_like(
                    learned_mask,
                    0.5,
                )
            else:
                blend_mask = learned_mask
        
            # Save the actually used mask
            mask_list[i] = blend_mask
        
            merged[i] = (
                merged[i][0] * blend_mask
                + merged[i][1] * (1.0 - blend_mask)
            )
        
        return flow_list, mask_list[2], merged
