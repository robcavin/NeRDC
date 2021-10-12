import _tkinter
import os

import numpy as np
import pytorch3d.transforms
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import torchvision
import torch
import torch.nn as nn
import math
import pytorch3d.transforms as py3d_transforms
import glob
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class NeRF(nn.Module):
    def __init__(self):
        super(NeRF, self).__init__()
        NUM_POS_CHANNELS = 63 # FIXME - Mismatch between paper and code
        NUM_VIEW_CHANNLES = 27

        self.fc1 = nn.Linear(NUM_POS_CHANNELS,256) # Positionally encoded inputs
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256,256)
        self.fc6_w_skip = nn.Linear(256 + NUM_POS_CHANNELS, 256) # Skip connection
        self.fc7 = nn.Linear(256,256)
        self.fc8 = nn.Linear(256,256)

        self.occupancy = nn.Linear(256,1) # Opacity
        self.feature = nn.Linear(256, 256) # Surface feature vector

        self.fc9 = nn.Linear(256 + NUM_VIEW_CHANNLES, 128) # Feature + encoded view inputs
        self.fc10 = nn.Linear(128, 3) # To RGB

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, encoded_positions, encoded_views):
        x = self.relu(self.fc1(encoded_positions))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))

        x = torch.cat((x, encoded_positions),dim=2)
        x = self.relu(self.fc6_w_skip(x))
        x = self.relu(self.fc7(x))
        x = self.relu(self.fc8(x))

        occupancy = self.relu(self.occupancy(x))
        x = self.feature(x)

        x = torch.cat((x,encoded_views), dim=2)

        x = self.relu(self.fc9(x))
        rgb = self.sigmoid(self.fc10(x))

        return rgb, occupancy


def encode(x, len) :
    freq_multiplier = 1
    # FIXME - The paper doesn't assume we pass in raw X, but the code does
    #  The right implementation should not need x in the brackets below...
    scaled_x = x / torch.tensor([2.5,1.7,1],device=device)
    scaled_x[...,-1] = (scaled_x[...,-1] - 0.2) / 1.8 * 2 - 1
    vals = [scaled_x]
    for i in range(len) :
        vals.append(torch.sin(freq_multiplier * math.pi * scaled_x)) # FIXME - Paper vs. code mismatch
        vals.append(torch.cos(freq_multiplier * math.pi * scaled_x))
        freq_multiplier *= 2
    return torch.cat(vals,dim=-1)


def get_rays(image,F,C,R,T):
    (H,W) = image.shape[1:3]
    x,y =torch.meshgrid(torch.linspace(0,H-1,H), torch.linspace(0,W-1,W))
    coords = torch.stack((y,x),dim=2)
    coords += 0.5
    rays_xy = (coords - C) / F
    rays = torch.cat((rays_xy, torch.ones((H, W, 1))),dim=2)

    rays_d = py3d_transforms.quaternion_apply(R,rays)
    assert((rays_d[:,:,2] > 0).all().item())

    # All rays are normalized so z=1 in world space, so
    #  that when multiplying by depth with sample uniformly
    #  in world z, not ray dir
    rays_d = rays_d / rays_d[:,:,2:]

    rays_o = T.tile((H,W,1))
    return rays_d, rays_o, image.permute(1,2,0)


def tx(R,T,v):
    return py3d_transforms.quaternion_apply(R,v) + T


def inv_tx(R,T,v):
    R_inv = py3d_transforms.quaternion_invert(R)
    return py3d_transforms.quaternion_apply(R_inv,v - T)



# FIXME - I don't understand this one yet.  The general gist is to invert
#  the probability as a function of the CDF, but I don't know how this code does that
def sample_pdf(bins, weights, N_samples, det=False):

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., N_samples, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=device)

    u = u.contiguous()

    # Invert CDF
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)

    #cdf_g = torch.gather(cdf, -1, inds_g, batch_dims=len(inds_g.shape)-2)
    #bins_g = torch.gather(bins, -1, inds_g, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1]-cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[..., 0])/denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1]-bins_g[..., 0])

    return samples



def render_samples(nerf, samples, view_dirs, near=0.0, far=1.0,):

    num_z_samples = samples.shape[-2]

    # FIXME - THIS IS WRONG.  Before encoding, samples should be in a -1 to 1
    #  range for the periodic function logic to make sense.
    #  We don't know the boudning volume for x and y, only z
    encoded_positions = encode(samples, 10)

    view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
    encoded_views = encode(view_dirs, 4).unsqueeze(1).expand(-1, num_z_samples, -1)

    (rgb, occupancy) = nerf(encoded_positions, encoded_views)

    # FIXME - Since the rays are not normalized, the distances t_i+1 - t_i != distance between two points
    #  Could rearrange this to save some compute - i.e. scale the sample distances for each normalized ray based on
    #  on it's direction.
    # FIXME - In the paper, dists are in z only?  Shouldn't it be energy along the ray?
    delta = (samples[:, 1:, 2:] - samples[:, :-1, 2:])
    dist = delta.norm(dim=2)

    # FIXME - Paper uses last distance to infinity.  This I guess captures any energy along the ray that
    #  wasn't caught by something else.  Does this work with my formulation?
    dist = torch.cat((dist, torch.tensor(1e10,device=device).expand(dist.shape[0], 1)), dim=-1)

    # FIXME - the first "delta"/"dist" should compare first point to point on ray at near dist.  Right now,
    #  I have one fewer distances than occupancies, and basically ignoring occupancy 0
    t = occupancy.squeeze(-1) * dist
    # accum = torch.zeros_like(t)
    # for i in range(num_z_samples - 1):  # FIXME - the -1 is because of the above issue
    #     accum[:, i:] += t[:, i, None]
    accum = torch.cumsum(t,-1)
    accum = torch.cat((torch.zeros((accum.shape[0], 1),device=device),accum[:,:-1]), -1)

    exp_accum = torch.exp(-accum)
    alpha = 1 - torch.exp(-t)

    weights = exp_accum * alpha

    # Their implementation - convert exp of sums to prod of exps
    # alpha = 1 - torch.exp(-t)
    #weights2 = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1),device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]

    c = torch.sum(weights.unsqueeze(-1) * rgb,
                  dim=1)  # FIXME - Skipping nearest sample as per above

    return c, weights


def render_rays(nerf_models, rays_d, rays_o, rand=True, near=0.0, far=1.0, num_coarse_samples=64, num_fine_samples=128):

    # Linearly sample along ray, sampling from uniform distribution in each bucket
    coarse_z_samples = torch.linspace(near, far, num_coarse_samples + 1, device=device)[:-1]
    if True: #  FIXME - What should we do with random samples?
        coarse_z_samples = coarse_z_samples + torch.rand((rays_d.shape[0], num_coarse_samples), device=device) * (
                far - near) / num_coarse_samples

    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1).expand(-1, num_coarse_samples, -1) * coarse_z_samples.unsqueeze(
        -1)

    rgb_coarse, weights_coarse = render_samples(nerf_models[0],samples,rays_d,near,far)

    # FIXME - The samplePDF function is lifted from the nerf pytorch code.  They
    #  also didn't include first or last samples... not sure why
    with torch.no_grad():
        z_vals_mid = .5 * (coarse_z_samples[...,1:] + coarse_z_samples[...,:-1])
        fine_z_samples = sample_pdf(z_vals_mid, weights_coarse[...,1:-1], num_fine_samples)

    # FIXME - Should we detach the fine_z samples??  Might not differentiate thorugh

    z_samples, _ = torch.sort(torch.cat([coarse_z_samples, fine_z_samples], -1), -1)

    samples = rays_o.unsqueeze(1) + rays_d.unsqueeze(1).expand(-1, num_coarse_samples + num_fine_samples, -1) * z_samples.unsqueeze(
        -1)
    rgb, weights = render_samples(nerf_models[1],samples,rays_d,near,far)

    # z_samples = coarse_z_samples
    # rgb = rgb_coarse
    # weights = weights_coarse

    #z_samples = z_samples[:,1:] #if rand else z_samples[1:] # FIXME - Still an off-by-one thing
    depth = torch.sum(weights * z_samples,dim=1)

    return rgb_coarse, rgb, weights, depth


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = (near - rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = 1. / (W / (2. * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = 1. / (H / (2. * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = 1. / (W / (2. * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = 1. / (H / (2. * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = 2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d



def train():
    files = glob.glob("/home/rob/Projects/datasets/rylie/corrected/img_*.png")
    files.sort()

    images = [torchvision.transforms.ToTensor()(Image.open(open(file,"rb"))) for file in files]

    device = "cuda:0"

    calib = torch.load("/home/rob/Projects/calib/iters/iter25600.th")
    F = calib["F"].detach()
    C = calib["C"].detach()
    D = calib["D"].detach().squeeze(1)
    R = calib["R"].detach()[:16]
    T = calib["T"].detach()[:16]

    R = R / R.norm(dim=1,keepdim=True)  # Ensure the quaternions are normalized

    # Fixup - Convert to cam2w.  R/T[0] = c02w, R/T[1:] = c02cX
    R_inv = py3d_transforms.quaternion_invert(R)
    T_inv = py3d_transforms.quaternion_apply(R_inv,-T)
    R_new = py3d_transforms.quaternion_multiply(R[0],R_inv)
    T_new = (py3d_transforms.quaternion_apply(R[0],T_inv) + T[0])


    all_rays_d = []
    all_rays_o = []
    all_colors = []
    cam_ids = []

    #debug = []
    for cam in range(len(F)) :
        rays_d, rays_o, colors = get_rays(images[cam],F[cam],C[cam],R_new[cam],T_new[cam])
        # debug.append(((rays_o[0][0], rays_d[0][0]),
        #                  (rays_o[799][0], rays_d[799][0]),
        #                  (rays_o[0][1279], rays_d[0][1279]),
        #                  (rays_o[799][1279], rays_d[799][1279])))
        all_rays_d.append(rays_d)
        all_rays_o.append(rays_o)
        all_colors.append(colors)
        cam_ids.append(torch.full(rays_d.shape[:-1],cam))
    #    centers.append((rays_o[400,640],rays_d[400,640]))

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # for x in debug:
    #     color = np.random.rand(3)
    #     for i in range(4):
    #         ray = torch.stack([x[i][0],x[i][0]+x[i][1]])
    #         ax.plot(ray[:,0],ray[:,1],ray[:,2],color=color)
    #
    # plt.show()

    test_rays_d = all_rays_d[0]
    test_rays_o = all_rays_o[0]

    rays_d = torch.stack(all_rays_d).reshape(-1,3)
    rays_o = torch.stack(all_rays_o).reshape(-1,3)
    colors = torch.stack(all_colors).reshape(-1,3)
    cam_ids = torch.stack(cam_ids).flatten()

    indices = torch.randperm(len(cam_ids))
    shuffled_rays_d = rays_d[indices].to(device)
    shuffled_rays_o = rays_o[indices].to(device)
    shuffled_colors = colors[indices].to(device)
    #shuffled_cam_ids = cam_ids[indices].to(device)

    batch_size = 4096
    near = 0.2
    far = 2
    num_coarse_samples = 64

    o, d = ndc_rays(800,1280,880,near,rays_o,rays_d)
    far_bounds = rays_o + rays_d * (far - rays_o[:, 2:])


    nerf_coarse = NeRF().to(device)
    nerf_fine = NeRF().to(device)

    #nerf_coarse.load_state_dict(torch.load("checkpoint_200000.th")["model_state_dict"])
    #nerf_fine.load_state_dict(torch.load("checkpoint_200000.th")["model_state_dict"])


    loss_fn = torch.nn.MSELoss()

    learning_rate = 5e-4
    optimizer = torch.optim.Adam(list(nerf_coarse.parameters()) + list(nerf_fine.parameters()), learning_rate)

    test_rays_d, test_rays_o,_ = get_rays(torch.zeros(1,800,800),F=torch.tensor([800,800]),C=torch.tensor([400,400]),R=torch.tensor([1.0,0,0,0]),T=torch.zeros(3))
    test_rays_o = test_rays_o.reshape(-1,3).to(device)
    test_rays_d = test_rays_d.reshape(-1,3).to(device)

    if os.path.exists("test"):
        import shutil
        shutil.rmtree('test')
    writer = SummaryWriter("test")

    iter = 0
    epoch = 0
    assert(len(shuffled_colors) % batch_size == 0)
    while iter < 500000:
        print("Epoch ",epoch)
        for batch_idx in range(0,len(shuffled_colors),batch_size) :
            optimizer.zero_grad()

            batch_rays_d = shuffled_rays_d[batch_idx : batch_idx + batch_size ]
            batch_rays_o = shuffled_rays_o[batch_idx : batch_idx + batch_size ]
            batch_colors = shuffled_colors[batch_idx : batch_idx + batch_size ]

            rgb_coarse, rgb, weights, depth = render_rays((nerf_coarse,nerf_fine), batch_rays_d, batch_rays_o, rand=True,
                                                          near=near, far=far, num_coarse_samples=num_coarse_samples)

            loss = loss_fn(rgb,batch_colors)
            loss += loss_fn(rgb_coarse,batch_colors)
            loss.backward()
            optimizer.step()


            # FIXME - This is specified as "IMPORTANT" in the original code, but in most
            #  cases it is suggested you only need to run 200k steps?
            decay_rate = 0.1
            decay_steps = 250 * 1000
            new_lrate = learning_rate * (decay_rate ** (iter / decay_steps))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lrate

            print(iter, loss.item())

            writer.add_scalar("loss",loss.item(),iter)

            if not ((iter+1) % 5000) :
                test_batch_size = len(test_rays_d) // 1000
                all_rgbs = []
                all_depths = []
                for test_idx in range(0,len(test_rays_d),test_batch_size) :
                    batch_rays_d = test_rays_d[test_idx:test_idx+test_batch_size]
                    batch_rays_o = test_rays_o[test_idx:test_idx+test_batch_size]
                    with torch.no_grad():
                        rgb_coarse, rgb, weights, depth = render_rays((nerf_coarse,nerf_fine), batch_rays_d, batch_rays_o, rand=False,
                                                                    near=near, far=far, num_coarse_samples=num_coarse_samples)
                    all_rgbs.append(rgb)
                    all_depths.append(depth)
                rgb = torch.cat(all_rgbs,dim=0).reshape(800,800,3)
                rgb = rgb.permute(2,0,1).clip(0,1)

                depth = torch.cat(all_depths,dim=0).reshape(1,800,800)
                normalized_depth = ((depth - near) / (far - near)).clip(0,1)

                writer.add_image('image',rgb, iter)
                writer.add_image('depth',normalized_depth, iter)

            if not ((iter+1) % 10000):
                torch.save({
                    'iter': iter,
                    'model_state_dict': nerf_fine.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, "checkpoint_{:06d}.th".format(iter+1))

            iter += 1
        epoch += 1


if __name__ == "__main__":
    train()
#
# x = [o[0] for o,d in centers]
# y = [o[1] for o,d in centers]
# z = [o[2] for o,d in centers]
# u = [d[0] for o,d in centers]
# v = [d[1] for o,d in centers]
# w = [d[2] for o,d in centers]
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.quiver(x,y,z,u,v,w, length=0.5, normalize=True)
# plt.show()

## forward = torch.tensor([0,0,1])
# base = torch.tensor([0,0,0])
#
# origin = tx(R_new,T_new,base)
# dir = tx(R_new,T_new,forward) - origin
#
# origin2 = tx(R[0],T[0],inv_tx(R,T,base))
# dir2 = tx(R[0],T[0],inv_tx(R,T,forward)) - origin2
#
# #dir = py3d_transforms.quaternion_apply(c2wR,forward - T) - origin
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# centers = torch.stack((origin,dir))
# ax.quiver(centers[0][:,0],centers[0][:,1],centers[0][:,2],centers[1][:,0],centers[1][:,1],centers[1][:,2], length=0.5, normalize=True)
#
# centers = torch.stack((origin2,dir2))
# ax.quiver(centers[0][:,0],centers[0][:,1],centers[0][:,2],centers[1][:,0],centers[1][:,1],centers[1][:,2], color="red",length=0.5, normalize=True)
#
# plt.show()
#
#
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# centers = []
# for cam in range(len(F)) :
#     rays_d,rays_o = get_rays(1280,800,F[cam],C[cam],R[cam],T[cam])
#     centers.append((rays_o[400,640],rays_d[400,640]))
#
# forward = torch.tensor([0,0,1])
# base = torch.tensor([0,0,0])
#
# #c2wR = py3d_transforms.quaternion_invert(R)
# #origin = py3d_transforms.quaternion_apply(c2wR,base - T)
#
# def tx(R,T,v):
#     return py3d_transforms.quaternion_apply(R,v) + T
#
# def inv_tx(R,T,v):
#     R_inv = py3d_transforms.quaternion_invert(R)
#     return py3d_transforms.quaternion_apply(R_inv,v - T)
#
# R_inv = py3d_transforms.quaternion_invert(R)
# T_inv = py3d_transforms.quaternion_apply(R_inv,-T)
# R_new = py3d_transforms.quaternion_multiply(R[0],R_inv)
# T_new = py3d_transforms.quaternion_apply(R[0],T_inv) + T[0]
#
# origin = tx(R[0],T[0],inv_tx(R,T,base))
# dir = tx(R[0],T[0],inv_tx(R,T,forward)) - origin
#
# #dir = py3d_transforms.quaternion_apply(c2wR,forward - T) - origin
# centers = torch.stack((origin,dir))
# ax.quiver(centers[0][:,0],centers[0][:,1],centers[0][:,2],centers[1][:,0],centers[1][:,1],centers[1][:,2], length=0.5, normalize=True)
# plt.show()
#
# x = [o[0] for o,d in centers]
# y = [o[1] for o,d in centers]
# z = [o[2] for o,d in centers]
# u = [d[0] for o,d in centers]
# v = [d[1] for o,d in centers]
# w = [d[2] for o,d in centers]
#
# ax.quiver(x,y,z,u,v,w, length=0.5, normalize=True)
# plt.show()
#
#
