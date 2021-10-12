import pytorch3d.transforms as py3d_transforms
import torch
import torch.nn as nn
import math


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


def encode(x, len, device='cuda:0') :
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


def get_rays(image,F,C,R,T, device='cuda:0'):
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


# FIXME - I don't understand this one yet.  The general gist is to invert
#  the probability as a function of the CDF, but I don't know how this code does that
def sample_pdf(bins, weights, N_samples, det=False, device='cuda:0'):

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


def render_samples(nerf, samples, view_dirs, near=0.0, far=1.0, device='cuda:0'):

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


def render_rays(nerf_models, rays_d, rays_o, rand=True, near=0.0, far=1.0, num_coarse_samples=64, num_fine_samples=128, device='cuda:0'):

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

