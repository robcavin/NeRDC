import os

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import torchvision
import torch
import pytorch3d.transforms as py3d_transforms
import glob
from PIL import Image

from NeRDC import NeRF, get_rays, render_rays

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# def tx(R,T,v):
#     return py3d_transforms.quaternion_apply(R,v) + T
#
#
# def inv_tx(R,T,v):
#     R_inv = py3d_transforms.quaternion_invert(R)
#     return py3d_transforms.quaternion_apply(R_inv,v - T)


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

    # o, d = ndc_rays(800,1280,880,near,rays_o,rays_d)
    # far_bounds = rays_o + rays_d * (far - rays_o[:, 2:])


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
