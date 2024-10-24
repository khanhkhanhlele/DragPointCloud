import torch


def get_handle_points(anchor_points, points, r):
    # anchor_point: list of [x0, y0, z0]
    # points: N x [xi, yi, zi] : (N, 3)
    # r: radius: int
    # return list handle points: list of [xi, yi, zi], mask: (N, 1)
    mask_list = []
    handle_points_list = []
    for i in range(len(anchor_points)):
        anchor = anchor_points[i]
        dis = torch.norm(points - anchor, dim=1) # can get dis here to scale down the d 
        mask = dis < r
        handle_points = points[mask]
        mask_list.append(mask)
        handle_points_list.append(handle_points)
    return handle_points_list, mask_list

def get_k_nearest_points(anchor_points, points, k):
    handle_points_list = []
    mask_list = []
    for i in range(len(anchor_points)):
        anchor = anchor_points[i]
        dis = torch.norm(points - anchor, dim=1)
        _, index = torch.topk(dis, k)
        mask = torch.zeros_like(dis, dtype=torch.bool)
        mask[index] = True
        handle_points = points[mask]
        mask_list.append(mask)
        handle_points_list.append(handle_points)

    return handle_points_list, mask_list


def get_index_nearest_anchor(anchor_points_user, points, r):
    list_index = []
    for i in range(len(anchor_points_user)):
        anchor = anchor_points_user[i]
        dis = torch.norm(points - anchor, dim=-1)
        index = torch.argmin(dis)
        list_index.append(index)
    return list_index
        


def check_anchor_reach_target(anchor_points, target_points, thres):
    all_dist = list(map(lambda p, q: torch.norm(p - q), anchor_points, target_points))
    print(all_dist)
    return (torch.tensor(all_dist) < thres).all(dim=0)


def point_tracking(anchor_points, points, target_points, thres):
    _, mask_list = get_handle_points(anchor_points, points, thres)
    for i in range(len(anchor_points)):
        pi, ti = anchor_points[i], target_points[i]

        # if torch.norm(ti - pi) < thres: 
        #     continue
        # di = (ti - pi) / torch.norm(ti - pi)
        di = (ti - pi) 
        mask = mask_list[i]
        points[mask] = points[mask] + di 
        anchor_points[i] = anchor_points[i] + di

    return points, anchor_points




from models.vae_adain import Model as VAE
from models.latent_points_ada_localprior import PVCNN2Prior as LocalPrior
from utils.diffusion_pvd import DiffusionDiscretized
from utils.vis_helper import plot_points
from utils.model_helper import import_model
from diffusers import DDPMScheduler, DDIMScheduler
import torch
from matplotlib import pyplot as plt
import pdb
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMInverseScheduler, AutoencoderKL, DDIMScheduler


## GOBAL: latent points, LOCAL: latent shape

class LION(object):
    def __init__(self, cfg):
        self.vae = VAE(cfg).cuda()
        GlobalPrior = import_model(cfg.latent_pts.style_prior)
        global_prior = GlobalPrior(cfg.sde, cfg.latent_pts.style_dim, cfg).cuda()
        local_prior = LocalPrior(cfg.sde, cfg.shapelatent.latent_dim, cfg).cuda()
        self.priors = torch.nn.ModuleList([global_prior, local_prior])
        # self.scheduler = DDPMScheduler(clip_sample=False,
        #                                beta_start=cfg.ddpm.beta_1, beta_end=cfg.ddpm.beta_T, beta_schedule=cfg.ddpm.sched_mode,
        #                                num_train_timesteps=cfg.ddpm.num_steps, variance_type=cfg.ddpm.model_var_type)
        self.scheduler = DDIMScheduler(
            clip_sample=False,
            beta_start=cfg.ddpm.beta_1,
            beta_end=cfg.ddpm.beta_T,
            beta_schedule=cfg.ddpm.sched_mode,
            num_train_timesteps=cfg.ddpm.num_steps
        )
        self.inverse_scheduler = DDIMInverseScheduler(
            clip_sample=False,
            beta_start=cfg.ddpm.beta_1,
            beta_end=cfg.ddpm.beta_T,
            beta_schedule=cfg.ddpm.sched_mode,
            num_train_timesteps=cfg.ddpm.num_steps
        )
        self.scheduler.set_timesteps(50, device='cuda')
        self.inverse_scheduler.set_timesteps(50, device='cuda')


        self.diffusion = DiffusionDiscretized(None, None, cfg)
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        # self.load_model(cfg)

    def load_model(self, model_path):
        # model_path = cfg.ckpt.path
        ckpt = torch.load(model_path)
        self.priors.load_state_dict(ckpt['dae_state_dict'])
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        print(f'INFO finish loading from {model_path}')


    #prepare condition
    @torch.no_grad()
    def prepare_condition(self, num_samples, latent_shape, timesteps, clip_feat=None):
        global_prior = self.priors[0]

        x_T_shape = [num_samples] + latent_shape[0]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')
        condition_input = None
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
            noise_pred = global_prior(x=x_noisy, t=t_tensor.float(), 
                    condition_input=condition_input, clip_feat=clip_feat)
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        return x_noisy

    # @torch.no_grad()
    def sample(self, num_samples=10, clip_feat=None, save_img=False):
        # self.scheduler.set_timesteps(1000, device='cuda')
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        local_prior = self.priors[1]
        sampled_list = []
        output_dict = {}

        condition_input = self.prepare_condition(num_samples, latent_shape, timesteps, clip_feat)
        condition_input = self.vae.global2style(condition_input)
        sampled_list.append(condition_input)
        output_dict['z_global'] = condition_input


        #Prepare anchor point list

        anchor_points_user = [torch.tensor([1.8, -0.24, 0.22]).to('cuda')]
        target_points = [torch.tensor([2., -0.18, 0.22]).to('cuda')]

        #ddim inverse
        
        with torch.no_grad():
            # point_cloud = torch.load('point_cloud_no_drag.pt')
            point_cloud = torch.load('lion_ckpt/unconditional/airplane/samples.pt').to('cuda')
            point_cloud = point_cloud[6].unsqueeze(0).to('cuda')
            anchor_points_index_list = get_index_nearest_anchor(anchor_points_user, point_cloud, 0.5)
            # import pdb; pdb.set_trace()
            anchor_points = [point_cloud[0][i] for i in anchor_points_index_list]
            print(f"User anchor points: {anchor_points_user}")
            print(f"Anchor points: {anchor_points}")
            _, _, latent_list = self.vae.encode(point_cloud)

            latents = latent_list[-1][0]
            # latents = torch.load('all_tensor_list.pt')[-1]
            inverse_list = [latents]
            print(latents.shape)
            for i, t in enumerate(reversed(timesteps)):
                t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
                model_input = local_prior(x=latents, t=t_tensor.float(), 
                        condition_input=condition_input, clip_feat=clip_feat)
                latents, pred_x0 = self.inverse_scheduler.step(model_input, t, latents, return_dict=False)
                inverse_list.append(latents)
            torch.save(inverse_list, 'all_tensor_list_inverse.pt')
            print("Saved sampled_list to all_tensor_list_inverse.pt")
            torch.cuda.empty_cache()

        with torch.no_grad():
            memory_bank = []
            x_noisy_denoise = latents.detach().clone()
            for i, t in enumerate(timesteps):
                t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
                noise_pred = local_prior(x=x_noisy_denoise, t=t_tensor.float(), 
                        condition_input=condition_input, clip_feat=clip_feat)
                x_noisy_denoise = self.scheduler.step(noise_pred, t, x_noisy_denoise).prev_sample

                if i > 0:
                    v = x_noisy_denoise - prev
                    v = v.reshape((-1, 4))[:, :3]
                    memory_bank.append(v)
                prev = x_noisy_denoise
            print(f"Len of memory_bank: {len(memory_bank)}")


        local_prior.train()
        x_T_shape = [num_samples] + latent_shape[1]
        x_noisy_0 = latents.detach().clone()
        x_noisy_0.requires_grad = True
        optimizer = torch.optim.Adam([x_noisy_0], lr=0.05)
        scaler = torch.cuda.amp.GradScaler()

        for e in range(15):  # epoch loop
            optimizer.zero_grad()  
            all = []
            loss = torch.tensor(0.0, device=x_noisy_0.device, requires_grad=False)  # Tích lũy loss cho toàn bộ các bước
            
            
            for i, t in enumerate(timesteps):  # timestep loop
                
                with torch.autocast(device_type='cuda'):
                    t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 1)
                    if i == 0:
                        noise_pred = local_prior(x=x_noisy_0, t=t_tensor.float(), 
                                                condition_input=condition_input, clip_feat=clip_feat)
                        x_noisy = self.scheduler.step(noise_pred, t, x_noisy_0).prev_sample
                    else:
                        noise_pred = local_prior(x=x_noisy, t=t_tensor.float(), 
                                                condition_input=condition_input, clip_feat=clip_feat)
                        x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample

                    # Start calculating loss from step 44
                    if i >= 44:
                        
                        
                        t_prev_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t + 2)
                        noise_prev_pred = local_prior(x=x_noisy, t=t_prev_tensor.float(), 
                                                    condition_input=condition_input, clip_feat=clip_feat)
                        x_noisy_prev = self.scheduler.step(noise_prev_pred, t, x_noisy).prev_sample

                        v_pred = x_noisy_prev - x_noisy
                        v = v_pred.reshape((-1, 4))[:, :3]

                        # Update anchor, handle points
                        
                        # anchor_points = [torch.tensor([-0.9, 2., 0.67]).to(x_noisy.device)]
                        # target_points = [torch.tensor([-0.9, 3., 0.67]).to(x_noisy.device)]
                        points = x_noisy.reshape((-1, 4))[:, :3]

                        
                        anchor_points = [points[i] for i in anchor_points_index_list]
                        _, mask_list = get_handle_points(anchor_points, points, 0.5)
                        #_, mask_list = get_k_nearest_points(anchor_points, points, 100)
                        print(f"Anchor points: {anchor_points}, Target points: {target_points}")
                        for j in range(len(anchor_points)):
                            anchor, target = anchor_points[j], target_points[j]
                            mask = mask_list[j]
                            if j == 0:
                                print(f"Num handle points: {mask.sum()}")
                            #d = (target - anchor) #/ torch.tensor(6.)

                            d = target - anchor_points_user[j]

                            # if density(anchor_points, 0.5) < density(anchor_points + d, 0.5):
                            #     new_anchor_points = cal_new_anchor_points(anchor_points, d, 0.5)

                            # anchor_points[j] = anchor + d / torch.tensor(60.)
                            
                            d = d.repeat(points.shape[0], 1)
                            loss2 = F.mse_loss(v[~mask], torch.zeros_like(v[~mask]))
                            # print(f"loss1: {loss} \t Loss2: {loss2}")
                            v_gt = memory_bank[i-1]
                            loss1 = F.mse_loss(v[mask], d[mask])
                            loss2 = nn.CosineEmbeddingLoss()(v[mask], d[mask], torch.ones(v[mask].size(0), device=v.device))
                            
                            loss4 = F.mse_loss(v[~mask], torch.zeros_like(v[~mask]))
                            loss5 = F.mse_loss(v[~mask], v_gt[~mask])
                            loss6 = F.l1_loss(10*torch.norm(v[mask],dim=1), torch.norm(d[mask], dim=1))
                            print(f"loss1: {loss1} \t loss2: {loss2} \t loss4: {loss4} \t loss5: {loss5} \t loss6: {loss6}")
                            loss = loss +  loss2# + loss6
                            print(f"loss: {loss}")
                    all.append(x_noisy)

            print(loss)
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
                    
            
        torch.save(all, 'all_tensor_list.pt')
        print("Saved sampled_list to all_tensor_list.pt")
        torch.cuda.empty_cache()

        #Updated init latent 
        x_noisy_0.requires_grad = False
        x_noisy = x_noisy_0

        with torch.no_grad():
            for i, t in enumerate(timesteps):
                t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
                noise_pred = local_prior(x=x_noisy, t=t_tensor.float(), 
                        condition_input=condition_input, clip_feat=clip_feat)
                x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample

            sampled_list.append(x_noisy)
            output_dict['z_local'] = x_noisy
            print(f"sampled_list len: {len(sampled_list)}")
            print(f"condition: {sampled_list[0].shape}, local: {sampled_list[1].shape}")
            
            # decode the latent
            output = self.vae.sample(num_samples=num_samples, decomposed_eps=sampled_list)
            if save_img:
                out_name = plot_points(output, "/tmp/tmp.png")
                print(f'INFO save plot image at {out_name}')
            output_dict['points'] = output
            return output_dict

    def get_mixing_component(self, noise_pred, t):
        # usage:
        # if global_prior.mixed_prediction:
        #     mixing_component = self.get_mixing_component(noise_pred, t)
        #     coeff = torch.sigmoid(global_prior.mixing_logit)
        #     noise_pred = (1 - coeff) * mixing_component + coeff * noise_pred

        alpha_bar = self.scheduler.alphas_cumprod[t]
        one_minus_alpha_bars_sqrt = np.sqrt(1.0 - alpha_bar)
        return noise_pred * one_minus_alpha_bars_sqrt

        


        


