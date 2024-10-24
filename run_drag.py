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

def check_anchor_reach_target(anchor_points, target_points, thres):
    all_dist = list(map(lambda p, q: torch.norm(p - q), anchor_points, target_points))
    print(all_dist)
    return (torch.tensor(all_dist) < thres).all(dim=0)


def point_tracking(anchor_points, points, target_points, thres):
    _, mask_list = get_handle_points(anchor_points, points, thres)
    for i in range(len(anchor_points)):
        pi, ti = anchor_points[i], target_points[i]

        if torch.norm(ti - pi) < thres: 
            continue
        di = (ti - pi) / torch.norm(ti - pi)
        mask = mask_list[i]
        points[mask] = points[mask] + di 
        anchor_points[i] = anchor_points[i] + di

    return points, anchor_points


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
        # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012,
        #             beta_schedule="scaled_linear", clip_sample=False,
        #             set_alpha_to_one=False, steps_offset=1)
        self.diffusion = DiffusionDiscretized(None, None, cfg)
        # self.load_model(cfg)

    def load_model(self, model_path):
        # model_path = cfg.ckpt.path
        ckpt = torch.load(model_path)
        self.priors.load_state_dict(ckpt['dae_state_dict'])
        self.vae.load_state_dict(ckpt['vae_state_dict'])
        print(f'INFO finish loading from {model_path}')

    @torch.no_grad()
    def sample(self, num_samples=10, clip_feat=None, save_img=False):
        # self.scheduler.set_timesteps(1000, device='cuda')
        self.scheduler.set_timesteps(25, device='cuda')
        timesteps = self.scheduler.timesteps
        latent_shape = self.vae.latent_shape()
        global_prior, local_prior = self.priors[0], self.priors[1]
        assert(not local_prior.mixed_prediction and not global_prior.mixed_prediction)
        sampled_list = []
        output_dict = {}

        # start sample global prior
        x_T_shape = [num_samples] + latent_shape[0]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')
        condition_input = None
        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
            noise_pred = global_prior(x=x_noisy, t=t_tensor.float(), 
                    condition_input=condition_input, clip_feat=clip_feat)
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        sampled_list.append(x_noisy)
        output_dict['z_global'] = x_noisy

        condition_input = x_noisy
        condition_input = self.vae.global2style(condition_input)

        # start sample local prior
        x_T_shape = [num_samples] + latent_shape[1]
        x_noisy = torch.randn(size=x_T_shape, device='cuda')

        # all = []

        for i, t in enumerate(timesteps):
            t_tensor = torch.ones(num_samples, dtype=torch.int64, device='cuda') * (t+1)
            noise_pred = local_prior(x=x_noisy, t=t_tensor.float(), 
                    condition_input=condition_input, clip_feat=clip_feat)
            x_noisy = self.scheduler.step(noise_pred, t, x_noisy).prev_sample
        #     all.append(x_noisy)
        # torch.save(all, 'all_tensor_list.pt')
        # print("Saved sampled_list to latents.npz")
        sampled_list.append(x_noisy)
        output_dict['z_local'] = x_noisy
        
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



def drag_diffusion_update(dae, init_code, t, anchor_points, target_points):
    assert len(anchor_points) == len(target_points)
    with torch.no_grad():
        x0_pred, x_prev_pred, epsilon_theta = dae.ddim_step(init_code, t)

    init_code.requires_grad = True 
    optimizer = torch.optim.Adam([init_code], lr=0.1) #replace with latent learning rate 
    
    #x_prev_pred shape ?
    points = init_code.reshape((-1, 4))[:, :3]
    handle_points, mask_list = get_handle_points(anchor_points, points, 0.3)
    #may change shape of the mask as same as the init_code 
    interp_mask = mask_list[0].reshape(-1, 1, 1, 1)
    using_mask = interp_mask.sum() != 0.0

    n_track_step = 80 
    scaler = torch.cuda.amp.GradScaler()
    for i in range(n_step_track):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            x0_pred, x_prev_pred, epsilon_theta = dae.ddim_step(init_code, t)
            points = init_code.reshape((-1, 4))[:, :3]
            if i != 0:
                points, anchor_points = point_tracking(anchor_points, points, target_points, 0.2)
            
            if check_anchor_reach_target(anchor_points, target_points, 0.2):
                break
            
            
            loss = F.mse_loss(points, x_prev_pred[:, :3])
     scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return init_code
            


                
                