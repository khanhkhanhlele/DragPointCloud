# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
    require diffusers-0.11.1
"""
import os
import clip
import torch
from PIL import Image
from default_config import cfg as config
from models.drag_point_utils_v2 import LION
from utils.vis_helper import plot_points
from huggingface_hub import hf_hub_download 

model_path = 'lion_ckpt/unconditional/airplane/checkpoints/model.pt'
model_config = 'lion_ckpt/unconditional/airplane/cfg.yml'

config.merge_from_file(model_config)
lion = LION(config)
lion.load_model(model_path)

device_str = 'cuda'
if config.clipforge.enable:
    input_t = ["a car"] 
    clip_model, clip_preprocess = clip.load(
                        config.clipforge.clip_model, device=device_str)    
    text = clip.tokenize(input_t).to(device_str)
    clip_feat = []
    clip_feat.append(clip_model.encode_text(text).float())
    clip_feat = torch.cat(clip_feat, dim=0)
    print('clip_feat', clip_feat.shape)
else:
    clip_feat = None
pc = []
for i in range(1):
    print(f'index: {i}')
    output = lion.sample(1 if clip_feat is None else clip_feat.shape[0], clip_feat=clip_feat,index=3,device = device_str)
    pts = output['points']
    plot_points(pts)
    pc.append(pts)
    
torch.save(pc, 'drag_point_cloud_v2.pt')

# img = Image.open(img_name)
# img.show()