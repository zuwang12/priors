import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import scipy.spatial
import cv2
import sys
from PIL import Image
import pandas as pd
import os

from unet import UNetModel
# from TSPDataset import TSPDataset
from model.TSPModel import TSPDataset
from diffusion import GaussianDiffusion
from tsp_utils import TSP_2opt, rasterize_tsp

import tqdm
import matplotlib.pyplot as plt
import reward_fns
from utils import calculate_distance_matrix2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--save_freq", type=int, default=2)
parser.add_argument("--num_cities", type=int, default=20)
parser.add_argument("--constraint_type", type=str, default='path')
parser.add_argument("--run_name", type=str, default='tsp_plug_and_play')
parser.add_argument("--start_idx", type=int, default=0)
parser.add_argument("--end_idx", type=int, default=1280)
args = parser.parse_args()

date_per_type = {
    'box' : '240710',
    'path' : '240711',
    'cluster' : '240721', 
}

print(args)

save_freq = 2
batch_size = 1
img_size = 64
constraint_type = args.constraint_type
reward_type = 'tsp_constraint'
reward_fn = getattr(reward_fns, reward_type)()
FILE_NAME = F'tsp{args.num_cities}_{args.constraint_type}_constraint_{date_per_type[args.constraint_type]}.txt'

device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
seed = 2024
deterministic = True

np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
if deterministic:
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
 

def normalize(cost, entropy_reg=0.1, n_iters=20, eps=1e-6):
    # Cost matrix is exp(-lambda*C)
    cost_matrix = -entropy_reg * cost # 0.1 * [1, 50, 50] (latent)
        
    cost_matrix -= torch.eye(cost_matrix.shape[-1], device=cost_matrix.device)*100000 # COST = COST - 100000*I
    cost_matrix = cost_matrix - torch.logsumexp(cost_matrix, dim=-1, keepdim=True)
    assignment_mat = torch.exp(cost_matrix)
    
    return assignment_mat # [1, 50, 50] (adj_mat)

class InferenceModel(nn.Module):
    def __init__(self):
        super(InferenceModel, self).__init__()
        
        # Latent variables (b,v,v) matrix
        self.latent = nn.Parameter(torch.randn(batch_size,points.shape[0],points.shape[0])) # (B, 50, 50)
        self.latent.requires_grad = True

        # Pre-compute edge images
        self.edge_images = []
        for i in range(points.shape[0]):
            node_edges = []
            for j in range(points.shape[0]):
                edge_img = np.zeros((img_size, img_size)) # (64, 64)
                cv2.line(edge_img, 
                         tuple(((img_size-1)*points[i,::-1]).astype(int)), # city position in 50x50 ex) (2, 39)
                         tuple(((img_size-1)*points[j,::-1]).astype(int)), 
                         color=test_dataset.line_color, thickness=test_dataset.line_thickness)
                edge_img = torch.from_numpy(edge_img).float().to(self.latent.device)

                node_edges.append(edge_img)
            node_edges = torch.stack(node_edges, dim=0)
            self.edge_images.append(node_edges)
        self.edge_images = torch.stack(self.edge_images, dim=0) # (50, 50, 64, 64) -> all edge connection image for each city
                        
    def encode(self):
        # Compute permutation matrix
        adj_mat = normalize(self.latent) # [1, 50, 50] -> [1, 50, 50]

        adj_mat_ = adj_mat
        all_edges = self.edge_images.view(1,-1,img_size,img_size).to(adj_mat.device)
        img = all_edges * adj_mat_.view(batch_size,-1,1,1) # [1, 2500, 64, 64] * [1, 50, 50] -> [1, 2500, 64, 64]
        img = torch.sum(img, dim=1, keepdims=True) # [1, 2500, 64, 64] -> [1, 1, 64, 64]
        
        img = 2*(img-0.5)               
        
        # Draw fixed points
        img[img_query.tile(batch_size,1,1,1) == 1] = 1
        
        return img

def runlat(model):
    opt = torch.optim.Adam(model.parameters(), lr=1, betas=(0, 0.9))
    scheduler = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1, end_factor=0.1, total_iters=1000)
    diffusion = GaussianDiffusion(T=1000, schedule='linear')
    # model.latent.data=temp

    steps = STEPS
    for i in range(steps):
        t = ((steps-i) + (steps-i)//3*math.cos(i/50))/steps*diffusion.T # Linearly decreasing + cosine

        t = np.clip(t, 1, diffusion.T)
        t = np.array([t for _ in range(batch_size)]).astype(int)

        # Denoise
        xt, epsilon = diffusion.sample(model.encode(), t) # get x_{ti} in Algorithm1 - (3 ~ 4)
        t = torch.from_numpy(t).float().view(batch_size)
        epsilon_pred = diffusion_net(xt.float(), t.to(device))

        loss = F.mse_loss(epsilon_pred, epsilon)

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        
        
STEPS=256

root_path = '/mnt/home/zuwang/workspace/diffusion_rl_tsp'
data_path = os.path.join(root_path, 'data')
input_path = os.path.join(data_path, FILE_NAME)
output_path = os.path.join(root_path, f'Results/{args.constraint_type}/{args.run_name}')
os.makedirs(output_path, exist_ok=True)

print('input_path : ', input_path)
print('output_path : ', output_path)

test_dataset = TSPDataset(data_file=input_path,
                          img_size=img_size,
                          point_radius=2, point_color=1, point_circle=True,
                          line_thickness=2, line_color=0.5,
                          constraint_type = args.constraint_type)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print('Created dataset')


diffusion_net = UNetModel(image_size=img_size, in_channels=1, out_channels=1, 
                          model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4),
                          attention_resolutions=[16,8], num_heads=4).to(device)

diffusion_net.load_state_dict(torch.load(f'models/unet50_64_8.pth', map_location=device))
diffusion_net.to(device)
diffusion_net.train()
print('Loaded model')


nn = torch.nn
results = []
nnn = 0

sample_idxes, basic_costs, gt_costs, penalty_counts = [], [], [], []

for batch in tqdm.tqdm(test_dataloader):
    nnn += 1
    img, _, _, sample_idx, _ = batch # [-1, 0, 1]로 이뤄진 GT image
    if not (args.start_idx <= int(sample_idx) < args.end_idx):
        continue
    _, points, gt_tour, constraint = test_dataset.rasterize(sample_idx[0].item())
    if args.constraint_type == 'box':
        distance_matrix, intersection_matrix = calculate_distance_matrix2(points, constraint)
        constraint=intersection_matrix
    tsp_solver = TSP_2opt(points)
    gt_cost = tsp_solver.evaluate([x-1 for x in gt_tour])

    img_query = torch.zeros_like(img)

    img_query[img == 1] = 1

    batch_idx=0
    
    model = InferenceModel().to(device)
    runlat(model)

    adj_mat = normalize((model.latent)).detach().cpu().numpy()[batch_idx] # model.latent : [1, 50, 50] -> adj_mat : (50, 50)
    adj_mat = adj_mat+adj_mat.T

    dists = np.zeros_like(adj_mat) # (50, 50)
    for i in range(dists.shape[0]):
        for j in range(dists.shape[0]):
            dists[i,j] = np.linalg.norm(points[i]-points[j])
            
    output = reward_fn(points, model.latent, dists, args.constraint_type, constraint = constraint)[1]
       
    basic_cost = output['basic_cost']
    penalty_count = output['penalty_count']
    
    sample_idxes.append(int(sample_idx))
    basic_costs.append(float(basic_cost))
    penalty_counts.append(int(penalty_count))
    gt_costs.append(float(gt_cost))
    
    if nnn%save_freq==0:
        result_df = pd.DataFrame({
        'sample_idx' : sample_idxes,
        'gt_cost' : gt_costs,
        'basic_cost' : basic_costs,
        'penalty_count' : penalty_counts,
        })
        result_df.to_csv(f'{output_path}/from{args.start_idx}_to{args.end_idx}.csv', index=False)
else:
    result_df = pd.DataFrame({
    'sample_idx' : sample_idxes,
    'gt_cost' : gt_costs,
    'basic_cost' : basic_costs,
    'penalty_count' : penalty_counts,
    })
    result_df.to_csv(f'{output_path}/from{args.start_idx}_to{args.end_idx}.csv', index=False)