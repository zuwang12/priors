import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
import scipy.spatial
import cv2
import sys

from unet import UNetModel
from TSPDataset import TSPDataset
from diffusion import GaussianDiffusion
from tsp_utils import TSP_2opt, rasterize_tsp

import tqdm
import matplotlib.pyplot as plt

STEPS=256

class TSPDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, img_size, point_radius=1, point_color=1, point_circle=True, line_thickness=2, line_color=0.5, max_points=100):
        self.data_file = data_file
        self.img_size = img_size
        self.point_radius = point_radius
        self.point_color = point_color
        self.point_circle = point_circle
        self.line_thickness = line_thickness
        self.line_color = line_color
        self.max_points = max_points
        
        self.file_lines = open(data_file).read().splitlines()
        print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')
        
    def __len__(self):
        return len(self.file_lines)
    
    def rasterize(self, idx):
        # Select sample
        line = self.file_lines[idx]
        # Clear leading/trailing characters
        line = line.strip()

        # Extract points
        points = line.split(' output ')[0]
        points = points.split(' ')
        points = np.array([[float(points[i]), float(points[i+1])] for i in range(0,len(points),2)])
        # Extract tour
        tour = line.split(' output ')[1]
        tour = tour.split(' ')
        tour = np.array([int(t) for t in tour])
        
        # Rasterize lines
        img = np.zeros((self.img_size, self.img_size))
        for i in range(tour.shape[0]-1):
            from_idx = tour[i]-1
            to_idx = tour[i+1]-1

            cv2.line(img, 
                     tuple(((img_size-1)*points[from_idx,::-1]).astype(int)), 
                     tuple(((img_size-1)*points[to_idx,::-1]).astype(int)), 
                     color=self.line_color, thickness=self.line_thickness)

        # Rasterize points
        for i in range(points.shape[0]):
            if self.point_circle:
                cv2.circle(img, tuple(((img_size-1)*points[i,::-1]).astype(int)), 
                           radius=self.point_radius, color=self.point_color, thickness=-1)
            else:
                row = round((img_size-1)*points[i,0])
                col = round((img_size-1)*points[i,1])
                img[row,col] = self.point_color
            
        # Rescale image to [-1,1]
        img = 2*(img-0.5)
            
        return img, points, tour

    def __getitem__(self, idx):
        img, points, tour = self.rasterize(idx)
            
        return img[np.newaxis,:,:], idx

device = torch.device(f'cuda:0')
batch_size = 1
img_size = 64

test_dataset = TSPDataset(data_file=f'data/tsp50_test_concorde.txt',
                          img_size=img_size,
                          point_radius=2, point_color=1, point_circle=True,
                          line_thickness=2, line_color=0.5)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
print('Created dataset')


diffusion_net = UNetModel(image_size=img_size, in_channels=1, out_channels=1, 
                          model_channels=64, num_res_blocks=2, channel_mult=(1,2,3,4),
                          attention_resolutions=[16,8], num_heads=4).to(device)

diffusion_net.load_state_dict(torch.load(f'models/unet50_64_8.pth'))
diffusion_net.to(device)
diffusion_net.train()
print('Loaded model')
                                         
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

nn = torch.nn
costs = []
nnn = 0
for batch in test_dataloader:
    nnn += 1
    img, sample_idx = batch # [-1, 0, 1]로 이뤄진 GT image

    _, points, gt_tour = test_dataset.rasterize(sample_idx[0].item())

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
    
    components = np.zeros((adj_mat.shape[0],2)).astype(int) # (50, 2)
    components[:] = np.arange(adj_mat.shape[0])[...,None] # (50, 1) | [[1], [2], ... , [49]]
    real_adj_mat = np.zeros_like(adj_mat) # (50, 50) 
    for edge in (-adj_mat/dists).flatten().argsort(): # [1715,  784, 1335, ..., 1326, 1224, 2499]) | 실제 거리(dists) 대비 adj_mat값이 가장 높은 순으로 iter
        a,b = edge//adj_mat.shape[0],edge%adj_mat.shape[0] # (34, 15)
        if not (a in components and b in components): continue
        ca = np.nonzero((components==a).sum(1))[0][0] # 34
        cb = np.nonzero((components==b).sum(1))[0][0] # 15
        if ca==cb: continue
        cca = sorted(components[ca],key=lambda x:x==a) # [34, 34]
        ccb = sorted(components[cb],key=lambda x:x==b) # [15, 15]
        newc = np.array([[cca[0],ccb[0]]]) # [34, 15]
        m,M = min(ca,cb),max(ca,cb) # (15, 34)
        real_adj_mat[a,b] = 1 # 연결됨
        components = np.concatenate([components[:m],components[m+1:M],components[M+1:],newc],0) # (49, 2)
        if len(components)==1: break
    real_adj_mat[components[0,1],components[0,0]] = 1 # 마지막 연결
    real_adj_mat += real_adj_mat.T # make symmetric matrix
    
    tour = [0]
    while len(tour)<adj_mat.shape[0]+1:
        n = np.nonzero(real_adj_mat[tour[-1]])[0]
        if len(tour)>1:
            n = n[n!=tour[-2]]
        tour.append(n.max())

    # Refine using 2-opt
    tsp_solver = TSP_2opt(points)
    solved_tour, ns = tsp_solver.solve_2opt(tour)

    def has_duplicates(l):
        existing = []
        for item in l:
            if item in existing:
                return True
            existing.append(item)
        return False

    assert solved_tour[-1] == solved_tour[0], 'Tour not a cycle'
    assert not has_duplicates(solved_tour[:-1]), 'Tour not Hamiltonian'

    gt_cost = tsp_solver.evaluate([i-1 for i in gt_tour])
    solved_cost = tsp_solver.evaluate(solved_tour)
    print(f'Ground truth cost: {gt_cost:.3f}')
    print(f'Predicted cost: {solved_cost:.3f} (Gap: {100*(solved_cost-gt_cost) / gt_cost:.4f}%)')
    costs.append((solved_cost, gt_cost, ns))
    if nnn % 1 == 0: 
        print((solved_cost-gt_cost)/gt_cost, sum(y[0] for y in costs)/sum(y[1] for y in costs)-1, ns)
print(costs)
print(sum(y[0] for y in costs), sum(y[1] for y in costs), sum(y[2] for y in costs)/len(costs))
