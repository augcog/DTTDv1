import torch
import numpy as np
from knn_cuda import KNN

def randla_processing(end_points, cfg):
    rndla_config = cfg.rndla_cfg
    n_ds_layers = rndla_config.num_layers
    pcld_sub_s_r = rndla_config.sub_sampling_ratio

    knn_local = KNN(k=rndla_config.k_n, transpose_mode=True)
    knn_interp = KNN(k=1, transpose_mode=True)

    cld = end_points["cloud"].clone().detach()

    #RANDLA NET STUFF
    # DownSample stage
    for i in range(n_ds_layers):
        _, nei_idx = knn_local(cld, cld)

        sub_pts = cld[:,::pcld_sub_s_r[i], :]

        pool_i = nei_idx[:,::pcld_sub_s_r[i], :]
        _, up_i = knn_interp(sub_pts, cld)

        end_points['RLA_xyz_%d'%i] = cld.clone().detach()
        end_points['RLA_neigh_idx_%d'%i] = nei_idx.clone().detach()
        end_points['RLA_sub_idx_%d'%i] = pool_i.clone().detach()
        end_points['RLA_interp_idx_%d'%i] = up_i.clone().detach()
        
        cld = sub_pts

    return end_points