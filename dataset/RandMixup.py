import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

def randmixup(bags, labels, alpha=1.0, n_pseudo_bags=4, gamma=0.75):
    n_batch = len(bags)
    if n_batch == 1:
        return bags, labels, labels, 1.0

    lmbda = np.random.beta(alpha, alpha)
    idxs = torch.randperm(n_batch)
   
    bags = [torch.chunk(bag, n_pseudo_bags, 0) for bag in bags]
    lam_temp = lmbda if lmbda != 1.0 else lmbda - 1e-5
    lam_temp = int(lam_temp * (n_pseudo_bags + 1))
    # mixup pseudo-bags: phenotype-based cutmix
    mixed_bags = []
    mixed_ratios = []
    for i in range(n_batch):
        bag_a  = fetch_pseudo_bags(bags[i], n_pseudo_bags, lam_temp)
        bag_b  = fetch_pseudo_bags(bags[idxs[i]], n_pseudo_bags, n_pseudo_bags - lam_temp)
        if bag_a is None or len(bag_a) == 0:
            bag_ab = bag_b
        elif bag_b is None or len(bag_b) == 0:
            bag_ab = bag_a
        else:
            if np.random.rand() < gamma:
                bag_ab = torch.cat([bag_a, bag_b], dim=0) 
            else:
                bag_ab = bag_a
        mixed_bags.append(bag_ab)
        mixed_ratios.append(lmbda)
    
    labels_a, labels_b = labels, [labels[idxs[i]] for i in range(n_batch)]

    return mixed_bags, labels_a, labels_b, mixed_ratios

def fetch_pseudo_bags(X, n, n_parts):
    assert n_parts <= n, 'the pseudo-bag number to fetch is invalid.'
    if n_parts == 0:
        return None

    ind_fetched = torch.randperm(n)[:n_parts]
    X_fetched = torch.cat([X[ind] for ind in ind_fetched], dim=0)

    return X_fetched
