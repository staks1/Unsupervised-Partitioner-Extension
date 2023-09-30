import torch
import torch.nn as nn
import numpy as np
import h5py

'''
Memory-compatible. 
Ranks of closest points not self.
Uses L2 (Euclidean) or Mahalanobis distance. If data is normalized, it uses cosine distance. 
Mahalanobis distance considers the covariance between variables to provide a more accurate measure of distance.
It is especially useful when dealing with multivariate data and data where variables have relationships.

Input: 
    -data_x: data tensors from which we want to compute distances
    -data_y: data to search in (default is None, in which case data_x is used)
    -k: number of top results we are interested in
    -largest: whether to pick the largest when ranking (default is False)
    -opt: options which may contain flags for normalizing data and using sift (default is None)
    -include_self: include the point itself in the final ranking (default is False)
    -data: type of data (default is 'mnist')
    -distance: type of distance to use ('euclidean' or 'mahalanobis', default is 'euclidean')

dist_rank function from: https://github.com/twistedcubic/learn-to-hash
'''
def dist_rank(data_x, k, data_y=None, largest=False, opt=None, include_self=False, data='mnist', distance_metric='euclidean'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if isinstance(data_x, np.ndarray):
        data_x = torch.from_numpy(data_x)

    if data_y is None:
        data_y = data_x
    else:
        if isinstance(data_y, np.ndarray):
            data_y = torch.from_numpy(data_y)
    k0 = k
    device_o = data_x.device
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    
    (data_x_len, dim) = data_x.size()
    data_y_len = data_y.size(0)
    #break into chunks. 5e6  is total for MNIST point size
    #chunk_sz = int(5e6 // data_y_len)
    chunk_sz = 16384
    chunk_sz = 300 #700 mem error. 1 mil points
    if data_y_len > 990000:
        chunk_sz = 90 #50 if over 1.1 mil
        #chunk_sz = 500 #1000 if over 1.1 mil 
    else:
        chunk_sz = 300 
    if k+1 > len(data_y):
        k = len(data_y) - 1
    #if opt is not None and opt.sift:
    if device == 'cuda':
        dist_mx = torch.cuda.LongTensor(data_x_len, k+1)
    else:
        dist_mx = torch.LongTensor(data_x_len, k+1)
    data_normalized = True if opt is not None and opt.normalize_data else False
    largest = True if largest else (True if data_normalized else False)
    
    #compute l2 dist <--be memory efficient by blocking
    total_chunks = int((data_x_len-1) // chunk_sz) + 1
    #print('total chunks ', total_chunks)
    y_t = data_y.t()
    
    if not data_normalized:
        if distance_metric == 'mahalanobis':
           mean_y = torch.mean(data_y, dim=0).to(device)
           covariance_y = torch.cov(data_y.t()).to(device)

           # Check if the covariance matrix is singular
           if torch.det(covariance_y).item() == 0:
              regularisation = torch.eye(covariance_y.shape[0]).to(device) * 1e-6 
              covariance_y += regularisation

           inv_covariance_y = torch.inverse(covariance_y)
           del covariance_y
        elif distance_metric == 'euclidean':
           y_norm = (data_y**2).sum(-1).view(1, -1)
    del data_y

    print('total_chunks')
    print(total_chunks)
    
    for i in range(total_chunks):
        if i % 500 == 0:
           print(str(i) + '/' + str(total_chunks))
        
        base = i*chunk_sz
        upto = min((i+1)*chunk_sz, data_x_len)
        cur_len = upto-base
        x = data_x[base : upto]
       
        if not data_normalized:
           if distance_metric == 'mahalanobis':
              diff = x - mean_y.to(device)
              dist = torch.sqrt((diff @ inv_covariance_y) * diff)
           elif distance_metric == 'euclidean':
              torch.cuda.synchronize()
              x_norm = (x**2).sum(-1).view(-1, 1)        
              dist = x_norm + y_norm        
              dist -= 2 * torch.mm(x, y_t)
              del x_norm
        else:
            dist = torch.mm(x, y_t)
            
        topk = torch.topk(dist, k=k+1, dim=1, largest=largest)[1]
        dist_mx[base:upto, :k+1] = topk #torch.topk(dist, k=k+1, dim=1, largest=largest)[1][:, 1:]
        del dist
        del x

    topk = dist_mx
    if k > 3 and opt is not None and opt.sift:

        # Check for duplicate points
        # authors said that sift contains duplicate points, and to not run this in general.
        identity_ranks = torch.LongTensor(range(len(topk))).to(topk.device)
        topk_0 = topk[:, 0]
        topk_1 = topk[:, 1]
        topk_2 = topk[:, 2]
        topk_3 = topk[:, 3]

        id_idx1 = topk_1 == identity_ranks
        id_idx2 = topk_2 == identity_ranks
        id_idx3 = topk_3 == identity_ranks

        if torch.sum(id_idx1).item() > 0:
            topk[id_idx1, 1] = topk_0[id_idx1]

        if torch.sum(id_idx2).item() > 0:
            topk[id_idx2, 2] = topk_0[id_idx2]

        if torch.sum(id_idx3).item() > 0:
            topk[id_idx3, 3] = topk_0[id_idx3]           

    
    if not include_self:
        topk = topk[:, 1:]
    elif topk.size(-1) > k0:
        topk = topk[:, :-1]
    
    return topk.to(device_o)
