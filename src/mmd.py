import torch

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes the Gaussian Kernel matrix between source and target.
    
    Args:
        source: Tensor of shape (B1, D)
        target: Tensor of shape (B2, D)
        kernel_mul: Multiplier for bandwidth variations
        kernel_num: Number of kernels to sum (multi-scale)
        fix_sigma: If provided, use fixed bandwidth
    
    Returns:
        kernel_val: The kernel matrix sum
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    # Compute L2 distance matrix (squared)
    # (x-y)^2 = x^2 + y^2 - 2xy
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    
    # Bandwidth heuristic (Median of distances)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    
    # Multi-scale kernels (essential for stability)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Calculates the Maximum Mean Discrepancy (MMD) between two distributions.
    
    Equation: MMD^2(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
    """
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    # Split kernel matrix into blocks: XX, YY, XY, YX
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    # Unbiased estimate of MMD (Gretton et al.)
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss


def pairwise_distances_chunked(X, Y, chunk_size=256):
    """
    Compute pairwise squared L2 distances between X and Y in chunks.
    
    Args:
        X: Tensor of shape (N, D)
        Y: Tensor of shape (M, D)
        chunk_size: Number of rows to process at once
        
    Returns:
        Distance matrix of shape (N, M)
    """
    N = X.shape[0]
    M = Y.shape[0]
    
    # Pre-compute squared norms
    X_sqnorm = (X ** 2).sum(dim=1)  # (N,)
    Y_sqnorm = (Y ** 2).sum(dim=1)  # (M,)
    
    # Initialize result
    distances = torch.zeros(N, M, dtype=X.dtype, device=X.device)
    
    # Compute in chunks to save memory
    for i in range(0, N, chunk_size):
        i_end = min(i + chunk_size, N)
        X_chunk = X[i:i_end]  # (chunk, D)
        
        # Compute distances for this chunk
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        XY = torch.mm(X_chunk, Y.t())  # (chunk, M)
        distances[i:i_end] = X_sqnorm[i:i_end].unsqueeze(1) + Y_sqnorm.unsqueeze(0) - 2 * XY
    
    return distances


def compute_bandwidth(source, target, chunk_size=256):
    """
    Compute bandwidth using chunked distance computation.
    """
    n1 = source.shape[0]
    n2 = target.shape[0]
    
    # Sample a subset for bandwidth estimation if too large
    max_samples = 1000
    if n1 > max_samples:
        idx = torch.randperm(n1)[:max_samples]
        source_sample = source[idx]
    else:
        source_sample = source
        
    if n2 > max_samples:
        idx = torch.randperm(n2)[:max_samples]
        target_sample = target[idx]
    else:
        target_sample = target
    
    # Compute distances on sample
    total = torch.cat([source_sample, target_sample], dim=0)
    n = total.shape[0]
    
    dist_sum = 0.0
    for i in range(0, n, chunk_size):
        i_end = min(i + chunk_size, n)
        chunk = total[i:i_end]
        dists = pairwise_distances_chunked(chunk, total, chunk_size)
        dist_sum += dists.sum().item()
    
    # Average (excluding diagonal)
    bandwidth = dist_sum / (n * n - n)
    return bandwidth


def mmd_rbf_chunked(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None, chunk_size=256):
    """
    Memory-efficient MMD computation using chunked kernel evaluation.
    
    This version computes the kernel matrix in chunks to avoid memory issues
    with large datasets.
    
    Args:
        source: Tensor of shape (N1, D) - first distribution samples
        target: Tensor of shape (N2, D) - second distribution samples
        kernel_mul: Multiplier for bandwidth variations
        kernel_num: Number of kernels to sum (multi-scale)
        fix_sigma: If provided, use fixed bandwidth
        chunk_size: Size of chunks for memory-efficient computation
        
    Returns:
        MMD^2 value (scalar tensor)
    """
    n1 = source.shape[0]
    n2 = target.shape[0]
    
    # Compute bandwidth
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = compute_bandwidth(source, target, chunk_size)
    
    # Multi-scale bandwidths
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    # Compute kernel expectations in chunks
    # E[k(x,x')]
    XX_sum = 0.0
    XX_count = 0
    for i in range(0, n1, chunk_size):
        i_end = min(i + chunk_size, n1)
        for j in range(0, n1, chunk_size):
            j_end = min(j + chunk_size, n1)
            
            dists = pairwise_distances_chunked(source[i:i_end], source[j:j_end], chunk_size)
            for bw in bandwidth_list:
                XX_sum += torch.exp(-dists / bw).sum().item()
            XX_count += (i_end - i) * (j_end - j)
    XX_mean = XX_sum / (XX_count * kernel_num)
    
    # E[k(y,y')]
    YY_sum = 0.0
    YY_count = 0
    for i in range(0, n2, chunk_size):
        i_end = min(i + chunk_size, n2)
        for j in range(0, n2, chunk_size):
            j_end = min(j + chunk_size, n2)
            
            dists = pairwise_distances_chunked(target[i:i_end], target[j:j_end], chunk_size)
            for bw in bandwidth_list:
                YY_sum += torch.exp(-dists / bw).sum().item()
            YY_count += (i_end - i) * (j_end - j)
    YY_mean = YY_sum / (YY_count * kernel_num)
    
    # E[k(x,y)]
    XY_sum = 0.0
    XY_count = 0
    for i in range(0, n1, chunk_size):
        i_end = min(i + chunk_size, n1)
        for j in range(0, n2, chunk_size):
            j_end = min(j + chunk_size, n2)
            
            dists = pairwise_distances_chunked(source[i:i_end], target[j:j_end], chunk_size)
            for bw in bandwidth_list:
                XY_sum += torch.exp(-dists / bw).sum().item()
            XY_count += (i_end - i) * (j_end - j)
    XY_mean = XY_sum / (XY_count * kernel_num)
    
    # MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    mmd_squared = XX_mean + YY_mean - 2 * XY_mean
    
    return torch.tensor(mmd_squared)