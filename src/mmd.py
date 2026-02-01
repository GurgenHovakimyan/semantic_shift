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