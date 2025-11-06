import torch 
import numpy as np 

def lcomb(n, k):
    return torch.lgamma(n+1)-torch.lgamma(k+1)-torch.lgamma(n-k+1)

def comb(n, k):
    """
    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> import scipy.special
        >>> n = torch.arange(5)
        >>> k = torch.arange(4) 
        >>> comb_torch = comb(n[:,None],k[None,:])
        >>> comb_torch
        tensor([[1.0000, 0.0000, 0.0000, 0.0000],
                [1.0000, 1.0000, 0.0000, 0.0000],
                [1.0000, 2.0000, 1.0000, 0.0000],
                [1.0000, 3.0000, 3.0000, 1.0000],
                [1.0000, 4.0000, 6.0000, 4.0000]])
        >>> comb_sp = scipy.special.comb(n[:,None].numpy(),k[None,:].numpy())
        >>> np.allclose(comb_sp,comb_torch.numpy())
        True
        >>> n = torch.tensor([torch.pi])
        >>> k = torch.tensor([torch.e])
        >>> np.allclose(comb(n,k),scipy.special.comb(n.numpy(),k.numpy()))
        True
        >>> np.allclose(comb(k,n),scipy.special.comb(k.numpy(),n.numpy()))
        True
    """
    v = torch.exp(lcomb(n,k))
    v[n<k] = 0
    return v

