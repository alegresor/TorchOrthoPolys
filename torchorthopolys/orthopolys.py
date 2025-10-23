import torch 
import numpy as np 
import scipy.special

class AbstractOrthoPolys(object):
    r"""
    Abstract class for [classic orthogonal polynomials](https://en.wikipedia.org/wiki/Classical_orthogonal_polynomials#Table_of_classical_orthogonal_polynomials). 
    """

    def __init__(
            self, 
            lower_limit, 
            upper_limit, 
    ):
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        if self.lower_limit is None: 
            self.lower_limit = -np.inf 
        if self.upper_limit is None: 
            self.upper_limit = np.inf
        self.lower_limit = float(self.lower_limit) 
        self.upper_limit = float(self.upper_limit)
    
    def __call__(self, n, x):
        r"""
        Evaluate polynomials. 

        Args:
            n (int): non-negative maximum degree of the polynomial.
            x (torch.Tensor): nodes at which to evaluate.

        Returns: 
            y (torch.Tensor): polynomial evaluations with shape `[n+1]+list(x.shape)`.
        """
        assert n>=0
        assert (x>=self.lower_limit).all()
        assert (x<=self.upper_limit).all()
        y = torch.empty([n+1]+list(x.shape))
        y[0] = self.c00
        if n==0: return y 
        y[1] = self.c11*x+self.c10
        if n==1: return y 
        t1,t2,t3 = self.recur_terms(n)
        for i in range(1,n):
            y[i+1] = (t1[i]*x+t2[i])*y[i]-t3[i]*y[i-1]
        return y
    
    def coeffs(self, n):
        r"""
        Evaluate coefficients. 

        Args:
            n (int): non-negative maximum degree of the polynomial.

        Returns: 
            c (torch.Tensor): coefficients with shape `[n+1,n+1]`.
        """
        assert n>=0 
        c = torch.zeros((n+1,n+1))
        c[0,0] = self.c00
        if n==0: return c
        c[1,0] = self.c10
        c[1,1] = self.c11
        if n==1: return c
        t1,t2,t3 = self.recur_terms(n)
        for i in range(1,n):
            c[i+1,:i] = -t3[i]*c[i-1,:i]
            c[i+1,:(i+1)] = c[i+1,:(i+1)]+t2[i]*c[i,:(i+1)]
            c[i+1,1:(i+2)] = c[i+1,1:(i+2)]+t1[i]*c[i,:(i+1)]
        return c
    
    def recur_terms(self, n):
        assert n>=0
        nrange = torch.arange(n+1)
        y = self._recur_terms(nrange)
        return y
    
    def lweight(self, x):
        r"""
        Log of the weight function. 

        Args:
            x (torch.Tensor): nodes at which to evaluate.

        Returns: 
            y (torch.Tensor): log-scaled weight evaluations with the same shape as `x`.
        """
        assert (x>=self.lower_limit).all()
        assert (x<=self.upper_limit).all()
        y = self._lweight(x)
        return y
    
    def lnorm(self, n):
        r"""
        Log of the normalization constants. 

        Args:
            n (int): non-negative maximum degree of the polynomial.

        Returns: 
            y (torch.Tensor): log-scaled normalization constants with shape `[n+1,]`.
        """
        assert n>=0
        nrange = torch.arange(n+1)
        y = self._lnorm(nrange)
        return y


class HermitePolys(AbstractOrthoPolys):

    r"""
    [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials)

    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(17)

        >>> p = HermitePolys()
        >>> n = 4
        >>> x = torch.rand((2,3),generator=rng)
        >>> y = p(n,x)
        >>> y.shape
        torch.Size([5, 2, 3])
        
        >>> torch.allclose(y[0],1+0*x)
        True
        >>> torch.allclose(y[1],2*x)
        True
        >>> torch.allclose(y[2],4*x**2-2)
        True
        >>> torch.allclose(y[3],8*x**3-12*x)
        True
        >>> torch.allclose(y[4],16*x**4-48*x**2+12)
        True

        >>> coeffs = p.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> xpows = x[...,None]**torch.arange(n+1)
        >>> xpows.shape
        torch.Size([2, 3, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,xpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 2, 3])
        >>> torch.allclose(y,yhat)
        True

        >>> HermitePolys().lnorm(3) 
        tensor([0.5724, 1.2655, 2.6518, 4.4436])

        >>> HermitePolys().lweight(torch.arange(-2,2)) 
        tensor([-4, -1,  0, -1])
    """

    def __init__(
            self,
    ):
        super().__init__(
            lower_limit = None, 
            upper_limit = None, 
        )
        self.c00 = 1
        self.c11 = 2 
        self.c10 = 0
    
    def _lnorm(self, nrange):
        return np.log(np.sqrt(np.pi))+nrange*np.log(2)+torch.lgamma(nrange+1)
    
    def _lweight(self, x):
        return -x**2
    
    def _recur_terms(self, nrange):
        t1 = 2+0*nrange
        t2 = 0*nrange
        t3 = 2*nrange
        return t1,t2,t3
    

class LaguerrePolys(AbstractOrthoPolys):

    r"""
    [Generalized Laguerre polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials)

    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(17)

        >>> alpha = np.pi 
        >>> p = LaguerrePolys(alpha=alpha)
        >>> n = 4
        >>> x = torch.rand((2,3),generator=rng)
        >>> y = p(n,x)
        >>> y.shape
        torch.Size([5, 2, 3])
        
        >>> torch.allclose(y[0],1+0*x)
        True
        >>> torch.allclose(y[1],-x+alpha+1)
        True
        >>> torch.allclose(y[2],1/2*(x**2-2*(alpha+2)*x+(alpha+1)*(alpha+2)))
        True
        >>> torch.allclose(y[3],1/6*(-x**3+3*(alpha+3)*x**2-3*(alpha+2)*(alpha+3)*x+(alpha+1)*(alpha+2)*(alpha+3)))
        True
        >>> torch.allclose(y[4],1/24*(x**4-4*(alpha+4)*x**3+6*(alpha+3)*(alpha+4)*x**2-4*(alpha+2)*(alpha+3)*(alpha+4)*x+(alpha+1)*(alpha+2)*(alpha+3)*(alpha+4)))
        True

        >>> coeffs = p.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> xpows = x[...,None]**torch.arange(n+1)
        >>> xpows.shape
        torch.Size([2, 3, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,xpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 2, 3])
        >>> torch.allclose(y,yhat)
        True

        >>> LaguerrePolys().lnorm(3) 
        tensor([0., 0., 0., 0.])
        >>> LaguerrePolys(alpha=np.pi).lnorm(3) 
        tensor([1.9724, 3.3935, 4.3377, 5.0542])

        >>> LaguerrePolys(alpha=np.pi).lweight(torch.arange(4)) 
        tensor([   -inf, -1.0000,  0.1776,  0.4514])
    """

    def __init__(
            self,
            alpha = 0,
    ):
        super().__init__(
            lower_limit = 0, 
            upper_limit = None, 
        )
        self.alpha = float(alpha) 
        assert self.alpha > -1
        self.c00 = 1
        self.c11 = -1 
        self.c10 = 1+self.alpha
    
    def _lnorm(self, nrange):
        return torch.lgamma(nrange+self.alpha+1) - torch.lgamma(nrange+1) 
    
    def _lweight(self, x):
        return self.alpha*torch.log(x)-x
    
    def _recur_terms(self, nrange):
        t1 = -1/(nrange+1)
        t2 = (2*nrange+1+self.alpha)/(nrange+1)
        t3 = (nrange+self.alpha)/(nrange+1)
        return t1,t2,t3


class JacobiPolys(AbstractOrthoPolys):

    r"""
    [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials)

    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(17)

        >>> alpha = -1/2 
        >>> beta = -3/4
        >>> p = JacobiPolys(alpha=alpha,beta=beta)
        >>> n = 4
        >>> x = torch.rand((2,3),generator=rng)
        >>> y = p(n,x)
        >>> y.shape
        torch.Size([5, 2, 3])
        
        >>> torch.allclose(y[0],1+0*x)
        True
        >>> torch.allclose(y[1],(alpha+1)+(alpha+beta+2)*(x-1)/2)
        True
        >>> torch.allclose(y[2],(alpha+1)*(alpha+2)/2+(alpha+2)*(alpha+beta+3)*(x-1)/2+(alpha+beta+3)*(alpha+beta+4)/2*((x-1)/2)**2)
        True

        >>> coeffs = p.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> xpows = x[...,None]**torch.arange(n+1)
        >>> xpows.shape
        torch.Size([2, 3, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,xpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 2, 3])
        >>> torch.allclose(y,yhat)
        True

        >>> torch.allclose(JacobiPolys(alpha=0,beta=0).lnorm(3),torch.log(2/(2*torch.arange(4)+1)))
        True
        >>> JacobiPolys(alpha=-1/2,beta=-3/4).lnorm(3)
        tensor([ 1.4838, -1.1552, -1.6942, -2.0527])

        >>> JacobiPolys(alpha=-1/2,beta=-3/4).lweight(torch.arange(-2,3)/2) 
        tensor([   inf, 0.3171, -0.0000, 0.0425,    inf])
    """
    
    def __init__(
            self,
            alpha = 0,
            beta = 0, 
    ):
        super().__init__(
            lower_limit = -1, 
            upper_limit = 1,
        )
        self.alpha = float(alpha) 
        self.beta = float(beta) 
        assert self.alpha > -1 
        assert self.beta > -1
        self.c00 = 1
        self.c11 = (self.alpha+self.beta+2)/2
        self.c10 = (self.alpha+1)-(self.alpha+self.beta+2)/2
    
    def _lnorm(self, nrange):
        t0 = (1+self.alpha+self.beta)*np.log(2)+scipy.special.gammaln(self.alpha+1)+scipy.special.gammaln(self.beta+1)-scipy.special.gammaln(self.alpha+self.beta+2)+np.log(scipy.special.betainc(1+self.alpha,1+self.beta,1/2)+scipy.special.betainc(1+self.beta,1+self.alpha,1/2))
        lognum = (self.alpha+self.beta+1)*np.log(2) + torch.lgamma(nrange[1:]+self.alpha+1)+torch.lgamma(nrange[1:]+self.beta+1)
        logdenom = torch.log(2*nrange[1:]+self.alpha+self.beta+1) + torch.lgamma(nrange[1:]+1) + torch.lgamma(nrange[1:]+self.alpha+self.beta+1)
        trest = lognum-logdenom
        return torch.hstack([t0*torch.ones(1),trest])
    
    def _lweight(self, x):
        return self.alpha*torch.log(1-x)+self.beta*torch.log(1+x)
    
    def _recur_terms(self, nrange):
        t1num = (2*nrange+1+self.alpha+self.beta)*(2*nrange+2+self.alpha+self.beta)
        t1denom = 2*(nrange+1)*(nrange+1+self.alpha+self.beta)
        t2num = (self.alpha**2-self.beta**2)*(2*nrange+1+self.alpha+self.beta)
        t2denom = 2*(nrange+1)*(2*nrange+self.alpha+self.beta)*(nrange+1+self.alpha+self.beta)
        t3num = (nrange+self.alpha)*(nrange+self.beta)*(2*nrange+2+self.alpha+self.beta)
        t3denom = (nrange+1)*(nrange+1+self.alpha+self.beta)*(2*nrange+self.alpha+self.beta)
        return t1num/t1denom,t2num/t2denom,t3num/t3denom
    
