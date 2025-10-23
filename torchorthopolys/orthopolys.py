import torch 
import numpy as np 
import scipy.special

class AbstractOrthoPolys(object):
    r"""
    Abstract class for [classic orthogonal polynomials](https://en.wikipedia.org/wiki/Classical_orthogonal_polynomials#Table_of_classical_orthogonal_polynomials). 
    """

    def __init__(
            self, 
            A, 
            B,
            a, 
            b,
    ):
        self.A = A
        self.B = B
        self.a = a 
        self.b = b 
    
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
        assert (x>=self.a).all()
        assert (x<=self.b).all()
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
        assert (x>=self.a).all()
        assert (x<=self.b).all()
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
    Orthonormal [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials)
    supported on $(-\infty,\infty)$ with the weight normalized to be a density function. 

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

        >>> p = HermitePolys(loc=np.pi,scale=np.exp(1))
        >>> p.a,p.b
        (-inf, inf)
        >>> p.A,p.B
        (0.36787944117144233, -1.1557273497909217)
    """

    def __init__(
            self,
            loc = 0, 
            scale = 1,
    ):
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert scale!=0
        loc = float(loc) 
        scale = float(scale)
        super().__init__(
            A = 1/scale,
            B = -loc/scale,
            a = -np.inf, 
            b = np.inf,
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
    Orthonormal [Generalized Laguerre polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials)
    supported on $[a,\infty)$ or $(-\infty,a]$ with the weight normalized to be a density function. 

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

        >>> p = LaguerrePolys(alpha=1/3,loc=np.pi,scale=np.exp(1))
        >>> p.a,p.b
        (3.141592653589793, inf)
        >>> p.A*p.a+p.B,p.A*p.b+p.B
        (0.0, inf)
        >>> p = LaguerrePolys(alpha=1/3,loc=np.pi,scale=-np.exp(1))
        >>> p.a,p.b
        (-inf, 3.141592653589793)
        >>> p.A*p.b+p.B,p.A*p.a+p.B
        (0.0, inf)
    """

    def __init__(
            self,
            alpha = 0,
            loc = 0, 
            scale = 1, 
    ):
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert scale!=0
        loc = float(loc) 
        scale = float(scale)
        super().__init__(
            A = 1/scale,
            B = -loc/scale,
            a = loc if scale>0 else -np.inf, 
            b = np.inf if scale>0 else loc,
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
    Orthonormal [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials) 
    supported on $[a,b]$ with the weight normalized to be a density function. 

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

        >>> p = JacobiPolys(alpha=-1/2,beta=-3/4,loc=np.pi,scale=np.exp(1))
        >>> p.a,p.b
        (3.141592653589793, 5.859874482048838)
        >>> p.A*p.a+p.B,p.A*p.b+p.B
        (-1.0, 1.0)
    """
    
    def __init__(
            self,
            alpha = 0,
            beta = 0,
            loc = -1, 
            scale = 2,
    ):
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert scale>0
        loc = float(loc) 
        scale = float(scale)
        super().__init__(
            A = 2/scale,
            B = -2*loc/scale-1,
            a = loc, 
            b = loc+scale,
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
    
