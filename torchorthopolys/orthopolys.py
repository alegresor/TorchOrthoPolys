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
        self.A = float(A)
        self.B = float(B)
        self.a = float(a) 
        self.b = float(b)
        self.factor_lweight = float(np.log(np.abs(self.A))+2*np.log(self.c00)-float(self.lnorm(0)))

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
        z = self.A*x+self.B
        lC = self.lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        y = torch.empty([n+1]+list(z.shape))
        y[0] = self.c00
        if n>0:
            y[1] = self.c11*z+self.c10
        if n>1: 
            t1,t2,t3 = self.recur_terms(n)
            for i in range(1,n):
                y[i+1] = (t1[i]*z+t2[i])*y[i]-t3[i]*y[i-1]
        return torch.einsum("i,i...->i...",v,y)
    
    def coeffs(self, n):
        r"""
        Evaluate coefficients. 

        Args:
            n (int): non-negative maximum degree of the polynomial.

        Returns: 
            c (torch.Tensor): coefficients with shape `[n+1,n+1]`.
        """
        assert n>=0 
        lC = self.lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        c = torch.zeros((n+1,n+1))
        c[0,0] = self.c00
        if n>0:
            c[1,0] = self.c10
            c[1,1] = self.c11
        if n>1:
            t1,t2,t3 = self.recur_terms(n)
            for i in range(1,n):
                c[i+1,:i] = -t3[i]*c[i-1,:i]
                c[i+1,:(i+1)] = c[i+1,:(i+1)]+t2[i]*c[i,:(i+1)]
                c[i+1,1:(i+2)] = c[i+1,1:(i+2)]+t1[i]*c[i,:(i+1)]
        return v[:,None]*c
    
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
        y = self.factor_lweight+self._lweight(self.A*x+self.B)
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

        >>> loc = -np.pi 
        >>> scale = np.sqrt(3) 
        >>> p = HermitePolys(loc=loc,scale=scale)

        >>> u = scipy.stats.qmc.Sobol(d=1,rng=7).random(2**16)[:,0]
        >>> x = torch.from_numpy(scipy.stats.norm.ppf(u,loc=loc,scale=scale))
        >>> n = 4
        
        >>> y = p(n,x)
        >>> y.shape
        torch.Size([5, 65536])
        >>> (y[:,None]*y[None,:]).mean(-1)
        tensor([[ 1.0000e+00,  5.7021e-07, -2.4570e-05,  1.2231e-05, -2.2798e-04],
                [ 5.7021e-07,  9.9997e-01,  2.1992e-05, -4.9851e-04,  1.5973e-04],
                [-2.4570e-05,  2.1992e-05,  9.9937e-01,  2.4418e-04, -4.3017e-03],
                [ 1.2231e-05, -4.9851e-04,  2.4418e-04,  9.9481e-01,  1.4077e-03],
                [-2.2798e-04,  1.5973e-04, -4.3017e-03,  1.4077e-03,  9.7405e-01]])
        
        >>> lrho = p.lweight(x) 
        >>> lrhohat = torch.from_numpy(scipy.stats.norm.logpdf(x.numpy(),loc=loc,scale=scale))
        >>> assert torch.allclose(lrho,lrhohat)

        >>> Cs = torch.exp(p.lnorm(n))
        >>> z = p.A*x+p.B
        >>> assert torch.allclose(p.c00*torch.sqrt(Cs[0]/Cs[0])*y[0],1+0*z)
        >>> assert torch.allclose(p.c00*torch.sqrt(Cs[1]/Cs[0])*y[1],2*z)
        >>> assert torch.allclose(p.c00*torch.sqrt(Cs[2]/Cs[0])*y[2],4*z**2-2)
        >>> assert torch.allclose(p.c00*torch.sqrt(Cs[3]/Cs[0])*y[3],8*z**3-12*z)
        >>> assert torch.allclose(p.c00*torch.sqrt(Cs[4]/Cs[0])*y[4],16*z**4-48*z**2+12)

        >>> coeffs = p.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> zpows = z[...,None]**torch.arange(n+1)
        >>> zpows.shape
        torch.Size([65536, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,zpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 65536])
        >>> torch.allclose(y,yhat)
        True
    """

    def __init__(
            self,
            loc = 0, 
            scale = 1/np.sqrt(2),
    ):
        self.c00 = 1
        self.c11 = 2 
        self.c10 = 0
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert scale>0
        loc = loc 
        scale = scale
        super().__init__(
            A = 1/(np.sqrt(2)*scale),
            B = -loc/(np.sqrt(2)*scale),
            a = -np.inf, 
            b = np.inf,
        )
    
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

        >>> alpha = -1/np.sqrt(3)
        >>> loc = -np.pi 
        >>> scale = -np.sqrt(3)
        >>> p = LaguerrePolys(alpha=alpha,loc=loc,scale=scale)

        >>> u = scipy.stats.qmc.Sobol(d=1,rng=7).random(2**16)[:,0]
        >>> x = -torch.from_numpy(scipy.stats.gamma.ppf(u,a=alpha+1,loc=-loc,scale=-scale))
        >>> n = 4

        >>> y = p(n,x)
        >>> y.shape
        torch.Size([5, 65536])
        >>> (y[:,None]*y[None,:]).mean(-1)
        tensor([[ 1.0000e+00,  1.1409e-05, -1.1488e-04,  4.2222e-04, -6.7890e-04],
                [ 1.1409e-05,  9.9967e-01,  2.4873e-03, -8.2370e-03,  1.2887e-02],
                [-1.1488e-04,  2.4873e-03,  9.8360e-01,  5.1614e-02, -8.0659e-02],
                [ 4.2222e-04, -8.2370e-03,  5.1614e-02,  8.3976e-01,  2.5730e-01],
                [-6.7890e-04,  1.2887e-02, -8.0659e-02,  2.5730e-01,  5.5508e-01]])
       
        >>> lrho = p.lweight(x) 
        >>> lrhohat = torch.from_numpy(scipy.stats.gamma.logpdf(-x.numpy(),a=alpha+1,loc=-loc,scale=-scale))
        >>> assert torch.allclose(lrho,lrhohat,atol=1e-3)

        >>> Cs = torch.exp(p.lnorm(n)) 
        >>> z = p.A*x+p.B
        >>> torch.allclose(p.c00*torch.sqrt(Cs[0]/Cs[0])*y[0],1+0*z)
        True
        >>> torch.allclose(p.c00*torch.sqrt(Cs[1]/Cs[0])*y[1],-z+alpha+1)
        True
        >>> ( (p.c00*torch.sqrt(Cs[2]/Cs[0])*y[2]) - (1/2*(z**2-2*(alpha+2)*x+(alpha+1)*(alpha+2))) ).abs().max()
        >>> torch.allclose(p.c00*torch.sqrt(Cs[2]/Cs[0])*y[2],1/2*(z**2-2*(alpha+2)*x+(alpha+1)*(alpha+2)),atol=1e-3)
        True
        >>> torch.allclose(p.c00*torch.sqrt(Cs[3]/Cs[0])*y[3],1/6*(-z**3+3*(alpha+3)*x**2-3*(alpha+2)*(alpha+3)*x+(alpha+1)*(alpha+2)*(alpha+3)),atol=1e-3)
        True
        >>> torch.allclose(p.c00*torch.sqrt(Cs[4]/Cs[0])*y[4],1/24*(z**4-4*(alpha+4)*x**3+6*(alpha+3)*(alpha+4)*x**2-4*(alpha+2)*(alpha+3)*(alpha+4)*x+(alpha+1)*(alpha+2)*(alpha+3)*(alpha+4)),atol=1e-3)
        True

        >>> coeffs = p.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> zpows = z[...,None]**torch.arange(n+1)
        >>> zpows.shape
        torch.Size([65536, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,zpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 65536])
        >>> torch.allclose(y,yhat)
        True
    """

    def __init__(
            self,
            alpha = 0,
            loc = 0, 
            scale = 1, 
    ):
        self.alpha = float(alpha) 
        assert self.alpha > -1
        self.c00 = 1
        self.c11 = -1 
        self.c10 = 1+self.alpha
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert scale!=0
        loc = loc
        scale = scale
        super().__init__(
            A = 1/scale,
            B = -loc/scale,
            a = loc if scale>0 else -np.inf, 
            b = np.inf if scale>0 else loc,
        )
    
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
        
        >>> Cs = torch.exp(p.lnorm(n))
        >>> torch.allclose(p.c00*torch.sqrt(Cs[0]/Cs[0])*y[0],1+0*x)
        True
        >>> torch.allclose(p.c00*torch.sqrt(Cs[1]/Cs[0])*y[1],(alpha+1)+(alpha+beta+2)*(x-1)/2)
        True
        >>> torch.allclose(p.c00*torch.sqrt(Cs[2]/Cs[0])*y[2],(alpha+1)*(alpha+2)/2+(alpha+2)*(alpha+beta+3)*(x-1)/2+(alpha+beta+3)*(alpha+beta+4)/2*((x-1)/2)**2)
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

        >>> alpha = -1/2
        >>> beta = -3/4 
        >>> loc = -np.pi 
        >>> scale = np.sqrt(3) 
        >>> p = JacobiPolys(alpha=alpha,beta=beta,loc=loc,scale=scale)
        >>> x = loc+scale*torch.rand((5,20),generator=rng)
        >>> lrho = p.lweight(x) 
        >>> lrho.shape
        torch.Size([5, 20])
        >>> lrhohat = torch.from_numpy(scipy.stats.beta.logpdf(x.numpy(),a=beta+1,b=alpha+1,loc=loc,scale=scale))
        >>> torch.allclose(lrho,lrhohat)
        True

        >>> u = scipy.stats.qmc.Sobol(d=1,rng=7).random(2**16)[:,0]
        >>> x = torch.from_numpy(scipy.stats.beta.ppf(u,a=beta+1,b=alpha+1,loc=loc,scale=scale))
        >>> y = p(4,x)
        >>> c = (y[:,None]*y[None,:]).mean(-1)
        >>> c.shape
        torch.Size([5, 5])
        >>> c
        tensor([[ 1.0000e+00, -1.3068e-09, -4.9899e-10, -1.1725e-09, -5.5519e-10],
                [-1.3068e-09,  1.0000e+00, -1.9337e-09, -1.2962e-09, -1.8712e-09],
                [-4.9899e-10, -1.9337e-09,  1.0000e+00, -1.8116e-09, -1.3395e-09],
                [-1.1725e-09, -1.2962e-09, -1.8116e-09,  1.0000e+00, -1.7744e-09],
                [-5.5519e-10, -1.8712e-09, -1.3395e-09, -1.7744e-09,  1.0000e+00]])
        
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
        self.alpha = float(alpha)
        self.beta = float(beta)
        assert self.alpha > -1 
        assert self.beta > -1
        self.c00 = 1
        self.c11 = (self.alpha+self.beta+2)/2
        self.c10 = (self.alpha+1)-(self.alpha+self.beta+2)/2
        assert np.isfinite(loc)
        assert np.isfinite(scale)
        assert scale>0
        loc = loc
        scale = scale
        super().__init__(
            A = 2/scale,
            B = -2*loc/scale-1,
            a = loc, 
            b = loc+scale,
        )
    
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
    
