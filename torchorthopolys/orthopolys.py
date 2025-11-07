import torch 
import numpy as np 
import scipy.special
from .util import comb 

class AbstractOrthoPolys(object):
    r"""
    Abstract class for [classic orthogonal polynomials](https://en.wikipedia.org/wiki/Classical_orthogonal_polynomials#Table_of_classical_orthogonal_polynomials). 
    """

    def __init__(self, scale_tilde=1, shift_tilde=0):
        self.factor_lweight = float(2*np.log(self.c00)-float(self._lnorm(0)))
        assert np.isfinite(scale_tilde)
        assert np.isfinite(shift_tilde)
        assert scale_tilde>0
        self.scale_tilde = scale_tilde
        self.shift_tilde = shift_tilde
        self.a = self.atilde*self.scale_tilde+self.shift_tilde
        self.b = self.btilde*self.scale_tilde+self.shift_tilde
        self.scale = 1/self.scale_tilde
        self.shift = -self.shift_tilde/self.scale_tilde
        assert np.allclose(self.atilde,self.a*self.scale+self.shift)
        assert np.allclose(self.btilde,self.b*self.scale+self.shift)
        self.logscale = np.log(self.scale)

    def __call__(self, n, x):
        r"""
        Evaluate polynomials. 

        Args:
            n (int): non-negative maximum degree of the polynomial.
            x (torch.Tensor): nodes at which to evaluate.

        Returns: 
            y (torch.Tensor): polynomial evaluations with shape `[n+1]+list(x.shape)`.
        """
        y = self._eval_unnormalized(n,x)
        lC = self._lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        return torch.einsum("i,i...->i...",v,y)
    
    def _eval_unnormalized(self, n, x):
        assert n>=0
        assert (x>=self.a).all()
        assert (x<=self.b).all()
        xt = self.scale*x+self.shift
        y = torch.empty([n+1]+list(xt.shape))
        y[0] = self.c00
        if n>0:
            y[1] = self.c11*xt+self.c10
        if n>1: 
            t1,t2,t3 = self._recur_terms(n)
            for i in range(1,n):
                y[i+1] = (t1[i]*xt+t2[i])*y[i]-t3[i]*y[i-1]
        return y
    
    def _coeffs_unnormalized(self, n):
        assert n>=0 
        lC = self._lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        c = torch.zeros((n+1,n+1))
        c[0,0] = self.c00
        if n>0:
            c[1,0] = self.c10
            c[1,1] = self.c11
        if n>1:
            t1,t2,t3 = self._recur_terms(n)
            for i in range(1,n):
                c[i+1,:i] = -t3[i]*c[i-1,:i]
                c[i+1,:(i+1)] = c[i+1,:(i+1)]+t2[i]*c[i,:(i+1)]
                c[i+1,1:(i+2)] = c[i+1,1:(i+2)]+t1[i]*c[i,:(i+1)]
        return v[:,None]*c
    
    def coeffs(self, n):
        r"""
        Evaluate coefficients. 

        Args:
            n (int): non-negative maximum degree of the polynomial.

        Returns: 
            c (torch.Tensor): coefficients with shape `[n+1,n+1]`.
        """
        C = self._coeffs_unnormalized(n)
        nrange = torch.arange(n+1)
        Apows = self.scale**nrange
        S = comb(nrange[:,None],nrange[None,:])
        Bpows = self.shift**torch.maximum(nrange[:,None]-nrange[None,:],torch.zeros(1))
        Cnew = torch.einsum("ij,jk,jk,k->ik",C,S,Bpows,Apows)
        return Cnew
    
    def _recur_terms(self, n):
        assert n>=0
        nrange = torch.arange(n+1)
        y = self._recur_terms_(nrange)
        return y
    
    def deriv(self, n, x):
        r"""
        Evaluate first derivative of polynomials. 

        Args:
            n (int): non-negative maximum degree of the polynomial.
            x (torch.Tensor): nodes at which to evaluate.

        Returns: 
            y (torch.Tensor): polynomial evaluations with shape `[n+1]+list(x.shape)`.
        """
        raise Exception("deriv not implemented by child class")
    
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
        y = self.logscale+self._lweight(x)
        return y
    
    def weight(self, x):
        r"""
        The weight function. 

        Args:
            x (torch.Tensor): nodes at which to evaluate.

        Returns: 
            y (torch.Tensor): weight evaluations with the same shape as `x`.
        """
        return torch.exp(self.lweight(x))
    
    def _lnorm(self, n):
        r"""
        Log of the normalization constants. 

        Args:
            n (int): non-negative maximum degree of the polynomial.

        Returns: 
            y (torch.Tensor): log-scaled normalization constants with shape `[n+1,]`.
        """
        assert n>=0
        nrange = torch.arange(n+1)
        y = self._lnorm_(nrange)
        return y
    
    def _lam_tilde(self, n):
        nrange = torch.arange(n+1) 
        return nrange*self._tau_tilde_1+nrange*(nrange-1)*self._sigma_tilde_2
    
    def lam(self, n):
        return self.scale**2*self._lam_tilde(n) 

    def _sigma_tilde(self, xt):
        return self._sigma_tilde_0+self._sigma_tilde_1*xt+self._sigma_tilde_2*xt**2
    
    def sigma(self, x):
        return self._sigma_tilde(self.scale*x+self.shift)
    
    def _tau_tilde(self, xt):
        return self._tau_tilde_0+self._tau_tilde_1*xt
    
    def tau(self, x):
        return self._tau_tilde(self.scale*x+self.shift)
    
    def integral(self, n, x):
        r"""
        Integral of the polynomials times the weight function from `self.a` to `x`
        
        Args:
            n (int): non-negative maximum degree of the polynomial.
        
        Returns:
            y (torch.Tensor): integral values `[n+1]+list(x.shape)`.
        """
        assert n>=0
        lam = self.lam(n=n)
        sigma = self.sigma(x=x) 
        dphi = self.deriv(n=n,x=x)
        w = self.weight(x)
        y = torch.einsum("i,i...->i...",1/lam,sigma*dphi*w)
        y[0] = self._cdf(x)
        return y

class Hermite(AbstractOrthoPolys):

    r"""
    Orthonormal [Hermite polynomials](https://en.wikipedia.org/wiki/Hermite_polynomials)
    supported on $(-\infty,\infty)$ with the weight normalized to be a density function. 

    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(17)

        >>> loc = -np.pi
        >>> scale = np.exp(1)
        >>> poly = Hermite(loc=loc,scale=scale)

        >>> u = scipy.stats.qmc.Sobol(d=1,rng=7).random(2**16)[:,0]
        >>> x = torch.from_numpy(scipy.stats.norm.ppf(u,loc=loc,scale=scale))
        >>> n = 4
        
        >>> y = poly(n,x)
        >>> y.shape
        torch.Size([5, 65536])
        >>> (y[:,None]*y[None,:]).mean(-1)
        tensor([[ 1.0000e+00,  5.7021e-07, -2.4570e-05,  1.2231e-05, -2.2798e-04],
                [ 5.7021e-07,  9.9997e-01,  2.1992e-05, -4.9851e-04,  1.5973e-04],
                [-2.4570e-05,  2.1992e-05,  9.9937e-01,  2.4418e-04, -4.3017e-03],
                [ 1.2231e-05, -4.9851e-04,  2.4418e-04,  9.9481e-01,  1.4077e-03],
                [-2.2798e-04,  1.5973e-04, -4.3017e-03,  1.4077e-03,  9.7405e-01]])
        
        >>> lrho = poly.lweight(x) 
        >>> lrhohat = torch.from_numpy(scipy.stats.norm.logpdf(x.numpy(),loc=loc,scale=scale))
        >>> assert torch.allclose(lrho,lrhohat)

        >>> Cs = torch.exp(poly._lnorm(n))
        >>> xt = poly.scale*x+poly.shift
        >>> assert torch.allclose(y[0],torch.sqrt(Cs[0]/Cs[0])/poly.c00*(1+0*xt))
        >>> assert torch.allclose(y[1],torch.sqrt(Cs[0]/Cs[1])/poly.c00*(2*xt))
        >>> assert torch.allclose(y[2],torch.sqrt(Cs[0]/Cs[2])/poly.c00*(4*xt**2-2))
        >>> assert torch.allclose(y[3],torch.sqrt(Cs[0]/Cs[3])/poly.c00*(8*xt**3-12*xt))
        >>> assert torch.allclose(y[4],torch.sqrt(Cs[0]/Cs[4])/poly.c00*(16*xt**4-48*xt**2+12))

        >>> coeffs = poly.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> coeffs
        tensor([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 1.1557,  0.3679,  0.0000,  0.0000,  0.0000],
                [ 0.2374,  0.6013,  0.0957,  0.0000,  0.0000],
                [-0.7853,  0.1513,  0.1916,  0.0203,  0.0000],
                [-0.6593, -0.5778,  0.0556,  0.0470,  0.0037]])
        >>> xpows = x[...,None]**torch.arange(n+1)
        >>> xpows.shape
        torch.Size([65536, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,xpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 65536])
        >>> assert torch.allclose(y,yhat)

        >>> yp = poly.deriv(n,x) 
        >>> yp.shape
        torch.Size([5, 65536])
        >>> xpowsm1 = x[...,None]**torch.arange(-1,n)
        >>> xpowsm1.shape
        torch.Size([65536, 5])
        >>> yphat = torch.einsum("ij,...j->i...",coeffs*torch.arange(n+1),xpowsm1) # generally unstable
        >>> yphat.shape
        torch.Size([5, 65536])
        >>> assert torch.allclose(yphat,yp)

        >>> x = torch.linspace(loc-2*scale,loc+2*scale,6)
        >>> n = 8
        >>> v = poly.integral(n,x)
        >>> v.shape
        torch.Size([9, 6])
        >>> vhat = torch.ones_like(v)
        >>> for i in range(len(x)):
        ...     ttrap = torch.linspace(loc-5*scale,x[i],100001)
        ...     ytrap = poly(n=n,x=ttrap)*poly.weight(ttrap)
        ...     vhat[:,i] = torch.trapezoid(ytrap,ttrap)
        >>> assert torch.allclose(vhat,v,atol=1e-3)
    """

    def __init__(self, loc=0, scale=1/np.sqrt(2)):
        r"""
        Args:
            loc (float): weight distribution will be `scipy.stats.norm(loc=loc,scale=scale)`
            scale (float): weight distribution will be `scipy.stats.norm(loc=loc,scale=scale)`
        """
        self.c00 = 1
        self.c11 = 2 
        self.c10 = 0
        self.atilde = float(-np.inf) 
        self.btilde = float(np.inf) 
        self.distrib = torch.distributions.Normal(loc=loc,scale=scale)
        self._sigma_tilde_0 = 1
        self._sigma_tilde_1 = 0 
        self._sigma_tilde_2 = 0
        self._tau_tilde_0 = 0 
        self._tau_tilde_1 = -2
        super().__init__(scale_tilde=np.sqrt(2)*scale,shift_tilde=loc)
    
    def _cdf(self, x):
        return self.distrib.cdf(x)
    
    def _lnorm_(self, nrange):
        return np.log(np.sqrt(np.pi))+nrange*np.log(2)+torch.lgamma(nrange+1)
    
    def _lweight(self, x):
        return self.distrib.log_prob(x)-self.logscale
    
    def _recur_terms_(self, nrange):
        t1 = 2+0*nrange
        t2 = 0*nrange
        t3 = 2*nrange
        return t1,t2,t3
    
    def deriv(self, n, x):
        y = self._eval_unnormalized(n,x)
        lC = self._lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        nrange = torch.arange(1,n+1)
        yp = torch.zeros_like(y)
        yp[1:] = torch.einsum("i,i...->i...",2*torch.exp(torch.lgamma(nrange+1)-torch.lgamma(nrange)+self.logscale)*v[1:],y[:-1])
        return yp
    

class Laguerre(AbstractOrthoPolys):

    r"""
    Orthonormal [Generalized Laguerre polynomials](https://en.wikipedia.org/wiki/Laguerre_polynomials#Generalized_Laguerre_polynomials)
    supported on $[0,\infty)$ with the weight normalized to be a density function. 

    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(17)

        >>> loc = -np.pi
        >>> scale = np.exp(1)
        >>> alpha = -1/np.sqrt(3)
        >>> poly = Laguerre(alpha=alpha,loc=loc,scale=scale)

        >>> u = scipy.stats.qmc.Sobol(d=1,rng=7).random(2**16)[:,0]
        >>> x = torch.from_numpy(scipy.stats.gamma.ppf(u,a=alpha+1,loc=loc,scale=scale))
        >>> n = 4

        >>> y = poly(n,x)
        >>> y.shape
        torch.Size([5, 65536])
        >>> (y[:,None]*y[None,:]).mean(-1)
        tensor([[ 1.0000e+00,  1.1409e-05, -1.1488e-04,  4.2222e-04, -6.7890e-04],
                [ 1.1409e-05,  9.9967e-01,  2.4873e-03, -8.2370e-03,  1.2887e-02],
                [-1.1488e-04,  2.4873e-03,  9.8360e-01,  5.1614e-02, -8.0659e-02],
                [ 4.2222e-04, -8.2370e-03,  5.1614e-02,  8.3976e-01,  2.5730e-01],
                [-6.7890e-04,  1.2887e-02, -8.0659e-02,  2.5730e-01,  5.5508e-01]])
       
        >>> lrho = poly.lweight(x) 
        >>> lrhohat = torch.from_numpy(scipy.stats.gamma.logpdf(x.numpy(),a=alpha+1,loc=loc,scale=scale))
        >>> assert torch.allclose(lrho,lrhohat,atol=1e-3)

        >>> Cs = torch.exp(poly._lnorm(n))
        >>> xt = poly.scale*x+poly.shift
        >>> assert torch.allclose(y[0],torch.sqrt(Cs[0]/Cs[0])/poly.c00*(1+0*xt))
        >>> assert torch.allclose(y[1],torch.sqrt(Cs[0]/Cs[1])/poly.c00*(-xt+alpha+1))
        >>> assert torch.allclose(y[2],torch.sqrt(Cs[0]/Cs[2])/poly.c00*(1/2*(xt**2-2*(alpha+2)*xt+(alpha+1)*(alpha+2))))
        >>> assert torch.allclose(y[3],torch.sqrt(Cs[0]/Cs[3])/poly.c00*(1/6*(-xt**3+3*(alpha+3)*xt**2-3*(alpha+2)*(alpha+3)*xt+(alpha+1)*(alpha+2)*(alpha+3))))
        >>> assert torch.allclose(y[4],torch.sqrt(Cs[0]/Cs[4])/poly.c00*(1/24*(xt**4-4*(alpha+4)*xt**3+6*(alpha+3)*(alpha+4)*xt**2-4*(alpha+2)*(alpha+3)*(alpha+4)*xt+(alpha+1)*(alpha+2)*(alpha+3)*(alpha+4))))

        >>> coeffs = poly.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> coeffs
        tensor([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [-1.1276, -0.5659,  0.0000,  0.0000,  0.0000],
                [-1.2323, -0.1791,  0.1234,  0.0000,  0.0000],
                [-0.7878,  0.3052,  0.1740, -0.0168,  0.0000],
                [-0.2235,  0.6433,  0.1274, -0.0413,  0.0017]])
        >>> xpows = x[...,None]**torch.arange(n+1)
        >>> xpows.shape
        torch.Size([65536, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,xpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 65536])
        >>> assert torch.allclose(y,yhat)

        >>> yp = poly.deriv(n,x) 
        >>> yp.shape
        torch.Size([5, 65536])
        >>> xpowsm1 = x[...,None]**torch.arange(-1,n)
        >>> xpowsm1.shape
        torch.Size([65536, 5])
        >>> yphat = torch.einsum("ij,...j->i...",coeffs*torch.arange(n+1),xpowsm1) # generally unstable
        >>> yphat.shape
        torch.Size([5, 65536])
        >>> assert torch.allclose(yphat,yp)

        >>> x = torch.linspace(poly.a,10,7)[1:]
        >>> n = 8
        >>> v = poly.integral(n,x)
        >>> v.shape
        torch.Size([9, 6])
        >>> vhat = torch.ones_like(v)
        >>> for i in range(len(x)):
        ...     ttrap = torch.linspace(poly.a,x[i],100001)[1:]
        ...     ytrap = poly(n=n,x=ttrap)*poly.weight(ttrap)
        ...     vhat[:,i] = torch.trapezoid(ytrap,ttrap)
        >>> assert torch.allclose(vhat,v,atol=2.5e-2)
    """

    def __init__(self, alpha=0, loc=0, scale=1):
        r"""
        Args:
            alpha (float): parameter $\alpha>-1$.
            loc (float): weight distribution will be `scipy.stats.gamma(a=alpha+1,loc=loc,scale=scale)`
            scale (float): weight distribution will be `scipy.stats.gamma(a=alpha+1,loc=loc,scale=scale)`
        """
        self.alpha = float(alpha) 
        assert self.alpha > -1
        self.c00 = 1
        self.c11 = -1 
        self.c10 = 1+self.alpha
        self.atilde = float(0) 
        self.btilde = float(np.inf)
        self.distrib = torch.distributions.Gamma(concentration=self.alpha+1,rate=1)
        self._sigma_tilde_0 = 0
        self._sigma_tilde_1 = 1 
        self._sigma_tilde_2 = 0
        self._tau_tilde_0 = self.alpha+1 
        self._tau_tilde_1 = -1
        super().__init__(scale_tilde=scale,shift_tilde=loc)
    
    def _cdf(self, x):
        return self.distrib.cdf(self.scale*x+self.shift)
    
    def _lnorm_(self, nrange):
        return torch.lgamma(nrange+self.alpha+1)-torch.lgamma(nrange+1) 
    
    def _lweight(self, x):
        xt = self.scale*x+self.shift
        return self.distrib.log_prob(xt)
    
    def _recur_terms_(self, nrange):
        t1 = -1/(nrange+1)
        t2 = (2*nrange+1+self.alpha)/(nrange+1)
        t3 = (nrange+self.alpha)/(nrange+1)
        return t1,t2,t3
    
    def deriv(self, n, x):
        self.alpha += 1
        self.c10 += 1
        y = self._eval_unnormalized(n=n,x=x)
        self.alpha -= 1
        self.c10 -= 1
        lC = self._lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        yp = torch.zeros_like(y)
        yp[1:] = -torch.einsum("i,i...->i...",self.scale*v[1:],y[:-1])
        return yp


class Jacobi(AbstractOrthoPolys):

    r"""
    Orthonormal [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials) 
    supported on $[-1,1]$ with the weight normalized to be a density function. 

    Examples:
        >>> torch.set_default_dtype(torch.float64)
        >>> rng = torch.Generator().manual_seed(17)

        >>> loc = -np.pi
        >>> scale = np.exp(1)
        >>> alpha = 1/2
        >>> beta = 3/4 
        >>> poly = Jacobi(alpha=alpha,beta=beta,loc=loc,scale=scale)

        >>> u = scipy.stats.qmc.Sobol(d=1,rng=7).random(2**16)[:,0]
        >>> x = torch.from_numpy(scipy.stats.beta.ppf(u,a=beta+1,b=alpha+1,loc=loc,scale=scale))
        >>> n = 4
        
        >>> y = poly(n,x)
        >>> y.shape
        torch.Size([5, 65536])
        >>> (y[:,None]*y[None,:]).mean(-1)
        tensor([[ 1.0000e+00,  1.4714e-08, -1.1409e-07,  1.9097e-07, -6.1552e-07],
                [ 1.4714e-08,  1.0000e+00,  2.2747e-07, -7.7976e-07,  1.0676e-06],
                [-1.1409e-07,  2.2747e-07,  1.0000e+00,  1.1397e-06, -2.8870e-06],
                [ 1.9097e-07, -7.7976e-07,  1.1397e-06,  1.0000e+00,  3.6799e-06],
                [-6.1552e-07,  1.0676e-06, -2.8870e-06,  3.6799e-06,  9.9999e-01]])

        >>> lrho = poly.lweight(x) 
        >>> lrhohat = torch.from_numpy(scipy.stats.beta.logpdf(x.numpy(),a=beta+1,b=alpha+1,loc=loc,scale=scale))
        >>> assert torch.allclose(lrho,lrhohat,1e-3)
        
        >>> Cs = torch.exp(poly._lnorm(n))
        >>> xt = poly.scale*x+poly.shift
        >>> assert torch.allclose(y[0],torch.sqrt(Cs[0]/Cs[0])/poly.c00*(1+0*xt))
        >>> assert torch.allclose(y[1],torch.sqrt(Cs[0]/Cs[1])/poly.c00*((alpha+1)+(alpha+beta+2)*(xt-1)/2))
        >>> assert torch.allclose(y[2],torch.sqrt(Cs[0]/Cs[2])/poly.c00*((alpha+1)*(alpha+2)/2+(alpha+2)*(alpha+beta+3)*(xt-1)/2+(alpha+beta+3)*(alpha+beta+4)/2*((xt-1)/2)**2))

        >>> coeffs = poly.coeffs(n)
        >>> coeffs.shape
        torch.Size([5, 5])
        >>> coeffs
        tensor([[ 1.0000,  0.0000,  0.0000,  0.0000,  0.0000],
                [ 2.5526,  1.5213,  0.0000,  0.0000,  0.0000],
                [ 5.7016,  7.7823,  2.2653,  0.0000,  0.0000],
                [12.4138, 27.3369, 17.4622,  3.3538,  0.0000],
                [26.8769, 82.2834, 83.9242, 34.5890,  4.9534]])
        >>> xpows = x[...,None]**torch.arange(n+1)
        >>> xpows.shape
        torch.Size([65536, 5])
        >>> yhat = torch.einsum("ij,...j->i...",coeffs,xpows) # generally unstable
        >>> yhat.shape
        torch.Size([5, 65536])
        >>> assert torch.allclose(y,yhat)
        
        >>> yp = poly.deriv(n,x) 
        >>> yp.shape
        torch.Size([5, 65536])
        >>> xpowsm1 = x[...,None]**torch.arange(-1,n)
        >>> xpowsm1.shape
        torch.Size([65536, 5])
        >>> yphat = torch.einsum("ij,...j->i...",coeffs*torch.arange(n+1),xpowsm1) # generally unstable
        >>> yphat.shape
        torch.Size([5, 65536])
        >>> assert torch.allclose(yphat,yp)

        >>> x = torch.linspace(poly.a,poly.b,6)
        >>> n = 8
        >>> v = poly.integral(n,x)
        >>> v.shape
        torch.Size([9, 6])
        >>> vhat = torch.ones_like(v)
        >>> for i in range(len(x)):
        ...     ttrap = torch.linspace(poly.a,x[i],100000)
        ...     ytrap = poly(n=n,x=ttrap)*poly.weight(ttrap)
        ...     vhat[:,i] = torch.trapezoid(ytrap,ttrap)
        >>> assert torch.allclose(vhat,v,atol=1e-5)
    """
    
    def __init__(self, alpha=0, beta=0, loc=-1, scale=2):
        r"""
        Args:
            alpha (float): parameter $\alpha>-1$.
            beta (float): parameter $\beta>-1$.
            loc (float): weight distribution will be `scipy.stats.beta(a=beta+1,b=alpha+1,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
            scale (float): weight distribution will be `scipy.stats.beta(a=beta+1,b=alpha+1,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
        """
        self.alpha = float(alpha)
        self.beta = float(beta)
        assert self.alpha > -1 
        assert self.beta > -1
        self.c00 = 1
        self.c11 = (self.alpha+self.beta+2)/2
        self.c10 = (self.alpha+1)-(self.alpha+self.beta+2)/2
        self.atilde = float(-1) 
        self.btilde = float(1) 
        self.distrib = torch.distributions.Beta(self.beta+1,self.alpha+1)
        self.distrib_scipy = scipy.stats.beta(a=self.beta+1,b=self.alpha+1,loc=loc,scale=scale)
        self._sigma_tilde_0 = 1
        self._sigma_tilde_1 = 0 
        self._sigma_tilde_2 = -1
        self._tau_tilde_0 = self.beta-self.alpha
        self._tau_tilde_1 = -(self.alpha+self.beta+2)
        super().__init__(scale_tilde=scale/2,shift_tilde=scale/2+loc)
    
    def _cdf(self, x):
        return torch.from_numpy(self.distrib_scipy.cdf(x.numpy()))
    
    def _lnorm_(self, nrange):
        t0 = (1+self.alpha+self.beta)*np.log(2)+scipy.special.gammaln(self.alpha+1)+scipy.special.gammaln(self.beta+1)-scipy.special.gammaln(self.alpha+self.beta+2)+np.log(scipy.special.betainc(1+self.alpha,1+self.beta,1/2)+scipy.special.betainc(1+self.beta,1+self.alpha,1/2))
        lognum = (self.alpha+self.beta+1)*np.log(2) + torch.lgamma(nrange[1:]+self.alpha+1)+torch.lgamma(nrange[1:]+self.beta+1)
        logdenom = torch.log(2*nrange[1:]+self.alpha+self.beta+1) + torch.lgamma(nrange[1:]+1) + torch.lgamma(nrange[1:]+self.alpha+self.beta+1)
        trest = lognum-logdenom
        return torch.hstack([t0*torch.ones(1),trest])
    
    def _lweight(self, x):
        xt = self.scale*x+self.shift
        return self.distrib.log_prob((xt+1)/2)-np.log(2)
    
    def _recur_terms_(self, nrange):
        t1num = (2*nrange+1+self.alpha+self.beta)*(2*nrange+2+self.alpha+self.beta)
        t1denom = 2*(nrange+1)*(nrange+1+self.alpha+self.beta)
        t2num = (self.alpha**2-self.beta**2)*(2*nrange+1+self.alpha+self.beta)
        t2denom = 2*(nrange+1)*(2*nrange+self.alpha+self.beta)*(nrange+1+self.alpha+self.beta)
        t3num = (nrange+self.alpha)*(nrange+self.beta)*(2*nrange+2+self.alpha+self.beta)
        t3denom = (nrange+1)*(nrange+1+self.alpha+self.beta)*(2*nrange+self.alpha+self.beta)
        return t1num/t1denom,t2num/t2denom,t3num/t3denom

    def deriv(self, n, x):
        self.alpha += 1 
        self.beta += 1
        self.c11 += 1
        y = self._eval_unnormalized(n,x)
        self.alpha -= 1 
        self.beta -= 1
        self.c11 -= 1
        lC = self._lnorm(n)
        v = torch.exp(lC[0]/2-lC/2-np.log(self.c00))
        nrange = torch.arange(1,n+1)
        yp = torch.zeros_like(y)
        yp[1:] = torch.einsum("i,i...->i...",torch.exp(torch.lgamma(self.alpha+self.beta+nrange+2)-torch.lgamma(self.alpha+self.beta+nrange+1)+self.logscale)/2*v[1:],y[:-1])
        return yp


class Gegenbauer(Jacobi):
    
    r"""
    Orthonormal [Gegenbauer polynomials](https://en.wikipedia.org/wiki/Gegenbauer_polynomials) 
    supported on $[-1,1]$ with the weight normalized to be a density function. 
    
    These are a special case of the Jacobi polynomials with $\alpha=\beta$.
    """

    def __init__(self, alpha=0, loc=-1, scale=2):
        r"""
        Args:
            alpha (float): parameter $\alpha>-1$.
            loc (float): weight distribution will be `scipy.stats.beta(a=alpha+1,b=alpha+1,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
            scale (float): weight distribution will be `scipy.stats.beta(a=alpha+1,b=alpha+1,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
        """
        self.alpha = float(alpha)
        super().__init__(alpha=alpha,beta=alpha,loc=loc,scale=scale)


class Chebyshev1(Gegenbauer):

    r"""
    Orthonormal [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) of the first kind
    supported on $[-1,1]$ with the weight normalized to be a density function. 
    
    These are a special case of the Gegenbauer polynomials with $\alpha=-1/2$.
    """

    def __init__(self, loc=-1, scale=2):
        r"""
        Args:
            loc (float): weight distribution will be `scipy.stats.beta(a=1/2,b=1/2,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
            scale (float): weight distribution will be `scipy.stats.beta(a=1/2,b=1/2,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
        """
        super().__init__(alpha=-1/2,loc=loc,scale=scale)


class Chebyshev2(Gegenbauer):

    r"""
    Orthonormal [Chebyshev polynomials](https://en.wikipedia.org/wiki/Chebyshev_polynomials) of the second kind
    supported on $[-1,1]$ with the weight normalized to be a density function. 
    
    These are a special case of the Gegenbauer polynomials with $\alpha=1/2$.
    """

    def __init__(self, loc=-1, scale=2):
        r"""
        Args:
            loc (float): weight distribution will be `scipy.stats.beta(a=3/2,b=3/2,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
            scale (float): weight distribution will be `scipy.stats.beta(a=3/2,b=3/2,loc=loc,scale=scale)` supported on `[loc,loc+scale]`
        """
        super().__init__(alpha=1/2,loc=loc,scale=scale)
    
class Legendre(Gegenbauer):

    r"""
    Orthonormal [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials) 
    supported on $[-1,1]$ with the weight normalized to be a density function. 
    
    These are a special case of the Gegenbauer polynomials with $\alpha=0$.
    """

    def __init__(self, loc=-1, scale=2):
        r"""
        Args:
            loc (float): weight distribution will be `scipy.stats.uniform(loc=loc,scale=scale)` supported on `[loc,loc+scale]`
            scale (float): weight distribution will be `scipy.stats.uniform(loc=loc,scale=scale)` supported on `[loc,loc+scale]`
        """
        super().__init__(alpha=0,loc=loc,scale=scale)
