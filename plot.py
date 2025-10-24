import torchorthopolys as top
import torch 
import numpy as np 
from matplotlib import pyplot 
from util import set_matplotlib_defaults

PW,DEFAULTFONTSIZE,MARKERS,LINESTYLES = set_matplotlib_defaults()

n = 6
polydict = {
    r"Hermite": (top.HermitePolys(), -1, 1),
    r"Laguerre": (top.LaguerrePolys(), 0, 10),
    r"Laguerre, $\alpha=\pi$": (top.LaguerrePolys(alpha=np.pi), 0, 10),
    r"Jacobi, $\alpha=-1/\sqrt{2}$, $\beta=-1/\sqrt{3}$": (top.JacobiPolys(alpha=-1/np.sqrt(2),beta=-1/np.sqrt(3)), -1, 1),
    r"Jacobi, $\alpha=1/\sqrt{2}$, $\beta=1/\sqrt{3}$": (top.JacobiPolys(alpha=1/np.sqrt(2),beta=1/np.sqrt(3)), -1, 1),
    r"Gegenbauer, $-\alpha=1/\sqrt{2}$": (top.Gegenbauer(alpha=1/np.sqrt(2)), -1, 1),
    r"Chebyshev $1^\mathrm{st}$ kind": (top.Chebyshev1(), -1, 1),
    r"Chebyshev $2^\mathrm{nd}$ kind": (top.Chebyshev2(), -1, 1),
    r"Legendre": (top.Legendre(), -1, 1),
}

ncols = 3 
nrows = int(np.ceil(len(polydict)/ncols))
fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(PW,PW/ncols*nrows))
for l,(name,(poly,a,b)) in enumerate(polydict.items()):
    i = l//ncols
    j = l%ncols
    x = torch.linspace(a,b,100)
    y = poly(n,x)
    for k in range(n+1):
        ax[i,j].plot(x,y[k],label=None if l>0 else r"$n = %d$"%k)
    ax[i,j].set_title(name)
    ax[i,j].set_xlim([a,b])
    ax[i,0].set_ylabel(r"$P_n(x)$")
    ax[-1,j].set_xlabel(r"$x$")
fig.legend(frameon=False,ncols=n+1,bbox_to_anchor=(.85,.95))
fig.suptitle(r"Orthonormal polynomials $P_n(x)$")
fig.savefig("polys.svg",bbox_inches="tight",transparent=False)

