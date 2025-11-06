def plot():
    import torchorthopolys as top
    import numpy as np 
    import torch 

    n = 6
    polydict = {
        "Hermite(loc=2,scale=3)": (top.Hermite(loc=2,scale=3), -4, 8),
        "Laguerre(loc=1,scale=2)": (top.Laguerre(loc=1,scale=2), 1, 20),
        "Laguerre(alpha=np.pi,loc=1,scale=2)": (top.Laguerre(alpha=np.pi,loc=1,scale=2), 1, 20),
        "Jacobi(alpha=-1/np.sqrt(2),beta=-1/np.sqrt(3),\nloc=2,scale=6)": (top.Jacobi(alpha=-1/np.sqrt(2),beta=-1/np.sqrt(3),loc=2,scale=6), 2, 8),
        "Jacobi(alpha=-1/2,beta=np.exp(-1),\nloc=2,scale=6)$": (top.Jacobi(alpha=-1/2,beta=np.exp(-1),loc=2,scale=6), 2, 8),
        "Gegenbauer(alpha=-1/np.pi,loc=2,scale=6)": (top.Gegenbauer(alpha=-1/np.pi,loc=2,scale=6), 2, 8),
        "Chebyshev1(loc=2,scale=6)": (top.Chebyshev1(loc=2,scale=6), 2, 8),
        "Chebyshev2(loc=2,scale=6)": (top.Chebyshev2(loc=2,scale=6), 2, 8),
        "Legendre(loc=2,scale=6)": (top.Legendre(loc=2,scale=6), 2, 8),
    }

    for name,(poly,a,b) in polydict.items():
        x = torch.linspace(a,b,100)
        y = poly(n,x)
        polydict[name] = (poly,a,b,x,y)

    from matplotlib import pyplot 
    from util import set_matplotlib_defaults
    PW,DEFAULTFONTSIZE,MARKERS,LINESTYLES = set_matplotlib_defaults()
    ncols = 3 
    nrows = int(np.ceil(len(polydict)/ncols))
    fig,ax = pyplot.subplots(nrows=nrows,ncols=ncols,figsize=(PW,PW/ncols*nrows))
    ax = np.atleast_2d(ax).reshape((nrows,ncols))
    for l,(name,(poly,a,b,x,y)) in enumerate(polydict.items()):
        i = l//ncols
        j = l%ncols
        for k in range(n+1):
            ax[i,j].plot(x,y[k],label=None if l>0 else r"$n = %d$"%k)
        ax[i,j].set_title(name)
        ax[i,j].set_xlim([a,b])
        ax[i,0].set_ylabel(r"$P_n(x)$")
        ax[-1,j].set_xlabel(r"$x$")
    fig.legend(frameon=False,ncols=n+1,bbox_to_anchor=(.85,.95))
    fig.suptitle(r"Orthonormal polynomials $P_n(x)$")
    fig.savefig("polys.svg",bbox_inches="tight",transparent=False)

if __name__=="__main__":
    plot()