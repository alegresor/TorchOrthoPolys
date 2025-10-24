def set_matplotlib_defaults():
    # import seaborn as sns 
    import matplotlib 
    from matplotlib import pyplot,cycler
    from matplotlib.colors import ListedColormap
    # import pandas as pd
    pyplot.style.use("seaborn-v0_8-whitegrid")
    # pyplot.style.use("mlqmcpy.mplstyle")
    MARKERS = ['o','s','^','d','.','*','X','1']
    # COLORS = ["xkcd:"+color[:-1] for color in pd.read_csv("./xkcd_colors.txt",comment="#",header=None).iloc[:,0].tolist()][::-1]
    # pyplot.rcParams['axes.prop_cycle'] = cycler(color=COLORS)
    LINESTYLES = ['solid','dotted','dashed','dashdot',(0, (1, 1))]
    DEFAULTFONTSIZE = 30
    pyplot.rcParams['xtick.labelsize'] = DEFAULTFONTSIZE
    pyplot.rcParams['ytick.labelsize'] = DEFAULTFONTSIZE
    pyplot.rcParams['ytick.labelsize'] = DEFAULTFONTSIZE
    pyplot.rcParams['axes.titlesize'] = DEFAULTFONTSIZE
    pyplot.rcParams['figure.titlesize'] = DEFAULTFONTSIZE
    pyplot.rcParams["axes.labelsize"] = DEFAULTFONTSIZE
    pyplot.rcParams['legend.fontsize'] = DEFAULTFONTSIZE
    pyplot.rcParams['font.size'] = DEFAULTFONTSIZE
    pyplot.rcParams['lines.linewidth'] = 5
    pyplot.rcParams['lines.markersize'] = 15
    PW = 30 # inches
    return PW,DEFAULTFONTSIZE,MARKERS,LINESTYLES
