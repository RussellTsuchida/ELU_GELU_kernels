# External libraries
import numpy as np
from scipy.special import erf
from scipy.stats import norm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['axes.linewidth'] = 5
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['grid.linewidth'] = 5
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mtick

x_max=2.5

acts_all        = ['gelu','relu','erf','Lrelu','prelu', 'elu']
acts_polished   = {'gelu':'GELU','relu':'ReLU','erf':'ERF','Lrelu':'LReLU',
                    'prelu':'PReLU','elu':'ELU'}
acts_dict={}
acts_dict['relu']  = lambda x: np.maximum(0,x)
acts_dict['gelu']  = lambda x: x*norm.cdf(x)
acts_dict['erf']   = lambda x: erf(x)
acts_dict['Lrelu'] = lambda x: np.maximum(0.2*x,x)
acts_dict['elu']   = lambda x: (x*[x>=0] + 1*(np.exp(x)-1)*[x<0]).flatten()

colours = ['r','b','g','k','darkorange']

x_grid = np.linspace(-x_max,x_max,100)

fig = plt.figure(figsize=(6, 4)) # w, l
ax = fig.add_subplot(111)

for j, act in enumerate(acts_dict.keys()):
    y = acts_dict[act](x_grid)
    ax.plot(x_grid, y, colours[j], marker=None,linewidth=3,
            label=acts_polished[act])

    ax.legend(loc='upper left', fontsize=20)
ax.set_xlim(-x_max,x_max)
ax.set_ylim(-1.1,x_max)
#ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
#ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
ax.set_yticklabels([])
ax.set_xticklabels([])

fig.savefig('act_fns.pdf',format='pdf', dpi=500, bbox_inches='tight')



