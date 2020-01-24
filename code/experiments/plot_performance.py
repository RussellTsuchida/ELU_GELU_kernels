import numpy as np
import sys

from .experiment_array import ExperimentArray
from .plotters import plot_samples


data = sys.argv[1]
kernel = sys.argv[2]

rmse = ExperimentArray((32, 50), kernel+data+'/rmse/').load_array()
nll = ExperimentArray((32, 50), kernel+data+'/nll/').load_array()

var_list = np.arange(0.1, 5.05, 0.1)

labels = [r'$l='+str(e)+'$' for e in range(1, 33)]
plot_samples(var_list, rmse, kernel+data+'/rmse_plot.pdf', plot_legend=True,
        markersize=0, labels=labels, linewidth=2, legend_fontsize=7,
        xlabel=r'$\sigma_w^2$', ylabel=r'RMSE')

plot_samples(var_list, nll, kernel+data+'/nll_plot.pdf', plot_legend=True,
        markersize=0, labels=labels, linewidth=2, legend_fontsize=7,
        xlabel=r'$\sigma_w^2$', ylabel=r'NLL')

