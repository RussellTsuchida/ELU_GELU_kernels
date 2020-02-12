import numpy as np

from .experiment_array import ExperimentArray
from ..plotters import plot_samples

OUTPUT_DIR  = 'code/experiments/outputs/deep_experiments/'

def plot_and_return_rmse(data, kernel):
    num_trials = 6
    rmse = 0
    nll = 0

    rmse_var = 0
    nll_var = 0
    rmse_mean_old = 0
    nll_mean_old = 0
    rmse_old = np.zeros((32,50))
    n = -1
    for trial in range(2, num_trials+1):
        rmse_new = ExperimentArray((32, 50), OUTPUT_DIR+kernel+data+'/rmse'+\
                str(trial) + '/').load_array()
        nll_new = ExperimentArray((32, 50), OUTPUT_DIR+kernel+data+'/nll'+\
                str(trial) + '/').load_array()
        if np.all(rmse_new == rmse_old):
            print(trial)
            print('Duplicate data...')

        if np.sum(rmse_new == 0) > 0:
            if np.sum(rmse_new==0) == 1600:
                print('Warning: all datapoints missing, skipping')
                print(trial)
                continue
            # Some settings failed due to numerical issues. Ignore these points
            rmse_new[rmse_new == 0] = 'nan'
        n += 1

        rmse += rmse_new
        nll += nll_new

        rmse_mean = rmse/(n+1)
        nll_mean = nll/(n+1)
        
        if n > 0:
            rmse_var = \
                ((n+1-2)*rmse_var+(rmse_new-rmse_mean)*(rmse_new-rmse_mean_old))/\
                (n+1-1)
            nll_var = \
                ((n+1-2)*nll_var+(nll_new-nll_mean)*(nll_new-nll_mean_old))/\
                (n+1-1)
        rmse_mean_old = rmse_mean
        nll_mean_old = nll_mean

        rmse_old = rmse_new

    var_list = np.arange(0.1, 5.05, 0.1)

    labels = [r'$l='+str(e)+'$' for e in range(1, 33)]
    rmse_fig = None
    nll_fig = None

    rmse_fig = plot_samples(var_list, rmse_mean, 
            kernel+data+'/rmse_plot.pdf', fig = rmse_fig,
            markersize=0, labels=labels, linewidth=1, legend_fontsize=7,
            xlabel=r'$\sigma_w^2$', ylabel=r'RMSE', fill_between=np.sqrt(rmse_var))
    rmse_fig = plot_samples(var_list, rmse_mean, kernel+data+'/rmse_plot.pdf', 
            plot_legend=True, fig = rmse_fig,
            markersize=0, labels=labels, linewidth=1, legend_fontsize=7,
            xlabel=r'$\sigma_w^2$', ylabel=r'RMSE')

    nll_fig = plot_samples(var_list, nll_mean, 
            kernel+data+'/nll_plot.pdf', fig = nll_fig,
            markersize=0, labels=labels, linewidth=1, legend_fontsize=7,
            xlabel=r'$\sigma_w^2$', ylabel=r'NLL', fill_between=np.sqrt(nll_var))
    nll_fig = plot_samples(var_list, nll, kernel+data+'/nll_plot.pdf', 
            plot_legend=True, fig = nll_fig,
            markersize=0, labels=labels, linewidth=1, legend_fontsize=7,
            xlabel=r'$\sigma_w^2$', ylabel=r'NLL')

    # Find the lowest rmse and corresponding variance, depth and rmse variance
    min_rmse = np.nanmin(rmse_mean)
    idx = np.where(rmse_mean == min_rmse)

    min_rmse_var = (rmse_var[idx])[0]
    var_w = (var_list[idx[1]])[0]
    depth = (idx[0])[0]+1

    return [min_rmse, np.sqrt(min_rmse_var), var_w, depth]

datasets = ['Boston', 'Concrete', 'Energy', 'Wine', 'Yacht']
kernels = ['ReLU', 'GELU', 'LReLU', 'ERF']

for data in datasets:
    for kernel in kernels:
        # Latex table formatting
        output = plot_and_return_rmse(data, kernel)
        print("& $ %.2f \pm %.2f $ & $ %.2f $ & $ %d $" % (output[0], output[1], output[2], output[3]))

