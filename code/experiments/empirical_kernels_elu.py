# External libraries
import numpy as np
import GPy
import sys
import time

# Internal imports
from ..networks.np_mlp import Mlp
from ..networks.np_initialisers import create_initialiser
from ..kernels.nn_kernel_elu import NNKernelElu
from .data.other_datasets import load_or_generate_x_xdash
from .data.other_datasets import load_or_generate_hypersphere
from .plotters import plot_samples


############################################
########## EXPERIMENT PARAMETERS ###########
############################################
L           = 32 # number of HIDDEN (not input or output) layers
NUM_X_POINTS= 100
X_DIMENSION = 2
ACTIVATIONS = lambda x: (x < 0)*(np.exp(x)-1.) + np.clip(x, 0, None)
WIDTH       = 3000
OUT_DIR     = 'code/experiments/outputs/kernels/'
############################################
############################################
# Weight and bias standard deviations and means
b_centre = 0; b_scale = 0;
#w_centres = [0, -1, -2, 1]
#w_scales= [np.sqrt(2)]*4
w_centres = [0]
w_scales = [1.4023779627897333]

##  || x ||       sigma
##   0.5       1.1746549260629595
##    1        1.2779600754047147
##    5        1.4023779627897333
############################################
############################################
############################################

assert len(w_centres) == len(w_scales)
# Load some random test data
theta_list = np.linspace(0.01, np.pi, NUM_X_POINTS)
data = load_or_generate_x_xdash(theta_list, X_DIMENSION)*5

for w_centre, w_scale in zip(w_centres, w_scales):
    fname = 'elu_kernel_' + str(w_centre) + '.pdf'
    if w_centre == 0:
        lw = 4
        markersize = 0
        legend = True
    else:
        lw = 0
        markersize = 6
        legend = False

    # Use random normal weights
    w_init = create_initialiser(lambda A, B, C, D: w_scale*D + w_centre,
            lambda A, B, C: w_centre)
    b_init = create_initialiser(lambda A, B, C, D: b_scale*D + b_centre,
            lambda A, B, C: b_centre)

    # Neural network, find the empirical normalised kernel
    mlp = Mlp([X_DIMENSION]+L*[WIDTH]+[1],  
            L*[ACTIVATIONS]+[lambda x: x],
            w_init, b_init)
    emp_kernel = mlp.empirical_normalised_kernel_given_data(data)

    labels = [r'$l='+str(2**e)+'$' for e in range(int(np.log2(L))+1)]
    fig = plot_samples(theta_list, emp_kernel, OUT_DIR + fname, 
            labels=labels, xlabel=r'$\theta^{(0)}$', 
            ylabel=r'$\cos\theta^{(l)}$', marker='o', 
            markersize=6, plot_legend=legend)

    # Find the theoretical normalised kernel
    kernel = NNKernelElu
    kern = kernel(X_DIMENSION, w_scale**2, w_centre, b_scale**2, b_centre, L)
    asym_kernel = kern.normalised_kernel_given_data(data)

    plot_samples(theta_list, asym_kernel, OUT_DIR + fname,
            xlabel=r'$\theta^{(0)}$', ylabel=r'$\cos\theta^{(l)}$', fig=fig, 
            marker='x', linewidth=lw, labels=labels, plot_legend=legend,
            markersize=markersize)


