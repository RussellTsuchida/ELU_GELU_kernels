# External imports
import numpy as np
import GPy

# Internal imports
from ..kernels.nn_kernel_relu import NNKernelRelu
from ..kernels.nn_kernel_gelu import NNKernelGelu
from ..kernels.nn_kernel_elu import NNKernelElu

from .data.other_datasets import load_or_generate_snelson
from .plotters import plot_gp

from ..mvn_mixture.diag_mvn_mixture import DiagMVNMixture

############################################
########## EXPERIMENT PARAMETERS ###########
############################################
NOISE_VAR   = 0.1 # If this setting is changed, 
OUT_DIR     = 'code/experiments/outputs/snelson/'
KERNEL      = 'elu' # elu or gelu. elu is currently bugged, working on fixing.
############################################
############################################
X, Y, X_test, Y_test = load_or_generate_snelson()
Xmax = np.amax(np.abs(X))
X = X/Xmax*0.01+10; X_test = X_test/Xmax*0.01+10


# Quantile p values for the # outside graph
quantile = [0.125, 0.875]

# Set up the model and likelihood

if KERNEL == 'gelu':
    kern = lambda w_centre, w_var, L: \
        NNKernelGelu(X.shape[1], variance_w=w_var, L=L, 
        mean_w=w_centre, 
        variance_b=w_var, mean_b = w_centre,
        standard_first_layer=False)
elif KERNEL == 'elu':
    kern = lambda w_centre, w_var, L: \
        NNKernelElu(X.shape[1], variance_w=w_var, L=L, 
        mean_w=w_centre, 
        variance_b=w_var, mean_b = w_centre,
        standard_first_layer=False)
#print(kern(0, 2, 1).K(X,X))

model = lambda w_centre, w_var, L: \
    GPy.models.GPRegression(X, Y, kern(w_centre, w_var, L),noise_var=NOISE_VAR)

likelihood = lambda w_centre, w_var, L: np.exp(\
        model(w_centre, w_var, L).log_likelihood())

# Instantiate model
# (If you wanted to do hyperparameter search you would loop over values here
# and evaluate the likelihood for each w_var)

w_var = 1**2
w_centre = 0 # VALUES NOT EQUAL TO ZERO ARE CURRENTLY NOT IMPLEMENTED
L = 2

m = model(w_centre, w_var, L)
mean, var = m.predict(X_test)
var = np.clip(var, 0, None)
plot_gp(mean, np.sqrt(var), X_test, Y_test, X, Y, name=OUT_DIR + 'gp_' + \
        KERNEL + '_' + str(L) + '.pdf', axis_on=True)
mvn_mix = DiagMVNMixture(mean, np.sqrt(var))
mse = mvn_mix.error_mse_to_mean(Y_test)
num_outside = mvn_mix.error_num_outside_quantile(Y_test, quantile)


