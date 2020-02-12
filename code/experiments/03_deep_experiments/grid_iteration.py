import sys
import numpy as np

# Internal imports
from ...kernels.nn_kernel_relu import NNKernelRelu
from ...kernels.nn_kernel_gelu import NNKernelGelu
from ...kernels.nn_kernel_elu import NNKernelElu
from ...kernels.nn_kernel_erf import NNKernelErf
from ...kernels.nn_kernel_lrelu import NNKernelLRelu

from .experiment_array import ExperimentArray
from ..data import datasets

DIR         = 'code/experiments/01_data/'
OUTPUT_DIR  = 'code/experiments/outputs/deep_experiments/'
NOISE_VAR   = 0.1
FORCE_NEW   = False

# One iteration of a hyperparameter grid search
###############################################################################
data        = sys.argv[1]
kern_str    = sys.argv[2]
var         = float(sys.argv[3])
var_idx     = int(sys.argv[4])
L           = int(sys.argv[5])
L_idx       = int(sys.argv[6])

rmse_data = ExperimentArray((32, 50), OUTPUT_DIR + kern_str+data+'/rmseX/')
nll_data = ExperimentArray((32, 50), OUTPUT_DIR + kern_str+data+'/nllX/')
# Skip points that have already been measured. 
# This means it is the experimenter's responsibility to clear disk if
# experiment parameters change.
if rmse_data[L_idx, var_idx] != 0:
    print("Already measured, exiting...")
    sys.exit()
###############################################################################
###################################### Set up data ############################
if data == 'Boston':
    dataset = datasets.Boston(DIR)
elif data == 'Concrete':
    dataset = datasets.Concrete(DIR)
elif data == 'Energy':
    dataset = datasets.Energy(DIR)
elif data == 'Kin8nm':
    dataset = datasets.Kin8nm(DIR)
elif data == 'Naval':
    dataset = datasets.Naval(DIR)
elif data == 'Power':
    dataset = datasets.Power(DIR)
elif data == 'Protein':
    dataset = datasets.Protein(DIR)
elif data == 'Wine':
    dataset = datasets.Wine(DIR)
elif data == 'Yacht':
    dataset = datasets.Yacht(DIR)

X_train, Y_train, X_test, Y_test = dataset.load_or_generate_data()

###################################### Set up kernel ##########################
if kern_str == 'ReLU':
    kern = NNKernelRelu(X_train.shape[1], var, 0, var, 0, L)
elif kern_str == 'LReLU':
    kern = NNKernelLRelu(X_train.shape[1], var, 0, var, 0, L)
elif kern_str == 'ERF':
    kern = NNKernelErf(X_train.shape[1], var, 0, var, 0, L)
elif kern_str == 'GELU':
    kern = NNKernelGelu(X_train.shape[1], var, 0, var, 0, L)
elif kern_str == 'ELU':
    kern = NNKernelElu(X_train.shape[1], var, 0, var, 0, L)

###################################### GP Regression ##########################
K_xx = kern.K(X_train,X_train)
K_x_xstar = kern.K(X_train,X_test)
K_xstar_x = K_x_xstar.T
K_xstar_xstar = kern.K(X_test,X_test)

mean = np.matmul(K_xstar_x,
    np.linalg.solve(K_xx+NOISE_VAR*np.eye(K_xx.shape[0]), Y_train))
var = K_xstar_xstar - np.matmul(K_xstar_x,
    np.linalg.solve(K_xx+NOISE_VAR*np.eye(K_xx.shape[0]), K_x_xstar))

###################################### Save performance ########################
RMSE = dataset.rmse_original_units(mean)

_, logdet = np.linalg.slogdet(K_xx+NOISE_VAR*np.eye(K_xx.shape[0]))
NLL = (0.5*np.matmul(Y_train.T,
        np.linalg.solve(K_xx+NOISE_VAR*np.eye(K_xx.shape[0]), Y_train))+\
      0.5*logdet)[0,0]

rmse_data[L_idx, var_idx] = RMSE
nll_data[L_idx, var_idx] = NLL
