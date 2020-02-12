import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.special import erf
import GPy
import abc

from .nn_kernel import NNKernel
from .nn_kernel_linear import NNKernelLinear
from .nn_kernel_abs import NNKernelAbs

class NNKernelErf(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_erf',
            standard_first_layer= False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        ERF activation function.
        
        Args:
            input_dim (int): Dimensionality of input.
            variance_w (float): Variance of the weights.
            mean_w (float): Mean of the weights.
            variance_b (float): Variance of the biases.
            mean_b (float): Mean of the biases.
            L (int): The number of hidden layers.
        """
        super().__init__(input_dim, variance_w, mean_w, variance_b, mean_b, L,
                name, standard_first_layer)

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer. 
        """
        bothnorms = x1norm*x2norm
        x_x2_norm = np.multiply(cos_theta, bothnorms)

        a = 2*x_x2_norm / np.sqrt((1+2*np.square(x1norm))*(1+2*np.square(x2norm)))

        return 2/np.pi*np.arcsin(a)

    def _single_layer_M(self, x1norm, x1sum):
        # not implemented
        return 0
