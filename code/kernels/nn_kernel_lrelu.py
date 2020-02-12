import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.special import erf
import GPy
import abc

from .nn_kernel import NNKernel
from .nn_kernel_linear import NNKernelLinear
from .nn_kernel_abs import NNKernelAbs

class NNKernelLRelu(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_lrelu',
            standard_first_layer= False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        a LReLU activation function.
        
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

        self.kernel_lin = NNKernelLinear(input_dim)
        self.kernel_abs = NNKernelAbs(input_dim)

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer
        """
        theta = np.arccos(cos_theta)
        a = 0.2 # leaky param

        return x1norm*x2norm*(np.square(1-a)/(2*np.pi)*\
                (np.sin(theta) + (np.pi-theta)*np.cos(theta)) + a*np.cos(theta))

    def _single_layer_M(self, x1norm, x1sum):
        # not implemented
        return 0
