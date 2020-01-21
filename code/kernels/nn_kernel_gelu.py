import numpy as np
import GPy
import abc

from .nn_kernel import NNKernel

class NNKernelGelu(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_gelu', 
            standard_first_layer = False):
        """
        Equivalent kernel of a neural network with L hidden layers and
        a GELU activation function.

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
        Kernel for a single layer
        """
        cos_theta = np.clip(cos_theta, -1, 1)
        sin_theta = np.sqrt(np.clip(1.-cos_theta**2, 0., 1.))
        cos_2theta = cos_theta**2 - sin_theta**2
        factor = x1norm**2*x2norm**2/(2*np.pi)

        first_term=(x1norm**2+x2norm**2+x1norm**2*x2norm**2*sin_theta**2+\
            0.5*(cos_2theta+3))/\
            ((1+x1norm**2)*(1+x2norm**2)*np.sqrt(1+x1norm**2+x2norm**2+\
            x1norm**2*x2norm**2*sin_theta**2))

        atan = np.arctan((cos_theta*x1norm*x2norm)/\
                np.sqrt(1+x1norm**2+x2norm**2+x1norm**2*x2norm**2*sin_theta**2))
        second_term = cos_theta*atan/(x1norm*x2norm)

        c = x1norm*x2norm/4

        k = factor*(first_term + second_term) + c*cos_theta
        
        return k

    def _single_layer_M(self, x1norm, x1sum):
        return 0
