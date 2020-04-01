import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.special import erf
import GPy
import abc

from .nn_kernel import NNKernel

class NNKernelTrig(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_lrelu',
            standard_first_layer= False, cos_or_sin = 'cos'):
        """
        Equivalent kernel of a neural network with L hidden layers and
        a cosine or sine activation function.
        
        Args:
            input_dim (int): Dimensionality of input.
            variance_w (float): Variance of the weights.
            mean_w (float): Mean of the weights.
            variance_b (float): Variance of the biases.
            mean_b (float): Mean of the biases.
            L (int): The number of hidden layers.
            cos_or_sin (str): 'cos' for cosine, 'sin' for sine, 'half' for
                half of the activations sqrt(2)*sin and the other half 
                sqrt(2)*cos
        """
        super().__init__(input_dim, variance_w, mean_w, variance_b, mean_b, L,
                name, standard_first_layer)
        if cos_or_sin == 'cos':
            self.trig = np.cos
        elif cos_or_sin == 'sin':
            self.trig = np.sin
        self.cos_or_sin = cos_or_sin

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer
        """
        if self.cos_or_sin == 'half':
            return  np.cos(x1sum-x2sum)*\
                    np.exp(-0.5*(x1norm**2+x2norm**2-2*x1norm*x2norm*cos_theta))

        term1 = np.cos(x1sum-x2sum)*np.exp(-0.5*\
                (x1norm**2+x2norm**2-2*x1norm*x2norm*cos_theta))
        term2 = np.cos(x1sum+x2sum)*np.exp(-0.5*\
                (x1norm**2+x2norm**2+2*x1norm*x2norm*cos_theta))
        if self.cos_or_sin == 'cos':
            return term1 + term2
        elif self.cos_or_sin == 'sin':
            return term1 - term2

    def _single_layer_M(self, x1norm, x1sum):
        if self.cos_or_sin == 'half':
            return  np.cos(x1sum)*np.exp(-x1norm**2/2) + \
                    np.sin(x1sum)*np.exp(-x1norm**2/2)
        else:
            return self.trig(x1sum)*np.exp(-x1norm**2/2)
