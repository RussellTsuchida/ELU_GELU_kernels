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
        Equivalent kernel of a neural network with a single hidden layer and
        a GELU activation function.

        input_dim (int): Dimensionality of input.
        variance_w (float): Variance of the weights.
        mean_w (float): Mean of the weights.
        variance_b (float): Variance of the biases.
        mean_b (float): Mean of the biases.
        L (int): The number of hidden layers.
        """
        super().__init__(input_dim, variance_w, mean_w, variance_b, mean_b, L,
                name)

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer
        """

        if isinstance(cos_theta, int) and (cos_theta == 1):
            return x1norm**2*x2norm**2*(x1norm**2+x2norm**2+2)/\
            (2*np.pi*(x1norm**2+1)*(x2norm**2+1)*np.sqrt(x1norm**2+x2norm**2+1))+\
            x1norm*x2norm/4*(1+2/np.pi*np.arcsin(x1norm*x2norm/(1+x1norm*x2norm)))
        else:
            cos_theta = np.clip(cos_theta, -1, 1)
            sin_theta = np.sqrt(np.clip(1.-cos_theta**2, 0., 1.))
            factor = x1norm**2*x2norm**2/(2*np.pi)
     
            x1_quad = (1+x1norm**2*sin_theta**2)
            x2_quad = (1+x2norm**2*sin_theta**2)

            first_term = np.divide((x1_quad*x2_quad - cos_theta**4),
                (sin_theta**2*(1+x1norm**2)*(1+x2norm**2)*\
                np.sqrt(1+x1norm**2+x2norm**2+x1norm**2*x2norm**2*sin_theta**2)),
                out = np.zeros_like(cos_theta, dtype=float),
                where = sin_theta != 0)

            atan = np.arctan(cos_theta*x1norm*x2norm/\
                (np.sqrt(1+x1norm**2+x2norm**2+x1norm**2*x2norm**2*sin_theta**2)))
            #atan[np.isnan(atan)] = np.arctan(np.inf)
            second_term = cos_theta*atan/(x1norm*x2norm)
            
            k = factor*(first_term+second_term)+x1norm*x2norm*cos_theta/4.
            # Evaluate on {0, pi}
            k[cos_theta == 1] = (x1norm**2*x2norm**2*(x1norm**2+x2norm**2+2)/\
            (2*np.pi*(x1norm**2+1)*(x2norm**2+1)*np.sqrt(x1norm**2+x2norm**2+1))+\
            x1norm*x2norm/4*(1+2/np.pi*np.arcsin(x1norm*x2norm/(1+x1norm*x2norm))))\
            [cos_theta == 1]

            k[cos_theta == -1] = (x1norm**2*x2norm**2*(x1norm**2+x2norm**2+2)/\
            (2*np.pi*(x1norm**2+1)*(x2norm**2+1)*np.sqrt(x1norm**2+x2norm**2+1))+\
            x1norm*x2norm/4*(1+2/np.pi*np.arcsin(x1norm*x2norm/(1+x1norm*x2norm)))-\
            x1norm*x2norm/2)[cos_theta == -1]

            return k 


    def _single_layer_M(self, x1norm, x1sum):
        return 0
