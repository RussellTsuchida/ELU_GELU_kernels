import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm
from scipy.special import erf
import GPy
import abc

from .nn_kernel import NNKernel
from .nn_kernel_relu import NNKernelRelu

class NNKernelElu(NNKernel):
    __metaclass__ = abc.ABCMeta
    def __init__(self, input_dim, variance_w = 1., mean_w = 0, 
            variance_b = 0, mean_b = 0, L=1, name='nn_elu',
            standard_first_layer= False):
        """
        Equivalent kernel of a neural network with a single hidden layer and
        a ReLU activation function.
        
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

        self.kernel_relu = NNKernelRelu(input_dim)

    def _single_layer_K(self, x1norm, x2norm, x1sum, x2sum, cos_theta, 
            x2_none=False):
        """
        Kernel for a single layer

        Args:
            x1norm (nparray): 
        """
        k_relu = self.kernel_relu._single_layer_K(x1norm, x2norm, x1sum, x2sum, 
                cos_theta, x2_none)

        e4_1 = self._E4(x1norm, x2norm, cos_theta, x2_none)
        e4_2 = self._E4(x1norm, 0*x2norm,      cos_theta, x2_none)
        if type(cos_theta) is int:
            e4_3 = self._E4(x2norm.T, 0*x1norm.T, cos_theta, x2_none).T
        else:
            e4_3 = self._E4(x2norm.T, 0*x1norm.T, cos_theta.T, x2_none).T
        e4_4 = self._E4(0*x1norm,      0*x2norm,      cos_theta, x2_none)

        E4 = e4_1 - e4_2 - e4_3 + e4_4

        E2 = self._E2(x1norm, x2norm, cos_theta, x2_none)
       
        if not (type(cos_theta) is int):
            E3 = self._E2(x2norm.T, x1norm.T, cos_theta.T, x2_none).T
        else:
            E3 = self._E2(x2norm, x1norm, cos_theta, x2_none)

        E1 = k_relu
        
        k = E1 + E2 + E3 + E4
        return k

    def _E4(self, a, b, cos_theta, x2_none):
        """
        Args:
            a (nparray): (n x 1)
            b (nparray): (1 x m)
            cos_theta (nparray): (n x m)
        Returns:
            nparray of size nxm
        """
        if type(cos_theta) is int:
            n = 1
            m = 1
        else:
            n = cos_theta.shape[0]
            m = cos_theta.shape[1]

        b = np.tile(b, (n,1))
        a = np.tile(a, (1,m))
       
        mu1 = a + b*cos_theta
        mu2 = a*cos_theta + b
        
        # Operate in log space to avoid numerical overflow
        #factor = np.exp(0.5*(mu1*a + mu2*b))
        log_factor = 0.5*(mu1*a+mu2*b)

        cdf, pdf = self._vectorised_bvn_cdf_pdf(-mu1, -mu2, cos_theta, 
                x2_none = x2_none, mus_are_vectors=False)
        log_cdf = np.log(cdf)

        return np.exp(log_factor+log_cdf)

        #return factor*cdf

    def _E2(self, xnorm1, xnorm2, cos_theta, x2_none):
        """
        Args:
            xnorm( nparray): (n x 1) or (1 x m)
            cos_theta (nparray): (n x m)
        Returns:
            nparray of size (n x m)
        """
        if type(cos_theta) is int:
            n = 1
            m = 1
        else:
            n = cos_theta.shape[0]
            m = cos_theta.shape[1]

        xnorm2 = np.tile(xnorm2, (n,1))
        xnorm1 = np.tile(xnorm1, (1,m))

        cdf, pdf = self._vectorised_bvn_cdf_pdf(xnorm2*cos_theta, -xnorm2,
                -cos_theta, x2_none = x2_none, mus_are_vectors = False)

        #return (-self._M(0*xnorm1, 0*xnorm1, cos_theta) + \
        #        (self._M(xnorm2*cos_theta, -xnorm2, cos_theta) + \
        #        xnorm2*cos_theta*cdf)*np.exp(xnorm2**2/2))*xnorm1
        # Operate in log space to avoid numerical overflow

        log_arg = self._M(-xnorm2*cos_theta, xnorm2, cos_theta) + \
                xnorm2*cos_theta*cdf
        log_factor = np.log(log_arg)
        log_exp = xnorm2**2/2

        second_term = np.exp(log_exp + log_factor)
        second_term[log_arg <= 0] = 0

        return (-self._M(0*xnorm1, 0*xnorm1, cos_theta) + \
                second_term)*xnorm1

    def _M(self, h, k, cos_theta):
        sin_theta = np.sqrt( np.clip(1-cos_theta**2, 0, 1) )

        if (type(cos_theta) is int):
            if sin_theta == 0:
                out = norm.pdf(h)*(1-(np.sign(k+h*cos_theta)+1)/2.) - \
                        norm.pdf(k)*cos_theta*(1-(np.sign(h+k*cos_theta)+1)/2.)
                return out

        out =   norm.pdf(h) * (1-norm.cdf((k+h*cos_theta)/sin_theta)) - \
      cos_theta*norm.pdf(k) * (1-norm.cdf((h+k*cos_theta)/sin_theta))

        if not (type(sin_theta) is int):
            out[sin_theta == 0] = \
                    (norm.pdf(h)*(1-(np.sign(k+h*cos_theta)+1)/2.)-\
                    cos_theta*norm.pdf(k)*(1-(np.sign(h+k*cos_theta)+1.)/2.))\
                    [sin_theta==0]

        return out

    def _single_layer_M(self, x1norm, x1sum):
        """
        return \
        self.kernel_relu._single_layer_M(x1norm,x1sum)
        """
        return np.zeros_like(x1sum)
