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
        """
        k_relu = self.kernel_relu._single_layer_K(x1norm, x2norm, x1sum, x2sum, 
                cos_theta, x2_none)

        if not (type(cos_theta) is int):
            E4 = \
                self._E4(x1norm, x2norm, cos_theta, x2_none) - \
                self._E4(x1norm, 0*x2norm,      cos_theta, x2_none) - \
                self._E4(0*x1norm.T,      x2norm.T, cos_theta.T, x2_none).T + \
                self._E4(0*x1norm,      0*x2norm,      cos_theta, x2_none)
        else:
            E4 = \
                self._E4(x1norm, x2norm, cos_theta, x2_none) - \
                self._E4(x1norm, 0*x2norm,      cos_theta, x2_none) - \
                self._E4(0*x1norm,      x2norm, cos_theta, x2_none) + \
                self._E4(0*x1norm,      0*x2norm,      cos_theta, x2_none)

    
        E2 = self._E2(x1norm, x2norm, cos_theta, x2_none)
        
        if not (type(cos_theta) is int):
            E3 = self._E2(x2norm.T, x1norm.T, cos_theta.T, x2_none).T
        else:
            E3 = self._E2(x2norm, x1norm, cos_theta, x2_none)

        E1 = k_relu
        
        return E1 + E2 + E3 + E4

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

        if b.shape[0] == n:
            b = np.tile(b, (1,m))
            a = np.tile(a, (n,1))
        elif b.shape[1] == m:
            b = np.tile(b, (n,1))
            a = np.tile(a, (1,m))

        mu1 = a + b*cos_theta
        mu2 = a*cos_theta + b

        factor = np.exp(0.5*(mu1*a + mu2*b))

        cdf, pdf = self._vectorised_bvn_cdf_pdf(-mu1, -mu2, cos_theta, 
                x2_none = x2_none, mus_are_vectors=False)

        return factor*cdf

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

        if xnorm2.shape[0] == n:
            xnorm2 = np.tile(xnorm2, (1,m))
            xnorm1 = np.tile(xnorm1, (n,1))
        elif xnorm2.shape[1] == m:
            xnorm2 = np.tile(xnorm2, (n,1))
            xnorm1 = np.tile(xnorm1, (1,m))

        cdf, pdf = self._vectorised_bvn_cdf_pdf(xnorm2*cos_theta, -xnorm2,
                -cos_theta, x2_none = x2_none, mus_are_vectors = False)

        return (-self._M(0*xnorm1, 0*xnorm1, cos_theta) + \
                (self._M(xnorm2*cos_theta, -xnorm2, cos_theta) + \
                xnorm2*cos_theta*cdf)*np.exp(xnorm2**2/2))*xnorm1

    def _M(self, h, k, cos_theta):
        sin_theta = np.sqrt( np.clip(1-cos_theta**2, 0, 1) )

        if (type(cos_theta) is int):
            if sin_theta == 0:
                out = norm.pdf(h)*(np.sign(k+h*cos_theta)+1)/2. - \
                        (np.sign(h+k*cos_theta)+1)/2.*norm.pdf(k)*cos_theta
                return out

        out =   norm.pdf(h) * norm.cdf((k+h*cos_theta)/sin_theta) - \
      cos_theta*norm.pdf(k) * norm.cdf((h+k*cos_theta)/sin_theta)

        if not (type(sin_theta) is int):
            out[sin_theta == 0] = (norm.pdf(h)*(np.sign(k+h*cos_theta)+1)/2.-\
                    cos_theta*norm.pdf(k)*(np.sign(h+k*cos_theta)+1.)/2.)\
                    [sin_theta==0]

        return out

    def _single_layer_M(self, x1norm, x1sum):
        """
        return \
        self.kernel_relu._single_layer_M(x1norm,x1sum)
        """
        return np.zeros_like(x1sum)
