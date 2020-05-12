import numpy as np
import abc
from scipy.stats import gennorm
from scipy.special import gamma
from scipy.stats import laplace

class RceInitialiser(object):
    __metaclass__ = abc.ABCMeta
    def __init__(self, apply_scale=True, use_normal_ABCD=True):
        """
        Args:
            apply_scale (Bool): True to scale the standard deviation and mean.
        """
        self.apply_scale = apply_scale
        self.use_normal_ABCD = use_normal_ABCD

    def _get_ABCD(self, shape):
        if self.use_normal_ABCD:
            A = np.random.normal(0, 1, size=(1,1))
            B = np.random.normal(0, 1, size=(shape[0], 1))
            C = np.random.normal(0, 1, size=(1, shape[1]))
            D = np.random.normal(0, 1, size=shape)
        else:
            A = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(1,1))
            B = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(shape[0], 1))
            C = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(1, shape[1]))
            D = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=shape)
      
        A = np.tile(A, [shape[0], shape[1]])
        B = np.tile(B, [1, shape[1]])
        C = np.tile(C, [shape[0], 1])

        return [A, B, C, D]

    def __call__(self, shape):
        A, B, C, D = self._get_ABCD(shape)

        if self.apply_scale:
            sqrt_n = np.sqrt(shape[1])
        else:
            sqrt_n = 1

        F = self._F(A,B,C,D)
        E_D_F = self._expected_over_D_of_F(A,B,C)
        return (F-E_D_F*(1-1./sqrt_n))/sqrt_n

    @abc.abstractmethod
    def _F(self, A, B, C, D):
        """
        Note: it is the user's responsibility to make sure _F and 
        _expected_over_D_of_F match.
        """
        pass

    @abc.abstractmethod
    def _expected_over_D_of_F(self, A, B, C):
        """
        Note: it is the user's responsibility to make sure _F and 
        _expected_over_D_of_F match.
        """
        pass

def create_initialiser(F, EF, apply_scale=True, use_normal_ABCD=True):
    """
    Created an RCE initialiser with a specified F and expected value of F over
    D. F should be a function of A, B, C, D and EF should be a function of A,
    B, C.

    Args:
        F (function):
        EF (function): 
        apply_scale (Bool): Apply scaling to mean and variance
        use_normal_ABCD (Bool): True to use Gaussian rvs for A, B, C, D .
            This is valid by the inverse transform theorem.
    """
    class LambdaRceInitialiser(RceInitialiser):
        """
        RceInitialiser that allows _F and _expected_over_D_of_F to be defined 
        with lambda functions.
        """
        def __init__(self):
            super().__init__(apply_scale, use_normal_ABCD)
            self._F = F
            self._expected_over_D_of_F = EF

    return LambdaRceInitialiser()

class GeneralisedNormal(object):
    """
    The generalised normal has a shape parameter beta that controls the tails of
    the pdf. When beta=2 we recover the Gaussian, beta = 1 is Laplace, beta->inf
    is uniform.

    https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.gennorm.html

    """
    def __init__(self, apply_scale=True, beta =2, std=1):
        self.apply_scale    = apply_scale
        self.beta           = beta
        self.std            = std

    def __call__(self, shape):
        if self.apply_scale:
            sqrt_n = np.sqrt(shape[1])
        else:
            sqrt_n = 1

        D = gennorm.rvs(beta=self.beta, size=shape)
        # This is the variance of d
        var = gamma(3./self.beta)/gamma(1./self.beta)
        # Make D have unit variance
        D = D/np.sqrt(var)
        
        # Return correctly scaled version of D
        return D/sqrt_n*self.std

class Laplace(object):
    def __init__(self, apply_scale=True, std=1, mu=0):
        self.apply_scale    = apply_scale
        self.std            = std
        self.mu             = mu

    def __call__(self, shape):
        if self.apply_scale:
            sqrt_n = np.sqrt(shape[1])
        else:
            sqrt_n = 1

        D = laplace.rvs(size=shape)
        # Scale to unit variance
        D = D / np.sqrt(2)

        # Return correctly scaled version of D
        return D/sqrt_n*self.std + self.mu/sqrt_n**2

class Uniform(object):
    def __init__(self, apply_scale=True, std=1, mu=0):
        self.apply_scale    = apply_scale
        self.std            = std
        self.mu             = mu

    def __call__(self, shape):
        if self.apply_scale:
            sqrt_n = np.sqrt(shape[1])
        else:
            sqrt_n = 1

        D = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=shape)

        # Return correctly scaled version of D
        return D/sqrt_n*self.std + self.mu/sqrt_n**2

class StudentT(object):
    def __init__(self, apply_scale=True, nu=2.5, mu=0):
        self.apply_scale    = apply_scale
        self.nu             = nu
        self.mu             = mu

    def __call__(self, shape):
        if self.apply_scale:
            sqrt_n = np.sqrt(shape[1])
        else:
            sqrt_n = 1

        D = np.random.standard_t(self.nu, size=shape)

        # Return correctly scaled version of D
        return D/sqrt_n + self.mu/sqrt_n**2

