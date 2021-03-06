from ..kernels.nn_kernel_elu import NNKernelElu
from ..kernels.nn_kernel_gelu import NNKernelGelu
from ..kernels.nn_kernel_relu import NNKernelRelu

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['axes.linewidth'] = 5
matplotlib.rcParams['lines.linewidth'] = 2
matplotlib.rcParams['grid.linewidth'] = 5
import numpy as np
import scipy.optimize as sci_opt
from scipy.stats import multivariate_normal

def sigma_star_plot(g, label, fig):
    norm_list = np.linspace(0.01, 8, 100)
    sigma_list = np.zeros_like(norm_list)
    for i, norm in enumerate(norm_list):
        root = sigma_star(g, norm)
        sigma_list[i] = root
    if fig is None:
        fig = plt.figure(figsize = (7,7))
    else:
        plt.figure(fig.number)
    plt.plot(norm_list, sigma_list, label=label)
    plt.ylabel(r'$\sigma^*$', fontsize=40)
    plt.xlabel(r'$\Vert \mathbf{x} \Vert$', fontsize=40)
    plt.legend(prop={'size':30})
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.savefig('code/dynamics/g_roots.pdf')

    return fig

def sigma_star(g, norm):
    """
    Find the root of g_||x||
    """
    root = sci_opt.brentq(lambda std: g(std, norm), 0.01, 2.) 
    return root

def norm_preserving_s(g, norm):
    """
    Find the value of s = ||x|| \sigma that corresponds to the value of 
    \sigma that preserves the expected squared norm
    """
    sigma = sigma_star(g, norm)
    return sigma*norm

def lambda_3_gelu(sigma, norm, theta):
    """
    Evaluate the lambda 1 at the value of s and norm, assuming sigma is at
    sigma star.
    """
    term1 = sigma**2/4*(1+2/np.pi*np.arcsin(\
            (sigma*norm)**2*np.cos(theta)/(1+sigma**2*norm**2)))
    term2 = sigma**4*norm**2*np.cos(theta)/\
            (np.pi*(norm**2*sigma**2+1)*\
            np.sqrt(sigma**4*norm**4*np.sin(theta)**2+2*norm**2*sigma**2+1))
    term3 = sigma**4*norm**2*np.cos(theta)\
            /(2*np.pi)*\
            (1+2*sigma**2*norm**2+sigma**4*norm**4*np.sin(theta)**2)**(-3/2)

    return term1 + term2 + term3

def lambda_3_elu(sigma, norm, theta):
    """
    Evaluate the lambda 1 at the value of s and norm, assuming sigma is at
    sigma star.
    """
    term1 = 1/(2*np.pi)*(np.pi-theta)

    rho = np.cos(theta)
    rv = multivariate_normal(mean = [0,0], cov=[[1, -rho], [-rho, 1]])
    term2 = 2*np.exp((sigma*norm)**2/2)*rv.cdf([-sigma*norm, sigma*norm*rho])

    rv = multivariate_normal(mean = [0,0], cov=[[1, rho], [rho, 1]])
    term3 = np.exp(0.5*(2*(sigma*norm)**2+2*(sigma*norm)**2*rho))*rv.cdf(\
            [-sigma*norm-sigma*norm*rho, -sigma*norm-sigma*norm*rho])
    
    return sigma**2*(term1+term2+term3)


# This code does two things:
# (1) Draw sigma star against ||x||
# (2) Plot the lambda_3 against theta for each activation function
sigma_star_fig = None
for mode in ['ELU', 'GELU', 'ReLU']:
    if mode == 'ELU':
        k_class = lambda std: NNKernelElu(1, std**2, 0, 0, 0, 1)
    elif mode == 'GELU':
        k_class = lambda std: NNKernelGelu(1, std**2, 0, 0, 0, 1)
    elif mode == 'ReLU':
        k_class = lambda std: NNKernelRelu(1, std**2, 0, 0, 0, 1)

    k_fun = lambda std, norm: k_class(std).\
            K(np.asarray([[norm]]), np.asarray([[norm]]))
    ## NOTE: we divide by variance because output of k multiplies by another
    ## variance for a linear output layer
    g = lambda std, norm: k_fun(std, norm)/(std**2) - norm**2

    # Plot sigma star
    sigma_star_fig = sigma_star_plot(g, mode, sigma_star_fig)

    # Plot lambda_3
    norm_list = [0.1, 0.5, 1, 2, 5]
    theta_list = np.linspace(0.001, np.pi-0.001, 500)
    plt.figure(figsize=(7,7))
    ylabel = r'$\lambda_3$'
    for norm in norm_list:
        # Find the value of sigma that preserves this norm
        s = norm_preserving_s(g, norm)
        sigma = s/norm
        print(mode)
        print(norm)
        print(sigma)

        # Now caluclate lambda_3 over the interval (0, pi)
        lambda_list = np.zeros_like(theta_list)
        for i, theta in enumerate(theta_list):
            if mode == 'ELU':
                lambda_3 = lambda_3_elu(sigma, norm, theta)
            elif mode == 'GELU':
                lambda_3 = lambda_3_gelu(sigma, norm, theta)
                ylabel = r'$\lambda_3$ lower bound'
            elif mode == 'ReLU':
                lambda_3 = (np.pi - theta)/np.pi
            lambda_list[i] = lambda_3
        plt.plot(theta_list, lambda_list, 
                label=r'$\Vert \mathbf{x} \Vert = ' + str(norm) + '$')
    plt.ylabel(ylabel, fontsize=40)
    plt.xlabel(r'$\theta$', fontsize=40)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 30)
    plt.legend(prop={'size':20})
    plt.plot(theta_list, np.ones_like(theta_list), 'k--')
    plt.savefig('code/dynamics/lambda_' + mode + '.pdf')
    plt.close()
