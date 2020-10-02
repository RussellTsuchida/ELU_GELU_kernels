import numpy as np

from .utils import *
from ...kernels.nn_kernel_relu import NNKernelRelu
from ...kernels.nn_kernel_gelu import NNKernelGelu
from ...kernels.nn_kernel_lrelu import NNKernelLRelu
from ...kernels.nn_kernel_elu import NNKernelElu
from ...kernels.nn_kernel_erf import NNKernelErf

class gp_model:
    def __init__(self, kernel_type, data_noise, b_0_var=1., w_0_var=1., depth=1):
        self.kernel_type = kernel_type
        self.data_noise = data_noise
        self.name_ = 'GP'

        # variance for step fn, relu, erf
        self.b_0_var = b_0_var # first layer bias variance        
        self.w_0_var = w_0_var # first layer weight variance

        # place holders
        self.mse_unnorm = 0.
        self.rmse = 0.
        self.nll = 0.

        self.depth = depth
        return

    def run_inference(self, x_train, y_train, x_predict, print_=False):
        ''' this is why we're here - do inference '''

        if self.kernel_type == 'relu' or self.kernel_type == 'softplus':
            kernel = NNKernelRelu
        elif self.kernel_type == 'Lrelu':
            kernel = NNKernelLRelu
        elif self.kernel_type == 'erf':
            kernel = NNKernelErf
        elif self.kernel_type == 'gelu':
            kernel = NNKernelGelu
        elif self.kernel_type == 'elu':
            kernel = NNKernelElu
        
        x_dim = x_train.shape[1]
        kern = kernel(x_dim, self.w_0_var, 0, 
                self.b_0_var, 0, L=self.depth)
        kernel_fn = lambda x1, x2: kern.K(x1, x2)

        # d is training data, x is test data
        if print_: print_w_time('beginning inference')
        cov_dd = kernel_fn(x_train, None) \
                + np.identity(x_train.shape[0])*self.data_noise

        if print_: print_w_time('compiled cov_dd')
        cov_xd = kernel_fn(x_predict, x_train)
        if print_: print_w_time('compiled cov_xd')
        cov_xx = kernel_fn(x_predict,x_predict)
        if print_: print_w_time('compiled cov_xx')

        # p 19 of Rasmussen & Williams
        L = np.linalg.cholesky(cov_dd)
        alpha = np.linalg.solve(L.T,np.linalg.solve(L,y_train))
        y_pred_mu = np.matmul(cov_xd,alpha)
        v = np.linalg.solve(L,cov_xd.T)
        cov_pred = cov_xx - np.matmul(v.T,v)

        y_pred_var = np.atleast_2d(np.diag(cov_pred) + self.data_noise).T
        y_pred_std = np.sqrt(y_pred_var)

        if print_: print_w_time('calculating log likelihood')
        marg_log_like = - np.matmul(y_train.T,alpha)/2 - \
                np.sum(np.log(np.diag(L))) - x_train.shape[0]*np.log(2*np.pi)/2

        if print_: print_w_time('matrix ops complete')

        self.cov_xx = cov_xx
        self.cov_dd = cov_dd
        self.cov_xd = cov_xd
        self.cov_xx = cov_xx
        self.cov_pred = cov_pred
        self.y_pred_mu = y_pred_mu
        self.y_pred_std = y_pred_std
        self.y_pred_var = y_pred_var
        self.x_train = x_train
        self.y_train = y_train
        self.x_predict = x_predict

        self.y_pred_mu = y_pred_mu
        self.y_pred_std = y_pred_std
        self.marg_log_like = marg_log_like

        return y_pred_mu, y_pred_std

    ############### Paper does not use any of the code below #################
    ##########################################################################

    def cov_visualise(self):
        ''' display heat map of cov matrix over 1-d input '''

        # plot cov matrix
        fig = plt.figure()
        plt.imshow(self.cov_xx, cmap='hot', interpolation='nearest')
        if self.kernel_type != 'rbf':
            title = self.kernel_type + ', cov matrix, b_0: ' + \
                    str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
        else:
            title = self.kernel_type + ', cov matrix, g_var: ' + \
                    str(self.g_var) + ', u_var: ' + str(self.u_var)
        plt.title(title)
        plt.colorbar()
        fig.show()

        return


    def priors_visualise(self, n_draws=10):
        # 1-D only, plot priors
        # we currently have data noise included in this, could remove it to get smooth

        print_w_time('getting priors')
        # get some priors
        y_samples_prior = np.random.multivariate_normal(
            np.zeros(self.x_predict.shape[0]), self.cov_xx, n_draws).T # mean, covariance, size

        # plot priors
        fig = plt.figure()
        plt.plot(self.x_predict, y_samples_prior, 'k',lw=0.5, label=u'Priors')
        plt.xlabel('$x$')
        plt.ylabel('$f(x)$')
        title = self.kernel_type + ', priors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
        plt.title(title)
        # plt.xlim(-6, 6)
        fig.show()

        return

    def priors_draw(self, n_draws=10):
        # 1-D only, plot priors
        # we currently have data noise included in this, could remove it to get smooth

        print_w_time('getting priors')
        # get some priors
        y_samples_prior = np.random.multivariate_normal(
            np.zeros(self.x_predict.shape[0]), self.cov_xx, n_draws).T # mean, covariance, size

        return y_samples_prior


    def posts_draw_visualise(self, n_draws=10, is_graph=True):
        # 1-D only, plot posteriors
        # we currently have data noise included in this, could remove it to get smooth

        # sample from posterior
        y_samples_post = np.random.multivariate_normal(
            self.y_pred_mu.ravel(), self.cov_pred, n_draws).T # mean, covariance, size

        # plot priors
        if is_graph:
            fig = plt.figure()
            plt.plot(self.x_predict, y_samples_post, color='k',alpha=0.5,lw=0.5, label=u'Priors')
            plt.plot(self.x_train, self.y_train, 'r.', markersize=14, label=u'Observations', markeredgecolor='k',markeredgewidth=0.5)
            plt.xlabel('$x$')
            plt.ylabel('$f(x)$')
            title = self.kernel_type + ', posteriors, b_0: ' + str(self.b_0_var) + ', w_0: ' + str(self.w_0_var)
            plt.title(title)
            # plt.xlim(-6, 6)
            fig.show()

        self.y_preds = y_samples_post.T

        y_pred_mu_draws = np.mean(self.y_preds,axis=0)
        y_pred_std_draws = np.std(self.y_preds,axis=0, ddof=1)

        # add on data noise
        # do need to add on for GP!
        y_pred_std_draws = np.sqrt(np.square(y_pred_std_draws) + self.data_noise)

        self.y_pred_mu_draws = y_pred_mu_draws
        self.y_pred_std_draws = y_pred_std_draws

        return





