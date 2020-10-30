from ...kernels.nn_kernel_elu import NNKernelElu
from ...kernels.nn_kernel_lrelu import NNKernelLRelu
from ...kernels.nn_kernel_gelu import NNKernelGelu

from .gp_mlp import gp_model
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rc
import numpy as np
from scipy import signal

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

types = ['relu', 'gelu']
NUM_TRAINING    = 20
NUM_TESTING     = 100
NOISE           = 0.1 # var of iid Gaussian noise
DEPTH           = 100
NORM            = 0.5
NUM_SAMPLES     = 10

viridis = cm.get_cmap('viridis', DEPTH)

f1 = lambda x: np.sin(x)
f2 = lambda x: signal.sawtooth(x)*2 + 5
f3 = lambda x: x**3 - 4
f4 = lambda x: np.sinc(x)
f5 = lambda x: np.exp(np.abs(x-np.pi))
f6 = lambda x: np.tan(x)

f_list = [f1, f2, f3, f4, f5, f6]

for f_count, f in enumerate(f_list):
    train_mse_list = np.zeros((DEPTH-1,NUM_SAMPLES,len(types)))
    test_mse_list = np.zeros((DEPTH-1,NUM_SAMPLES,len(types)))
    for s in range(NUM_SAMPLES):
        theta_train = 2*np.pi*np.random.uniform(0, 1, (NUM_TRAINING, 1)).reshape((-1,1))
        x_train = np.hstack([np.cos(theta_train), np.sin(theta_train)])*NORM

        y_train = (f(theta_train) + np.sqrt(NOISE)\
            *np.random.normal(0,1, theta_train.shape))\
            .reshape((-1,1))

        theta_test = np.linspace(0, 2*np.pi, NUM_TESTING).reshape((-1,1))
        x_test = np.hstack([np.cos(theta_test), np.sin(theta_test)])*NORM
        y_test = (f(theta_test) + np.sqrt(NOISE)\
            *np.random.normal(0,1, theta_test.shape))\
            .reshape((-1,1))

        for act_count, act in enumerate(types):
            print(act)

            plt.figure(figsize=(7,7))
            for l in range(1, DEPTH):
                if act == 'relu':
                    wvar = 2
                else:
                    wvar = 1.5906**2
                gp = gp_model(act, NOISE, b_0_var=0., w_0_var=wvar,depth=l)
                mean, var = gp.run_inference(x_train, y_train, x_test)
                mean_train, _ = gp.run_inference(x_train, y_train, x_train)
                plt.plot(theta_test, mean, color=viridis(float(l)/DEPTH), lw=2)
                plt.savefig(str(f_count) + act + '.pdf')

                test_mse_list[l-1, s, act_count] = np.average( (mean - y_test)**2)
                train_mse_list[l-1, s, act_count] = np.average( (mean_train - y_train)**2)

            # Plot the toy plots
            plt.scatter(theta_train, y_train, zorder=10, c='k')
            plt.xlabel(r'$\gamma$', fontsize=40)
            plt.xlim([0, 2*np.pi])
            plt.ylim([np.amin(y_train)*1.1, np.amax(y_train)*1.1])
            ax = plt.gca()
            ax.tick_params(axis = 'both', which = 'major', labelsize = 30)
            plt.savefig(str(f_count) + act + '.pdf')
            plt.close()

    for act_count, act in enumerate(types):
        # Plot training and testing error
        mean_train = np.average(train_mse_list[:,:,act_count], axis=1)
        std_train = np.std(train_mse_list[:,:,act_count], axis=1)
        plt.plot(np.asarray(range(1,DEPTH)), mean_train, 'g')
        plt.fill_between(np.asarray(range(1,DEPTH)), mean_train+2*std_train, 
                mean_train-2*std_train, color='g', alpha=0.3)

        mean_test = np.average(test_mse_list[:,:,act_count], axis=1)
        std_test = np.std(test_mse_list[:,:,act_count], axis=1)
        plt.plot(np.asarray(range(1,DEPTH)), mean_test, 'b')
        plt.fill_between(np.asarray(range(1,DEPTH)), mean_test+2*std_test, 
                mean_test-2*std_test, color='b', alpha=0.3)

        plt.legend(['Train', 'Test'], fontsize=20, loc='upper left')
        plt.xlabel(r'$L$', fontsize=30)
        plt.ylabel(r'MSE', fontsize=30)
        plt.ylim([0,0.6])
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 20)
        plt.savefig(str(f_count) + act + 'train_test.pdf')
        plt.close()

