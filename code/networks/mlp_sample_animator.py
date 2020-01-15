import matplotlib.pyplot as plt
import matplotlib
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
from .np_mlp import Mlp
from ..kernels.nn_kernel_relu import NNKernelRelu

class MlpGPAnimator(object):
    def __init__(self, mlpgp):
        self.mlpgp = mlpgp

    def make_animation(self, dir_name, num_samples = 5, grid_res = 100, 
            num_timesteps = 100):
        """
        Make an animation of the GP prior. To do this, sample some draws from
        the GP prior, then "update" them by adding a new iid draw to each 
        sample. Normalise the resulting sum, and do this in a loop.

        Args:
            dir_name (str): directory name to save animation frames
        """
        x = np.linspace(-5, 5, grid_res)
        t = np.linspace(0, 2*np.pi, num_timesteps)
        data = np.transpose([np.tile(x, len(t)), np.repeat(t, len(x))])
        data = np.hstack((data[:,0].reshape((-1,1)), 
                np.hstack(( 5*np.cos(data[:,1]).reshape((-1,1)), 
                            5*np.sin(data[:,1]).reshape(-1,1)))))

        if type(self.mlpgp) is Mlp:
            data = data.T
            all_samples = self.mlpgp.sample_functions(data, n = num_samples)
        elif type(self.mlpgp) is NNKernelRelu:
            all_samples = self.mlpgp.sample_prior(data, num_samples)
        all_samples = all_samples.reshape((num_samples, len(t), len(x)))

        # make all_samples circular
        for t in range(num_timesteps):
            fig, ax = plt.subplots()
            for s in range(all_samples.shape[0]):
                sample = all_samples[s, t, :]
                ax.plot(x, sample)
            ax.axis('off')
            ax.set_ylim(-10, 10)
            plt.savefig(dir_name + '/animation_' + str(t) + '.pdf')
            plt.close()

if __name__ == '__main__':
    hidden_layers = 5
    activations = [lambda z: np.clip(z, 0, None)]*hidden_layers + [lambda z: z]
    layer_widths = [3] + [4]*hidden_layers + [1]

    model = Mlp(layer_widths, activations)
    model = NNKernelRelu(3, 2, 0, 2, 0, hidden_layers)

    animator = MlpGPAnimator(model)
    animator.make_animation('code/networks/animations')
