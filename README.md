# ELU_GELU_kernels
This code accompanies the paper 

    'Avoiding Kernel Fixed Points: Computing with ELU and GELU Infinite Networks'

A link to a permanent repository will be added to the paper upon acceptance.

## Installing dependencies
We recommend you use a virtual environment. Inside your environment, type
`pip install --user --requirement requirements.txt`

You might need to compile a piece of Fortran code into a Python module. If you
receive errors when trying to run the commands below, follow the directions
in `code/experiments/fortran_module/instructions.txt`

## Reproducing the results in the paper
Each of these commands should be run from the top directory, which is called 
`ELU_GELU_kernels` by default.

Figure 1 may be reproduced by typing
`python3 -m code.experiments.activation_plotter`

Figure 2 may be reproduced by typing 
`python3 -m code.experiments.empirical_kernels_gelu`

Figure 3 may be reproduced by typing
`python3 -m code.experiments.empirical_kernels_elu`

Figures 4 and 5 may be reproduced by typing
`python3 -m code.dynamics.dynamics`

Note: Figure 6 runs all experiments in series, and Figure 7 runs all 
experiments in parallel. It is possible to change this set-up based on what
you need.

Figure 6 may be reproduced by typing
`python3 -m code.experiments.02_shallow_experiments.main_exp_bench`
NOTE: This runs all the experiment IN SERIES. If desired, you may want to 
run these in parallel.

Figure 7 may be reproduced through a multi-step process. 
1. Edit lines 28 and 29 of
   `code/experiments/03_deep_experiments/grid_iteration.py` 
   <pre><code>
   rmse_data = ExperimentArray((32, 50), OUTPUT_DIR + kern_str+data+'/rmseX/')
   nll_data = ExperimentArray((32, 50), OUTPUT_DIR + kern_str+data+'/nllX/')
   </code></pre>
   replacing X with 1. 
2. Run 
   `./grid_loop.sh Boston GELU &`
   By default this is set up to run on a system using SLURM to schedule jobs.
   If you don't use SLURM, edit the file as needed.
3. Run the command above with every dataset and kernel
   combination (Boston, Concrete, Energy, Wine, Yacht) and (GELU, ReLU, LReLU,
   Yacht, Wine).
4. You have now run each combination through one data shuffle. The data is 
   stored in `code/experiments/outputs/deep_experiments`. To shuffle and
   run again, delete all the .npy files in `code/experiments/01_data/` and 
   go back to step 1, replacing X with 2, 3, ... etc.
5. Once you have enough data, edit lines 9 and 19 of 
   `code/experiments/03_deep_experiments/plot_performance` to reflect how much
   data you gathered, then run 
   `python3 -m code.experiments.03_deep_experiments.plot_performance` to
   aggregate all the data.
