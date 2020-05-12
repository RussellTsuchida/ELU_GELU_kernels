# -*- coding: utf-8 -*-
###### this script produces banchmark results for single layer NN GPs ########
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
import datetime
import pickle

from .utils import *
from .DataGen import *
from .module_gp_gelu import *

start_time = datetime.datetime.now()
print_w_time('started')

# avoid the dreaded type 3 fonts...
plt.rcParams['pdf.fonttype']    = 42
plt.rcParams['ps.fonttype']     = 42
plt.rcParams['text.usetex']     = True
plt.rcParams.update({'font.size': 10})

np.random.seed(101)

# -- inputs --
train_prop      = 0.9        # test/train split 
is_save_results = 1         # save to pickle file

# -- redundant inputs --
n_samples = 12                # only used for synthetic datasets
u_var = 1                    # var for rbf params as -> inf, goes to stationary cov dist
g_var = 1                    # var for rbf params

if is_save_results:
    # create results file
    filename_save = 'code//experiments//outputs//benchmarks//reg_results_' + \
            datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + '.p'
    print('saving at ', filename_save)


for activation_fn in ['gelu','relu','erf','Lrelu']:
    print_w_time(str('\n\n// - working on '+activation_fn+ ' - //\n\n'))
    for data_set in ['~boston']:#,'~concrete','~energy','~kin8','~naval','~power', 
            #'~protein','~wine','~yacht']:
        print_w_time(str('\n\n// - working on '+data_set+ ' - //\n\n'))

        if data_set=='~protein':
            n_runs = 5
        else:
            n_runs = 20

        gp_results=[]; ens_results=[]; single_results=[]; mc_results=[]; 
        hmc_results=[]; sk_results=[]; unc_ens_results=[]; 
        run_kls=[]
        for run_ in range(n_runs):
            print('  ====== run:',run_, '======', end='\r')

            # -- create data --
            Gen = DataGenerator(type_in=data_set)
            X_train, y_train, X_val, y_val = Gen.CreateData\
                (n_samples=n_samples, seed_in=run_+1000,train_prop=train_prop)
            n_samples = X_train.shape[0]
            X_dim = X_train.shape[1]
            y_dim = 1

            # for larger datasets when running GP only use 10k rows to train
            if data_set=='~protein':
                X_train = X_train[0:10000]
                y_train = y_train[0:10000]

            # really slow to do hyperparam tuning every run, only do on first one
            if run_ == 0:
                # do hyperparam tuning
                n_hyp = int(X_train.shape[0]*7/9) # need 70% total train for hyperparam tuning
                X_train_hyp = X_train[:n_hyp]
                y_train_hyp = y_train[:n_hyp]
                X_val_hyp = X_train[n_hyp:]
                y_val_hyp = y_train[n_hyp:]

                if data_set in ['~yacht', '~energy', '~naval', '~wine']:
                    # save some time as I know these datasets are low noise from
                    # previous runs
                    data_noise_vals=[0.001,0.0001,0.00001]
                else:
                    # higher data noise
                    data_noise_vals=[1.0,0.1,0.01]
                b_0_var_vals=[10,3.5,3.0,2.5,2.0,1.5,1.0,0.5,0.1]
                #b_0_var_vals = [1.0, 2.0]
                w_0_var_vals=b_0_var_vals
                w_1_var_vals=b_0_var_vals

                results=[]
                best_nll = 1e9 # init very bad
                best_rmse = 1e9
                best_margll = -1e9
                nll_results = np.zeros((len(data_noise_vals),len(b_0_var_vals)))
                rmse_results = np.zeros_like(nll_results)
                margll_results = np.zeros_like(nll_results)
                # Grid over noise, hidden weights and biases, and output weights
                for i, data_noise in enumerate(data_noise_vals):
                    for j, b_0_var in enumerate(b_0_var_vals):
                        for jw, w_0_var in enumerate(w_0_var_vals):
                            for k, w_1_var in enumerate(w_1_var_vals):
                                # -- gp model --
                                gp = gp_model\
                                    (kernel_type=activation_fn, 
                                    data_noise=data_noise, 
                                    b_0_var=b_0_var, 
                                    w_0_var=w_0_var, 
                                    w_1_var=w_1_var, 
                                    u_var=5., 
                                    g_var=1.)
                                y_pred_mu, y_pred_std = gp.run_inference\
                                        (x_train=X_train_hyp, 
                                        y_train=y_train_hyp,
                                        x_predict=X_val_hyp, 
                                        print=0)

                                metrics_calc(y_val_hyp, y_pred_mu, y_pred_std, 
                                    Gen.scale_c, b_0_var, w_0_var, data_noise, 
                                    gp, is_print=False)

                                nll_results[i,j]    = gp.nll
                                rmse_results[i,j]   = gp.rmse
                                margll_results[i,j] = gp.marg_log_like

                                if gp.nll < best_nll:
                                    best_nll_params = [ b_0_var, w_0_var, 
                                                        w_1_var, data_noise]
                                    best_nll = gp.nll.copy()
                                if gp.rmse < best_rmse:
                                    best_rmse_params = [b_0_var, w_0_var, 
                                                        w_1_var, data_noise]
                                    best_rmse = gp.rmse.copy()
                                if gp.marg_log_like > best_margll:
                                    best_margll_params = [  b_0_var, w_0_var, 
                                                            w_1_var, data_noise]
                                    best_margll = gp.marg_log_like.copy()

                                results.append(np.array((gp.mse_unnorm, 
                                                        gp.rmse, 
                                                        gp.nll, 
                                                        gp.marg_log_like)))
                results = np.array(results)

                # choose best hyperparam setting according to nll or margll
                best_params = best_nll_params


                b_0_var_tuned = best_params[0]
                w_0_var_tuned = best_params[1]
                w_1_var_tuned = best_params[2]
                data_noise_tuned = best_params[3]

            # -- gp model --
            # do actual testing
            gp = gp_model\
                (kernel_type=activation_fn, data_noise=data_noise_tuned, 
                b_0_var=b_0_var_tuned, w_0_var=w_0_var_tuned,  
                w_1_var=w_1_var_tuned, u_var=5., g_var=1.)
            y_pred_mu, y_pred_std = gp.run_inference\
                (x_train=X_train, y_train=y_train, x_predict=X_val, print=False)

            metrics_calc(y_val, y_pred_mu, y_pred_std, Gen.scale_c, 
                    b_0_var_tuned, w_0_var_tuned, data_noise_tuned, gp, 
                    is_print=False)
            gp_results.append(np.array((gp.mse_unnorm, gp.rmse, gp.nll)))

        gp_results = np.array(gp_results)
        print('\n\n___ GP RESULTS ___')
        print('data', data_set, ', act_fn', activation_fn, ', b_0_var', 
                b_0_var_tuned, ', w_0_var', w_0_var_tuned,', w_1_var', 
                w_1_var_tuned, 'd_noise',data_noise_tuned)
        metric_names= ['MSE_un','RMSE', 'NLL']
        print('runs')
        print(n_runs)
        print('\tavg\tstd_err\tstd_dev')
        metrics_save=[] # to be pickled
        for i in range(0,len(metric_names)): 
            avg = np.mean(gp_results[:,i])
            std_dev = np.std(gp_results[:,i], ddof=1)
            std_err = std_dev/np.sqrt(n_runs)
            print(metric_names[i], '\t', round(avg,3), 
                '\t', round(std_err,3),
                '\t', round(std_dev,3))
            metrics_save.append([metric_names[i],avg,std_err])

        if is_save_results:
            try:
                curr_results = pickle.load(open(filename_save, 'rb'))
            except (OSError, IOError) as e:
                curr_results = []
                pickle.dump(curr_results, open(filename_save, 'wb'))

            curr_results.append\
                    ([activation_fn, data_set, 'nll', gp_results, metrics_save])
            pickle.dump(curr_results, open(filename_save, 'wb'))

# -- tidy up --
if is_save_results:
    print('saved at ', filename_save)
print_w_time('finished')
end_time = datetime.datetime.now()
total_time = end_time - start_time
print('seconds taken:', round(total_time.total_seconds(),1),
    '\nstart_time:', start_time.strftime('%H:%M:%S'), 
    'end_time:', end_time.strftime('%H:%M:%S'))
