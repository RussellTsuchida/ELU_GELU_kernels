# External libraries
import numpy as np
import sys
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pickle

# this is to read in pickle results file and plot some stuff
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 10})
plt.rcParams['text.usetex'] = True

is_save = 1
which = 'RMSE' # Set to RMSE or NLL

file_name = 'code//experiments//outputs//benchmarks//reg_results_01_24_2020_17_13_26.p'
save_name = 'allexps_finegrid_' # what to call addition of saved figures

results_list = pickle.load(open(file_name, 'rb'))

# Convert to dictionaries
dict_results_rmse={}
dict_results_nll={}
for result_exp in results_list:
    dict_results_rmse[result_exp[0] + '_' + result_exp[1][1:]] = result_exp[-1][1]
    dict_results_nll[result_exp[0] + '_' + result_exp[1][1:]] = result_exp[-1][2]

# List of activations and datasets
acts_all = ['gelu','relu','Lrelu','erf']
datasets_all = ['boston','concrete','energy','kin8','naval','power','protein','wine','yacht']

# Pretty names
datasets_polished = {'boston':'Boston',
                    'concrete': 'Concrete',
                    'energy': 'Energy',
                    'kin8': 'Kin8nm',
                    'naval': 'Naval',
                    'power': 'Power',
                    'protein': 'Protein',
                    'wine': 'Wine',
                    'yacht':'Yacht'}
acts_polished = {   'gelu':'GELU',
                    'relu':'ReLU',
                    'erf':'ERF',
                    'Lrelu':'L. ReLU',
                    'prelu':'PReLU'}
colours = {'gelu':'r','relu':'b','erf':'g','Lrelu':'k','prelu':'gray'}

if which == 'NLL':
    dict_in = dict_results_nll
elif which == 'RMSE':
    dict_in = dict_results_rmse

# Individual plots for each dataset
for i, dataset in enumerate(datasets_all):
    fig = plt.figure(figsize=(1.5, 3)) # originally (1.5, 4)
    ax = fig.add_subplot(111)
    ax.set_xticks([])
    ax.set_xticklabels([])
    for j, act in enumerate(acts_all):
        mean = dict_in[act+'_'+dataset][1]
        std_err = dict_in[act+'_'+dataset][2]
        ax.plot(j, mean, colours[act], marker='_',markersize=15,linewidth=0.)
        ax.plot([j,j], [mean+2*std_err,mean-2*std_err], colours[act], 
                marker=None,linewidth=2.0,label=act)
        ax.plot(j, mean+2*std_err, colours[act], marker='_',markersize=10,
                linewidth=0.)
        ax.plot(j, mean-2*std_err, colours[act], marker='_',markersize=10,
                linewidth=0.)
        ax.text(x=j/(len(acts_all))+0.07,y=-0.05,s=acts_polished[act],
                rotation=90,transform=ax.transAxes)

    ax.set_title(datasets_polished[dataset] + ', '+which)
    ax.set_xlim(-0.5,len(acts_all)-1+0.5)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2f'))

    fig.show()
    if is_save:
        fig.savefig('experiments//outputs//benchmarks//01_bench_graphs//' + \
            save_name+which+'_'+dataset+'.pdf',format='pdf', dpi=500, 
            bbox_inches='tight')

# one big plot aggregating all datasets
fig = plt.figure(figsize=(5, 5)) # originally (1.5, 4)
ax = fig.add_subplot(111)
ax.set_xticks([])
ax.set_xticklabels([])
for i, dataset in enumerate(datasets_all):
    for j, act in enumerate(acts_all):
        mean = dict_in[act+'_'+dataset][1]
        std_err = dict_in[act+'_'+dataset][2]
        ax.plot(i*len(acts_all)*1.5+j, mean, colours[act], marker='_',
                markersize=15,linewidth=0.)
        ax.plot([i*len(acts_all)*1.5+j,i*len(acts_all)*1.5+j], 
                [mean+2*std_err,mean-2*std_err], colours[act], marker=None,
                linewidth=2.0,label=act)
        ax.plot(i*len(acts_all)*1.5+j, mean+2*std_err, colours[act], 
                marker='_',markersize=10,linewidth=0.)
        ax.plot(i*len(acts_all)*1.5+j, mean-2*std_err, colours[act], 
                marker='_',markersize=10,linewidth=0.)
        # ax.text(x=j-0.2,y=0.,s=act,rotation=90)
    # ax.text(x=i,y=0.,s=datasets_polished[dataset],rotation=90)

    ax.text(x=i/len(datasets_all) + 0.3/len(datasets_all),y=-0.05,
            s=datasets_polished[dataset],rotation=90,transform=ax.transAxes)
fig.show()

if is_save:
    fig.savefig('experiments//outputs//benchmarks//01_bench_graphs//overview_'+\
            save_name+which+'.pdf',format='pdf', dpi=500, bbox_inches='tight')

# convert to latex formatting
for j, act in enumerate(acts_all):
    if j==0:
        string_exps = acts_polished[act] 
    else:
        string_exps = string_exps + ' & ' + acts_polished[act]
print(string_exps) # header

dict_in = dict_results_rmse # dict_results_nll dict_results_rmse
# dict_in = dict_results_nll # dict_results_nll dict_results_rmse
for i, dataset in enumerate(datasets_all):
    string_exps = datasets_polished[dataset]
    for j, act in enumerate(acts_all):
        # mean = np.around(dict_in[act+'_'+dataset][1],decimals=2)
        mean = '%.2f' %dict_in[act+'_'+dataset][1]
        # std_err = np.around(dict_in[act+'_'+dataset][2],decimals=2)
        std_err = '%.2f' %dict_in[act+'_'+dataset][2]
        string_exps = string_exps + ' & ' + str(mean) + ' $\pm$ ' + \
                str(std_err) + ''
    print(string_exps + ' \\\\') # header

print()

# plot both nll and rmse
for i, dataset in enumerate(datasets_all):
    string_exps = datasets_polished[dataset]
    for dict_in in [dict_results_nll,dict_results_rmse]:
        for j, act in enumerate(acts_all):
            mean = '%.2f' %dict_in[act+'_'+dataset][1]
            std_err = '%.2f' %dict_in[act+'_'+dataset][2]
            string_exps = string_exps + ' & ' + str(mean) + ' $\pm$ ' + \
                    str(std_err) + ''
    print(string_exps + ' \\\\') # header



