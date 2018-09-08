import sys
import subprocess
from sbatch import launch_exp

nb_expes = 10 # number of experiments per set of parameters

interactive = False

if not interactive:
    runOpt = {'env_name': 'test', # name of the environment to be activated
              'use_slurm': True, # if True, read the slurm options
              'interactive': False, # must be True if you run an interactive job
              # if use_slurm is True and interactive is False,
              #     run the script with sbatch
              'command': 'python', # 'python', 'ipython -i'
              'script': 'main.py',
              'temp_file': 'temp_run.sh',
              'keep_temp_file': False}
else:
   runOpt = {'env_name': 'test', # name of the environment to be activated
              'use_slurm': True, # if True, read the slurm options
              'interactive': True, # must be True if you run an interactive job
              # if use_slurm is True and interactive is False,
              #     run the script with sbatch
              'command': 'ipython -i',
              'script': 'main.py',
              'temp_file': 'temp_run.sh',
              'keep_temp_file': True} 
"""
Notes:
  1) if your job is interactive, it is NOT launched automatically; this program
     outputs the commands you have to run
  2) to launch a grid of experiments, you must set use_slurm=True and
     interactive=False
  3) if you already run the command 'srun <...> --pty bash', you must set
     use_slurm=False
  4) if you set 'use_slurm' to False, your job is supposed to be interactive
"""

# List of options added when launching 'py_file_name'
argsDict = {'epochs': 1000,
            'early_stopping': True,
#            'size_multiplier': 3,
            'model_name': 'GoogLeNet',#'SENet18'
#            'optimizer': 'Adam',
#            'lr': .025,
#            'use_switch': True,
#            'minLR': -5,
#            'maxLR': 1,
            'nb_class': 10,
            'data_augm': True,
#            'batchnorm': 'BN',

#            'penalty': True,
#            'dropOut': 0, # def: 0

            'suffix': '',#'sqrtmomentum',#, # def: ''
            'exp_number': -1, # def: -1
            'stats': False}


# List of slurm options
sbatchOpt = ['--job-name=mixed_lr',
             '--gres=gpu:1',
             '--time=2-00:00:00',
             '-n1',
#             '--no-kill',
             '--error=AAA-err-%j.txt',
             '--output=AAA-out-%j.txt',
#             '-w titanic-2']
             '-C pascal'] #,
#             '--exclude=titanic-2']


# Exp Test
#lr_list = [1e-6, 1e-5, 1e-4, 0.1, 0.5, 1., 10.]
#lr_list_orig = [.025, .05, .01, .005, .001]

#MINI, MAXI = -6, 2

argsDict['size_multiplier'] = 1
#argsDict['model_name'] = 'GoogLeNet'


# MINI, MAXI = -6, 0
# #lr_list = [10 ** k for k in range(MINI, MAXI + 1)]

# lr_list = [10 ** (-k) for k in range(2,6)]
# minmaxlr = [(MINI, MAXI)]
    
# for i in range(1, nb_expes):
#     argsDict['exp_number'] = i
#     argsDict['use_switch'] = False
        
#     for lr in lr_list:
#         argsDict['lr'] = lr
#         launch_exp(runOpt, sbatchOpt, argsDict)

# for i in range(nb_expes):
#     argsDict['use_switch'] = True
#     for (minLR, maxLR) in minmaxlr:
#         argsDict['minLR'] = minLR
#         argsDict['maxLR'] = maxLR
#         launch_exp(runOpt, sbatchOpt, argsDict)


for model_name in ['VGG19', 'SENet18', "MobileNetV2"]:
    argsDict['model_name'] = model_name
    #

    #  Adam
    argsDict['optimizer'] = 'Adam'
    MINI, MAXI = -6, 0
    lr_list = [10 ** k for k in range(MINI, MAXI + 1)]
    minmaxlr = [(MINI, MAXI)]
    
    for i in range(1):
        argsDict['exp_number'] = i
        argsDict['use_switch'] = False
        
        for lr in lr_list:
            argsDict['lr'] = lr
            launch_exp(runOpt, sbatchOpt, argsDict)

    for i in range(nb_expes):
        argsDict['use_switch'] = True
        for (minLR, maxLR) in minmaxlr:
            argsDict['minLR'] = minLR
            argsDict['maxLR'] = maxLR
            launch_exp(runOpt, sbatchOpt, argsDict)


    #  SGD
    argsDict['optimizer'] = 'SGD'
    MINI, MAXI = -5, 1
    lr_list = [10 ** k for k in range(MINI, MAXI + 1)]
    minmaxlr = [(MINI, MAXI)]
    
    for i in range(1):
        argsDict['exp_number'] = i
        argsDict['use_switch'] = False
        
        for lr in lr_list:
            argsDict['lr'] = lr
            launch_exp(runOpt, sbatchOpt, argsDict)

    for i in range(nb_expes):
        argsDict['use_switch'] = True
        for (minLR, maxLR) in minmaxlr:
            argsDict['minLR'] = minLR
            argsDict['maxLR'] = maxLR
            launch_exp(runOpt, sbatchOpt, argsDict)

