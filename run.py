import sys
import subprocess
from sbatch import launch_exp

nb_expes = 1 # number of experiments per set of parameters

runOpt = {'env_name': 'test', # name of the environment to be activated
          'use_slurm': True, # if True, read the slurm options
          'interactive': False, # must be True if you run an interactive job
                               # if use_slurm is True and interactive is False,
                               #     run the script with sbatch
          'command': 'python', # 'python', 'ipython -i'
          'script': 'main.py',
          'temp_file': 'temp_run.sh',
          'keep_temp_file': False}

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
argsDict = {'epochs': 10000,
            'early_stopping': True,
#            'size_multiplier': 3,
            'model_name':'GoogLeNet',
            'optimizer': 'Adam',
#            'lr': .025,
            'use_switch': True,
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

lr_list = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1., 10.]
argsDict['size_multiplier'] = 1
argsDict['model_name'] = 'GoogLeNet'
argsDict['use_switch'] = False
for i in range(1):
    argsDict['exp_number'] = i
    for lr in lr_list:
        argsDict['lr'] = lr
        launch_exp(runOpt, sbatchOpt, argsDict)

            
minmaxlr = [(mini, maxi) for mini in range(-6,2) for maxi in range(mini,2)]
#minmaxlr = [(-6, maxi) for maxi in range(-6, 2)] + [(mini, 1) for mini in range(-5,2)]
# minmaxlr = [(mini, mini) for mini in range(-7,3)]
for i in range(1):
    argsDict['exp_number'] = i
    
    argsDict['lr'] = 0.025
    argsDict['use_switch'] = True

    for (minLR, maxLR) in minmaxlr:
        argsDict['minLR'] = minLR
        argsDict['maxLR'] = maxLR
        launch_exp(runOpt, sbatchOpt, argsDict)

            
    

        # argsDict['use_switch'] = False
        # argsDict['size_multiplier'] = 1
        # for lr in lr_list:
        #     argsDict['lr'] = lr
        #     for i in range(3):
        #         argsDict['exp_number'] = i
        #         launch_exp(runOpt, sbatchOpt, argsDict)
        
        # argsDict['use_switch'] = False
        # argsDict['size_multiplier'] = 3
        # for lr in lr_list:
        #     argsDict['lr'] = lr
        #     for i in range(3):
        #         argsDict['exp_number'] = i
        #         launch_exp(runOpt, sbatchOpt, argsDict)



            
# Send the tasks
"""
gridDict = {'lr': [.0001, .001, .01, .1]}
for argsDict['lr'] in gridDict['lr']:
    if nb_expes > 1:
        for i in range(nb_expes):
            argsDict['exp_number'] = i
            launch_exp(runOpt, sbatchOpt, argsDict)
    else:
        launch_exp(runOpt, sbatchOpt, argsDict)
"""

"""
gridDict = {'size_multiplier': [1, 2, 3],
            'nb_class': [1, 5, 10]}
for argsDict['size_multiplier'] in gridDict['size_multiplier']:
    for argsDict['nb_class'] in gridDict['nb_class']:
        if nb_expes > 1:
            for i in range(nb_expes):
                argsDict['exp_number'] = i
                launch_exp(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict)
        else:
            launch_exp(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict)
"""
