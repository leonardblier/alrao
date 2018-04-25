import sys
import subprocess
from sbatch import launch_exp

nb_expes = 1 # number of experiments per set of parameters

runOpt = {'env_name': 'pytorch', # name of the environment to be activated
          'use_slurm': False, # if True, executes the script automatically with sbatch
          'interactive': False,
          'command': 'python', # 'python', 'ipython -i'
          'script': 'main.py',
          'temp_file': 'temp_run.sh',
          'keep_temp_file': False}


# List of options added when launching 'py_file_name'
argsDict = {'epochs': 1,
            'size_multiplier': 1,

            'optimizer': 'SGD',
            'lr': .001,
            'use_switch': False,
            'minLR': -5,
            'maxLR': 0,
            'nb_class': 1,
            'data_augm': True,
#            'batchnorm': 'BN',

#            'penalty': True,
#            'dropOut': 0, # def: 0

            'suffix': '', # def: ''
            'exp_number': -1, # def: -1
            'stats': False}

# List of slurm options
sbatchOpt = ['--job-name=mixed_lr',
             '--gres=gpu:1',
             '--time=2-00:00:00',
             '-n1',
             '--no-kill',
             '--error=AAA-err-%j.txt',
             '--output=AAA-out-%j.txt',
             '-C pascal'] #,
#             '--exclude=titanic-2']

# Exp Test
launch_exp(runOpt, sbatchOpt, argsDict)

# Send the tasks
"""
gridDict = {'lr': [.0001, .001, .01, .1]}
for argsDict['lr'] in gridDict['lr']:
    if nb_expes > 1:
        for i in range(nb_expes):
            argsDict['exp_number'] = i
            launch_exp(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict)
    else:
        launch_exp(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict)
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
