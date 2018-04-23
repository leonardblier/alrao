import sys
import subprocess
from sbatch import launch_exp

py_file_name = 'main.py'
nb_expes = 1 # number of experiment per set of parameters

# List of options added when launching 'py_file_name'
argsDict = {'epochs': 1,
            'size_multiplier': 1,

#            'optimizer': 'Adam',
            'lr': .001, # def: .01
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

# Name of the temporary file to be launched with slurm (sbatch)
temp_file_name = 'temp_run.sh'

# Send the tasks
if nb_expes > 1:
    for i in range(nb_expes):
        argsDict['exp_number'] = i
        launch_exp(py_file_name, temp_file_name, sbatchOpt, argsDict)
else:
    launch_exp(py_file_name, temp_file_name, sbatchOpt, argsDict)


"""
gridDict = {'size_multiplier': [1, 2, 3],
            'nb_class': [1, 5, 10]}
for argsDict['size_multiplier'] in gridDict['size_multiplier']:
    for argsDict['nb_class'] in gridDict['nb_class']:
        if nb_expes > 1:
            for i in range(nb_expes):
                argsDict['exp_number'] = i
                launch_exp(py_file_name, temp_file_name, sbatchOpt, argsDict)
        else:
            launch_exp(py_file_name, temp_file_name, sbatchOpt, argsDict)
"""
