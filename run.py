import sys
import subprocess
from sbatch import launch_exp

# Options
py_file_name = 'main.py'
nb_expes = 1
argsDict = {'epochs': 1,
            'size_multiplier': 1,

#            'optimizer': 'Adam',
            'lr': .005, # def: .01
            'minLR': -5,
            'maxLR': 0,
            'nb_class': 11,
            'data_augm': True,
#            'batchnorm': 'BN',

#            'penalty': True,
#            'dropOut': 0, # def: 0

            'suffix': '', # def: ''
            'exp_number': -1, # def: -1
            'stats': False}

sbatchOpt = ['--job-name=mixed_lr',
             '--gres=gpu:1',
             '--time=2-00:00:00',
             '-n1',
             '--no-kill',
             '--error=AAA-err-%j.txt',
             '--output=AAA-out-%j.txt',
             '-C pascal'] #,
#             '--exclude=titanic-2']

temp_file_name = 'temp_run.sh'

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
