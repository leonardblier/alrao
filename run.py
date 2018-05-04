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
argsDict = {'epochs': 5,
            'size_multiplier': 1,
            'model_name':'GoogLeNet',
            'optimizer': 'SGD',
            'lr': .025,
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
#for model_name in ['GoogLeNet', 'MobileNetV2', 'SENet18', 'DPN92']:
for model_name in ['GoogLeNet']:
    argsDict['model_name'] = model_name
    for i in range(5):
        argsDict['exp_number'] = i
        launch_exp(runOpt, sbatchOpt, argsDict)

    
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
