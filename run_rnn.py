import sys
import subprocess
from sbatch import launch_exp

# Options
py_file_name = 'pytorch_cifar.py'
interactive = True

if not interactive:
    runOpt = {'env_name': 'pytorch', # name of the environment to be activated
              'use_slurm': True, # if True, read the slurm options
              'interactive': False, # must be True if you run an interactive job
              # if use_slurm is True and interactive is False,
              #     run the script with sbatch
              'command': 'python', # 'python', 'ipython -i'
              'script': 'main_rnn.py',
              'temp_file': 'temp_run.sh',
              'keep_temp_file': False}
else:
   runOpt = {'env_name': 'pytorch', # name of the environment to be activated
              'use_slurm': True, # if True, read the slurm options
              'interactive': True, # must be True if you run an interactive job
              # if use_slurm is True and interactive is False,
              #     run the script with sbatch
              'command': 'python',
              'script': 'main_rnn.py',
              'temp_file': 'temp_run.sh',
              'keep_temp_file': True}

argsDict = {'epochs': 10,
            'early_stopping': True,

            'dropOut': .2,

            'rnn_type': 'LSTM',
            'rnn_data_path': './ptb', # './ptb' './wikitext-2-raw'
            'rnn_nlayers': 2,

            'rnn_emsize': 100,
            'rnn_nhid': 100,
            'rnn_batch_size': 20,
            'rnn_bptt': 70,

            'use_switch': True,
            'nb_class': 6,
            'optimizer': 'SGD',
            'minLR': -3,
            'maxLR': 2,
            'lr': .001,
            'momentum': 0,
            'rnn_clip': .25,

            'suffix': 'test'}

sbatchOpt = ['--job-name=rLR_RNN_test',
             '--gres=gpu:1',
             '--time=2-00:00:00',
             '-n1',
             '--no-kill',
             '--error=AAA-err-%j.txt',
             '--output=AAA-out-%j.txt',
             '-C pascal'] #,
#             '--exclude=titanic-2']

temp_file_name = 'temp_run.sh'

# Launch experiment
launch_exp(runOpt, sbatchOpt, argsDict)
