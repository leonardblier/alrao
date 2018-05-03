import os
import sys
import subprocess
from pathlib import Path

# Utility
def getLastDot(U_str):
    for i in reversed(range(len(U_str))):
        if U_str[i] == '.':
            return i
    return -1

def createTempFileName(temp_file_name):
    i_dot = getLastDot(temp_file_name)
    if i_dot == -1:
        temp_type = ''
    else:
        l = len(temp_file_name)
        temp_type = temp_file_name[(i_dot - l):]
        ret_file_name = temp_file_name[:i_dot]
    temp_number = 0
    temp_suffix = ''
    while Path(ret_file_name + temp_suffix + temp_type).is_file():
        temp_number += 1
        temp_suffix = '_' + repr(temp_number)
    ret_file_name += temp_suffix + temp_type

    return ret_file_name

def send_proc(bashCMD):
    proc = subprocess.Popen(bashCMD.split(), stdout = subprocess.PIPE)
    output, error = proc.communicate()

    if not output is None:
        print(str(output, 'utf-8').strip())
    if not error is None:
        print(str(error, 'utf-8').strip(), file = sys.stderr)

# Script command
def build_args(argsDict):
    lstArgs = ''
    for name, key in argsDict.items():
        if isinstance(key, bool):
            if key:
                lstArgs += ' --' + name
        elif isinstance(key, str):
            if key != '':
                lstArgs += ' --' + name + '=' + key
        else:
            lstArgs += ' --' + name + '=' + repr(key)
    return lstArgs

def build_script_command(runOpt, argsDict):
    str_args = build_args(argsDict)
    if 'ipython' in runOpt['command']:
        str_args = ' --' + str_args
    return runOpt['command'] + ' ' + runOpt['script'] + str_args


# General definitions
def build_bash_part(f_sbatch, runOpt, argsDict):
    f_sbatch.write('#TEMPORARY_FILE\n')
    if runOpt['keep_temp_file']: f_sbatch.write('#KEEP_FILE\n')
    f_sbatch.write('\n')

    f_sbatch.write('source activate ' + runOpt['env_name'] + '\n')
    f_sbatch.write(build_script_command(runOpt, argsDict) + '\n')
    f_sbatch.write('source deactivate ' + runOpt['env_name'] + '\n')
    f_sbatch.write('\n')

    if not (runOpt['use_slurm'] and not runOpt['interactive']):
        f_sbatch.write('echo "#USED_FILE" >> $0\n')
        if not runOpt['keep_temp_file']: f_sbatch.write('rm $0\n')

def build_sbatch(runOpt, sbatchOpt, argsDict):
    runOpt['temp_file'] = createTempFileName(runOpt['temp_file'])

    f_sbatch = open(runOpt['temp_file'], 'w')
    f_sbatch.write('#!/bin/bash\n\n')

    # Slurm options
    for opt in sbatchOpt:
        f_sbatch.write('#SBATCH ' + opt + '\n')
    f_sbatch.write('\n')

    # Bash part
    build_bash_part(f_sbatch, runOpt, argsDict)

    f_sbatch.close()

def build_srun(runOpt, sbatchOpt, argsDict):
    # Slurm options
    command_srun = 'srun'
    for opt in sbatchOpt:
        if ('--output' not in opt) and ('--error' not in opt):
            command_srun += ' ' + opt
    command_srun += ' --pty bash'

    # Bash part
    runOpt['temp_file'] = createTempFileName(runOpt['temp_file'])

    f_sbatch = open(runOpt['temp_file'], 'w')
    f_sbatch.write('#!/bin/bash\n\n')

    build_bash_part(f_sbatch, runOpt, argsDict)

    f_sbatch.close()
    return command_srun

def launch_exp(runOpt, sbatchOpt, argsDict):
    if runOpt['use_slurm']:
        if runOpt['interactive']:
            command_srun = build_srun(runOpt, sbatchOpt, argsDict)
            print('run: ' + command_srun)
            print('run: bash ' + runOpt['temp_file'])
        else:
            build_sbatch(runOpt, sbatchOpt, argsDict)
            send_proc('sbatch ' + runOpt['temp_file'])
            if not runOpt['keep_temp_file']:
                send_proc('rm ' + runOpt['temp_file'])
            else:
                f_sbatch = open(runOpt['temp_file'], 'a')
                f_sbatch.write('#USED_FILE\n')
                f_sbatch.close()
    else:
        runOpt['interactive'] = True
        command_srun = build_srun(runOpt, sbatchOpt, argsDict)
        print('bash ' + runOpt['temp_file'])
