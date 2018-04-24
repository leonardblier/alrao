import sys
import subprocess

# General definitions
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

def build_sbatch(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict):
    f_sbatch = open(temp_file_name, 'w')
    f_sbatch.write('#!/bin/bash\n\n')

    for opt in sbatchOpt:
        f_sbatch.write('#SBATCH ' + opt + '\n')
    f_sbatch.write('\n')

    f_sbatch.write('source activate ' + env_name + '\n')
    f_sbatch.write('python ' + py_file_name + build_args(argsDict) + '\n')
    f_sbatch.write('source deactivate ' + env_name + '\n')

    f_sbatch.close()

def launch_exp(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict):
    build_sbatch(py_file_name, env_name, temp_file_name, sbatchOpt, argsDict)
    bashCMD = 'sbatch ' + temp_file_name
    proc = subprocess.Popen(bashCMD.split(), stdout = subprocess.PIPE)
    output, error = proc.communicate()

    if not output is None:
        print(str(output, 'utf-8').strip())
    if not error is None:
        print(str(error, 'utf-8').strip(), file = sys.stderr)
