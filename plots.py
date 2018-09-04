import matplotlib.pyplot as plt
matplotlib.use('Agg')

import os
import re
import numpy as np
from collections import defaultdict
from itertools import chain

colors = ['r', 'b', 'g', 'm', 'y', 'k']

def load_data_files(directory='Results/'):
    listfile = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not re.match('Data.*', filename):
            raise ValueError


        expname = re.sub('Data_|_exp-.|.txt','', filename) 
        listfile[expname].append(filename)


    explist = []
    for expname in listfile:
        expdict = {}
        expnsplit = expname.split('_')
        if expnsplit[0] == 'da':
            expdict['da'] = True
            expnsplit = expnsplit[1:]
        elif expnsplit[0][-2:] == 'da':
            expdict['da'] = True
            expnsplit[0] = expnsplit[0][:-2]
        else:
            expdict['da'] = False

        expdict['modelname'] = expnsplit.pop(0)
        expdict['lr'] = float(expnsplit.pop(0).split('-')[1])

        if 'mLR' in expnsplit[0]:
            expdict['alrao'] = True
            expdict['lr'] = 0.
            
            mlr = expnsplit.pop(0).split('-')[1]
            minlr, maxlr = int(mlr[1]), int(mlr[3])
            minlr *= (1 if mlr[0] == 'p' else -1)
            maxlr *= (1 if mlr[2] == 'p' else -1)
            expdict['minlr'], expdict['maxlr'] = minlr, maxlr

            expdict['sw'] = int(expnsplit.pop(0).split('-')[1])
        else:
            expdict['alrao'] = False

        expdict['opt'] = expnsplit.pop(0)
        expdict['K'] = int(expnsplit.pop(0)[1])
        expdict['ep'] = expnsplit.pop(0)

        expdict['suffix'] = expnsplit[0]

        expdict['files'] = listfile[expname]

        explist.append(expdict)

    for expdict in explist:
        files = expdict['files']
        expdict['metrics'] = []
        
        dictmetrics = defaultdict(list)
        for fname in files:
            expdict['metrics'].append({'t':[], 'ltrain':[]. 'ltrain':[],
                'lvalid':[], 'avalid':[], 'ltest':[], 'atest':[]})
            with open(fname, 'r') as f:
                next(f)
                for l in f:
                    _, t, ltrain, atrain, lvalid, avalid, ltest, atest = [float(x) for x in l.split()]
                    expdict['metrics'][-1]['t'].append(t)
                    expdict['metrics'][-1]['ltrain'].append(ltrain)
                    expdict['metrics'][-1]['atrain'].append(atrain)
                    expdict['metrics'][-1]['lvalid'].append(ltrain)
                    expdict['metrics'][-1]['avalid'].append(atrain)
                    expdict['metrics'][-1]['ltest'].append(ltest)
                    expdict['metrics'][-1]['atest'].append(atest)
                    
    return explist


def plot_learning_curves(explist, namefile='learningcurves.eps'):
    minidx, maxidx = 0, 50

    fig, axes = plt.subplots(2, 2, figsize=(10.,8.))
    
    for title, ax in zip(["ltrain", "atrain", "ltest", "atest"], axes.flat):
        ax.set_title(title)
        #lrcolordict = dict((lr, 'C'+str(i)) for (i,lr) in enumerate(['0.0010', '0.0050', '0.0500', '0.0100', '0.1000', '0.0250']))
        #lrcolordict['mLR'] = 'k'
        for exp in sorted(explist, key=lambda d: d['lr']):
            if exp['alrao']:
                label = 'alrao'
            else:
                label='lr{:.0e}'.format(exp['lr']
            metlist = [[] for _ in range(max(len(exprun[title]))) for exprun in exp['metrics']]
            mean = []
            std = []
            
            for exprun in exp['metrics']:
                for i, x in enumerate(exprun[title]):
                    metlist[i].append(x)

            for i, metep in enumerate(metlist):
                metep = np.array(metep)
                mean.append(metep.mean())
                std.append(metep.std())

            
            ax.plot(mean[minidx:maxidx], label=m)
            
        ax.legend()
        plt.tight_layout()
        plt.savefig(namefile, format="eps")



explist = load_data_files()
plot_learning_curves(explist)
