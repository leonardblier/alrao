import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import os
import re
import numpy as np
from collections import defaultdict
from itertools import chain, product

colors = ['r', 'b', 'g', 'm', 'y', 'k']

def load_data_files(directory='Results/'):
    listfile = defaultdict(list)
    
    for filename in os.listdir(directory):
        if not re.match('Data.*', filename):
            raise ValueError


        expname = re.sub('Data_|_exp-.|.txt','', filename)
        #print("filename:{}\t expname:{}".format(filename, expname))
        listfile[expname].append(filename)

    print("Expname: {}".format(list(listfile.keys())))
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
        expdict['lr'] = float(expnsplit.pop(0)[3:])

        if 'mLR' in expnsplit[0]:
            expdict['alrao'] = True
            expdict['lr'] = 0.
            
            mlr = expnsplit.pop(0).split('-')[1]
            minlr, maxlr = int(mlr[1]), int(mlr[3])
            minlr *= (1 if mlr[0] == 'p' else -1)
            maxlr *= (1 if mlr[2] == 'p' else -1)
            expdict['minlr'], expdict['maxlr'] = minlr, maxlr
            expdict['sw'] = int(expnsplit.pop(0)[3:])
        else:
            expdict['alrao'] = False

        expdict['opt'] = expnsplit.pop(0)
        expdict['K'] = int(expnsplit.pop(0)[1])
        expdict['ep'] = expnsplit.pop(0)

        if len(expnsplit) > 0:
            expdict['suffix'] = expnsplit[0]
        else:
            expdict['suffix'] = ''

        expdict['files'] = listfile[expname]

        explist.append(expdict)

    for expdict in explist:
        files = expdict['files']
        expdict['metrics'] = []
        
        dictmetrics = defaultdict(list)
        for fname in files:
            expdict['metrics'].append({'t':[]})
            for tvt, m in product(['l', 'a'], ['train', 'valid', 'test']):
                expdict['metrics'][-1][tvt+m] = []
            
            with open(os.path.join(directory, fname), 'r') as f:
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


def results(explist):
    
    for exp in explist:
        
        
        mykeys = ["lvalid","avalid","ltest","atest"]
        bestmet = dict((k, []) for k in mykeys)
        exp["averagedmet"] = {}
        for exprun in exp['metrics']:
            bestepoch = min(enumerate(exprun["lvalid"]), key=lambda x:x[1])[0]
            for k in mykeys:
                bestmet[k].append(exprun[k][bestepoch])

        for k in mykeys:
            values = np.array(bestmet[k])
            mean, std = values.mean(), values.std()
            #print("{}\t {} ± {}".format(k, mean, std))
            exp["averagedmet"][k] = (mean,std)

    print('----------')
    for exp in sorted(explist, key=lambda exp:exp["averagedmet"]["ltest"][0]):
        print(exp["files"][0])
        for k in mykeys:
            mean, std = exp["averagedmet"][k]
            print("{}\t {} ± {}".format(k, mean, std))

        print('--')
        
        


def plot_learning_curves(explist, namefile='learningcurves.eps'):
    minidx, maxidx = 0, 100

    
    fig, axes = plt.subplots(2, 2, figsize=(12.,10.))
    
        
    for title, ax in zip(["ltrain", "atrain", "ltest", "atest"], axes.flat):
        ax.set_title(title)
        #lrcolordict = dict((lr, 'C'+str(i)) for (i,lr) in enumerate(['0.0010', '0.0050', '0.0500', '0.0100', '0.1000', '0.0250']))
        #lrcolordict['mLR'] = 'k'
        for exp in sorted(explist, key=lambda d: d['lr']):
            if exp['alrao']:
                label = 'alrao'
                c = 'K'
                alpha=1.
                linestyle='-'
                linewidth=1.
            else:
                label='lr={:.0e}'.format(exp['lr'])
                c = None
                alpha=.1
                linestyle='-'
                linewidth=1.
                
            metlist = [[]  for _ in range(max(len(exprun[title]) for exprun in exp['metrics']))]
            mean = []
            std = []

            
            for exprun in exp['metrics']:
                for i, x in enumerate(exprun[title]):
                    metlist[i].append(x)

            for i, metep in enumerate(metlist):
                metep = np.array(metep)
                mean.append(metep.mean())
                std.append(metep.std())


            ax.errorbar(list(range(len(mean[minidx:maxidx]))), mean[minidx:maxidx], yerr=std[minidx:maxidx],  label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            #ax.plot(mean[minidx:maxidx], label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)

        if title[0] == 'l':
            ax.set_ylim(0., 3.)
        else:
            ax.set_ylim(0., 1.)
        ax.legend()
        plt.tight_layout()
        plt.savefig(namefile, format="eps")



explist = load_data_files('FinalResults/')
results(explist)
plot_learning_curves(explist)
