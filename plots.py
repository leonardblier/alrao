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
            #raise ValueError
            continue


        expname = re.sub('Data_|_exp-.|.txt','', filename)
        #expname = re.sub('Data_|.txt','', filename)
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


def load_switch_weights(directory='ResultsSwitch/', prefix='switch_weights_e', ncl=10):
    postlist = []
    listfile = []
    for filename in os.listdir(directory):
        if filename.startswith(prefix):
            listfile.append(directory+filename)
    listfile = sorted(listfile, key=lambda fn: int(fn.split('_')[-1].split('.')[0][1:]))
    postlist = [[] for _ in range(ncl)]
    for filename in listfile:
        f = open(filename, 'r')
        for l in f:
            for cl, p in zip(postlist, l.split(';')):
                cl.append(float(p))
    return postlist


def plot_switch_weights(postlist, namefile='switchplot.eps'):
    y = np.vstack(postlist)
    x = np.arange(len(postlist[0])) * 32. / 40000

    # labels = ["Fibonacci ", "Evens", "Odds"]

    fig, ax = plt.subplots(figsize=(10.,4.))

    cnorm = matplotlib.colors.Normalize(0., 1.)
    cm = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='viridis')
    minlr, maxlr = 10 ** (-5), 10
    classifiers_lr = [np.exp(\
        np.log(minlr) + k / 9 * (np.log(maxlr) - np.log(minlr)) \
        ) for k in range(10)]

    im = ax.stackplot(x, y,
        colors=[cm.to_rgba(u/9) for u in range(10)],
        labels=["$a_{{{}}}$ : $\eta_{{{}}}$ = {:.1e}".format(
            u+1, u+1, classifiers_lr[u]) for u in range(10)])
    ax.set_xscale('log')
    ax.set_xlabel('Epochs (log-scale)')
    ax.legend()
    # ax.legend(loc='upper left')

    ax.set_ylim(0., 1.)
    ax.set_ylabel("Model averaging weights $a_j$")
    ax.set_xlim(0., len(postlist[0]) * 32 // 40000)
    fig.tight_layout()
    plt.savefig(namefile, format="eps")



def results(explist, latex=False):
    mykeys = ["lvalid","avalid","ltest","atest"]
    for exp in explist:

        bestmet = dict((k, []) for k in mykeys)
        for k in ['tcv', 'epcv', 'tbest', 'epbest']:
            bestmet[k] = []
        bestmet["epcv"] = []
        exp["averagedmet"] = {}
        for exprun in exp['metrics']:
            bestepoch = min(enumerate(exprun["lvalid"]), key=lambda x:x[1])[0]
            for k in mykeys:
                bestmet[k].append(exprun[k][bestepoch])
            bestmet['tcv'].append(exprun['t'][-1])
            bestmet['epcv'].append(len(exprun['t']))
            bestmet['tbest'].append(exprun['t'][bestepoch])
            bestmet['epbest'].append(bestepoch)

        for k in mykeys + ['tcv', 'epcv', 'tbest', 'epbest']:
            values = np.array(bestmet[k])
            mean, std = values.mean(), values.std()
            #print("{}\t {} ± {}".format(k, mean, std))
            exp["averagedmet"][k] = (mean,std)


    modelnameslist = set(exp["modelname"] for exp in explist)
    #opt = ["SGD"]
    opt = ["SGD", "Adam"]

    for modelname, opt in product(modelnameslist, opt):
        if not latex:
            print("Experiments {} with {}".format(modelname, opt))
            subexplist = [exp for exp in explist if exp["modelname"] == modelname and exp["opt"] == opt]

            if any(not exp["alrao"] for exp in subexplist):
                print("Best lr")
                expbestlr = min([exp for exp in subexplist if not exp["alrao"]],
                                key = lambda exp:exp["averagedmet"]["lvalid"][0])

                if opt == "Adam" and any(exp['lr'] == 1.e-3 for exp in subexplist):
                    print("1e-3")
                    print_exp(next(exp for exp in subexplist if exp['lr'] == 1.e-3))
                print_exp(expbestlr)

            alraoexp = [exp for exp in subexplist if exp["alrao"]]
            if len(alraoexp) != 0:
                for exp in alraoexp:
                    print_exp(exp)

            print(' ')
        else:
            textlist = [modelname, opt]
            subexplist = [exp for exp in explist if exp["modelname"] == modelname and exp["opt"] == opt]

            if any(not exp["alrao"] for exp in subexplist):
                expbestlr = min([exp for exp in subexplist if not exp["alrao"]],
                                key = lambda exp:exp["averagedmet"]["lvalid"][0])



                textlist.append(print_exp(expbestlr, True))
            else:
                textlist.append('& &')

            alraoexp = [exp for exp in subexplist if exp["alrao"]]
            if len(alraoexp) != 0:
                textlist.append(print_exp(alraoexp[0], True))

            print(' & '.join(textlist) + '\\\\')




def print_exp(exp, latex=False):
    textlist = []
    if not latex:
        if exp['alrao']:
            textlist.append("Alrao 1e{} -> 1e{}".format(exp['minlr'], exp['maxlr']))
        else:
            textlist.append("lr:{:.1e}".format(exp['lr']))

        for k in ["lvalid","avalid","ltest","atest"]:
            textlist.append("{}: {:.4f} +- {:.4f}".format(k, *exp["averagedmet"][k]))
        textlist.append('\n')
        for k in ['tcv', 'epcv', 'tbest', 'epbest']:
            textlist.append("{}: {:.0f} +- {:.0f}".format(k, *exp["averagedmet"][k]))

        print('\t'.join(textlist))
    else:

        if not exp['alrao']:
            textlist.append("$10^{{ {} }}$".format(int(np.log10(exp['lr']))))


        textlist.append("${:.2f} \\pm {:.2f}$".format(*exp["averagedmet"]["ltest"]))
        atest, stdatest = exp["averagedmet"]["atest"]
        textlist.append("${:.1f} \\pm {:.1f}$".format(100*atest, 100*stdatest))
        return ' & '.join(textlist)



def plot_triangle(explist, modelname, opt, epoch=1000, namefile='triangle.eps'):

    fig, axes = plt.subplots(1,2, figsize=(10.,4.))
    subexplist = [exp for exp in explist if exp["modelname"] == modelname and exp["opt"] == opt]

    MINLR = min([int(np.log10(exp['lr'])) for exp in subexplist if not exp['alrao']] + \
                [exp['minlr']for exp in subexplist if exp['alrao']])
    MAXLR = max([int(np.log10(exp['lr'])) for exp in subexplist if not exp['alrao']] + \
                [exp['maxlr']for exp in subexplist if exp['alrao']])


    for title, ax in zip(["ltrain","ltest"], axes.flat):
        titledict = {'a': "Accuracy", 'l': "Loss"}
        #ax.set_title(titledict[title[0]] + ' ' + title[1:])

        met2d = np.zeros((MAXLR - MINLR + 1,MAXLR - MINLR + 1))
        met2d[:,:] = None
        for exp in subexplist:
            print("BEWARE!!!")
            # mean = np.mean([exprun[title][min(len(exprun[title]) - 1, epoch)] for exprun in exp['metrics']])
            mean = [exprun[title][min(len(exprun[title]) - 1, epoch)] for exprun in exp['metrics']][0]
            # if mean > 5. or np.isnan(mean):
            #     mean = None
            if exp['alrao']:
                minlr, maxlr = exp['minlr'], exp['maxlr']
                if minlr >= maxlr:
                    continue

                met2d[minlr-MINLR, maxlr-MINLR] = mean
            else:
                lr = int(np.log10(exp['lr']))
                print(lr, exp['lr'])
                met2d[lr - MINLR, lr - MINLR] = mean
        #cmap = 'viridis'

        cnorm = matplotlib.colors.Normalize(vmin=0., vmax=3.)
        cmap = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='viridis_r').cmap
        # if title[0] == 'a':
        #     cmap = matplotlib.cm.viridis
        # else:
        #     cmap = matplotlib.cm.viridis_r
        cmap.set_bad('white',None)
        cmap.set_over('red')

        vmin, vmax = None, 3.
        im = ax.imshow(np.clip(met2d,0., 3.), cmap=cmap, origin='upper')
        fig.colorbar(im, ax=ax)
        ax.set_xticks(np.arange(MAXLR - MINLR + 1))
        ax.set_yticks(np.arange(MAXLR - MINLR + 1))
        ax.set_xticklabels(['1e{}'.format(x+MINLR) for x in range(10)], rotation = 45)
        ax.set_yticklabels(['1e{}'.format(x+MINLR) for x in range(10)])
        ax.tick_params(labelsize=10, top=True, bottom=False,
                    labeltop=True, labelbottom=False)
        ax.set_xlabel('Maximum learning rate $\eta_\max$')
        ax.xaxis.set_label_position('top')

        ax.set_ylabel('Minimum learning rate $\eta_\min$')
        ax.legend()
    print(met2d)
    plt.tight_layout()
    plt.savefig(namefile, format="eps")


def plot_learning_curves(explist, namefile='learningcurves.eps'):
    minidx, maxidx = 0, 50


    fig, axes = plt.subplots(1,2, figsize=(10.,4.))

    lrlist = [np.log(exp['lr']) for exp in explist if not exp['alrao']]
    if len(lrlist) > 0:
        cnorm = matplotlib.colors.Normalize(vmin=min(lrlist), vmax=max(lrlist))
        cm = matplotlib.cm.ScalarMappable(norm=cnorm, cmap='plasma')


    for title, ax in zip(["ltrain","ltest"], axes.flat):
        titledict = {'a': "Accuracy", 'l': "Loss"}
        ax.set_title(titledict[title[0]] + ' ' + title[1:])


        for exp in sorted(explist, key=lambda d: (d["opt"], d['lr'])):
            if exp['alrao']:
                label = 'alrao'.format(exp["minlr"], exp["maxlr"])
                c = 'K'
                # c = None
                alpha=1.
                linestyle='--'
                # linestyle='-'
                linewidth=1.
            elif exp["opt"] == 'Adam':
                label = 'Adam default'
                c = 'r'
                # c = None
                alpha=1.
                linestyle='--'
                # linestyle='-'
                linewidth=1.
            else:
                label='lr={:.0e}'.format(exp['lr'])
                c = cm.to_rgba(np.log(exp['lr']))
                alpha=.1
                # linestyle='-'
                linestyle='-'
                linewidth=1.

            metlist = [[]  for _ in range(max(len(exprun[title]) for exprun in exp['metrics']))]
            mean = []
            std = []


            for exprun in exp['metrics']:
                for i, x in enumerate(exprun[title]):
                    metlist[i].append(x)
                # ax.plot(exprun[title])

            for metep in metlist:
                metep = np.array(metep)
                mean.append(metep.mean())
                std.append(metep.std())


            #ax.errorbar(list(range(len(mean[minidx:maxidx]))), mean[minidx:maxidx], yerr=std[minidx:maxidx],  label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)
            ax.plot(mean[minidx:maxidx], label=label, c=c, alpha=alpha, linestyle=linestyle, linewidth=linewidth)

        if title[0] == 'l':
            ax.set_ylim(0., 2.4)
            ax.set_ylabel("loss")
        else:
            ax.set_ylim(0., 1.)
            ax.set_ylabel("accuracy")
        ax.set_xlim(minidx, maxidx)
        ax.set_xlabel("epochs")

        if "train" in title:
            ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(namefile, format="eps")

# postlist = load_switch_weights()
# plot_switch_weights(postlist)

# explist = load_data_files('./ResultsTriangle/')
# #explist = [exp for exp in explist if exp['suffix'] == 'triangle']
# plot_triangle(explist, 'GoogLeNet', 'SGD', epoch=30, namefile='triangle.eps')


explist = load_data_files('FinalResults/')

explist = [exp for exp in explist if exp["modelname"] != "SENet18"]
#explist = load_data_files('./')
results(explist, latex=False)

# subexplist = [exp for exp in explist if exp["modelname"] == "VGG19" and exp["opt"] == "Adam" and exp["alrao"]]
subexplist = [exp for exp in explist if exp["modelname"] == "MobileNetV2" and (exp["opt"] == "SGD" or (not exp["alrao"] and exp["lr"] == 1.e-3))]
plot_learning_curves(subexplist, namefile='totalmobile.eps')



# results(explist, latex=True)
#
#
# modelnameslist = set(exp["modelname"] for exp in explist)
# opt = ["SGD", "Adam"]
#
# for modelname, opt in product(modelnameslist, opt):
#     subexplist = [exp for exp in explist if exp["modelname"] == modelname and exp["opt"] == opt]
#     if len(subexplist) == 0:
#         continue
#     plot_learning_curves(subexplist, namefile='learningcurves_{}_{}.eps'.format(modelname, opt))
#
#print("Len explist: {}".format(len(explist)))
