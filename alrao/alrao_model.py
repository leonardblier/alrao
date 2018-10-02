import torch
import torch.nn as nn
import torch.optim as optim
from switch import Switch


class AlraoModel(nn.Module):
    r"""
    AlraoModel is the class transforming a pre-classifier into a model learnable with Alrao.

    Arguments:
        preclassifier: pre-classifier preclassifier to train. It is an entire network, given without its last layer.
        nclassifiers: number of classifiers to use with the model averaging method
        nclasses: number of classes in the classification task
        classifier_gen: python class to use to construct the classifiers
    """
    def __init__(self, preclassifier, nclassifiers, classifier_gen, *args, **kwargs):
        super(AlraoModel, self).__init__()
        self.switch = Switch(nclassifiers, save_cl_perf=True)
        self.preclassifier = preclassifier
        self.nclassifiers = nclassifiers

        for i in range(nclassifiers):
            classifier = classifier_gen(*args, **kwargs)
            setattr(self, "classifier" + str(i), classifier)

    def method_fwd_preclassifier(self, method_name_src, method_name_dst = None):
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_dst, None) is None, \
            'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        method = getattr(self.preclassifier, method_name_src)
        def f(*args, **kwargs):
            return method(*args, **kwargs)
        f.__doc__ = method.__doc__
        f.__name__ = method_name_dst
        setattr(self, f.__name__, f)

    def method_fwd_classifiers(self, method_name_src, method_name_dst = None):
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_src, None) is None, \
            'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        lst_methods = [getattr(cl, method_name_src) for cl in self.classifiers()]
        

        def f(*args, **kwargs):
            return [method(*args, **kwargs) for method in lst_methods]

        f.__doc__ = lst_methods[0].__doc__
        f.__name__ = method_name_dst
        setattr(self, f.__name__, f)

    def reset_parameters(self):
        self.preclassifier.reset_parameters()
        for cl in self.classifiers():
            cl.reset_parameters()

    def forward(self, *args, **kwargs):
        r"""
        input x -> x = self.preclassifier(x)
        either 'x' is a scalar or a tuple:
            - 'x' is a scalar: 'x' is used as input of each classifier
            - 'x' is a tuple: 'x[0]' is used as input of each classifier
        """
        x = self.preclassifier(*args, **kwargs)

        z = x
        if isinstance(z, tuple):
            z = x[0]
        
        lst_logpx = [cl(z) for cl in self.classifiers()]
        self.last_x, self.last_lst_logpx = z, lst_logpx
        out = self.switch.forward(lst_logpx)
        
        if isinstance(x, tuple):
            out = (out,) + x[1:]
        return out

    def update_switch(self, y, x=None, catch_up=False):
        """
        Updates the model averaging weights
        """
        if x is None:
            lst_px = self.last_lst_logpx
        else:
            lst_px = [cl(x) for cl in self.classifiers()]
        self.switch.Supdate(lst_px, y)

        if catch_up:
            self.hard_catch_up()

    def hard_catch_up(self, threshold=-20):
        """
        The hard catch up allows to reset all the classifiers with low performance, and to
        set their weights to the best classifier ones. This can be done periodically during learning
        """
        logpost = self.switch.logposterior
        weak_cl = [cl for cl, lp in zip(self.classifiers(), logpost) if lp < threshold]
        if len(weak_cl) == 0:
            return None

        mean_weight = torch.stack(
            [cl.fc.weight * p for (cl, p) in zip(self.classifiers(), logpost.exp())],
            dim=-1).sum(dim=-1).detach()
        mean_bias = torch.stack(
            [cl.fc.bias * p for (cl, p) in zip(self.classifiers(), logpost.exp())],
            dim=-1).sum(dim=-1).detach()
        for cl in weak_cl:
            cl.fc.weight.data = mean_weight.clone()
            cl.fc.bias.data = mean_bias.clone()

    def parameters_preclassifier(self):
        return self.preclassifier.parameters()

    def classifiers(self):
        for i in range(self.nclassifiers):
            yield getattr(self, "classifier"+str(i))

    def classifiers_parameters_list(self):
        return [cl.parameters() for cl in self.classifiers()]

    def posterior(self):
        return self.switch.logposterior.exp()

    def classifiers_predictions(self, x=None):
        if x is None:
            return self.last_lst_logpx
        x = self.preclassifier(x)
        lst_px = [cl(x) for cl in self.classifiers()]
        self.last_lst_logpx = lst_px
        return lst_px

    def repr_posterior(self):
        post = self.posterior()
        bars = u' ▁▂▃▄▅▆▇█'
        res = "|"+"".join(bars[int(px)] for px in post/post.max() * 8) + "|"
        return res

    def print_norm_parameters(self):
        print("Pre-Classifier: {:.0e}".format(\
            sum(float(p.norm()) for p in self.parameters_preclassifier())))
        for (i, c) in enumerate(self.classifiers()):
            print("Classifier {}: {:.0e}".format(i,
                sum(float(p.norm()) for p in c.parameters())))

