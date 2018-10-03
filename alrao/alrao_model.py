import torch
import torch.nn as nn
from .switch import Switch


class AlraoModel(nn.Module):
    r"""
    AlraoModel is the class transforming a pre-classifier into a model learnable with Alrao.

    Arguments:
        preclassifier: pre-classifier preclassifier to train.
                        It is an entire network, given without its last layer.
        nclassifiers: number of classifiers to use with the model averaging method
        nclasses: number of classes in the classification task
        classifier_gen: python class to use to construct the classifiers
        *args, **kwargs: arguments to be passed to the constructor of 'classifier_gen'
    """
    def __init__(self, preclassifier, nclassifiers, classifier_gen, *args, **kwargs):
        super(AlraoModel, self).__init__()
        self.switch = Switch(nclassifiers, save_cl_perf=True)
        self.preclassifier = preclassifier
        self.nclassifiers = nclassifiers

        for i in range(nclassifiers):
            classifier = classifier_gen(*args, **kwargs)
            setattr(self, "classifier" + str(i), classifier)

        self.last_x, self.last_lst_logpx = None, None

    def method_fwd_preclassifier(self, method_name_src, method_name_dst=None):
        r"""
        Allows the user to call directly a method of the pre-classifier.

        Creates a new method for the called instance of AlraoModel named 'method_name_dst'.
        Calling this method is exactly equivalent to calling the method of
        'self.preclassifier' named 'method_name_src'.
        If 'method_name_dst' is left to 'None', 'method_name_dst' is set to 'method_name_src'.

        Example:
            am = AlraoModel(precl, ncl, cl_gen)
            am.method_fwd_preclassifier('some_method')
            # call 'some_method' by the usual way
            am.preclassifier.some_method(some_args)
            # call 'some_method' using forwarding
            am.some_method(som_args)

        Arguments:
            method_name_src: name of the method of the pre-classifier to bind
            method_name_dst: name of the method to call
        """
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_dst, None) is None, \
            'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        method = getattr(self.preclassifier, method_name_src)
        def forwarded_method(*args, **kwargs):
            return method(*args, **kwargs)
        forwarded_method.__doc__ = method.__doc__
        forwarded_method.__name__ = method_name_dst
        setattr(self, forwarded_method.__name__, forwarded_method)

    def method_fwd_classifiers(self, method_name_src, method_name_dst=None):
        r"""
        Allows the user to call directly a method of the classifiers.

        Creates a new method for the called instance of AlraoModel named 'method_name_dst'.
        Calling this method is exactly equivalent to calling the method of the classifiers named 'method_name_src'
            on each classifier.
        If 'method_name_dst' is left to 'None', 'method_name_dst' is set to 'method_name_src'.

        Example:
            am = AlraoModel(precl, ncl, cl_gen)
            am.method_fwd_classifiers('some_method')
            # call 'some_method' by the usual way
            for cl in am.classifiers():
                cl.some_method(some_args)
            # call 'some_method' using forwarding
            am.some_method(som_args)

        Arguments:
            method_name_src: name of the method of the classifiers to bind
            method_name_dst: name of the method to call
        """
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_src, None) is None, \
            'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        lst_methods = [getattr(cl, method_name_src) for cl in self.classifiers()]

        def forwarded_method(*args, **kwargs):
            return [method(*args, **kwargs) for method in lst_methods]

        forwarded_method.__doc__ = lst_methods[0].__doc__
        forwarded_method.__name__ = method_name_dst
        setattr(self, forwarded_method.__name__, forwarded_method)

    def reset_parameters(self):
        """
        Resets both the classifiers and preclassifier's parameters.
        """
        self.preclassifier.reset_parameters()
        for cl in self.classifiers():
            cl.reset_parameters()

    def forward(self, *args, **kwargs):
        r"""
        Gives an input to the pre-classifier, then gives its output to each classifier,
            averages their output with 'switch', a model averaging method.

        The output 'x' of the pre-classifier is either a scalar or a tuple:
            - 'x' is a scalar: 'x' is used as input of each classifier
            - 'x' is a tuple: 'x[0]' is used as input of each classifier

        Arguments:
            *args, **kwargs: arguments to be passed to the forward method of the pre classifier
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
        if not weak_cl:
            return

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
        """
        Iterator over the preclassifier parameters
        """
        return self.preclassifier.parameters()

    def classifiers(self):
        """
        Iterator over the classifiers
        """
        for i in range(self.nclassifiers):
            yield getattr(self, "classifier"+str(i))

    def classifiers_parameters_list(self):
        """
        List of iterators, each one over a classifier parameters list.
        """
        return [cl.parameters() for cl in self.classifiers()]

    def posterior(self):
        """
        Return the switch posterior over the classifiers
        """
        return self.switch.logposterior.exp()

    def classifiers_predictions(self, x=None):
        """
        Return all the predictions, for each classifier.
        If x is None, the last predictions are returned.
        """
        if x is None:
            return self.last_lst_logpx
        x = self.preclassifier(x)
        lst_px = [cl(x) for cl in self.classifiers()]
        self.last_lst_logpx = lst_px
        return lst_px

    def repr_posterior(self):
        """
        Compact string representation of the posterior
        """
        post = self.posterior()
        bars = u' ▁▂▃▄▅▆▇█'
        res = "|"+"".join(bars[int(px)] for px in post/post.max() * 8) + "|"
        return res
