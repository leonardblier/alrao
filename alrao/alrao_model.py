import torch
import torch.nn as nn
from .switch import Switch


class AlraoModel(nn.Module):
    r"""
    AlraoModel is the class transforming a internal NN into a model learnable with Alrao.

    Arguments:
        internal_nn: part of the neural network preceding the last layer.
        n_last_layers: number of parallel last layers to use with the model averaging method
        n_classes: number of classes in the classification task
        last_layer_gen: python class to use to construct the last layers
        task: either 'classification' or 'regression'
        loss: loss used in the model (subclass of pytorch's '_Loss')
            loss(output, target, size_average = False) returns the loss embedded into a 0-dim tensor
            the option 'size_average = True' returns the averaged loss
        *args, **kwargs: arguments to be passed to the constructor of 'last_layer_gen'
    """
    def __init__(self, internal_nn, n_last_layers, last_layer_gen, task, loss, *args, **kwargs):
        super(AlraoModel, self).__init__()
        self.task = task
        self.loss = loss
        self.switch = Switch(n_last_layers, save_ll_perf = True, task = task, loss = loss)
        self.internal_nn = internal_nn
        self.n_last_layers = n_last_layers

        for i in range(n_last_layers):
            last_layer = last_layer_gen(*args, **kwargs)
            setattr(self, "last_layer_" + str(i), last_layer)

        self.last_x, self.last_lst_logpx = None, None

    def method_fwd_internal_nn(self, method_name_src, method_name_dst = None):
        r"""
        Allows the user to call directly a method of the internal NN.

        Creates a new method for the called instance of AlraoModel named 'method_name_dst'.
        Calling this method is exactly equivalent to calling the method of
        'self.internal_nn' named 'method_name_src'.
        If 'method_name_dst' is left to 'None', 'method_name_dst' is set to 'method_name_src'.

        Example:
            am = AlraoModel(intnn, nll, ll_gen)
            am.method_fwd_internal_nn('some_method')
            # call 'some_method' by the usual way
            am.internal_nn.some_method(some_args)
            # call 'some_method' using forwarding
            am.some_method(som_args)

        Arguments:
            method_name_src: name of the method of the internal NN to bind
            method_name_dst: name of the method to call
        """
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_dst, None) is None, \
            'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        method = getattr(self.internal_nn, method_name_src)
        def forwarded_method(*args, **kwargs):
            return method(*args, **kwargs)
        forwarded_method.__doc__ = method.__doc__
        forwarded_method.__name__ = method_name_dst
        setattr(self, forwarded_method.__name__, forwarded_method)

    def method_fwd_last_layers(self, method_name_src, method_name_dst = None):
        r"""
        Allows the user to call directly a method of the last layers.

        Creates a new method for the called instance of AlraoModel named 'method_name_dst'.
        Calling this method is exactly equivalent to calling the method of the last layers named 'method_name_src'
            on each of them.
        If 'method_name_dst' is left to 'None', 'method_name_dst' is set to 'method_name_src'.

        Example:
            am = AlraoModel(intnn, nll, ll_gen)
            am.method_fwd_last_layers('some_method')
            # call 'some_method' by the usual way
            for ll in am.last_layers():
                ll.some_method(some_args)
            # call 'some_method' using forwarding
            am.some_method(som_args)

        Arguments:
            method_name_src: name of the method of the last layers to bind
            method_name_dst: name of the method to call
        """
        if method_name_dst is None:
            method_name_dst = method_name_src
        assert getattr(self, method_name_src, None) is None, \
            'The method {} cannot be forwarded: an attribute with the same name already exists.'.format(method_name_dst)
        lst_methods = [getattr(ll, method_name_src) for ll in self.last_layers()]

        def forwarded_method(*args, **kwargs):
            return [method(*args, **kwargs) for method in lst_methods]

        forwarded_method.__doc__ = lst_methods[0].__doc__
        forwarded_method.__name__ = method_name_dst
        setattr(self, forwarded_method.__name__, forwarded_method)

    def reset_parameters(self):
        """
        Resets both the last layers and internal NN's parameters.
        """
        self.internal_nn.reset_parameters()
        for ll in self.last_layers():
            ll.reset_parameters()

    def forward(self, *args, **kwargs):
        r"""
        Gives an input to the internal NN, then gives its output to each last layer,
            averages their output with 'switch', a model averaging method.

        The output 'x' of the internal NN is either a scalar or a tuple:
            - 'x' is a scalar: 'x' is used as input of each last layer
            - 'x' is a tuple: 'x[0]' is used as input of each last layer

        Arguments:
            *args, **kwargs: arguments to be passed to the forward method of the internal NN
        """
        x = self.internal_nn(*args, **kwargs)
        if not torch.isfinite(x).all():
            raise ValueError

        z = x
        if isinstance(z, tuple):
            z = x[0]

        lst_ll_out = [ll(z) for ll in self.last_layers()]
        self.last_x, self.last_lst_ll_out = z, lst_ll_out
        out = self.switch.forward(lst_ll_out)

        if isinstance(x, tuple):
            out = (out,) + x[1:]
        return out

    def update_switch(self, y, x = None, catch_up = False):
        """
        Updates the model averaging weights

        Arguments: 
            y: tensor of targets
            x: tensor of outputs of the internal NN
                if x is None, the stored outputs of the last layers are used
        """
        if x is None:
            lst_ll_out = self.last_lst_ll_out
        else:
            lst_ll_out = [ll(x) for ll in self.last_layers()]
        self.switch.Supdate(lst_ll_out, y)

        if catch_up:
            self.hard_catch_up()

    def hard_catch_up(self, threshold = -20):
        """
        The hard catch up allows to reset all the last layers with low performance, and to
        set their weights to the best last layer ones. This can be done periodically during learning
        """
        logpost = self.switch.logposterior
        weak_ll = [ll for ll, lp in zip(self.last_layers(), logpost) if lp < threshold]
        if not weak_ll:
            return

        mean_weight = torch.stack(
            [ll.fc.weight * p for (ll, p) in zip(self.last_layers(), logpost.exp())],
            dim=-1).sum(dim=-1).detach()
        mean_bias = torch.stack(
            [ll.fc.bias * p for (ll, p) in zip(self.last_layers(), logpost.exp())],
            dim=-1).sum(dim=-1).detach()
        for ll in weak_ll:
            ll.fc.weight.data = mean_weight.clone()
            ll.fc.bias.data = mean_bias.clone()

    def parameters_internal_nn(self):
        """
        Iterator over the internal NN parameters
        """
        return self.internal_nn.parameters()

    def last_layers(self):
        """
        Iterator over the last layers
        """
        for i in range(self.n_last_layers):
            yield getattr(self, "last_layer_" + str(i))

    def last_layers_parameters_list(self):
        """
        List of iterators, each one over a last layer parameters list.
        """
        return [ll.parameters() for ll in self.last_layers()]

    def posterior(self):
        """
        Return the switch posterior over the last_layers
        """
        return self.switch.logposterior.exp()

    def last_layers_predictions(self, x = None):
        """
        Return all the predictions, for each last layer.
        If x is None, the last predictions are returned.
        """
        if x is None:
            return self.last_lst_logpx
        x = self.internal_nn(x)
        lst_px = [ll(x) for ll in self.last_layers()]
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
