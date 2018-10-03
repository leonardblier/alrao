# Learning with Random Learning Rates

Authors' implementation of "Learning with Random Learning Rates" (2018) in PyTorch (0.4 and higher).

Paper: https://arxiv.org/abs/1810.01322.

Authors: Léonard Blier, Pierre Wolinski, Yann Ollivier.

## Feedforward neural networks

TODO

## Recurrent Neural Networks

TODO

## Features

### Provided models

pre-classifiers: VGG, GoogLeNet, LSTM, ... (TODO).

classifiers: linear classifier (feddforward), linear classifier (RNN).

### Custom models

An user can use its custom model with some modifications.

First, the custom model must be split into a pre-classifier class and a classifier class in order to be integrated into the class `AlraoModel`. Note that the classifier is supposed to return log-probabilities.

Then the `forward` method of the pre-classifier is assumed to return either one value or a tuple. If a tuple `(x, a, b, ...)` is returned, the first element `x` is supposed to be taken as input of the classifiers. Their outputs are then averaged with a model averaging method (here, the `Switch` class), which returns `y`. Thus, the `forward` method of `AlraoModel` returns a tuple `(y, a, b, ...)`.

### Method forwarding

To make Alrao easier to integrate in a given project, method forwarding is provided. Suppose a model class named `Model` has a method `f`, which is regularly called in a code with `x, y = some_model.f(a, b)`. This model has just to be processed as indicated above, and: 
```
some_model = Model(...)
...
x, y = some_model.f(a, b)
```
to be replaced by:
```
some_model = AlraoModel(...)
# if 'f' is part of the preclassifier
some_model.method_fwd_preclassifier('f')
# if 'f' is part of the classifier
#some_model.method_fwd_preclassifier('f')
...
x, y = some_model.f(a, b)
``` 

