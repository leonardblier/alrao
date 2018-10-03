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

 * pre-classifiers: VGG, GoogLeNet, LSTM, ... (TODO);
 * classifiers: linear classifier (feedforward), linear classifier (RNN).

### Custom models

Custom models can be used with some modifications.

First, the custom model has to be split into a pre-classifier class (e.g. `PreClassif`) and a classifier class (e.g. `Classif`) in order to be integrated into the class `AlraoModel`. Note that the classifier is supposed to return log-probabilities. Once done, an instance of `AlraoModel` can be created with:
```python
preclassif = PreClassif(<args_of_the_preclassifier>)
alrao_model = AlraoModel(preclassif, nb_classifiers, Classif, <args_of_the_classifiers>)
```

Then the `forward` method of the pre-classifier is assumed to return either one value or a tuple. If a tuple `(x, a, b, ...)` is returned, the first element `x` is supposed to be taken as input of the classifiers. Their outputs are then averaged with a model averaging method (here, the `Switch` class), which returns `y`. Thus, the `forward` method of `AlraoModel` returns a tuple `(y, a, b, ...)`.

### Method forwarding

To make Alrao easier to integrate in a given project, method forwarding is provided. Suppose a model class named `Model` has a method `f`, which is regularly called in a code with `x, y = some_model.f(a, b)`. This model has just to be processed as indicated above, and the code is to be changed from: 
```python
some_model = Model(...)
...
x, y = some_model.f(a, b)
```
to:
```python
some_model = AlraoModel(...)
# if 'f' is a preclassifier method:
some_model.method_fwd_preclassifier('f')
# if 'f' is a classifier method:
#some_model.method_fwd_classifiers('f')
...
x, y = some_model.f(a, b)
``` 

