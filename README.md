# Learning with Random Learning Rates

Authors' implementation of "Learning with Random Learning Rates" (2018) in PyTorch.

Paper: https://arxiv.org/abs/1810.01322.

Authors: Léonard Blier, Pierre Wolinski, Yann Ollivier.

## Requirements
The requirements are:
* pytorch (0.4 and higher)
* numpy, scipy
* tqdm

## Tutorial
A tutorial on how to use Alrao with custom models is in `tutorial.ipynb`.

## Sample script for using Alrao with convolutional models on CIFAR10
The script `main_cnn.py` trains convolutional neural networks on CIFAR10.

The main options are:
```
  --no-cuda             disable cuda
  --epochs EPOCHS       number of epochs for phase 1 (default: 50)
  --model_name MODEL_NAME
                        Model {VGG19, GoogLeNet, MobileNetV2, SENet18}
  --optimizer OPTIMIZER
                        optimizer (default: SGD) {Adam, SGD}
  --lr LR               learning rate, when used without alrao
  --use_alrao           multiple learning rates
  --minLR MINLR         log10 of the minimum LR in alrao (log_10 eta_min)
  --maxLR MAXLR         log10 of the maximum LR in alrao (log_10 eta_max)
  --nb_class NB_CLASS   number of classifiers used in Alrao (default 10)
```
More options are available. Check it by running `python main_cnn.py --help`.
For example, to use the script with Alrao on the interval (10**-5, 10) with GoogLeNet, run:
```
python main_cnn.py --use_alrao --minLR -5 --maxLR 1 --nb_class 10 --model_name GoogLeNet
```

If you want to train the same model but with SGD with a learning rate 10**-3, run:
```
python main_cnn.py --lr 0.001 --model_name GoogLeNet
```

The available models are VGG19, GoogLeNet, MobileNetV2, SENet18.

## Sample script for using Alrao with recurrent models on PTB
The script `main_rnn.py` trains a recurrent neural networks on PTB with a LSTM.

The main options are:
```
--no-cuda             disable cuda
--epochs EPOCHS       number of epochs for phase 1 (default: 50)
--optimizer OPTIMIZER
                      optimizer (default: SGD) {Adam, SGD}
--lr LR               learning rate, when used without alrao
--use_alrao           multiple learning rates
--minlr MINLR         minimum LR in alrao (eta_min)
--maxlr MAXLR         maximum LR in alrao (eta_max)
--nb_class NB_CLASS   number of classifiers before the switch
```

More options are available. Check it by running `python main_rnn.py --help`.


## How to use Alrao on custom models


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
