Introduction
============

Hyperparameter tuning is a notable source of computational cost with
deep learning models [@zoph2016neural]. One of the most critical
hyperparameters is the learning rate of the gradient descent
[@theodoridis2015machine p. 892]. With too large learning rates, the
model does not learn; with too small learning rates, optimization is
slow and can lead to local minima and poor generalization
[@jastrzkebski2017three; @Kurita2018; @Mack2016; @Surmenok2017].
Although popular optimizers like Adam [@Kingma2015] come with default
hyperparameters, fine-tuning and scheduling of the Adam learning rate is
still frequent [@denkowski2017stronger], and we suspect the default
setting might be somewhat specific to current problems and architecture
sizes. Such hyperparameter tuning takes up a lot of engineering time.
These and other issues largely prevent deep learning models from working
out-of-the-box on new problems, or on a wide range of problems, without
human intervention (AutoML setup, [@guyon2016brief]).

We propose *All Learning Rates At Once* (Alrao), an alteration of
standard optimization methods for deep learning models. Alrao uses
multiple learning rates at the same time in the same network. By
sampling one learning rate per feature, Alrao reaches performance close
to the performance of the optimal learning rate, without having to try
multiple learning rates. Alrao can be used on top of various
optimization algorithms; we tested SGD and Adam [@Kingma2015]. Alrao
with Adam typically led to strong overfit with good train but poor test
performance (see Sec. [4](#sec:discussion){reference-type="ref"
reference="sec:discussion"}), and our experimental results are obtained
with Alrao on top of SGD.

Alrao could be useful when testing architectures: an architecture could
first be trained with Alrao to obtain an approximation of the
performance it would have with an optimal learning rate. Then it would
be possible to select a subset of promising architectures based on
Alrao, and search for the best learning rate on those architectures,
fine-tuning with any optimizer.

Alrao increases the size of a model on the output layer, but not on the
internal layers: this usually adds little computational cost unless most
parameters occur on the output layer. This text comes along with a
Pytorch implementation usable on a wide set of architectures.

#### Related Work. {#sec:related-works}

Automatically using the "right" learning rate for each parameter was one
motivation behind "adaptive" methods such as RMSProp
[@tieleman2012lecture], AdaGrad [@adagrad] or Adam [@Kingma2015]. Adam
with its default setting is currently considered the default go-to
method in many works [@wilson2017marginal], and we use it as a baseline.
However, further global adjustement of the learning rate in Adam is
common [@liu2017progressive]. Many other heuristics for setting the
learning rate have been proposed, e.g., [@pesky]; most start with the
idea of approximating a second-order Newton step to define an optimal
learning rate [@lecun1998efficient].

Methods that directly set per-parameter learning rates are equivalent to
preconditioning the gradient descent with a diagonal matrix.
Asymptotically, an arguably optimal preconditioner is either the Hessian
of the loss (Newton method) or the Fisher information matrix [@Amari98].
These can be viewed as setting a per-direction learning rate after
redefining directions in parameter space. From this viewpoint, Alrao
just replaces these preconditioners with a random diagonal matrix whose
entries span several orders of magnitude.

Another approach to optimize the learning rate is to perform a gradient
descent on the learning rate itself through the whole training procedure
(for instance [@maclaurin2015gradient]). This can be applied online to
avoid backpropagating through multiple training rounds
[@masse2015speed]. This idea has a long history, see, e.g.,
[@schraudolph1999local] or [@mahmood2012tuning] and the references
therein.

The learning rate can also be treated within the framework of
architecture search, which can explore both the architecture and
learning rate at the same time (e.g., [@real2017large]). Existing
methods range from reinforcement learning
[@zoph2016neural; @baker2016designing] to bandits [@li2017hyperband],
evolutionary algorithms (e.g.,
[@stanley2002evolving; @jozefowicz2015empirical; @real2017large]),
Bayesian optimization [@bergstra2013making] or differentiable
architecture search [@liu2018darts]. These powerful methods are
resource-intensive and do not allow for finding a good learning rate in
a single run.

#### Motivation. {#sec:motivation}

Alrao was inspired by the intuition that not all units in a neural
network end up being useful. Hopefully, in a large enough network, a
sub-network made of units with a good learning rate could learn well,
and hopefully the units with a wrong learning rate will just be ignored.
(Units with a too large learning rate may produce large activation
values, so this assumes the model has some form of protection against
those, such as BatchNorm or sigmoid/tanh activations.)

Several lines of work support the idea that not all units of a network
are useful or need to be trained. First, it is possible to *prune* a
trained network without reducing the performance too much (e.g.,
[@lecun1990; @Han2015; @Han2015a; @See]). [@Li2018] even show that
performance is reasonable if learning only within a very
small-dimensional affine subspace of the parameters, *chosen in advance
at random* rather than post-selected.

Second, training only some of the weights in a neural network while
leaving the others at their initial values performs reasonably well (see
experiments in Appendix [11](#sec:alrao-bernouilli){reference-type="ref"
reference="sec:alrao-bernouilli"}). So in Alrao, units with a very small
learning rate should not hinder training.

Alrao is consistent with the *lottery ticket hypothesis*, which posits
that "large networks that train successfully contain subnetworks
that---when trained in isolation---converge in a comparable number of
iterations to comparable accuracy" [@Frankle2018]. This subnetwork is
the *lottery ticket winner*: the one which had the best initial values.
Arguably, given the combinatorial number of subnetworks in a large
network, with high probability one of them is able to learn alone, and
will make the whole network converge.

Viewing the per-feature learning rates of Alrao as part of the
initialization, this hypothesis suggests there might be enough
sub-networks whose initialization leads to good convergence.

All Learning Rates At Once: Description {#sec:idea}
=======================================

[\[sec:our-method\]]{#sec:our-method label="sec:our-method"}

#### Alrao: principle.

Alrao starts with a standard optimization method such as SGD, and a
range of possible learning rates $(\eta_{\min}, \eta_{\max})$. Instead
of using a single learning rate, we sample once and for all one learning
rate for each *feature*, randomly sampled log-uniformly in
$(\eta_{\min}, \eta_{\max})$. Then these learning rates are used in the
usual optimization update: $$\label{eq:alraoprinciple}
  \theta_{l,i} \leftarrow \theta_{l,i} - \eta_{l,i} \cdot \nabla_{\theta_{l,i}}\ell(\Phi_\theta(x), y)$$
where $\theta_{l,i}$ is the set of parameters used to compute the
feature $i$ of layer $l$ from the activations of layer $l-1$ (the
*incoming* weights of feature $i$). Thus we build "slow-learning" and
"fast-learning" features, in the hope to get enough features in the
"Goldilocks zone".

What constitutes a *feature* depends on the type of layers in the model.
For example, in a fully connected layer, each component of a layer is
considered as a feature: all incoming weights of the same unit share the
same learning rate. On the other hand, in a convolutional layer we
consider each convolution filter as constituting a feature: there is one
learning rate per filter (or channel), thus keeping
translation-invariance over the input image. In LSTMs, we apply the same
learning rate to all components in each LSTM unit (thus in the
implementation, the vector of learning rates is the same for input
gates, for forget gates, etc.).

However, the update
[\[eq:alraoprinciple\]](#eq:alraoprinciple){reference-type="eqref"
reference="eq:alraoprinciple"} cannot be used directly in the last
layer. For instance, for regression there may be only one output
feature. For classification, each feature in the final classification
layer represents a single category, and so using different learning
rates for these features would favor some categories during learning.
Instead, on the output layer we chose to duplicate the layer using
several learning rate values, and use a (Bayesian) model averaging
method to obtain the overall network output
(Fig. [2](#fig:archi){reference-type="ref" reference="fig:archi"}).

We set a learning rate *per feature*, rather than per parameter.
Otherwise, every feature would have some parameters with large learning
rates, and we would expect even a few large incoming weights to be able
to derail a feature. So having diverging parameters within a feature is
hurtful, while having diverging features in a layer is not necessarily
hurtful since the next layer can choose to disregard them. Still, we
tested this option; the results are compatible with this intuition
(Appendix [10](#sec:lr-sampling){reference-type="ref"
reference="sec:lr-sampling"}).

#### Definitions and notation. {#sec:notations}

We now describe Alrao more precisely for deep learning models with
softmax output, on classification tasks (the case of regression is
similar).

Let $\mathcal{D} = \{(x_{1}, y_{1}), ..., (x_{N}, y_{N})\}$, with $y_{i}
\in \{1, ..., K\}$, be a classification dataset. The goal is to predict
the $y_{i}$ given the $x_{i}$, using a deep learning model
$\Phi_{\theta}$. For each input $x$, $\Phi_{\theta}(x)$ is a probability
distribution over $\{1, ..., K\}$, and we want to minimize the
categorical cross-entropy loss $\ell$ over the dataset:
$\frac{1}{N}\sum_{i}\ell(\Phi_{\theta}(x_{i}), y_{i})$.

A deep learning model for classification $\Phi_{\theta}$ is made of two
parts: a *pre-classifier* $\phi_{\theta^{\text{pc}}}$ which computes
some quantities fed to a final *classifier layer*
$C_{\theta^{\text{c}}}$, namely,
$\Phi_{\theta}(x)=C_{\theta^{\mathrm{cl}}}(\phi_{\theta^{\text{pc}}}(x))$.
The classifier layer $C_{\theta^{\text{c}}}$ with $K$ categories is
defined by $C_{\theta^{\text{c}}} = \text{softmax}\circ\left(W^{T}x +
b\right)$ with $\theta^{\mathrm{cl}} = (W, b)$, and
$\text{softmax}(x_{1}, ...,
x_{K})_{k} = {e^{x_{k}}}/\left({\sum_{i} e^{x_{i}}}\right).$The
*pre-classifier* is a computational graph composed of any number of
*layers*, and each layer is made of multiple *features*.

We denote $\logunif(\cdot ; \eta_{\min}, \eta_{\max})$ the *log-uniform*
probability distribution on an interval $(\eta_{\min}, \eta_{\max})$:
namely, if $\eta \sim \logunif(\cdot ; \eta_{\min},
\eta_{\max})$, then $\log \eta$ is uniformly distributed between $\log
\eta_{\min}$ and $\log \eta_{\max}$. Its density function is
$$\label{eq:logunif}
  \logunif(\eta; \eta_{\min}, \eta_{\max}) = \frac{\mathbbm{1}_{\eta_{\min} \leq \eta \leq \eta_{\max}}}{\eta_{\max} - \eta_{\min}}\times\frac{1}{\eta}$$

#### Alrao for the pre-classifier: A random learning rate for each feature.

In the pre-classifier, for each feature $i$ in each layer $l$, a
learning rate $\eta_{l,i}$ is sampled from the probability distribution
$\logunif(.; \eta_{\min}, \eta_{\max})$, once and for all at the
beginning of training.[^4] Then the incoming parameters of each feature
in the preclassifier are updated in the usual way with this learning
rate (Eq. [\[eq:updatepc\]](#eq:updatepc){reference-type="ref"
reference="eq:updatepc"}).

\centering
![Left: a standard fully connected neural network for a classification
task with three classes, made of a pre-classifier and a classifier
layer. Right: Alrao version of the same network. The single classifier
layer is replaced with a set of parallel copies of the original
classifier, averaged with a model averaging method. Each unit uses its
own learning rate for its incoming weights (represented by different
styles of
arrows).[]{label="fig:archi"}](img/beforealrao.eps "fig:"){#fig:archi
width="0.48\linewidth"} ![Left: a standard fully connected neural
network for a classification task with three classes, made of a
pre-classifier and a classifier layer. Right: Alrao version of the same
network. The single classifier layer is replaced with a set of parallel
copies of the original classifier, averaged with a model averaging
method. Each unit uses its own learning rate for its incoming weights
(represented by different styles of
arrows).[]{label="fig:archi"}](img/newalrao.eps "fig:"){#fig:archi
width="0.48\linewidth"}

#### Alrao for the classifier layer: Model averaging from classifiers with different learning rates. {#sec:parall-class}

In the classifier layer, we build multiple clones of the original
classifier layer, set a different learning rate for each, and then use a
model averaging method from among them. The averaged classifier and the
overall Alrao model are: $$\label{eq:parall-class}
  C^{\text{Alrao}}_{\theta^{\mathrm{cl}}}(z) \mathrel{\mathop{:}}=
  \sum_{j=1}^{N_{\mathrm{cl}}}a_{j} \, C_{\theta^{\mathrm{cl}}_{j}}(z),
  \qquad
\Phi^{\text{Alrao}}_{\theta}(x) \mathrel{\mathop{:}}=
C^{\text{Alrao}}_{\theta^{\mathrm{cl}}}(\phi_{\theta^{\text{pc}}}(x))$$
where the $C_{\theta^{\mathrm{cl}}_{j}}$ are copies of the original
classifier layer, with non-tied parameters, and
$\theta^{\mathrm{cl}} \mathrel{\mathop{:}}=(\theta^{\mathrm{cl}}_{1}, ...,
\theta^{\mathrm{cl}}_{N_{\mathrm{cl}}})$. The $a_{j}$ are the parameters
of the model averaging, and are such that for all $j$,
$0 \leq a_{j} \leq 1$, and $\sum_{j}a_{j} = 1$. These are not updated by
gradient descent, but via a model averaging method from the literature
(see below).

For each classifier $C_{\theta^{\mathrm{cl}}_{j}}$, we set a learning
rate $\log \eta_{j} = \log \eta_{\min} +
\frac{j-1}{N_{\mathrm{cl}}-1}\log(\eta_{\max}/ \eta_{\min})$, so that
the classifiers' learning rates are log-uniformly spread on the interval
$(\eta_{\min}, \eta_{\max})$.

Thus, the original model $\Phi_{\theta}(x)$ leads to the Alrao model
$\Phi^{\text{Alrao}}_{\theta}(x)$. Only the classifier layer is
modified, the pre-classifier architecture being unchanged.

#### The Alrao update.

\algsetblock[Name]{Forall}{Stop}{3}{1cm}
$a_{j} \leftarrow 1/N_{\mathrm{cl}}$ for each
$1\leq j \leq N_{\mathrm{cl}}$

$\Phi^{\text{Alrao}}_{\theta}(x) :=
    \sum_{j=1}^{N_{\mathrm{cl}}}a_{j}\,C_{\theta^{\mathrm{cl}}_{j}}(
    \phi_{\theta^{\mathrm{pc}}}(x))$ Sample
$\eta_{l,i} \sim \logunif(.; \eta_{\min}, \eta_{\max})$. Define
$\log \eta_{j} = \log\eta_{\min} + \frac{j-1}{N_{\mathrm{cl}}-1}\log \frac{\eta_{\max}}{\eta_{\min}}$.

\While{Convergence ?}
$z_{t} \leftarrow \phi_{\theta^{\mathrm{pc}}}(x_{t})$
$\theta_{l,i} \leftarrow \theta_{l,i} - \eta_{l,i} \cdot \nabla_{\theta_{l,i}}\ell(\Phi^{\text{Alrao}}_{\theta}(x_{t}), y_{t})$
$\theta^{\mathrm{cl}}_{j} \leftarrow
        \theta^{\mathrm{cl}}_{j} - \eta_{j}\cdot
        \nabla_{\theta^{\mathrm{cl}}_{j}} \,
        \ell(C_{\theta^{\mathrm{cl}}_{j}}(z_{t}), y_{t})$

$a \leftarrow \mathtt{ModelAveraging}(a, (C_{\theta^{\mathrm{cl}}_{i}}(z_{t}))_{i}, y_{t})$
$t \leftarrow t+1$ mod $N$

Alg. [\[algo:alrao\]](#algo:alrao){reference-type="ref"
reference="algo:alrao"} presents the full Alrao algorithm for use with
SGD (other optimizers like Adam are treated similarly). The updates for
the pre-classifier, classifier, and model averaging weights are as
follows.

-   The update rule for the pre-classifier is the usual SGD one, with
    per-feature learning rates. For each feature $i$ in each layer $l$,
    its incoming parameters are updated as: $$\label{eq:updatepc}
      \theta_{l,i} \leftarrow \theta_{l,i} - \eta_{l,i} \cdot \nabla_{\theta_{l,i}}\ell(\Phi^{\text{Alrao}}_\theta(x), y)$$

-   The parameters $\theta^{\mathrm{cl}}_j$ of each classifier clone $j$
    on the classifier layer are updated as if this classifier alone was
    the only output of the model: $$\begin{aligned}
        \label{eq:updatec}
      \theta^{\mathrm{cl}}_{j} \leftarrow & \;
    %   \theta^{\mathrm{cl}}_{j}  - \frac{\eta_{j}}{a_{j}} \cdot \nabla_{\theta^{\mathrm{cl}}_{j}}\,\ell(\Phi^{\text{Alrao}}_\theta(x), y)
    %   \\
    %   =&\;
      \theta^{\mathrm{cl}}_{j}  - \eta_{j} \cdot
      \nabla_{\theta^{\mathrm{cl}}_{j}}\,\ell(C_{\theta^{\mathrm{cl}}_{j}}(\phi_{\theta^{\mathrm{pc}}}(x)), y)\end{aligned}$$
    (still sharing the same pre-classifier
    $\phi_{\theta^{\mathrm{pc}}}$). This ensures classifiers with low
    weights $a_j$ still learn, and is consistent with model averaging
    philosophy. Algorithmically this requires differentiating the loss
    $N_{\mathrm{cl}}$ times with respect to the last layer (but no
    additional backpropagations through the preclassifier).

-   To set the weights $a_j$, several model averaging techniques are
    available, such as Bayesian Model Averaging [@Wasserman2000]. We
    decided to use the *Switch* model averaging [@VanErven2011], a
    Bayesian method which is both simple, principled and very responsive
    to changes in performance of the various models. After each sample
    or mini-batch, the switch computes a modified posterior distribution
    $(a_j)$ over the classifiers. This computation is directly taken
    from [@VanErven2011] and explained in
    Appendix [6](#sec:switch){reference-type="ref"
    reference="sec:switch"}. The observed evolution of this posterior
    during training is commented in
    Appendix [7](#sec:posterior){reference-type="ref"
    reference="sec:posterior"}.

#### Implementation. {#sec:implementation}

We release along with this paper a Pytorch [@paszke2017automatic]
implementation of this method. It can be used on an existing model with
little modification. A short tutorial is given in
Appendix [13](#sec:tutorial){reference-type="ref"
reference="sec:tutorial"}. The *features* (sets of weights which will
share the same learning rate) need to be defined for each layer type:
for now we have done this for linear, convolutional, and LSTMs layers.

Experiments {#sec:experiments}
===========

We tested Alrao on various convolutional networks for image
classification (CIFAR10), and on LSTMs for text prediction. The
baselines are SGD with an optimal learning rate, and Adam with its
default setting, arguably the current default method
[@wilson2017marginal].

#### Image classification on CIFAR10. {#sec:cifar10}

For image classification, we used the CIFAR10 dataset [@Krizhevsky2009].
It is made of 50,000 training and 10,000 testing data; we split the
training set into a smaller training set with 40,000 samples, and a
validation set with 10,000 samples. For each architecture, training on
the smaller training set was stopped when the validation loss had not
improved for 20 epochs. The epoch with best validation loss was selected
and the corresponding model tested on the test set. The inputs are
normalized. Training used data augmentation (random cropping and random
horizontal flipping). The batch size is always 32. Each setting was run
10 times: the confidence intervals presented are the standard deviation
over these runs.

We tested Alrao on three architectures known to perform well on this
task: GoogLeNet [@szegedy2015going], VGG19 [@Simonyan14c] and MobileNet
[@howard2017mobilenets]. The exact implementation for each can be found
in our code.

The Alrao learning rates were sampled log-uniformly from
$\eta_{\min} = 10^{-5}$ to $\eta_{\max} = 10$. For the output layer we
used 10 classifiers with switch model averaging
(Appendix [6](#sec:switch){reference-type="ref"
reference="sec:switch"}); the learning rates of the output classifiers
are deterministic and log-uniformly spread in
$[\eta_{\min},\eta_{\max}]$.

In addition, each model was trained with SGD for every learning rate in
the set
$\{10^{-5},\linebreak[1] 10^{-4},\linebreak[1] 10^{-3},\linebreak[1] 10^{-2},\linebreak[1] 10^{-1},\linebreak[1] 1.,\linebreak[1] 10.$$\}$.
The best SGD learning rate is selected on the validation set, then
reported in Table [\[tab:results\]](#tab:results){reference-type="ref"
reference="tab:results"}. We also compare to Adam with its default
hyperparameters ($\eta=10^{-3}, \beta_1 = 0.9, \beta_2 = 0.999$).

The results are presented in
Table [\[tab:results\]](#tab:results){reference-type="ref"
reference="tab:results"}. Learning curves with various SGD learning
rates, with Adam, and with Alrao are presented in
Fig. [\[fig:firstepochs\]](#fig:firstepochs){reference-type="ref"
reference="fig:firstepochs"}. Fig. [5](#fig:trig){reference-type="ref"
reference="fig:trig"} tests the influence of $\eta_{\min}$ and
$\eta_{\max}$.

\centering
![GoogLeNet[]{label="fig:firstepochs-googlenet"}](img/learningcurvesGooglenet.eps){#fig:firstepochs-googlenet
width="\linewidth"}

![MobileNetV2[]{label="fig:firstepochs-mobilenet"}](img/learningcurvesmobilenet.eps){#fig:firstepochs-mobilenet
width="\linewidth"}

\fontsize{7pt}{7pt}
\selectfont
  ------------------------- -------- ------------------- ---------------- ------------------- ---------------- ----------------- ---------------- --
  Model                                                                                                                                           
  (lr)2-4 (lr)5-6 (lr)7-8   LR              Loss             Acc (%)             Loss             Acc (%)            Loss            Acc (%)      
  MobileNet                 $1e$-1     $0.37 \pm 0.01$    $90.2 \pm 0.3$    $1.01 \pm 0.95$     $78 \pm 11$     $0.42 \pm 0.02$   $88.1 \pm 0.6$  
  GoogLeNet                 $1e$-2     $0.45 \pm 0.05$    $89.6 \pm 1.0$    $0.47 \pm 0.04$    $89.8 \pm 0.4$   $0.47 \pm 0.03$   $88.9 \pm 0.8$  
  VGG19                     $1e$-1     $0.42 \pm 0.02$    $89.5 \pm 0.2$    $0.43 \pm 0.02$    $88.9 \pm 0.4$   $0.45 \pm 0.03$   $87.5 \pm 0.4$  
  LSTM (PTB)                $1$       $1.566 \pm 0.003$   $66.1 \pm 0.1$   $1.587 \pm 0.005$   $65.6 \pm 0.1$   $1.67 \pm 0.01$   $64.1 \pm 0.2$  
  ------------------------- -------- ------------------- ---------------- ------------------- ---------------- ----------------- ---------------- --

  :  Performance of Alrao-SGD, of SGD with optimal learning rate from
  $\{10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1., 10.\}$, and of
  Adam with its default setting. Three convolutional models are reported
  for image classifaction (CIFAR10) and one recurrent model for
  character prediction (Penn Treebank). For Alrao the learning rates lie
  in $[\eta_{\min};\eta_{\max}] = [10^{-5};10]$ (CIFAR10) or
  $[10^{-3};10^2]$ (PTB). Each experiment is run 10 times (CIFAR10) or 5
  times (PTB); the confidence intervals report the standard deviation
  over these runs.

-0.1in [\[tab:results\]]{#tab:results label="tab:results"}

\centering
![Performance of Alrao with a GoogLeNet model, depending on the interval
$(\eta_\min, \eta_\max)$. Left: loss on the train set; right: on the
test set. Each point with coordinates $(\eta_\min, \eta_\max)$ above the
diagonal represents the loss after 30 epochs for Alrao with this
interval. Points $(\eta, \eta)$ on the diagonal represent standard SGD
with learning rate $\eta$ after 50 epochs. Standard SGD with
$\eta = 10^2$ is left blank to due numerical divergence (NaN). Alrao
works as soon as $(\eta_\min, \eta_\max)$ contains at least one suitable
learning rate.[]{label="fig:trig"}](img/triangle.eps){#fig:trig
width="\linewidth"}

#### Recurrent learning on Penn Treebank. {#sec:penn-tree-bank}

To test Alrao on a different kind of architecture, we used a recurrent
neural network for text prediction on the Penn Treebank [@Marcus1993]
dataset. The experimental procedure is the same, with
$(\eta_{\min}, \eta_{\max}) =
(0.001, 100)$ and $6$ output classifiers for Alrao. The results appear
in Table [\[tab:results\]](#tab:results){reference-type="ref"
reference="tab:results"}, where the loss is given in bits per character
and the accuracy is the proportion of correct character predictions.

The model was trained for *character* prediction rather than word
prediction. This is technically easier for Alrao implementation: since
Alrao uses copies of the output layer, memory issues arise for models
with most parameters on the output layer. Word prediction (10,000
classes on PTB) requires more output parameters than character
prediction; see Section [4](#sec:discussion){reference-type="ref"
reference="sec:discussion"} and
Appendix [9](#sec:number-parameters){reference-type="ref"
reference="sec:number-parameters"}.

The model is a two-layer LSTM [@hochreiter1997long] with an embedding
size of 100 and with 100 hidden features. A dropout layer with rate
$0.2$ is included before the decoder. The training set is divided into
20 minibatchs. Gradients are computed via truncated backprop through
time [@werbos1990backpropagation] with truncation every 70 characters.

#### Comments.

As expected, Alrao performs slightly worse than the best learning rate.
Still, even with wide intervals $(\eta_\min, \eta_\max)$, Alrao comes
reasonably close to the best learning rate, across all setups; hence
Alrao's possible use as a quick assessment method. Although Adam with
its default parameters almost matches optimal SGD, this is not always
the case, for example with the MobileNet model
(Fig.[4](#fig:firstepochs-mobilenet){reference-type="ref"
reference="fig:firstepochs-mobilenet"}). This confirms a known risk of
overfit with Adam [@wilson2017marginal]. In our setup, Alrao seems to be
a more stable default method.

Our results, with either SGD, Adam, or SGD-Alrao, are somewhat below the
art: in part this is because we train on only 40,000 CIFAR samples, and
do not use stepsize schedules.

Limitations, further remarks, and future directions {#sec:discussion}
===================================================

[\[sec:strengths-weaknesses\]]{#sec:strengths-weaknesses
label="sec:strengths-weaknesses"}

#### Increased number of parameters for the classification layer.

Alrao modifies the output layer of the optimized model. The number of
parameters for the classification layer is multiplied by the number of
classifier copies used (the number of parameters in the pre-classifier
is unchanged). On CIFAR10 (10 classes), the number of parameters
increased by less than 5% for the models used. On Penn Treebank, the
number of parameters increased by $15\%$ in our setup (working at the
character level); working at word level it would have increased
threefold (Appendix [9](#sec:number-parameters){reference-type="ref"
reference="sec:number-parameters"}).

This is clearly a limitation for models with most parameters in the
classifier layer. For output-layer-heavy models, this can be mitigated
by handling the copies of the classifiers on distinct computing units:
in Alrao these copies work in parallel given the pre-classifier.

Still, models dealing with a very large number of output classes usually
rely on other parameterizations than a direct softmax, such as a
hierarchical softmax (see references in [@jozefowicz2016exploring]);
Alrao could be used in conjunction with such methods.

#### Adding two hyperparameters.

We claim to remove a hyperparameter, the learning rate, but replace it
with two hyperparameters $\eta_{\min}$ and $\eta_{\max}$.

Formally, this is true. But a systematic study of the impact of these
two hyperparameters (Fig. [5](#fig:trig){reference-type="ref"
reference="fig:trig"}) shows that the sensitivity to $\eta_{\min}$ and
$\eta_{\max}$ is much lower than the original sensitivity to the
learning rate. In our experiments, convergence happens as soon as
$(\eta_{\min};\eta_{\max})$ contains a reasonable learning rate
(Fig. [5](#fig:trig){reference-type="ref" reference="fig:trig"}).

A wide range of values of $(\eta_{\min};\eta_{\max})$ will contain one
good learning rate and achieve close-to-optimal performance
(Fig. [5](#fig:trig){reference-type="ref" reference="fig:trig"}).
Typically, we recommend to just use an interval containing all the
learning rates that would have been tested in a grid search, e.g.,
$10^{-5}$ to $10$.

So, even if the choice of $\eta_{\min}$ and $\eta_{\max}$ is important,
the results are much more stable to varying these two hyperparameters
than to the learning rate. For instance, standard SGD fails due to
numerical issues for $\eta =
100$ while Alrao with $\eta_\max = 100$ works with any $\eta_\min\leq 1$
(Fig. [5](#fig:trig){reference-type="ref" reference="fig:trig"}), and is
thus stable to relatively large learning rates. We would still expect
numerical issues with very large $\eta_\max$, but this has not been
observed in our experiments.

#### Alrao with Adam.

Alrao is much less reliable with Adam than with SGD. Surprisingly, this
occurs mostly for test performance, which can even diverge, while
training curves mostly look good
(Appendix [8](#sec:alrao-adam){reference-type="ref"
reference="sec:alrao-adam"}). We have no definitive explanation for this
at present. It might be that changing the learning rate in Adam also
requires changing the momentum parameters in a correlated way. It might
be that Alrao does not work on Adam because Adam is more sensitive to
its hyperparameters. The stark train/test discrepancy might also suggest
that Alrao-Adam performs well as a pure optimization method but
exacerbates the underlying risk of overfit of Adam
[@wilson2017marginal; @keskar2017improving].

[\[sec:remarks\]]{#sec:remarks label="sec:remarks"}
[\[sec:future-directions\]]{#sec:future-directions
label="sec:future-directions"}

#### Increasing network size.

With Alrao, neurons with unsuitable learning rates will not learn: those
with a too large learning rate might learn nothing, while those with too
small learning rates will learn too slowly to be used. Thus, Alrao may
reduce the *effective size* of the network to only a fraction of the
actual architecture size, depending on $(\eta_{\min}, \eta_{\max})$.

Our first intuition was that increasing the width of the network was
going to be necessary with Alrao, to avoid wasting too many units. In a
fully connected network, the number of weights is quadratic in the
width, so increasing width (beyond a factor three in our experiments)
can be bothersome. Comparisons of Alrao with increased width are
reported in Appendix [12](#sec:increased-width){reference-type="ref"
reference="sec:increased-width"}. Width is indeed a limiting factor for
the models considered, even without Alrao
(Appendix [12](#sec:increased-width){reference-type="ref"
reference="sec:increased-width"}). But to our surprise, Alrao worked
well even without width augmentation.

#### Other optimization algorithms, other hyperparameters, learning rate schedulers\...

Using a learning rate schedule instead of a fixed learning rate is often
effective [@bengio2012practical]. We did not use learning rate
schedulers here; this may partially explain why the results in
Table [\[tab:results\]](#tab:results){reference-type="ref"
reference="tab:results"} are worse than the state-of-the-art. Nothing
prevents using such a scheduler within Alrao, e.g., by dividing all
Alrao learning rates by a time-dependent constant; we did not experiment
with this yet.

The idea behind Alrao could be used on other hyperparameters as well,
such as momentum. However, if more hyperparameters are initialized
randomly for each feature, the fraction of features having all their
hyperparameters in the "Goldilocks zone" will quickly decrease.

Conclusion {#sec:conclusion}
==========

Applying stochastic gradient descent with random learning rates for
different features is surprisingly resilient in our experiments, and
provides performance close enough to SGD with an optimal learning rate,
as soon as the range of random learning rates contains a suitable one.
This could save time when testing deep learning models, opening the door
to more out-of-the-box uses of deep learning.

Acknowledgments {#acknowledgments .unnumbered}
===============

We would like to thank Corentin Tallec for his technical help, and his
many remarks and advice. We thank Olivier Teytaud for pointing useful
references.

\bibliographystyle{abbrv}
\vfill
\pagebreak
\appendix
Model Averaging with the Switch {#sec:switch}
===============================

As explained is
Section [\[sec:our-method\]](#sec:our-method){reference-type="ref"
reference="sec:our-method"}, we use a model averaging method on the
classifiers of the output layer. We could have used the Bayesian Model
Averaging method [@Wasserman2000]. But one of its main weaknesses is the
*catch-up phenomenon* [@VanErven2011]: plain Bayesian posteriors are
slow to react when the relative performance of models changes over time.
Typically, for instance, some larger-dimensional models need more
training data to reach good performance: at the time they become better
than lower-dimensional models for predicting current data, their
Bayesian posterior is so bad that they are not used right away (their
posterior needs to "catch up" on their bad initial performance). This
leads to very conservative model averaging methods.

The solution from [@VanErven2011] against the catch-up phenomenon is to
*switch* between models. It is based on previous methods for prediction
with expert advice (see for instance
[@herbster1998tracking; @volf1998switching] and the references in
[@koolen2008combining; @VanErven2011]), and is well rooted in
information theory. The switch method maintains a Bayesian posterior
distribution, not over the set of models, but over the set of *switching
strategies* between models. Intuitively, the model selected can be
adapted online to the number of samples seen.

We now give a quick overview of the switch method from [@VanErven2011]:
this is how the model averaging weights $a_j$ are chosen in Alrao.

Assume that we have a set of prediction strategies
$\mathcal{M} = \{p^{j}, j \in
\mathcal{I}\}$. We define the set of *switch sequences*,
$\mathbb{S}= \{((t_{1},
j_{1}), ..., (t_{L}, j_{L})), 1 = t_{1} < t_{2} < ... < t_{L}\; , \;  j
\in \mathcal{I}\}$. Let $s = ((t_{1}, j_{1}), ..., (t_{L}, j_{L}))$ be a
switch sequence. The associated prediction strategy
$p_{s}(y_{1:n}|x_{1:n})$ uses model $p^{j_i}$ on the time interval
$[t_i;t_{i+1})$, namely $$\begin{aligned}
 \label{eq:defswitch}
p_{s}(y_{1:i+1}|x_{1:i+1},y_{1:i}) =
p^{K_{i}}(y_{i+1}|x_{1:i+1},y_{1:i}) \end{aligned}$$ where $K_{i}$ is
such that $K_{i} = j_{l}$ for $t_{l} \leq i < t_{l+1}$. We fix a prior
distribution $\pi$ over switching sequences. In this work,
${\cal I} = \{1, ..., N_C\}$ the prior is, for a switch sequence
$s = ((t_{1}, j_{1}), ..., (t_{L}, j_{L}))$: $$\label{eq:priorswitch}
  \pi(s) = \pi_L(L)\pi_K(j_1)\prod_{i=2}^{L} \pi_T(t_{i}|t_i > t_{i-1})\pi_K(j_i)$$
with $\pi_L(L) = \frac{\theta^L}{1-\theta}$ a geometric distribution
over the switch sequences lengths, $\pi_K(j) = \frac{1}{N_C}$ the
uniform distribution over the models (here the classifiers) and
$\pi_T(t) = \frac{1}{t(t+1)}$.

This defines a Bayesian mixture distribution: $$\label{eq:2}
  p_{sw}(y_{1:T}|x_{1:T}) = \sum_{s\in\mathbb{S}}\pi(s)p_{s}(y_{1:T}|x_{1:T})$$
Then, the model averaging weight $a_{j}$ for the classifier $j$ after
seeing $T$ samples is the posterior of the switch distribution:
$\pi(K_{T+1}=j|y_{1:T}, x_{1:T})$. $$\label{eq:1}
  a_{j} = p_{sw}(K_{T+1}=j|y_{1:T}, x_{1:T}) = \frac{p_{sw}(y_{1:T}, K_{T+1}=j | x_{1:T})}{p_{sw}(y_{1:T}| x_{1:T})}$$
These weights can be computed online exactly in a quick and simple way
[@VanErven2011], thanks to dynamic programming methods from hidden
Markov models.

The implementation of the switch used in Alrao exactly follows the
pseudo-code from [@NIPS2007_3277], with hyperparameter $\theta = 0.999$
(allowing for many switches a priori). It can be found in the
accompanying online code.

Evolution of the Posterior {#sec:posterior}
==========================

The evolution of the model averaging weights can be observed during
training. In Figure [6](#fig:posterior){reference-type="ref"
reference="fig:posterior"}, we can see their evolution during the
training of the GoogLeNet model with Alrao, 10 classifiers, with
$\eta_\min = 10^{-5}$ and $\eta_\max=10^1$.

We can make several observations. First, after only a few gradient
descent steps, the model averaging weights corresponding to the three
classifiers with the largest learning rates go to zero. This means that
their parameters are moving too fast, and their loss is getting very
large.

Next, for a short time, a classifier with a moderately large learning
rate gets the largest posterior weight, presumably because it is the
first to learn a useful model.

Finally, after the model has seen approximately 4,000 samples, a
classifier with a slightly smaller learning rate is assigned a posterior
weight $a_j$ close to 1, while all the others go to 0. This means that
after a number of gradient steps, the model averaging method acts like a
model selection method.

\centering
![Model averaging weights during training. During the training of the
GoogLeNet model with Alrao, 10 classifiers, with $\eta_\min = 10^{-5}$
and $\eta_\max=10^1$, we represent the evolution of the model averaging
weights $a_j$, depending on the corresponding classifier's learning
rate. []{label="fig:posterior"}](img/switchplot.eps){#fig:posterior
width="\linewidth"}

Alrao-Adam {#sec:alrao-adam}
==========

In Figure [\[fig:adam-alrao\]](#fig:adam-alrao){reference-type="ref"
reference="fig:adam-alrao"}, we report our experiments with Alrao-Adam.
As explained in
Section [\[sec:strengths-weaknesses\]](#sec:strengths-weaknesses){reference-type="ref"
reference="sec:strengths-weaknesses"}, Alrao is much less reliable with
Adam than with SGD.

This is especially true for the test performance, which can even diverge
while training performance remains either good or acceptable
(Fig. [\[fig:adam-alrao\]](#fig:adam-alrao){reference-type="ref"
reference="fig:adam-alrao"}). Thus Alrao-Adam seems to send the model
into atypical regions of the search space.

\centering
![Alrao-Adam on GoogLeNet: Alrao-Adam compared with standard Adam with
various learning rates. Alrao uses 10 classifiers and learning rates in
the interval $[10^{-6}, 1]$. Each plot is averaged on 10 experiments. We
observe that optimization with Alrao-Adam is efficient, since train loss
is comparable to the usual Adam methods. But the model starkly overfits,
as the test loss
diverges.[]{label="fig:googlenet-adamalrao"}](img/adamalraogoogle.eps){#fig:googlenet-adamalrao
width="\textwidth"}

\vfill
![Alrao-Adam on MobileNet: Alrao-Adam with two different learning rate
intervals, with 10 classifiers. Each plot is averaged on 10 experiments.
Exactly as with GoogLeNet model, optimization itself is efficient (for
both intervals). For the interval with the smallest $\eta_\max$, the
test loss does not converge and is very unstable. For the interval with
the largest $\eta_\max$, the test loss
diverges.[]{label="fig:mobilenet-adamalrao"}](img/adamalraomobile.eps){#fig:mobilenet-adamalrao
width="\textwidth"}

\vfill
![Alrao-Adam on VGG19: Alrao-Adam on the interval $[10^{-6}, 1]$, with
10 classifiers. The 10 plots are 10 runs of the same experiments. While
9 of them do converge and generalize, the last one exhibits wide
oscillations, both in train and
test.[]{label="fig:vgg-adamalrao"}](img/adamalraovgg.eps){#fig:vgg-adamalrao
width="\textwidth"}

Number of Parameters {#sec:number-parameters}
====================

As explained in
Section [\[sec:strengths-weaknesses\]](#sec:strengths-weaknesses){reference-type="ref"
reference="sec:strengths-weaknesses"}, Alrao increases the number of
parameters of a model, due to output layer copies. The additional number
of parameters is approximately equal to $(N_{\mathrm{cl}} - 1)\times K
\times d$ where $N_{\mathrm{cl}}$ is the number of classifier copies
used in Alrao, $d$ is the dimension of the output of the pre-classifier,
and $K$ is the number of classes in the classification task (assuming a
standard softmax output; classification with many classes often uses
other kinds of output parameterization instead).

\fontsize{9pt}{9pt}
\selectfont
  ----------- -------------- ------------
  Model                      
               Without Alao   With Alrao
  GoogLeNet       6.166M        6.258M
  VGG            20.041M       20.087M
  MobileNet       2.297M        2.412M
  LSTM (C)        0.172M        0.197M
  LSTM (W)        2.171M        7.221M
  ----------- -------------- ------------

  : Comparison between the number of parameters in models used without
  and with Alrao. LSTM (C) is a simple LSTM cell used for character
  prediction while LSTM (W) is the same cell used for word prediction.

-0.1in [\[tab:nparams\]]{#tab:nparams label="tab:nparams"}

The number of parameters for the models used, with and without Alrao,
are in Table [\[tab:nparams\]](#tab:nparams){reference-type="ref"
reference="tab:nparams"}. We used 10 classifiers in Alrao for
convolutional neural networks, and 6 classifiers for LSTMs. Using Alrao
for classification tasks with many classes, such as word prediction
(10,000 classes on PTB), increases the number of parameters noticeably.

For those model with significant parameter increase, the various
classifier copies may be done on parallel GPUs.

Other Ways of Sampling the Learning Rates {#sec:lr-sampling}
=========================================

In Alrao we sample a learning rate for each feature. Intuitively, each
feature (or neuron) is a computation unit of its own, using a number of
inputs from the previous layer. If we assume that there is a "right"
learning rate for learning new features based on information from the
previous layer, then we should try a learning rate per feature; some
features will be useless, while others will be used further down in the
network.

An obvious variant is to set a random learning rate per weight, instead
of for all incoming weights of a given feature. However, this runs the
risk that *every* feature in a layer will have a few incoming weights
with a large rate, so intuitively every feature is at risk of diverging.
This is why we favored per-feature rates.

Still, we tested sampling a learning rate for each weight in the
pre-classifier (while keeping the same Alrao method for the classifier
layer).

\centering
![Loss for various intervals $(\eta_\min,
  \eta_\max)$, as a function of the sampling method for the learning
rates, per feature or per weight. The model is a two-layer LSTM trained
for 20 epochs only, for character prediction. Each curves represents 10
runs. (Losses are much higher than the results reported in
Table [\[tab:results\]](#tab:results){reference-type="ref"
reference="tab:results"} because the full training for
Table [\[tab:results\]](#tab:results){reference-type="ref"
reference="tab:results"} takes approximately 300
epochs.)[]{label="fig:other-lr-sampler"}](img/lr_sampler_rnn.eps){#fig:other-lr-sampler
width="\linewidth"}

In our experiments on LSTMs, per-weight learning rates sometimes perform
well but are less stable and more sensitive to the interval $(\eta_\min,
\eta_\max)$: for some intervals $(\eta_\min, \eta_\max)$ with very large
$\eta_\max$, results with per-weight learning rates are a lot worse than
with per-feature learning rates. This is consistent with the intuition
above.

Learning a Fraction of the Features {#sec:alrao-bernouilli}
===================================

\centering
![Loss of a model where only a random fraction $p$ of the features are
trained, and the others left at their initial value, as a function of
$p$. The architecture is GoogLeNet, trained on
CIFAR10.[]{label="fig:fraction-feature"}](img/DataBox_Exp_2018_09_23_prop_lr_GoogLeNet_p_test_nll.pdf){#fig:fraction-feature
width="0.7\linewidth"}

As explained in the introduction, several works support the idea that
not all units are useful when learning a deep learning model. Additional
results supporting this hypothesis are presented in
Figure [11](#fig:fraction-feature){reference-type="ref"
reference="fig:fraction-feature"}. We trained a GoogLeNet architecture
on CIFAR10 with standard SGD with learning rate $\eta_0$, but learned
only a random fraction $p$ of the features (chosen at startup), and kept
the others at their initial value. This is equivalent to sampling each
learning rate $\eta$ from the probability distribution
$P(\eta = \eta_0) = p$ and $P(\eta = 0) = 1 - p$.

We observe that even with a fraction of the weights not being learned,
the model's performance is close to its performance when fully trained.

When training a model with Alrao, many features might not learn at all,
due to too small learning rates. But Alrao is still able to reach good
results. This could be explained by the resilience of neural networks to
partial training.

Increasing Network Size {#sec:increased-width}
=======================

As explained in Section [4](#sec:discussion){reference-type="ref"
reference="sec:discussion"}, learning with Alrao reduces the *effective
size* of the network to only a fraction of the actual architecture size,
depending on $(\eta_{\min}, \eta_{\max})$. We first tought that
increasing the width of each layer was going to be necessary in order to
use Alrao. However, our experiments show that this is not necessary.

Alrao and SGD experiments with increased width are reported in
Figure [\[fig:K1K3\]](#fig:K1K3){reference-type="ref"
reference="fig:K1K3"}. As expected, Alrao with increased width has
better performance, since the *effective size* increases. However,
increasing the width also improves performance of standard SGD, by
roughly the same amount.

Thus, width is still a limiting factor both for GoogLeNet and MobileNet.
This shows that Alrao can perform well even when network size is a
limiting factor; this runs contrary to our initial intuition that Alrao
would require very large networks in order to have enough features with
suitable learning rates.

\centering
![GoogLeNet[]{label="fig:googlenet-K1K3"}](img/K1K3GoogLeNet.eps){#fig:googlenet-K1K3
width="\textwidth"}

\vfill
![MobileNet[]{label="fig:mobilenet-K1K3"}](img/K1K3MobileNetV2.eps){#fig:mobilenet-K1K3
width="\textwidth"}

\newpage
Tutorial {#sec:tutorial}
========

In this section, we briefly show how Alrao can be used in practice on an
already implemented method in Pytorch. The code will be available soon.

The first step is to build the preclassifier. Here, we use the VGG19
architecture. The model is built without a classifier. Nothing else is
required for Alrao at this step.

``` {language="python"}
class VGG(nn.Module):
    def __init__(self, cfg):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        # The dimension of the preclassier's output need to be specified.
        self.linearinputdim = 512

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        # The model do not contain a classifier layer.
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

preclassifier = VGG([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', \
        512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
```

Then, we can build the Alrao-model with this preclassifier, sample the
learning rates for the model, and define the Alrao optimizer

``` {language="python"}
# We define the interval in which the learning rates are sampled
minlr = 10 ** (-5)
maxlr = 10 ** 1

# nb_classifiers is the number of classifiers averaged by Alrao.
nb_classifiers = 10
nb_categories = 10

net = AlraoModel(preclassifier, nb_categories, preclassifier.linearinputdim, nb_classifiers)

# We spread the classifiers learning rates log-uniformly on the interval.
classifiers_lr = [np.exp(np.log(minlr) + \
    k /(nb_classifiers-1) * (np.log(maxlr) - np.log(minlr)) \
    ) for k in range(nb_classifiers)]

# We define the sampler for the preclassifier's features.
lr_sampler = lr_sampler_generic(minlr, maxlr)
lr_preclassifier = generator_randomlr_neurons(net.preclassifier, lr_sampler)

# We define the optimizer
optimizer = SGDAlrao(net.parameters_preclassifier(),
                      lr_preclassifier,
                      net.classifiers_parameters_list(),
                      classifiers_lr)
```

Finally, we can train the model. The only differences here with the
usual training procedure is that each classifier needs to be updated as
if it was alone, and that we need to update the model averaging weights,
here the switch weights.

``` {language="python"}
def train(epoch):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        # We update the model averaging weights in the optimizer
        optimizer.update_posterior(net.posterior())
        optimizer.zero_grad()

        # Forward pass of the Alrao model
        outputs = net(inputs)
        loss = nn.NLLLoss(outputs, targets)

        # We compute the gradient of all the model's weights
        loss.backward()

        # We reset all the classifiers gradients, and re-compute them with
        # as if their were the only output of the network.
        optimizer.classifiers_zero_grad()
        newx = net.last_x.detach()
        for classifier in net.classifiers():
            loss_classifier = criterion(classifier(newx), targets)
            loss_classifier.backward()

        # Then, we can run an update step of the gradient descent.
        optimizer.step()

        # Finally, we update the model averaging weights
        net.update_switch(targets, catch_up=False)
```

[^1]: Facebook AI Research, Paris, France

[^2]: Université Paris Sud, INRIA, equipe TAU, Gif-sur-Yvette, France

[^3]: Equal contribution

[^4]: With learning rates resampled at each time, each step would be, in
    expectation, an ordinary SGD step with learning rate
    $\mathbb{E}\eta_{l,i}$, thus just yielding an ordinary SGD
    trajectory with more noise.
