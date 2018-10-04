<script type="text/x-mathjax-config">
   MathJax.Hub.Config({
     tex2jax: {
       skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
       inlineMath: [['$','$']],
       displayMath: [ ['$$','$$'] ],
     }
   });
 </script>
 <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


Introduction
============

We propose *All Learning Rates At Once* (Alrao), an alteration of
standard optimization methods for deep learning models. Alrao uses
multiple learning rates at the same time in the same network. By
sampling one learning rate per feature, Alrao reaches performance close
to the performance of the optimal learning rate, without having to try
multiple learning rates. Alrao can be used on top of various
optimization algorithms; our experimental results are obtained
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

#### Motivation.

Alrao was inspired by the intuition that not all units in a neural
network end up being useful. Hopefully, in a large enough network, a
sub-network made of units with a good learning rate could learn well,
and hopefully the units with a wrong learning rate will just be ignored.
(Units with a too large learning rate may produce large activation
values, so this assumes the model has some form of protection against
those, such as BatchNorm or sigmoid/tanh activations.)

Several lines of work support the idea that not all units of a network
are useful or need to be trained, for example on *pruning* trained networks
[@lecun1990; @Han2015; @Han2015a; @See], or the *lottery ticket hypothesis* [@Frankle2018].

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
