================================================
Introduction to Recurrent Neural Networks (RNNs)
================================================

.. contents:: Table of Contents
    :depth: 2


Mathematical Formulation
------------------------

Recurrent neural networks, also known as RNNs, are a class of neural networks that allow previous outputs to be used as
part of inputs while having hidden states. They are typically as follows:

.. figure:: ../img/architecture-rnn-ltr.png
    :align: center

For each timestep :math:`t` the activation :math:`a^{\langle t \rangle}` and the output :math:`y^{\langle t \rangle}`
are expressed as follows:

.. math::

    h^{\langle t \rangle} = g_1\left( W_{hh}h^{\langle t - 1 \rangle} + W_{hx}x^{\langle t \rangle} + b_h \right)

    y^{\langle t \rangle} = g_2\left( W_{yh}h^{\langle t \rangle} + b_y \right)

where :math:`W_{hx}`, :math:`W_{hh}`, :math:`W_{yh}`, :math:`b_h`, :math:`b_y` are coefficients that are shared
temporally and :math:`g_1`, :math:`g_2` are activation functions. The diagram below visualizes the 2 formula above:

.. figure:: ../img/description-block-rnn-ltr.png
    :align: center

Our target is to learn the following 3 weight matrices

1. :math:`W_{hx}`
2. :math:`W_{hh}`
3. :math:`W_{yh}`

and 2 bias vectors:

1. :math:`b_h`
2. :math:`b_y`

Initialization
--------------

Before training, we need to initialize the :math:`W`, :math:`b`, and, don't forget, :math:`h_0`, the initial hidden
state. Each of them can be initialized using either of the 2 approaches:

1. All-zero
2. Random

Common literature [#f1]_ [#f2]_ tends to initialize

* *small* random values to :math:`W`
* all-zero to :math:`b`

Training
--------

.. admonition:: Prerequisite

    This section requires familiarity of basic Artificial Neural Network concepts, which are drawn from *Chapter 4 -
    Artificial Neural Networks* (p. 81) of `MACHINE LEARNING by Mitchell, Thom M. (1997)`_ Paperback. Please, if
    possible, read the chapter beforehand and refer to it if something looks confusing in the discussion of this section

What is the Error Function of RNN?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the discussion of Back `MACHINE LEARNING by Mitchell, Thom M. (1997)`_












In the case of a recurrent neural network, we are essentially backpropagation through time, which means that we are
forwarding through entire sequence to compute losses, then backwarding through entire sequence to compute gradients.
Formally, the `loss function`_ :math:`\mathcal{L}` of all time steps is defined as the sum of
the loss at every time step:

.. math::

    \mathcal{L}\left( \hat{y}, y \right) = \sum_{t = 1}^{T_y}\mathcal{L}\left( \hat{y}^{<t>}, y^{<t>} \right)

However, this becomes problematic when we want to train a sequence that is very long. For example, if we were to train a
a paragraph of words, we have to iterate through many layers before we can compute one simple gradient step. In
practice, for the back propagation, we examine how the output at the very *last* timestep affects the weights at the
very first time step. Then we can compute the gradient of loss function, the details of which can be found in the
`Vanilla RNN Gradient Flow & Vanishing Gradient Problem`_

.. admonition:: Gradient Clipping

    Gradient clipping is a technique used to cope with the `exploding gradient`_ problem sometimes encountered when
    performing backpropagation. By capping the maximum value for the gradient, this phenomenon is controlled in
    practice.

    .. figure:: ../img/gradient-clipping.png
        :align: center

    In order to remedy the vanishing gradient problem, specific gates are used in some types of RNNs and usually have a
    well-defined purpose. They are usually noted :math:`\Gamma` and are defined as

    .. math::

        \Gamma = \sigma(Wx^{<t>} + Ua^{<t - 1>} + b)

    where :math:`W`, :math:`U`, and :math:`b` are coefficients specific to the gate and :math:`\sigma` is the sigmoid
    function

LSTM Formulation
^^^^^^^^^^^^^^^^

Now we know that Vanilla RNN has Vanishing/exploding gradient problem, `LSTM Formulation`_ discusses the theory of LSTM
which is used to remedy this problem.

Applications of RNNs
--------------------

RNN models are mostly used in the fields of natural language processing and speech recognition. The different
applications are summed up in the table below:

.. list-table:: Applications of RNNs
   :widths: 20 60 20
   :align: center
   :header-rows: 1

   * - Type of RNN
     - Illustration
     - Example
   * - | One-to-one
       | :math:`T_x = T_y = 1`
     - .. figure:: ../img/rnn-one-to-one-ltr.png
     - Traditional neural network
   * - | One-to-many
       | :math:`T_x = 1`, :math:`T_y > 1`
     - .. figure:: ../img/rnn-one-to-many-ltr.png
     - Music generation
   * - | Many-to-one
       | :math:`T_x > 1`, :math:`T_y = 1`
     - .. figure:: ../img/rnn-many-to-one-ltr.png
     - Sentiment classification
   * - | Many-to-many
       | :math:`T_x = T_y`
     - .. figure:: ../img/rnn-many-to-many-same-ltr.png
     - Named entity recognition
   * - | Many-to-many
       | :math:`T_x \ne T_y`
     - .. figure:: ../img/rnn-many-to-many-different-ltr.png
     - Machine translation

.. rubric:: Footnotes

.. [#f1] https://lucyliu-ucsb.github.io/posts/Backpropagation-of-a-vanilla-RNN/#initialize-parameters
.. [#f2] https://gist.github.com/karpathy/d4dee566867f8291f086


.. _`exploding gradient`: https://qubitpi.github.io/stanford-cs231n.github.io/rnn/#vanilla-rnn-gradient-flow--vanishing-gradient-problem

.. _`MACHINE LEARNING by Mitchell, Thom M. (1997)`: https://a.co/d/bjmsEOg

.. _`loss function`: https://qubitpi.github.io/stanford-cs231n.github.io/neural-networks-2/#losses
.. _`LSTM Formulation`: https://qubitpi.github.io/stanford-cs231n.github.io/rnn/#lstm-formulation

.. _`Vanilla RNN Gradient Flow & Vanishing Gradient Problem`: https://qubitpi.github.io/stanford-cs231n.github.io/rnn/#vanilla-rnn-gradient-flow--vanishing-gradient-problem
