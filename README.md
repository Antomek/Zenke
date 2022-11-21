# README

A sandbox repo to play around with automatic differentiation.

The goal is to reproduce [this tutorial from Zenke](https://github.com/fzenke/spytorch/blob/main/notebooks/SpyTorchTutorial1.ipynb).

A starting point is in ``minimal_example.jl``,
where we aim to show that surrogate gradients work for a two coupled neurons.

We define a LIF neuron with dynamics:

$$
\begin{aligned}
\frac{dV_i}{dt} &= (-V_i + I_i) / \tau_V, \\
S_i &= H(V_i - \theta), \\
I_i &= -I_i / \tau_{\text{syn}} + \sum_j w_{ij} S_j(t).
\end{aligned}
$$

Furthermore the neurons are equipped with a reset:

$$ V \ge \theta: V_{\text{rest}} \leftarrow V.$$

(We work with $\theta = 1$ and $V_{\text{rest}} = 0$.)

In order for the gradients to flow,
we have to replace the gradients of the reset and those of the Heaviside step function $H(x)$.

The main simulation is in ``Zenke.jl``,
with some helper functions defined in the other files,
which are imported in this main file.
