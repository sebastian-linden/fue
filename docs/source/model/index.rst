Mathematical Background & Modeling
==================================

This section details the mathematical and algorithmic architecture used by the ``fue`` package to estimate forecast uncertainty. 

Rather than computing post-hoc statistical variances or classic standard deviations around a single machine learning prediction, this package reframes uncertainty estimation entirely. We treat uncertainty as a **primary target variable** by directly training estimators to predict the real-world absolute error magnitude of numerical weather simulations.

The Mathematical Formulation
-----------------------------

Let $y_{t, h}$ represent the raw numerical forecast generated for a specific weather variable at calendar day $t$ with a look-ahead horizon of $h$ days (where $h = \text{delta\_days}$). Let $\tilde{y}_{t}$ represent the actual ground-truth observation recorded for that calendar day.

We define the physical absolute forecast error, $e_{t, h}$, as:

.. math::

   e_{t, h} = | y_{t, h} - \tilde{y}_{t} |

Our models are explicitly optimized to find a predictive mapping function $f(\cdot)$ such that:

.. math::

   \hat{e}_{t, h} = f(X_{t, h})

Where $X_{t, h}$ is our design matrix containing predictors like seasonal timing (``day_of_year``), look-ahead time constraints (``delta_days``), and simulated atmospheric point states (like maximum temperature or precipitation sums).

The Base Architecture (``UncertaintyModel``)
--------------------------------------------

The abstract base class ``UncertaintyModel`` standardizes how features and targets flow into concrete mathematical operations. It isolates algorithm implementations from data preparation tasks by enforcing a strict two-stage cycle:

1. **Stateful Data Preparation:** The base class passes the raw input data frame through the ``Preprocessor``. Features are transformed according to your ``config.json`` rules (e.g., standardizing columns, applying Box-Cox corrections, or parsing circular wind vectors into sine-cosine coordinates).
2. **Subclass Hook Execution:** The cleaned data matrices are forwarded down to internal mathematical hooks (``_fit_internal`` and ``_predict_internal``) overridden by individual model variants.

Linear Modeling Approach
------------------------

The baseline algorithm implemented is the ``LinearUncertaintyModel``. This model uses standard multi-output ordinary least squares (OLS) linear regression to map features to absolute errors.

The linear system can be expressed as:

.. math::

   \mathbf{E} = \mathbf{X}_{processed} \cdot \mathbf{W} + \mathbf{B}

Where $\mathbf{E}$ represents our multi-column target error matrix, $\mathbf{W}$ is the weights coefficient matrix, and $\mathbf{B}$ is the bias intercept vector.

.. note::
   Because standard linear lines extend to infinity, the OLS equation can mathematically output negative values for short forecast horizons ($h \approx 0$). Since a physical error magnitude can never drop below zero, the model automatically clips outputs using an element-wise maximum constraint: $\max(0, \hat{e})$.

Nonlinear Machine Learning Approach
-----------------------------------

To model complex, multi-variable weather interactions that straight linear paths miss, we implement the ``MLUncertaintyModel``. This architecture utilizes a multi-layer neural network (Multi-Layer Perceptron Regressor) backed by an ensemble framework.

Neural Network Structure
^^^^^^^^^^^^^^^^^^^^^^^^
Features traverse fully connected hidden processing layers using Rectified Linear Unit (ReLU) activations to learn highly flexible nonlinear boundaries:

.. math::

   \mathbf{H}_1 = \text{ReLU}(\mathbf{X}_{processed} \cdot \mathbf{W}_1 + \mathbf{b}_1)

The network inherently excels at handling variable saturation—meaning that as forecast look-ahead horizons extend past major prediction limits (e.g., beyond 10-14 days), the network’s outputs naturally flatten into a stable error ceiling matching historical climatological variance.

Ensemble Aggregation
^^^^^^^^^^^^^^^^^^^^
Neural networks are highly sensitive to random weight initializations, especially on smaller data footprints. To stabilize our variance ceilings, the model trains an **Ensemble Pool** of size $N$ (configured in ``config.json``).

Each ensemble member is initialized with a distinct random seed ($S_i = \text{base\_seed} + i$). The final predicted uncertainty bound is calculated as the simple arithmetic average of the individual sub-model predictions:

.. math::

   \hat{e}_{final} = \frac{1}{N} \sum_{i=1}^{N} f_{MLP, i}(\mathbf{X}_{processed})

Hyperparameter Optimization (HPO)
---------------------------------

To locate the ideal structural setup for the neural networks, the ``HyperparameterTuner`` executes an automated grid search sweep across your parameter ranges.

The tuner collapses multi-variable prediction metrics down to a single optimization objective. When configured for a ``global`` target, it computes the mean error across all metrics simultaneously:

.. math::

   \text{Objective} = \frac{1}{M} \sum_{m=1}^{M} \text{Metric}_m

Where $M$ is the number of target weather variables and $\text{Metric}$ represents an error score vector like Root Mean Squared Error (``RMSE``) or Mean Absolute Error (``MAE``) calculated against unseen validation partitions.