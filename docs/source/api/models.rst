Models API Reference
====================

This section details the statistical and machine learning estimators used 
to model absolute deviations and calculate forecast uncertainty constraints.
The parent class ``UncertaintyModel`` implements functionality, that is 
independent from the exact mathematical model like error evaluation, saving
a model with its parameters and loading the model with its learnt parameters. 
The specific mathematical models inherit from this parent class and are only 
responsible for fitting and predicting.

.. automodule:: fue.models
   :members:
   :undoc-members:
   :show-inheritance: