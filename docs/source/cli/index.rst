======================
Command Line Interface
======================

As I mentioned in the Data Management Workflow, setting up instances and running pipelines in Python works perfectly fine, but using the terminal is often the much more convenient solution. 

The Forecast Uncertainty Estimation (FUE) package comes with a built-in Command Line Interface (CLI) powered by Typer. It gives you terminal-based execution for all the core pipeline tasks: fetching data, checking your inventory, training models, and generating those final uncertainty bounds.

Below is a breakdown of the available commands and how they fit into the workflow.

Downloading Data
----------------

Estimating the expected error of a forecast relies entirely on comparing past forecasts with the true weather outcomes. Because weather models overwrite their data every 3-6 hours, you have to conserve old forecasts by fetching and storing them regularly. 

To pull the newest available forecasts for your target cities via the Open-Meteo API, simply use:

.. code-block:: bash

   fue download

This command fetches the data, ensures it has the right format, discards duplicates, and stores it. 

**Note:** The CLI command doesn't have the option to pass a custom path. It uses the default location (``data/raw/forecasts.csv``). If you want to store your data somewhere else, you will either need to modify ``cli.py`` or write your own Python script using the ``Data`` object.

Checking Your Inventory
-----------------------

If you don't have any data yet, you will have to spend a considerable amount of time collecting it. Before you can really start predicting uncertainty bounds, you need to know if you have enough chronological records to actually train a model. 

.. code-block:: bash

   fue dataset-summary

This evaluates your local data storage and outputs a formatted table of your pipeline readiness. By default, it requires a minimum of 100 valid records for a city to be considered "ACTIVE". If you want to change that threshold, you can pass it directly:

.. code-block:: bash

   fue dataset-summary --threshold 150

Training and Tuning Models
--------------------------

Once you have enough active cities, you can move on to training. The training command ingests your pooled meteorological data and applies a strict chronological test-train partition. This is important to protect against temporal leakage (we don't want to predict the past using future weather patterns).

.. code-block:: bash

   fue train --model ml --layers "16,8" --alpha 0.01

You can choose between a Multi-Layer Perceptron (``--model ml``) or Multi-Output Linear Regression (``--model linear``). When it finishes, it dumps the binary artifacts and run metadata into a local ``runs/`` directory. In other words, it stores your trained model using a unique run_id.

If you don't want to guess the best parameters, you can use the following command to perform a hyper parameter optimization (HPO) automatically. Here, the default model is the `MLUncertaintyModel`. You can adjust the search space in the `Tuner()` class if you want.

.. code-block:: bash

   fue tune

This executes a grid search across your active dataset. It keeps the terminal quiet to avoid flooding your screen, instead sending detailed iteration logs to ``tune_debug.log``. The best performing "champion" model is automatically stored to your ``runs/`` directory.

Evaluating Performance
----------------------

Did the model actually learn anything useful? You can validate a saved model's predictive skill against your validation dataset using:

.. code-block:: bash

   fue evaluate

This looks for the latest run in your ``runs/`` folder (though you can specify one with ``--run-id``) and prints a tabular layout of absolute residual scores, breaking down the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for each meteorological target.

Generating Forecasts
--------------------

Finally, once your model is trained, you can pull live weather predictions for a specific city and run them through your checkpoint model to get forward operational forecasts decorated with estimated uncertainty bounds.

.. code-block:: bash

   fue forecast --city aachen --days 7

This maps out a daily terminal layout highlighting expected physical forecast fluctuations (like ± temperature or precipitation). 

If you prefer visuals over terminal tables, you can pop open an interactive Matplotlib window instead by adding the plot flag:

.. code-block:: bash

   fue forecast --city aachen --days 7 --plot