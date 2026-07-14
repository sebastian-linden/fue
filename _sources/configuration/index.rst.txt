Configuration
=============

The source of configuration settings in this package is the ``config.json`` file located in the root directory. When you initialize an instance of the ``Config`` class, it reads this file, shares those settings with the other package components and provides built-in methods to update settings and save those changes back to the ``config.json`` file. On this page, I will walk you through each configuration entry so you know how to choose the right parameters for your run.

The ``config.json`` file uses a standard JSON key-value format that maps cleanly to a Python dictionary. When downloading weather data from the Open-Meteo API, the package passes a dictionary with the parameters specifying the location(s), the number of days the forecast should cover, and the meteorological variables you are interested in tracking. We also pass an explicit timezone setting, because capturing the exact local time boundary for each prediction block is important for our data alignment steps.

Initially the ``Config`` class was designed to handle only the parameters which are required by the Open-Meteo API. However, as the package grew, more and more settings were outsourced to the ``config.json`` file to preserve modularity and avoid hard-coding machine learning design choices directly into the Python code. 

In the following sections I walk you through each element of the configuration file so you can customize it for your own use.

Open-Meteo Parameters
---------------------

``cities``
^^^^^^^^^^
The Open-Meteo API can handle multiple locations simultaneously and they are passed internally as two lists containing latitudes and longitudes. For convenience, this package pairs the coordinates with the associated location name. For consistency city names are written in all lowercase letters. The cities dictionary looks as follows:

.. code-block:: json

    "cities": {
        "aachen": {
            "lat": 50.7753,
            "lon": 6.0839
        },
        "munich": {
            "lat": 48.1351,
            "lon": 11.582
        }
    }

``timezone``
^^^^^^^^^^^^
Choose the timezone corresponding to your local time. The API automatically adjusts the timestamp of the forecasts. To know which string to specify, have a look at the available options at `www.open-meteo.com/en/docs <https://open-meteo.com/en/docs>`_.

``forecast_days``
^^^^^^^^^^^^^^^^^
This parameter specifies how many days the forecast covers. A number of 1 means that the forecast returns estimated forecasts for today (0) and tomorrow (1). The default is 7, but many weather models produce forecasts of up to 14 days. This depends on the location(s) you specified. For more information look at the table on their website: `https://open-meteo.com/en/docs#data_sources <https://open-meteo.com/en/docs#data_sources>`_.

``past_days``
^^^^^^^^^^^^^
Open-Meteo also provides an option to fetch forecast for past days. If you pass a value larger than 0 the API returns historical forecasts alongside the future forecasts. In this package, we use these past days as our "ground truth" observations to calculate our forecast error targets. Open-Meteo continuously updates past days' fields with high-resolution reanalysis data (observed weather mapped back onto a grid) once the calendar day is complete.

``daily``
^^^^^^^^^
This is a list where you specify exactly which daily weather variables you want to download from the API. The string names you choose must match Open-Meteo’s official naming conventions (like ``temperature_2m_max`` or ``precipitation_sum``). The client uses the order of this list to map the incoming raw arrays correctly, so make sure they are valid. You can check the complete list of variables directly on the Open-Meteo documentation page: `https://open-meteo.com/en/docs#daily_weather_variables <https://open-meteo.com/en/docs#daily_weather_variables>`_.

Package Pipeline Settings
-------------------------

``preprocessing``
^^^^^^^^^^^^^^^^^
This section is a dictionary map where you assign custom data cleaning and math transformation rules to your weather columns. Before training a model, the ``Preprocessor`` looks at this dictionary to normalize data scales or handle tricky distributions. You can map variables to stateful scalers like ``"standard"`` or ``"min-max"``, apply adjustments like ``"log"``, ``"square"``, or ``"sqrt"``, or transform circular parameters (like angles) into coordinate pairs using ``"sin-cos"``. For example:

.. code-block:: json

    "preprocessing": {
        "precipitation_sum": "log",
        "wind_direction_10m_dominant": "sin-cos"
    }

``default_feature_columns``
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This is the baseline list of input features that our machine learning and linear models look at to learn when forecasts tend to be wrong. It acts as the default blueprint for your model's design matrix, packing together metrics like ``day_of_year`` (to capture seasonal changes), ``delta_days`` (how many days in advance the forecast was made), and specific predicted weather states.

``default_target_columns``
^^^^^^^^^^^^^^^^^^^^^^^^^^
This list defines the exact targets your models are trying to predict. In our case, these are the calculated physical absolute errors of the weather forecasts, which always carry the ``"abs_diff__"`` prefix (for example, ``abs_diff__temperature_2m_max``). The baseline ``UncertaintyModel`` checks this list during training and inference to map out multi-target predictions simultaneously.