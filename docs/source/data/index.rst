Data Pipeline & Architecture
============================

Welcome to the data layer of the package. Before we can train any uncertainty estimators, we need a robust way to download, store, clean, and organize our meteorological data. Because weather data is highly dependent on time and geography, we cannot just throw raw numbers into a standard machine learning algorithm. We have to handle it with care to prevent data leakage and ensure our models learn genuine physical relationships.

On this page, I will walk you through the core data components, where files are stored, and exactly how we transform raw API downloads into strict, training-ready datasets.

General Overview of Components
------------------------------

Our data architecture is built around three main components:

``Data`` Class
^^^^^^^^^^^^^^
Think of this as the main engine of the data pipeline. It orchestrates everything. When you initialize an instance of the ``Data`` class, it manages reading from your local hard drive, automatically fixing data types, merging datasets, and performing the complex chronological and geographical splitting required for our machine learning models.

``OpenMeteoClient`` Class
^^^^^^^^^^^^^^^^^^^^^^^^^
This is our dedicated network messenger. It handles the actual internet communication with the Open-Meteo API. Instead of writing raw web requests, this class uses a caching session (so we don't accidentally spam the API and get blocked) and a retry mechanism to handle random network drops smoothly. It downloads the data and formats it directly into a clean Pandas DataFrame for the ``Data`` class to consume.

Raw Forecast Data
^^^^^^^^^^^^^^^^^
This is our persistent historical archive. Every time you pull new forecasts from the API, they are appended to a central dataset on your local machine. Over time, this file grows, capturing both what the weather models predicted and what actually happened.

Raw Data Format and File Location
---------------------------------

By default, all of your raw weather data is stored as a standard CSV file located in your project workspace at ``data/raw/forecasts.csv``. 

Every row in this file represents a single weather snapshot. To make our uncertainty math work, the two most important columns in this file are:
* ``forecasted_on``: The exact timestamp when the prediction was generated.
* ``forecast_for``: The future timestamp that the prediction is actually describing.

The Data Processing Pipeline
----------------------------

To train our models, we need to know exactly how wrong a forecast was. But raw API data doesn't explicitly tell us the error margin — we have to calculate it ourselves. When you call the ``generate_dataset()`` method, the ``Data`` class runs a strict assembly line to build our final training matrix.

Here is how it works under the hood:

1. **The 12-Hour Boundary Rule:** We split the raw data into two piles based on the time difference between ``forecasted_on`` and ``forecast_for``. If the look-ahead horizon is longer than 12 hours, we treat it as a **forecast**. If the horizon is shorter than 12 hours, we treat it as a **ground-truth measurement**. (Because Open-Meteo updates past days with high-resolution reanalysis data, these short-horizon rows act as highly accurate historical observations).
2. **Calendar Merging:** We strip away the exact hours and match the "forecast" rows with their corresponding "ground-truth" rows for that exact calendar day. 
3. **Calculating Absolute Error:** We subtract the true observed weather from the forecasted weather and take the absolute value. This gives us our exact target variables (e.g., ``abs_diff__temperature_2m_max``).
4. **Feature Engineering:** We inject a few extra helper columns to give our models context. We calculate ``delta_days`` (how many days in advance the forecast was made) and ``day_of_year`` (to help the model understand seasonal atmospheric changes).
5. **Cleanup:** Finally, we drop any incomplete rows and strip out the temporary mathematical columns, leaving us with a matrix ready for training.

Data Splitting (Preventing Leakage)
-----------------------------------

When it comes to weather, you cannot just randomly shuffle your data into 80% training and 20% validation sets. Weather is highly correlated. If it is raining in Munich today, it is probably raining in neighboring cities too. If we randomly shuffle neighboring cities into both sets, the model will "cheat" by memorizing a specific regional storm rather than actually learning how to predict uncertainty.

When you call ``split_dataset()``, the package uses a specialized spatiotemporal strategy:

* **Dynamic Filtering:** First, the pipeline ignores any cities that haven't accumulated enough historical data (the default threshold is 100 valid rows).
* **Spatial Pairing:** Next, it calculates the geographic distance between all considered cities using the Haversine formula and groups neighboring cities into pairs.
* **Local Splits:** It then splits the pairs! One city goes entirely into the training set, and its neighbor goes entirely into the validation set. This forces the model to prove it can generalize its uncertainty rules to an "unseen" microclimate, guaranteeing that our validation scores are honest and robust.

Other Important Remarks
-----------------------

* **Duplicate Handling:** Because you might run the download command multiple times a day, your raw CSV might accidentally pull overlapping predictions. The ``Data`` class handles this safely by rounding coordinates to a tiny decimal tolerance and silently dropping exact duplicate rows before saving to disk.
* **Unit Safety:** Pay attention to your weather units. For example, raw sunshine duration is provided by the API in seconds, which creates massive numbers that can break machine learning gradients. The ``Data`` class automatically converts these into standard hours behind the scenes to keep the math stable.

Data Source
===========

.. raw:: html

   <p>Weather data by <a href="https://open-meteo.com/">Open-Meteo.com</a></p>