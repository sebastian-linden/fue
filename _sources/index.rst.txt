fue documentation
=================

Welcome to the documentation for the Forecast Uncertainty Estimation (fue) package. 
It is built around the idea that a weather forecasts are reliable only to a certain
degree. There might be significant uncertainty attached to forecasts, that are not
clearly stated in the apps we use daily. This hobby project aims at finding these
deviations between the forecasted weather and the true realization of the weather
several days later, when weather station measurements show us how accurate a forecast
really was. I had the idea for this project for a while and now that I am taking a 
course on sustainable computational engineering (aka how to write code that others 
can understand and use), I finally had a good reason to realize the project.

The project is split into two modules: the core **fue** and the **fue.models** module.
The core **fue** module includes fetching forecasting data from the open-meteo.com API, cleaning
and preprocessing of data and the generation of valid traing and validation datasets.
**fue.models** is focused on using a mathematical model to map a certain forecast
to attached uncertainties.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Home <self>
   Quickstart Guide <notebooks/quickstart>
   Forecast Data <data/index>
   Mathematical Background & Modeling <model/index>
   Configuration Guide <configuration/index>

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/fue
   api/models