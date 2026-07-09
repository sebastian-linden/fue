Phase 1: Core Modeling & Internal Auditing (The Foundation)

1. Verification / Validation

Why it's #1: Your architecture is uniquely prone to chronological data leakage (cross-validating sequentially over look-ahead horizons, dates, and microclimates across 50 European cities). Before tracking runs or optimizing hyperparameters, your multi-city train/validation splitting engine must be rock-solid. Testing validation behavior guarantees that the statistics you log later are truthful.

2. Logging

Why it's #2: Application logging acts as your primary internal diagnostic tool. By implementing the NullHandler() architecture inside the package modules now, you provide immediate debugging context for your data splitting loops, API requests, and model iterations. This needs to exist before you wrap optimization engines or CLIs around it.

3.  Store Models & Experiment Tracking (Run Artifacts)

Why it's #3: Now that the pipeline prints clear diagnostic statements and splits data cleanly, you need to persist model weights alongside their evaluation metrics (meta.json error norms). Saving the model checkpoints (e.g., using pickle or joblib) indexed under the exact same run_id as your performance metrics bridges standard application behavior with scientific experimentation.

Phase 2: Engine Expansion & Execution (The Workbench)

4. Hyperparameter Optimization (HPO)

Why it's #4: You shouldn't try to build an automated search matrix (like Optuna or GridSearch) until your code naturally produces a run_id and saves its metrics automatically. Because Phase 1 sets up the structured recording architecture, your HPO routine becomes incredibly clean: it loops over configurations, triggers model.fit(), records validation logs natively via your tracking subsystem, and chooses the best iteration effortlessly.

5. Command Line Interface (CLI)

Why it's #5: Your existing typer scaffold already contains commands like download. Now that the engine handles training, tracking, and evaluation autonomously behind the scenes, you can expose these as explicit upper-level workflow commands (e.g., fue train, fue evaluate, fue compare, fue tune). The CLI simply acts as a shallow user-facing wrapper around your core API methods.

Phase 3: Communication & Dissemination (The Interface)

6. Documentation

Why it's #6: This involves completing detailed inline docstrings (such as the NumPy/Google format you've been establishing) and potentially compiling API layouts with a framework like Sphinx or MkDocs. Writing this after stabilizing your model storage and CLI endpoints prevents you from wasting time updating changing method interfaces.

7. README / Demo

Why it's #7: A pristine README.md acts as your project's front page. It requires a code example illustrating a complete workflow (Initializing Data $\rightarrow$ Training an UncertaintyModel $\rightarrow$ Querying a Forecast visualization). It must be written when the workflow syntax is locked in and will not undergo further breaking updates.

8. PyPI Upload

Why it's #8: Uploading your source distribution to PyPI using modern tools like hatch, flit, or twine is the definitive final step. It requires a complete metadata manifest, an unchangeable version identifier, and a final documentation build. You only publish to an external ecosystem once your software is completely verified, packaged, and documented locally.










Looking at you suggestions, I agree. I did actually try to implement a scheduler in Ubuntu a few months ago, but it always broke after a few days. Explain what you mean with the GitHub actions scheduler! I am using git + GitHub, but how would I use the online runner to fetch data and somehow place it in the right spot on my machine?

I attached some scripts, that use my package to give you an idea of how to implement some of your functions. Here is a list of CLI functions, I want you to implement:
- fue train --model [ml|linear]: This command should use Data() to read raw data, turn it into a dataset split (train_df, val_df). This should also store the model with a run_id, because I guess otherwise it will just be gone.
- fue evaluate: Loads the latest model (can be identified via the run_id) and evaluates it against a data_val. Now of course there is the question of whether to generate another dataset from the data and use the val_df from that to evaluate or to somehow store the val_df? But I guess this is against the spirit of a CLI, right? If no model exists, ask the user to run the "fue train..." command.
- fue forecast --loc [aachen|...]: This fetches a forecast for the next days with uncertainty estimated using the latest model. If no model exists, ask the user to run the "fue train..." command.

Before writing any code, answer my questions and write a rough structure of how those commands will be implemented.

---

The folder structure is like this:
root/runs/run_id/model.joblib
root/runs/run_id/meta.json
root/scripts/...
root/src/fue/...
root/src/fue/models/...
Commands should be executed form the root directory and the runs/ folder in the current directory should be used as a source for our previously trained/stored models. Also be aware, that linear and ml models take different types of parameters. We could solve this by optionally pass parameters for any of the two models, but depending on which model is chosen in the command options, the relevant parameters are used. Also I agree that we should have an hpo command with optional parameters corresponding to the tuner() class. Also what is more common as a name for this, "tune" or "hpo"? In this last cli command, I want to establish a logging level of DEBUG as default, but somehow give the user a feedback regarding the iteration. Is there a way of implementing a progress bar or a percentage number going up dynamically as we have more and more iterations?

Now, implement these commands. For each, implement a small section, that prints some information about whether the command has worked or not. If there is
any problem, I will then provide you with the respective print messages.

---

the `fue forecast` command should fetch the forecast for the location, apply the model (compute uncertainties) and print the forecast including a uncertainty bound (+- abs_diff) This should be tabular with a row for each day and a column for the different target variables. Right now the target variables are:
    "default_target_columns": [    
        "abs_diff__temperature_2m_max",
        "abs_diff__temperature_2m_min",
        "abs_diff__precipitation_sum",
        "abs_diff__wind_speed_10m_mean",
        "abs_diff__precipitation_probability_mean"
    ]
Those strings should be abreviated with something like t_max, t_min, prcp_sum, prcp_prob and wind_mean. Those are just suggestions. Come up with the best / most established set of abbreviations here, but if you like mine, take those. For each entry in the table write "forecasted value (+- abs_diff) unit" like "23 (+- 2.3) °C". I did define units in the Forecast() class:

import logging
import warnings

import matplotlib.pyplot as plt

from .config import Config
from .data import Data
from .utils import daylength

logger = logging.getLogger(__name__)


class Forecast:
    """A class, that applies a fitted uncertainty model to a forecast, that can be either
    given or fetched from the Open-Meteo API via the OpenMeteoClient() class. This class also provides
    methods to visualize the forecast and its uncertainty predictions, if a model is given.

    Attributes:
        forecast: DataFrame with forecast data (set by fetch_forecast)
        uncertainty_model: Fitted uncertainty model (set by compute_uncertainties)
        uncertainty_predictions: DataFrame with uncertainty predictions (set by compute_uncertainties)
        past_days: Number of past days included in forecast
    """

    # Class-level plot specifications for known variables
    _KNOWN_COLORS = {
        # Raw features
        "temperature_2m_max": "crimson",
        "temperature_2m_min": "royalblue",
        "precipitation_sum": "teal",
        "sunshine_duration": "gold",
        "wind_speed_10m_mean": "slategray",
        "precipitation_probability_mean": "cornflowerblue",
        # Target variables
        "abs_diff__temperature_2m_max": "darkred",
        "abs_diff__temperature_2m_min": "navy",
        "abs_diff__precipitation_sum": "darkcyan",
        "abs_diff__sunshine_duration": "darkorange",
        "abs_diff__wind_speed_10m_mean": "dimgray",
        "abs_diff__precipitation_probability_mean": "mediumpurple",
    }
    _KNOWN_UNITS = {
        # Raw features
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "precipitation_sum": "mm",
        "sunshine_duration": "h",
        "wind_speed_10m_mean": "m/s",
        "precipitation_probability_mean": "%",
        # Target variables
        "abs_diff__temperature_2m_max": "°C",
        "abs_diff__temperature_2m_min": "°C",
        "abs_diff__precipitation_sum": "mm",
        "abs_diff__sunshine_duration": "h",
        "abs_diff__wind_speed_10m_mean": "m/s",
        "abs_diff__precipitation_probability_mean": "%",
    }
    _KNOWN_TITLES = {
        # Raw features
        "temperature_2m_max": "Maximum Temperature (2m)",
        "temperature_2m_min": "Minimum Temperature (2m)",
        "precipitation_sum": "Total Daily Precipitation",
        "sunshine_duration": "Daily Sunshine Duration",
        "wind_speed_10m_mean": "Mean Wind Speed (10m)",
        "precipitation_probability_mean": "Mean Precipitation Probability",
        # Target variables
        "abs_diff__temperature_2m_max": "Max Temperature Forecast Absolute Error",
        "abs_diff__temperature_2m_min": "Min Temperature Forecast Absolute Error",
        "abs_diff__precipitation_sum": "Precipitation Forecast Absolute Error",
        "abs_diff__sunshine_duration": "Sunshine Duration Forecast Absolute Error",
        "abs_diff__wind_speed_10m_mean": "Mean Wind Speed Forecast Absolute Error",
        "abs_diff__precipitation_probability_mean": "Precipitation Probability Forecast Absolute Error",
    }



----

The following script works, just make sure your cli implementation works similarly:
import logging

from fue.config import Config
from fue.data import Data
from fue.models.ml_model import MLUncertaintyModel
from fue.tuner import HyperparameterTuner
from fue.utils import generate_run_id

# 1. Configure the global logger for the execution script
# This binds the internal package loggers to your console output
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing HPO Pipeline...")

    # 2. Setup Configuration and Features
    config = Config()
    feature_columns = config.default_feature_columns
    target_columns = config.default_target_columns

    # 3. Data Ingestion & Pooling
    data = Data()
    data.read_raw()
    dataset = data.generate_dataset()
    
    # 4. Strict Chronological Split
    logger.info("Executing chronological train/validation split to prevent leakage...")
    train_df, val_df = data.split_dataset(dataset, val_fraction=0.2)
    logger.info("Split complete. Train records: %d | Val records: %d", len(train_df), len(val_df))

    # 5. Define the Hyperparameter Grid
    param_grid = {
        "hidden_layer_sizes": [(32, 16), (16, 8), (16,)],
        "alpha": [0.001, 0.01, 0.1],
        "ensemble_size": [3, 5],
        "max_iter": [250, 500, 1000, 2000] # Kept static to cap execution time
    }

    # 6. Initialize the Tuner
    tuner = HyperparameterTuner(
        model_class=MLUncertaintyModel,
        param_grid=param_grid,
        optimization_target="global",
        optimization_metric="RMSE"
    )

    # 7. Generate Tracker ID and Execute
    run_id = generate_run_id(purpose="hpo_mlp")
    logger.info("Execution tracker ID generated: %s", run_id)
    
    best_model, history = tuner.search(
        train_df=train_df,
        val_df=val_df,
        feature_columns=feature_columns,
        target_columns=target_columns,
        run_id=run_id
    )

    logger.info("HPO Pipeline completed successfully!")
    logger.info("Champion model saved safely to '.fue/runs/%s'", run_id)


if __name__ == "__main__":
    main()

Also here is the actual tuner class as a reference:

import logging
from typing import Type, Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from .models import MLUncertaintyModel
from .utils import generate_run_id

# Initialize module-scoped logger
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Orchestrates in-memory hyperparameter optimization for FUE Uncertainty Models.
    
    This tuner safely respects temporal data splits by taking pre-separated 
    training and validation dataframes. It performs an exhaustive grid search 
    in memory, tracks validation metrics for all configurations, and writes 
    only the highest-performing model to disk.
    """

    def __init__(
        self,
        model_class: Type[MLUncertaintyModel],
        param_grid: Dict[str, list],
        optimization_target: str = "global",
        optimization_metric: str = "RMSE"
    ):
        """
        Initializes the tuner with a model class and a search space.

        Parameters
        ----------
        model_class : Type[UncertaintyModel]
            The uninstantiated class of the model to tune (e.g., MLUncertaintyModel).
        param_grid : Dict[str, list]
            Dictionary with parameters names (`str`) as keys and lists of parameter 
            settings to try as values.
        optimization_target : str, optional
            The specific target column to optimize for (e.g., 'abs_diff__temperature_2m_max').
            If set to "global" (default), minimizes the average error across all targets.
        optimization_metric : str, optional
            The metric key to minimize. Usually "RMSE" (default) or "MAE".
        """
        self.model_class = model_class
        self.param_grid = param_grid
        self.optimization_target = optimization_target
        self.optimization_metric = optimization_metric

    def _compute_objective(self, metrics_dict: Dict[str, Dict[str, float]]) -> float:
        """
        Collapses a multi-target validation metrics dictionary into a single float scalar.

        Parameters
        ----------
        metrics_dict : Dict[str, Dict[str, float]]
            The nested dictionary returned by the `evaluate()` method.

        Returns
        -------
        float
            The scalar value to be minimized.
        """
        if self.optimization_target == "global":
            # Extract the specific metric (e.g., RMSE) for every target variable and average it
            scores = [
                target_metrics[self.optimization_metric] 
                for target_metrics in metrics_dict.values()
            ]
            return float(np.mean(scores))
        else:
            # Route directly to the user-specified priority target
            return metrics_dict[self.optimization_target][self.optimization_metric]

    def search(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame, 
        feature_columns: list, 
        target_columns: list,
        run_id: str | None = None
    ) -> Tuple[MLUncertaintyModel, List[Dict[str, Any]]]:
        """
        Executes the grid search loop, tracking configurations in memory.

        Parameters
        ----------
        train_df : pd.DataFrame
            The chronologically split training dataset.
        val_df : pd.DataFrame
            The chronologically split validation dataset.
        feature_columns : list
            List of raw feature column names to feed to the model.
        target_columns : list
            List of absolute difference target column names to predict.
        save_run_id : str, optional
            The tracker key under which the champion model will be persisted.

        Returns
        -------
        Tuple[UncertaintyModel, List[Dict[str, Any]]]
            A tuple containing the fitted champion model object and the complete 
            history of all evaluated configurations and their scores.
        """
        grid = list(ParameterGrid(self.param_grid))
        total_runs = len(grid)

        if run_id is None:
            run_id = generate_run_id(purpose="hpo")
            logger.debug("No explicit run_id provided for HPO. Auto-generated identifier: %s", run_id)

        logger.info("Starting HPO sweep. Total configurations to evaluate: %d", total_runs)

        best_score = float("inf")
        best_model = None
        best_metrics = None
        history = []

        for i, params in enumerate(grid):
            # 1. Instantiate the model dynamically with the current grid slice
            model = self.model_class(**params)
            
            # 2. Train on the 80% pool
            model.fit(train_df, feature_columns, target_columns)
            
            # 3. Evaluate on the 20% unseen pool
            metrics = model.evaluate(val_df)
            
            # 4. Collapse metrics to a single optimization scalar
            current_score = self._compute_objective(metrics)
            
            # 5. Track leaderboard logic
            is_best = False
            if current_score < best_score:
                best_score = current_score
                best_model = model
                best_metrics = metrics
                is_best = True

            # 6. One-Liner State Logging
            # Output: "Iter 4/48 | Score (RMSE): 0.8421 | Best: 0.8101 | Params: {'alpha': 0.01, ...}"
            mark = "⭐" if is_best else "  "
            logger.info(
                "[%s] Iter %d/%d | Score (%s): %.4f | Best: %.4f | Params: %s",
                mark, i + 1, total_runs, self.optimization_metric, current_score, best_score, params
            )

            # Store iteration results strictly in memory
            history.append({
                "iteration": i + 1,
                "params": params,
                "score": current_score,
                "metrics": metrics
            })

        # 7. Persist the Champion Model to disk
        logger.info("HPO complete. Writing champion model to disk under run_id: '%s'.", run_id)
        
        # Package the HPO search metadata alongside the standard validation metrics
        tracking_metadata = {
            "hpo_winning_score": best_score,
            "hpo_optimization_metric": self.optimization_metric,
            "hpo_optimization_target": self.optimization_target,
            "validation_metrics": best_metrics
        }
        
        # We leverage Phase 1's save logic!
        best_model.save(run_id=run_id, metrics=tracking_metadata)

        return best_model, history

Finally, it would be great if you could make a model option available. Currently my tuner expectes a ml model "model_class: Type[MLUncertaintyModel]". Could we generalize this? Or does HPO only make sense with machine learning models anyways and not with regression models? The script above that performes the hpo spits out a bunch of warnings like this:
me/linden/code/uni/sce/fue/.venv/lib/python3.14/site-packages/sklearn/neural_network/_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.
  warnings.warn(
/home/linden/code/uni/sce/fue/.venv/lib/python3.14/site-packages/sklearn/neural_network/_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.
  warnings.warn(
/home/linden/code/uni/sce/fue/.venv/lib/python3.14/site-packages/sklearn/neural_network/_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.
  warnings.warn(
/home/linden/code/uni/sce/fue/.venv/lib/python3.14/site-packages/sklearn/neural_network/_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.
  warnings.warn(
/home/linden/code/uni/sce/fue/.venv/lib/python3.14/site-packages/sklearn/neural_network/_multilayer_perceptron.py:785: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (250) reached and the optimization hasn't converged yet.
I would say that we mute them somehow, since they are indirectly considered if the output model is not optimal. If a model that has not converged is in fact optimal, then we don't really care about this warning neither.
