import logging

from fue.config import Config
from fue.data import Data
from fue.models.ml_model import MLUncertaintyModel
from fue.tuner import HyperparameterTuner
from fue.utils import generate_run_id

# 1. Configure the global logger for the execution script
# This binds the internal package loggers to your console output
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"
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
        "max_iter": [250, 500, 1000, 2000],  # Kept static to cap execution time
    }

    # 6. Initialize the Tuner
    tuner = HyperparameterTuner(
        model_class=MLUncertaintyModel, param_grid=param_grid, optimization_target="global", optimization_metric="RMSE"
    )

    # 7. Generate Tracker ID and Execute
    run_id = generate_run_id(purpose="hpo_mlp")
    logger.info("Execution tracker ID generated: %s", run_id)

    best_model, history = tuner.search(
        train_df=train_df, val_df=val_df, feature_columns=feature_columns, target_columns=target_columns, run_id=run_id
    )

    logger.info("HPO Pipeline completed successfully!")
    logger.info("Champion model saved safely to '.fue/runs/%s'", run_id)


if __name__ == "__main__":
    main()
