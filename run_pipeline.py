import os
import sys
import warnings
from pathlib import Path

# make project root importable on any OS
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
from pipelines.training_pipeline import maintenance_training_pipeline

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    # If MONGODB_URI is set in .env, ingest pulls from Mongo and data_path is ignored.
    # Otherwise it falls back to this CSV path.
    DATA_PATH = os.getenv(
        "SEED_CSV_PATH",
        "./data/raw_data/predictive_maintenance_dataset.csv",
    )
    maintenance_training_pipeline.with_options(
        config_path="config.yaml",
        enable_cache=False,
    )(data_path=DATA_PATH)
