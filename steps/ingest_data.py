import os
import logging
import pandas as pd
from zenml import step

logging.basicConfig(level=logging.INFO)


@step
def ingest_df(data_path: str = "") -> pd.DataFrame:
    """Ingest data. Prefers MongoDB (if MONGODB_URI is set), else falls back to a local CSV.

    Why this design:
    - In production, data lives in a database, not a CSV on someone's laptop. Reading from
      MongoDB makes the pipeline "production-shaped": new rows added to the collection are
      automatically picked up next run -> this is your "data changes -> pipeline adapts" story.
    - The CSV fallback keeps the pipeline runnable on a fresh machine / in CI where no DB exists.
    """
    try:
        mongo_uri = os.getenv("MONGODB_URI", "").strip()

        if mongo_uri:
            # ---- Production path: pull from MongoDB Atlas ----
            from pymongo import MongoClient  # lazy import so CSV-only users don't need pymongo

            db_name = os.getenv("MONGO_DB", "predictive_maintenance")
            col_name = os.getenv("MONGO_COLLECTION", "sensor_readings")

            logging.info(f"Ingesting from MongoDB -> db='{db_name}', collection='{col_name}'")
            client = MongoClient(mongo_uri)
            collection = client[db_name][col_name]

            # exclude Mongo's internal _id so the dataframe matches the original schema exactly
            documents = list(collection.find({}, {"_id": 0}))
            client.close()

            if not documents:
                raise ValueError("MongoDB collection is empty. Run seed_mongodb.py first.")

            df = pd.DataFrame(documents)
            logging.info(f"Fetched {len(df)} rows from MongoDB.")
            return df

        # ---- Fallback path: local CSV ----
        if not data_path:
            raise ValueError("No MONGODB_URI set and no data_path provided.")
        logging.info(f"MONGODB_URI not set. Ingesting from local CSV: {data_path}")
        df = pd.read_csv(data_path)
        return df

    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
