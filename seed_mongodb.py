"""
seed_mongodb.py — one-time loader: pushes your CSV into MongoDB Atlas.

Setup (free):
  1. Create a free MongoDB Atlas M0 cluster at https://www.mongodb.com/cloud/atlas
  2. Create a database user + allow your IP (or 0.0.0.0/0 for dev).
  3. Copy the connection string and put it in a .env file:
        MONGODB_URI="mongodb+srv://user:pass@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority"
  4. Run:  python seed_mongodb.py

After seeding, the ZenML pipeline's ingest step reads from MongoDB automatically.
"""
import os
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

CSV_PATH = os.getenv(
    "SEED_CSV_PATH",
    "./data/raw_data/predictive_maintenance_dataset.csv",
)
DB_NAME = os.getenv("MONGO_DB", "predictive_maintenance")
COLLECTION = os.getenv("MONGO_COLLECTION", "sensor_readings")


def main():
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise SystemExit("MONGODB_URI not set. Put it in a .env file (see header).")

    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"Rows to insert: {len(df)}")

    client = MongoClient(uri)
    col = client[DB_NAME][COLLECTION]

    # idempotent: clear then insert, so re-running doesn't duplicate rows
    col.delete_many({})
    # insert in batches to stay light on the free tier
    records = df.to_dict("records")
    BATCH = 5000
    for i in range(0, len(records), BATCH):
        col.insert_many(records[i:i + BATCH])
        print(f"  inserted {min(i + BATCH, len(records))}/{len(records)}")

    print(f"Done. {col.count_documents({})} documents in {DB_NAME}.{COLLECTION}")
    client.close()


if __name__ == "__main__":
    main()
