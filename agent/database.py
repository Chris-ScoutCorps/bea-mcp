from dotenv import load_dotenv
load_dotenv()

import os
from enum import Enum
from typing import Optional
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

class Collections(Enum):
    """Enum for MongoDB collection names"""
    DATASETS = "datasets"

# MongoDB connection
def get_mongo_client() -> MongoClient:
    """Get MongoDB client using MONGO_URI from environment variables"""
    mongo_uri = os.getenv('MONGO_URI')
    if not mongo_uri:
        raise ValueError("MONGO_URI environment variable is not set")
    
    try:
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ismaster')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        raise ConnectionError(f"Failed to connect to MongoDB: {e}")

def get_database() -> Database:
    """Get the database from MongoDB client"""
    return get_mongo_client()['BEA']

def ensure_collection(name: str) -> Collection:
    """Ensure the collection exists in the database"""
    db = get_database()
    
    # Check if collection exists
    if name not in db.list_collection_names():
        # Create the collection
        db.create_collection(name)

    return db[name]

def upsert_dataset(dataset_name: str, dataset_description: str) -> bool:
    """
    Upsert a dataset in the datasets collection.
    
    Args:
        dataset_name: Unique name for the dataset
        dataset_description: Description of the dataset
    
    Returns:
        True if successful, False otherwise
    """
    collection = ensure_collection(Collections.DATASETS.value)
    
    try:
        collection.update_one(
            {'DatasetName': dataset_name},
            {
                '$set': {
                    'DatasetName': dataset_name,
                    'DatasetDescription': dataset_description
                }
            },
            upsert=True
        )
        return True
    except Exception as e:
        print(f"Error upserting dataset: {e}")
        return False
