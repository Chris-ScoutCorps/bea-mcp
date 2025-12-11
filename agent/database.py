from dotenv import load_dotenv
load_dotenv()

import os
from enum import Enum
from typing import Optional, List, Dict, Any
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError, OperationFailure

from embeddings import embed_documents
from logger import info

class Collections(Enum):
    """Enum for MongoDB collection names"""
    DATASETS = "datasets"
    DATA_LOOKUP = "data_lookup"

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
    """Return a handle to the configured database (created lazily on first write).

    Uses env var MONGO_DB (default 'BEA'). MongoDB creates a database implicitly when a
    collection is first written to; no explicit 'create database' command exists.
    """
    db_name = os.getenv('MONGO_DB', 'BEA')
    client = get_mongo_client()
    return client[db_name]

def ensure_database_exists() -> Database:
    """Force creation of the database by creating a lightweight marker collection if empty.

    If the database has zero collections we create a temporary 'db_init_marker' collection
    (if not present) and immediately drop it. This guarantees the database files exist.
    """
    db = get_database()
    try:
        if not db.list_collection_names():
            temp_name = 'db_init_marker'
            if temp_name not in db.list_collection_names():
                db.create_collection(temp_name)
            db[temp_name].drop()
    except Exception as e:
        info(f"Warning: ensure_database_exists encountered error (non-fatal): {e}")
    return db

def ensure_collection(name: str) -> Collection:
    """Ensure the collection exists in the database (creating database implicitly if needed)."""
    db = ensure_database_exists()
    
    # Check if collection exists
    if name not in db.list_collection_names():
        # Create the collection
        db.create_collection(name)

    return db[name]

def upsert_dataset(dataset_name: str, dataset: Dict[str, str]) -> bool:
    """
    Upsert a dataset in the datasets collection.
    
    Args:
        dataset_name: Unique name for the dataset
        dataset: Dictionary containing dataset information
    
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
                    'DatasetDescription': dataset.get('DatasetDescription', ''),
                    'Parameters': dataset.get('Parameters', [])
                }
            },
            upsert=True
        )
        return True
    except Exception as e:
        info(f"Error upserting dataset: {e}")
        return False

def get_all_datasets() -> List[Dict[str, str]]:
    """
    Get all datasets from the database.
    
    Returns:
        List of dictionaries containing dataset information
    """
    collection = ensure_collection(Collections.DATASETS.value)
    
    try:
        datasets = list(collection.find({}, {'_id': 0}))  # Exclude _id field
        return datasets
    except Exception as e:
        info(f"Error retrieving datasets: {e}")
        return []

def get_data_lookup() -> List[Dict[str, str]]:
    """
    Get data lookup information from the database.
    
    Returns:
        List of dictionaries containing data lookup information
    """
    collection = ensure_collection(Collections.DATA_LOOKUP.value)
    
    try:
        lookup = list(collection.find({}, {'_id': 0}))  # Exclude _id field
        return lookup
    except Exception as e:
        info(f"Error retrieving data lookup: {e}")
        return []

def get_tables_for_dataset(dataset_name: str) -> List[Dict[str, str]]:
    """
    Get all tables for a specific dataset from the database.

    Returns:
        List of dictionaries containing table information
    """
    collection = ensure_collection(Collections.DATA_LOOKUP.value)

    try:
        tables = list(collection.find({ 'dataset_name': dataset_name }, {'_id': 0, 'embedding': 0}))  # Exclude _id field
        return tables
    except Exception as e:
        info(f"Error retrieving tables: {e}")
        return []

def list_datasets_descriptions() -> List[str]:
    """
    Get a simplified list of datasets with name and description.
    
    Returns:
        List of dicts with DatasetName and DatasetDescription
    """
    collection = ensure_collection(Collections.DATASETS.value)
    
    try:
        datasets = list(collection.find({}, {'_id': 0, 'DatasetDescription': 1}))
        return [dataset['DatasetDescription'] for dataset in datasets]
    except Exception as e:
        info(f"Error listing datasets: {e}")
        return []

def refresh_data_lookup(documents: List[Dict]) -> bool:
    """
    Clear the data_lookup collection and insert new documents.
    
    Args:
        documents: List of documents to insert into the collection
    
    Returns:
        True if successful, False otherwise
    """
    collection = ensure_collection(Collections.DATA_LOOKUP.value)
    result = collection.delete_many({})
    info(f"Cleared {result.deleted_count} existing documents from data_lookup")
    # If documents lack embeddings, attempt to embed them (best-effort)
    if documents and "embedding" not in documents[0]:
        try:
            documents = embed_documents(documents)
            info("Embeddings generated for documents.")
        except Exception as e:
            info(f"Warning: could not embed documents automatically: {e}")
    result = collection.insert_many(documents)
    info(f"Inserted {len(result.inserted_ids)} new documents into data_lookup")
    # Attempt to (re)create vector search index (Atlas / Atlas Local only)
    try:
        create_vector_search_index()
    except Exception as e:
        info(f"Warning: could not create vector search index: {e}")
    # Ensure weighted legacy text index (always available)
    try:
        ensure_text_index()
    except Exception as e:
        info(f"Warning: could not create text index: {e}")
    return True

def create_vector_search_index(
    collection_name: str = Collections.DATA_LOOKUP.value,
    index_name: str = "data_lookup_vector",
    embedding_field: str = "embedding",
    num_dimensions: int = 1536,
    similarity: str = "cosine",
    dynamic: bool = True,
    extra_string_fields: Optional[List[str]] = None,
) -> bool:
    """Create or replace a vector + text capable Atlas Search index (Atlas / Atlas Local).

    This uses the `createSearchIndexes` command available only when Atlas Search is enabled.
    If the environment is a plain self-managed mongod without Atlas Search, this will fail
    gracefully with an OperationFailure / unknown command error.

    Args:
        collection_name: Target collection.
        index_name: Name of the Atlas Search index to create.
        embedding_field: Field name containing the embedding array (floats).
        num_dimensions: Length of each embedding vector.
        similarity: Similarity metric ("cosine", "dotProduct", "euclidean") supported by Atlas.
        dynamic: Whether to allow dynamic mapping of other fields.
        extra_string_fields: Optional explicit list of string fields to map; if None and dynamic=True,
            they will still be searchable via dynamic mapping (if supported).

    Returns:
        True if index creation command succeeded, False otherwise.
    """
    db = get_database()

    # Build mapping definition
    # Atlas / Atlas Local expects 'knnVector' (NOT 'vector') and 'dimensions' key.
    # Example valid spec:
    # "embedding": { "type": "knnVector", "dimensions": 1536, "similarity": "cosine" }
    fields: Dict[str, Any] = {
        embedding_field: {
            "type": "knnVector",
            "dimensions": num_dimensions,
            "similarity": similarity,
        }
    }

    if extra_string_fields:
        for f in extra_string_fields:
            if f != embedding_field:  # avoid overwrite
                fields[f] = {"type": "string"}

    definition: Dict[str, Any] = {
        "mappings": {
            "dynamic": dynamic,
            "fields": fields,
        }
    }

    cmd: Dict[str, Any] = {
        "createSearchIndexes": collection_name,
        "indexes": [
            {
                "name": index_name,
                "definition": definition,
            }
        ],
    }

    try:
        result = db.command(cmd)
        if result.get("ok") == 1:
            info(f"Vector search index '{index_name}' ensured on collection '{collection_name}'.")
            return True
        info(f"Unexpected response creating search index '{index_name}': {result}")
        return False
    except OperationFailure as oe:
        # Provide clearer guidance if common field type error occurs
        msg = str(oe)
        if "must be one of" in msg and "knnVector" not in msg:
            info("Atlas Search mapping error did not list 'knnVector'; available types changed? Message:", msg)
        else:
            info(f"OperationFailure creating search index '{index_name}': {oe}")
        return False
    except Exception as e:
        info(f"Error creating search index '{index_name}': {e}")
        return False

def vector_search(
    query_vector: List[float],
    collection_name: str = Collections.DATA_LOOKUP.value,
    index_name: str = "data_lookup_vector",
    embedding_field: str = "embedding",
    num_candidates: int = 200,
    limit: int = 10,
    project: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """Execute a vector similarity search using $vectorSearch stage if available.

    Falls back to returning an empty list if the stage is unsupported.
    """
    db = get_database()
    # Attempt native $vectorSearch first
    pipeline_vs: List[Dict[str, Any]] = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": embedding_field,
                "queryVector": query_vector,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        }
    ]
    if project:
        pipeline_vs.append({"$project": project})
    try:
        return list(db[collection_name].aggregate(pipeline_vs))
    except OperationFailure as oe:
        msg = str(oe)
        # Fallback: try knnBeta inside $search (older Atlas Search syntax)
        if "Unrecognized pipeline stage name: '$vectorSearch'" in msg or "Location40324" in msg:
            pipeline_knn: List[Dict[str, Any]] = [
                {
                    "$search": {
                        "index": index_name,
                        "knnBeta": {
                            "path": embedding_field,
                            "vector": query_vector,
                            "k": limit if limit < num_candidates else min(num_candidates, 1000)
                        }
                    }
                },
                {"$limit": limit}
            ]
            if project:
                pipeline_knn.append({"$project": project})
            try:
                return list(db[collection_name].aggregate(pipeline_knn))
            except OperationFailure as oe2:
                info(f"Vector search fallback (knnBeta) failed: {oe2}")
                return []
        info(f"Vector search failed: {oe}")
        return []

def detect_vector_capability() -> Dict[str, bool]:
    """Detect available vector search capabilities.

    Returns dict flags: { 'vectorSearchStage': bool, 'knnBeta': bool }
    Lightweight heuristic: run a dummy aggregation that will fail fast if unsupported.
    """
    flags = {"vectorSearchStage": False, "knnBeta": False}
    db = get_database()
    coll = db[Collections.DATA_LOOKUP.value]
    # Try $vectorSearch
    try:
        coll.aggregate([
            {"$vectorSearch": {"index": "__nonexistent__", "path": "embedding", "queryVector": [0.0], "numCandidates": 1, "limit": 1}}
        ]).close()
        flags["vectorSearchStage"] = True
    except OperationFailure as oe:
        if "Unrecognized pipeline stage name" not in str(oe):
            # Some other error means stage exists but parameters bad
            if "failed to parse" in str(oe) or "Failed to find index" in str(oe):
                flags["vectorSearchStage"] = True
    except Exception:
        pass
    # Try knnBeta via $search
    try:
        coll.aggregate([
            {"$search": {"index": "__nonexistent__", "knnBeta": {"path": "embedding", "vector": [0.0], "k": 1}}},
            {"$limit": 1}
        ]).close()
        flags["knnBeta"] = True
    except OperationFailure as oe:
        if "Failed to parse" in str(oe) or "index not found" in str(oe):
            flags["knnBeta"] = True
    except Exception:
        pass
    info(f"Vector capability detection: {flags}")
    return flags

def ensure_text_index(
    collection_name: str = Collections.DATA_LOOKUP.value,
    index_name: str = "text_weighted_index",
    important_fields: Optional[Dict[str, int]] = None,
    secondary_field: str = "other_parameters",
) -> bool:
    """Create (or ensure) a weighted legacy $text index.

    We treat dataset_name, table_name, dataset_description, table_description as important.
    The 'other_parameters' field (or provided secondary_field) is included at low weight (1).

    Args:
        collection_name: Target collection.
        index_name: A logical name we log (Mongo text index name will differ unless we specify name).
        important_fields: Optional override mapping of field -> weight for important fields.
        secondary_field: Field containing concatenated less-important parameters.
    Returns:
        True if index exists or is created; False on error.
    """
    coll = get_database()[collection_name]
    if important_fields is None:
        important_fields = {
            "dataset_name": 10,
            "table_name": 10,
            "dataset_description": 6,
            "table_description": 6,
            # Include common misspelling variant so it still gets weight
            "table_desription": 6,
        }

    weights = {**important_fields}
    # Add the secondary field with minimal weight if present in docs
    if secondary_field and secondary_field not in weights:
        weights[secondary_field] = 1

    try:
        # Build a single compound text index over all weighted fields.
        index_spec = [(field, "text") for field in weights.keys()]
        coll.create_index(index_spec, name=index_name, default_language="english", weights=weights)
        info(f"Text index '{index_name}' ensured with weights: {weights}")
        return True
    except Exception as e:
        info(f"Failed to create text index '{index_name}': {e}")
        return False

def hybrid_text_vector_search(
    text_query: Optional[str] = None,
    query_vector: Optional[List[float]] = None,
    collection_name: str = Collections.DATA_LOOKUP.value,
    vector_index: str = "data_lookup_vector",
    vector_field: str = "embedding",
    text_fields: Optional[List[str]] = None,
    limit: int = 25,
    num_candidates: int = 200,
    mode: str = "atlas_compound",
    dataset_name_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Hybrid search combining text and vector relevance when possible.

        Modes:
      - "atlas_compound": Uses Atlas Search compound (requires Atlas Search) with knnBeta + text.
      - "sequential": Run vector then filter/refine with text score (fallback when no compound).
      - "text_only" / "vector_only": Force a single modality.

    If neither Atlas Search nor vector is supported, returns text-only or empty list.

        dataset_name_filter:
                If provided, results will be restricted to documents whose 'dataset_name' exactly matches.
                Pushdown: For atlas_compound we attempt an equals filter; for fallback paths we filter after retrieval.
    """
    db = get_database()
    coll = db[collection_name]
    text_fields = text_fields or ["dataset_name", "table_name", "dataset_description", "table_description", "other_parameters"]

    # Vector-only fast path
    if mode == "vector_only" and query_vector is not None:
        return vector_search(query_vector, collection_name=collection_name, index_name=vector_index, embedding_field=vector_field, num_candidates=num_candidates, limit=limit)
    if mode == "text_only" and text_query:
        return list(coll.find({"$text": {"$search": text_query}}).limit(limit))

    if mode == "atlas_compound" and (text_query or query_vector is not None):
        # Attempt compound using Atlas Search
        compound: Dict[str, Any] = {"must": [], "should": []}
        if dataset_name_filter:
            # Use filter clause for exact match if supported
            # 'equals' operator (Atlas Search) is more explicit than text for exact string equality
            # Fallback will occur automatically if not supported (caught by exception below)
            compound["filter"] = [
                {"equals": {"path": "dataset_name", "value": dataset_name_filter}}
            ]
        if text_query:
            compound["must"].append({
                "text": {
                    "query": text_query,
                    "path": text_fields
                }
            })
        if query_vector is not None:
            compound["should"].append({
                "knnBeta": {
                    "path": vector_field,
                    "vector": query_vector,
                    "k": limit if limit < num_candidates else min(num_candidates, 1000)
                }
            })
        pipeline: List[Dict[str, Any]] = [
            {
                "$search": {
                    "index": vector_index,
                    "compound": compound
                }
            },
            {"$limit": limit}
        ]
        try:
            results = list(coll.aggregate(pipeline))
            # Post-filter safeguard (in case equals not supported or mismatch)
            if dataset_name_filter:
                results = [r for r in results if r.get("dataset_name") == dataset_name_filter]
            return results
        except OperationFailure:
            # Fall back to sequential approach
            pass
        except Exception as e:
            info(f"Hybrid compound search error, falling back sequential: {e}")

    # Sequential fallback logic
    vector_results: List[Dict[str, Any]] = []
    if query_vector is not None:
        vector_results = vector_search(query_vector, collection_name=collection_name, index_name=vector_index, embedding_field=vector_field, num_candidates=num_candidates, limit=num_candidates)
    if text_query:
        text_cursor = coll.find({"$text": {"$search": text_query}}, {"score": {"$meta": "textScore"}})
        text_cursor = text_cursor.sort([("score", {"$meta": "textScore"})]).limit(num_candidates)
        text_docs = list(text_cursor)
    else:
        text_docs = []

    # Apply post-filter if dataset_name_filter specified
    if dataset_name_filter:
        if vector_results:
            vector_results = [d for d in vector_results if d.get("dataset_name") == dataset_name_filter]
        if text_docs:
            text_docs = [d for d in text_docs if d.get("dataset_name") == dataset_name_filter]

    if vector_results and text_docs:
        # Simple merge heuristic: map _id to best combined rank sum
        rank_map: Dict[Any, float] = {}
        for i, doc in enumerate(vector_results):
            rank_map[doc.get("_id")] = rank_map.get(doc.get("_id"), 0.0) + 1.0 / (1 + i)
        for i, doc in enumerate(text_docs):
            rank_map[doc.get("_id")] = rank_map.get(doc.get("_id"), 0.0) + 1.0 / (1 + i)
        # Fetch unique docs in descending combined score
        sorted_ids = sorted(rank_map.items(), key=lambda x: x[1], reverse=True)[:limit]
        id_set = {sid for sid, _ in sorted_ids}
        merged_docs: Dict[Any, Dict[str, Any]] = {}
        for d in vector_results + text_docs:
            if d.get("_id") in id_set:
                merged_docs[d.get("_id")] = d
        return [merged_docs[_id] for _id, _ in sorted_ids if _id in merged_docs]
    # If only one modality produced results
    if vector_results:
        return vector_results[:limit]
    return text_docs[:limit]

