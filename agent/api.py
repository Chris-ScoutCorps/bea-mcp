from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from typing import Dict, List, Optional
from database import upsert_dataset

def build_bea_api_url(method: str) -> str:
    """Build the BEA API URL"""
    # Get API key from environment
    api_key = os.getenv('BEA_API_KEY')
    if not api_key:
        raise ValueError("BEA_API_KEY environment variable is not set")
    
    return f"https://apps.bea.gov/api/data?&UserID={api_key}&method={method}&ResultFormat=JSON"

def fetch_from_bea_api(method: str) -> Dict:
    """Fetch data from BEA API for a given method"""
    url = build_bea_api_url(method)
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"API request failed: {e}")
        return {"error": str(e)}

def fetch_and_upsert_bea_datasets() -> Dict[str, int]:
    """
    Fetch BEA datasets from API and upsert them to the database.
    
    Returns:
        Dict with counts of successful and failed upserts
    """

    try:
        data = fetch_from_bea_api("GETDATASETLIST")
        if 'error' in data:
            print(f"Error fetching data: {data['error']}")
            return data

        # Navigate to the datasets
        datasets = data.get('BEAAPI', {}).get('Results', {}).get('Dataset', [])
        
        # Iterate through each dataset and upsert
        for dataset in datasets:
            dataset_name = dataset.get('DatasetName', '')
            dataset_description = dataset.get('DatasetDescription', '')
            upsert_dataset(dataset_name, dataset_description)

        print(f"Found {len(datasets)} data sets")
        return {'success': len(datasets)}
        
    except Exception as e:
        print(f"Error processing datasets: {e}")
        return {'error': str(e)}
