from dotenv import load_dotenv
load_dotenv()

import os
import json
import requests
from typing import Dict, List, Optional
from database import upsert_dataset

def build_bea_api_url(method: str, params: Optional[Dict[str, str]] = None) -> str:
    """Build the BEA API URL"""
    # Get API key from environment
    api_key = os.getenv('BEA_API_KEY')
    if not api_key:
        raise ValueError("BEA_API_KEY environment variable is not set")
    
    # Start with base URL and required parameters
    url = f"https://apps.bea.gov/api/data?&UserID={api_key}&method={method}&ResultFormat=JSON"
    
    # Add additional parameters if provided
    if params:
        for key, value in params.items():
            url += f"&{key}={value}"
    
    return url

def fetch_from_bea_api(method: str, itemname: str, params: Optional[Dict[str, str]] = None) -> Dict:
    """Fetch data from BEA API for a given method and parameters"""
    def _fetch():
        print(f"Fetching data for method: {method} with params: {params}")
        url = build_bea_api_url(method, params)
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return {"error": str(e)}
    
    data = _fetch()
    if 'error' in data:
        raise ValueError(f"Error fetching data: {data['error']}")
    if data.get('BEAAPI', {}).get('Error', None):
        raise ValueError(f"BEA API returned an error: {json.dumps(data['BEAAPI']['Error'], indent=2)}")

    # Navigate to the items
    items = data.get('BEAAPI', {}).get('Results', {}).get(itemname, [])
    
    # Return items if it's a list, if it's an object, return a new list with that one thing in it
    if isinstance(items, list):
        return items
    else:
        return [items] 

def fetch_data_from_bea_api(params: Optional[Dict[str, str]] = None) -> Dict:
    return fetch_from_bea_api('GetData', 'Data', params)

def fetch_data_from_bea_api_url(params: Optional[Dict[str, str]] = None) -> Dict:
    return build_bea_api_url('GetData', params)

def fetch_and_upsert_bea_datasets() -> Dict[str, int]:
    """
    Fetch BEA datasets from API and upsert them to the database.
    
    Returns:
        Dict with counts of successful and failed upserts
    """

    try:
        datasets = fetch_from_bea_api("GetDatasetList", "Dataset")
        
        # Iterate through each dataset and upsert
        for dataset in datasets:
            dataset_name = dataset.get('DatasetName', '')
            
            if dataset_name:
                # Get parameter list for this dataset
                parameters = fetch_from_bea_api("GetParameterList", "Parameter", {"DatasetName": dataset_name})
                
                # For each parameter, get its values
                for parameter in parameters:
                    parameter_name = parameter.get('ParameterName', '')
                    if parameter_name:
                        try:
                            parameter_values = fetch_from_bea_api("GetParameterValues", "ParamValue", {
                                "DatasetName": dataset_name,
                                "ParameterName": parameter_name
                            })
                            parameter['Values'] = parameter_values
                            print(f"Fetched {len(parameter_values)} values for parameter {parameter_name} in dataset {dataset_name}")
                        except Exception as e:
                            print(f"Failed to fetch values for parameter {parameter_name} in dataset {dataset_name}: {e}")
                            parameter['Values'] = []
                
                dataset['Parameters'] = parameters

            upsert_dataset(dataset_name, dataset)

        print(f"Found {len(datasets)} data sets")
        return datasets
        
    except Exception as e:
        print(f"Error processing datasets: {e}")
        return {'error': str(e)}

