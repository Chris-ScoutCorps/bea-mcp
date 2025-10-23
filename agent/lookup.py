from typing import List, Dict, Any

from logger import info

def build_lookup_documents(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate documents for each table in datasets that have TableName parameters.
    
    Args:
        datasets: List of dataset dictionaries with metadata
        
    Returns:
        List of documents containing table and parameter information
    """
    documents = []
    
    for dataset in datasets:
        dataset_name = dataset.get('DatasetName', '')
        dataset_description = dataset.get('DatasetDescription', '')
        parameters = dataset.get('Parameters', [])
        
        # Find TableName parameter, with TableID as fallback
        table_parameter = None
        tableid_parameter = None
        other_parameters = []
        
        for param in parameters:
            param_name = param.get('ParameterName', '')
            if param_name.lower() == 'tablename':
                table_parameter = param
            elif param_name.lower() == 'tableid':
                tableid_parameter = param
            else:
                other_parameters.append(param)
        
        # Use TableName if available, otherwise fall back to TableID
        if not table_parameter and tableid_parameter:
            table_parameter = tableid_parameter
        
        # If we found a TableName parameter, process its values
        if table_parameter:
            table_values = table_parameter.get('Values', [])
            
            for table_value in table_values:
                table_name = table_value.get('TableName', table_value.get('Key', ''))
                table_description = table_value.get('Description', table_value.get('Desc', ''))
                
                # Create document for this table
                document = {
                    'dataset_name': dataset_name,
                    'dataset_description': dataset_description,
                    'table_name': table_name,
                    'table_description': table_description,
                    'other_parameters': []
                }
                
                # Add other parameters with their names and descriptions
                for param in other_parameters:
                    param_info = {
                        'parameter_name': param.get('ParameterName', ''),
                        'parameter_description': param.get('ParameterDescription', '')
                    }
                    document['other_parameters'].append(param_info)
                
                documents.append(document)
                info(f"Created document for table '{table_name}' in dataset '{dataset_name}'")
        else:
            # No TableName parameter, create one entry for the dataset
            document = {
                'dataset_name': dataset_name,
                'dataset_description': dataset_description,
                'other_parameters': []
            }
            
            # Add all parameters as other_parameters
            for param in other_parameters:
                param_info = {
                    'parameter_name': param.get('ParameterName', ''),
                    'parameter_description': param.get('ParameterDescription', '')
                }
                document['other_parameters'].append(param_info)
            
            documents.append(document)
            info(f"Created document for dataset '{dataset_name}' (no tables)")
    
    info(f"Generated {len(documents)} table documents from {len(datasets)} datasets")
    return documents
