from typing import List, Dict, Any, Optional
from dataclasses import asdict, dataclass
import re

from logger import info

@dataclass
class ParsedTable:
    is_annual: bool
    is_quarterly: bool
    is_monthly: bool
    section_number: int
    subsection_number: int
    sub_subsection_letter: Optional[str]
    table_number: Optional[int]
    section: str
    subsection: str


# NIPA Section mapping based on first number of table
NIPA_SECTIONS = {
    1: "Domestic Product and Income",
    2: "Personal Income and Outlays",
    3: "Government Current Receipts and Expenditures",
    4: "Foreign Transactions",
    5: "Saving and Investment",
    6: "Income and Employment by Industry",
    7: "Supplemental Tables",
    8: "Not Seasonally Adjusted"
}

# NIPA Metric mapping based on last number of table
NIPA_METRICS = {
    1: "Percent change from preceding period in real estimates (most at annual rates)",
    2: "Contributions to percent change in real estimates",
    3: "Real estimates, quantity indexes",
    4: "Price indexes",
    5: "Current dollars",
    6: "Real estimates, chained dollars",
    7: "Percent change in prices",
    8: "Contributions to percent change in prices",
    9: "Implicit price deflators",
    10: "Percentage shares of GDP",
    11: "Percent change from (quarter or month) one year ago"
}

# Examples of NIPA tables
"""
- Table 1.2.1. Percent Change From Preceding Period in Real Gross Domestic Product by Major Type of Product (A) (Q)
- Table 1.2.2. Contributions to Percent Change in Real Gross Domestic Product by Major Type of Product (A) (Q)
- Table 1.2.3. Real Gross Domestic Product by Major Type of Product, Quantity Indexes (A) (Q)
- Table 1.2.4. Price Indexes for Gross Domestic Product by Major Type of Product (A) (Q)
- Table 1.2.5. Gross Domestic Product by Major Type of Product (A) (Q)
- Table 1.2.6. Real Gross Domestic Product by Major Type of Product, Chained Dollars (A) (Q)

Common substring / subsection name: "Gross Domestic Product by Major Type of Product"
"""

def extract_table_content(table_name: str) -> Optional[str]:
    """Extract the content between 'Table X.Y.Z. ' and frequency indicators like '(A) (Q)'."""
    # Remove table number prefix
    content_match = re.search(r'Table\s+\d+\.\d+\.\d+\.\s+(.+)', table_name)
    if not content_match:
        return None
    
    content = content_match.group(1)
    
    # Remove frequency indicators (A), (Q), (M) from the end
    content = re.sub(r'\s*\([AMQ]\)\s*', ' ', content).strip()
    
    return content

def longest_common_substring(strings: List[str]) -> str:
    """Find the longest contiguous substring common to all strings."""
    if not strings:
        return ""
    
    if len(strings) == 1:
        return strings[0].strip()
    
    # Start with the first string as reference
    reference = strings[0]
    if not reference:
        return ""
    
    # Find all possible substrings of the reference string
    longest = ""
    
    for i in range(len(reference)):
        for j in range(i + 1, len(reference) + 1):
            substring = reference[i:j]
            # Check if this substring appears in all other strings
            if all(substring in s for s in strings[1:]):
                if len(substring) > len(longest):
                    longest = substring
    
    return longest.strip()

def parse_nipa_table_desc(table_name: str, subsection_tables: List[str]) -> Optional[ParsedTable]:
    """
    Parse NIPA table name to extract section, frequency flags, and data item name.
    
    Args:
        table_name: NIPA table name (e.g., "Table 1.2.1. Percent Change... (A) (Q)" or "Table 1.12. National Income... (A) (Q)")
        subsection_tables: List of all table names in the same subsection (X.Y) to compute common data item name
        
    Returns:
        ParsedTable with parsed information, or None if not a valid NIPA table
    """
    # Try three-part number like "1.2.1"
    number_match = re.search(r'(\d+)\.(\d+)\.(\d+)', table_name)
    
    if number_match:
        # Three-part number: X.Y.Z
        section_num = int(number_match.group(1))
        subsection_num = int(number_match.group(2))
        table_num = int(number_match.group(3))
        sub_subsection_letter = None
        
        # Get section name from mapping
        section = NIPA_SECTIONS.get(section_num, f"Section {section_num}")
        
        # Extract data item name as longest common substring of all tables in subsection
        data_item_name = ""
        contents = [extract_table_content(t) for t in subsection_tables]
        contents = [c for c in contents if c]  # Filter out None values
        
        if contents:
            data_item_name = longest_common_substring(contents)
    else:
        # Try two-part number like "1.12" or "6.21D"
        two_part_match = re.search(r'Table\s+(\d+)\.(\d+)([A-Z])?\.\s+(.+)', table_name)
        if not two_part_match:
            return None
        
        section_num = int(two_part_match.group(1))
        subsection_num = int(two_part_match.group(2))
        sub_subsection_letter = two_part_match.group(3)  # Will be None if no letter
        table_num = None
        
        # Get section name from mapping
        section = NIPA_SECTIONS.get(section_num, f"Section {section_num}")
        
        # Extract content between table number and frequency indicators
        content = two_part_match.group(4)
        content = re.sub(r'\s*\([AMQ]\)\s*', ' ', content).strip()
        data_item_name = content
    
    # Check frequency indicators
    is_annual = '(A)' in table_name
    is_quarterly = '(Q)' in table_name
    is_monthly = '(M)' in table_name
    
    return ParsedTable(
        subsection=data_item_name,
        is_annual=is_annual,
        is_quarterly=is_quarterly,
        is_monthly=is_monthly,
        section_number=section_num,
        subsection_number=subsection_num,
        sub_subsection_letter=sub_subsection_letter,
        table_number=table_num,
        section=section
    )

def get_nipa_table_subsection_id(table_name: str) -> Optional[str]:
    """Extract subsection ID (X.Y) from NIPA table name."""
    match = re.search(r'(\d+)\.(\d+)\.\d+', table_name)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return None

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
            
            if dataset_name == 'NIPA':
                table_descs = [table_value.get('Description', table_value.get('Desc', '')) for table_value in table_values]
                subsection_tables_map: Dict[str, List[str]] = {}
                for d in table_descs:
                    key = get_nipa_table_subsection_id(d)
                    subsection_tables_map[key] = subsection_tables_map.get(key, []) + [d]

            for table_value in table_values:
                table_name = table_value.get('TableName', table_value.get('Key', ''))
                table_description = table_value.get('Description', table_value.get('Desc', ''))
                
                # Create document for this table
                document = {
                    'dataset_name': dataset_name,
                    'dataset_description': dataset_description,
                    'table_name': table_name,
                    'table_description': table_description,
                    'other_parameters': [],
                }

                if dataset_name == 'NIPA':
                    key = get_nipa_table_subsection_id(table_description)
                    parsed = parse_nipa_table_desc(table_description, subsection_tables_map.get(key, []))
                    document['meta'] = asdict(parsed) if parsed else None

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
