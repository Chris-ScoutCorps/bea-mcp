import json
import sys
import subprocess
import time
import re
from pathlib import Path

# Add parent directory to path so we can import mcp_server
sys.path.insert(0, str(Path(__file__).parent.parent))

import mcp_server


def send_json_rpc_request(request_dict, timeout=60):
    """Send a JSON-RPC request to the actual MCP server and return the response"""
    cmd = ['poetry', 'run', 'python', 'mcp_server.py']
    proc = subprocess.Popen(
        cmd, 
        stdin=subprocess.PIPE, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True,
        cwd=Path(__file__).parent.parent
    )
    
    request_json = json.dumps(request_dict) + '\n'
    stdout, stderr = proc.communicate(input=request_json, timeout=timeout)
    
    if not stdout.strip():
        raise RuntimeError(f"No response from server. stderr: {stderr}")
    
    try:
        return json.loads(stdout.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON response: {stdout}. Error: {e}")


def test_tools_list():
    """Test that tools/list returns all expected tools"""
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {}
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 1
    assert "result" in response
    
    tools = response["result"]
    tool_names = [tool["name"] for tool in tools]
    
    expected_tools = ["ask_bea", "get_all_datasets", "get_tables_for_dataset", "fetch_data_from_bea_api"]
    for expected_tool in expected_tools:
        assert expected_tool in tool_names, f"Tool {expected_tool} not found in {tool_names}"


def test_get_all_datasets():
    """Test the get_all_datasets tool returns actual dataset data"""
    request = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "get_all_datasets",
            "params": {}
        }
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 2
    assert "result" in response
    
    datasets = response["result"]
    assert isinstance(datasets, list)
    assert len(datasets) > 0, "Should return at least one dataset"
    
    # Check that datasets have expected structure
    first_dataset = datasets[0]
    assert "DatasetName" in first_dataset
    assert "DatasetDescription" in first_dataset
    assert isinstance(first_dataset["DatasetName"], str)
    assert isinstance(first_dataset["DatasetDescription"], str)
    
    # Assert that there is a dataset with DatasetName
    dataset_names = [dataset["DatasetName"] for dataset in datasets]
    assert "NIPA" in dataset_names, "Should have at least one dataset with a DatasetName"


def test_get_tables_for_dataset():
    """Test getting tables for the NIPA dataset"""
    request = {
        "jsonrpc": "2.0", 
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_tables_for_dataset",
            "params": {
                "dataset_name": "NIPA"
            }
        }
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 3
    assert "result" in response
    
    tables = response["result"]
    assert isinstance(tables, list)
    # NIPA should have tables if the database is populated
    
    # Assert that there's a table with table_name T10101
    table_names = [table.get("table_name") for table in tables if isinstance(table, dict)]
    assert "T10101" in table_names, f"Table T10101 not found in NIPA tables: {table_names}"


def test_get_tables_for_invalid_dataset():
    """Test error handling for invalid dataset"""
    request = {
        "jsonrpc": "2.0",
        "id": 4, 
        "method": "tools/call",
        "params": {
            "name": "get_tables_for_dataset",
            "params": {
                "dataset_name": "NONEXISTENT_DATASET"
            }
        }
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 4
    assert "result" in response
    
    # Should return empty list for nonexistent dataset
    assert response["result"] == []


def test_fetch_data_from_bea_api():
    """Test direct BEA API call with real parameters"""
    request = {
        "jsonrpc": "2.0",
        "id": 5,
        "method": "tools/call", 
        "params": {
            "name": "fetch_data_from_bea_api",
            "params": {
                "params": {
                    "DatasetName": "NIPA",
                    "TableName": "T10101", 
                    "Year": "2023",
                    "Frequency": "A"
                }
            }
        }
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 5
    
    # Should either succeed with data or fail with a clear error
    if "result" in response:
        # Success case
        data = response["result"]
        assert isinstance(data, list)
    else:
        # Error case
        assert "error" in response
        assert "message" in response["error"]


def test_ask_bea_integration():
    """Test the main ask_bea functionality end-to-end"""

    # Note that we're giving it a ridiculously easy and objective question because this is a TEST not an EVAL

    request = {
        "jsonrpc": "2.0",
        "id": 6,
        "method": "tools/call",
        "params": {
            "name": "ask_bea", 
            "params": {
                "question": "What was the US GDP according to Table 1.1.5. Gross Domestic Product (A) (Q) in year 2020?"
            }
        }
    }
    
    # Use longer timeout for ask_bea since it involves LLM processing
    response = send_json_rpc_request(request, timeout=300)  # 5 minutes
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 6
    assert "result" in response
    
    result = response["result"]
    assert "question" in result
    assert result["question"] == "What was the US GDP according to Table 1.1.5. Gross Domestic Product (A) (Q) in year 2020?"
    assert "fetch_status" in result
    
    # Should have made some attempt to answer
    assert result["fetch_status"] in ["ok", "error", "no_datasets"]
    
    # If successful, validate the GDP value is approximately $21.375 trillion
    if result["fetch_status"] == "ok" and "answer" in result:
        answer = result["answer"]
        assert isinstance(answer, str), "Answer should be a string"
        
        # Pattern to match various forms of ~$21.375 trillion GDP
        # Matches: $21,375,281 million, $21,375 billion, $21.375 trillion, etc.
        gdp_patterns = [
            r'\$21[,.]?375[,.]?281\s*million',  # $21,375,281 million
            r'\$21[,.]?375\s*billion',          # $21,375 billion  
            r'\$21\.375\s*trillion',            # $21.375 trillion
            r'\$21\.4\s*trillion',              # $21.4 trillion (rounded)
            r'21[,.]?375[,.]?281',              # Without $ symbol
            r'21[,.]?375\s*billion',            # 21,375 billion
            r'21\.375\s*trillion'               # 21.375 trillion
        ]
        
        found_match = False
        for pattern in gdp_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                found_match = True
                break
        
        assert found_match, f"Expected GDP value (~$21.375 trillion) not found in answer: {answer}"


def test_invalid_tool():
    """Test error handling for unknown tools"""
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {
            "name": "nonexistent_tool",
            "params": {}
        }
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0"
    assert response["id"] == 7
    assert "error" in response
    assert "Unknown tool" in response["error"]["message"]


def test_malformed_request():
    """Test error handling for malformed JSON-RPC requests"""
    request = {
        "jsonrpc": "2.0",
        "id": 8,
        "method": "tools/call",
        # Missing params
    }
    
    response = send_json_rpc_request(request)
    
    assert response["jsonrpc"] == "2.0" 
    assert response["id"] == 8
    # Should handle gracefully, either with result or error
