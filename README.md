# bea-mcp

## About

This is an open source project, designed as a demonstration of an end-to-end workflow.  
The ask_bea tool in particular should be considered a starting point / prototype, and not a production-ready function.  

Initial development funded by [BrightQuery](https://brightquery.com/)  
Ongoing maintenance on volunteer time from [Scout Corps](https://www.scoutcorpsllc.com/)

### Development Status

- The "basic" tools, get_all_datasets, get_tables_for_dataset, and fetch_data_from_bea_api should behave deterministically and are essentially complete
- The "advanced" tool uses those three (with a bunch of keyword and vector search functionality and LLM agents) to attempt to answer natural language questions using data from the BEA's API
  - It's not deterministic
  - It is NOT done, but rather should be considered a demonstration of an end-to-end workflow
  - It completes the assignment in that it reads the question, picks a dataset, forms a request, and interpret the results
  - It doesn't always pick the right data set, though. Most of the TODO's below pertain to improving that.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- BEA API key (register for free at https://apps.bea.gov/API/signup/)
- OpenAI API key (requires an OpenAI account) https://platform.openai.com/api-keys

### Setup
1. **Environment Configuration**
   ```bash
   cp .env.template .env
   ```
   Edit `.env` and populate the required variables:
   - `BEA_API_KEY`: Your BEA API key
   - `OPENAI_API_KEY`: Your OpenAI API key (for the ask_bea tool)

2. **Start Services**
   ```bash
   docker-compose up -d
   ```
   This starts MongoDB Atlas Local and builds/runs the Python agent.

3. **Access the Container**
   ```bash
   docker-compose exec agent bash
   ```
   You'll get a bash prompt inside the agent container where you can run the MCP server or CLI tools.

4. **Test the Server**
   From inside the container:
   ```bash
   # Test basic functionality
   python cli.py "What was the change in US gross domestic product over the past decade?"
   
   # Execute Tests
   pytest

   # Or run the MCP server directly
   poetry run python mcp_server.py
   ```

5. **Stop the Server**
   ```bash
   # Remove volumes
   docker-compose down -v
   ```

   Note: You currently need to remove volumes, because the Mongo Atlas Docker image doesn't handle startup initialization well from existing volumes. It's brand new, so hopefully this issue won't be around for long. It only takes a few seconds to re-populate the database cache from scratch, anyway.

## MCP Server

This project exposes a minimal Model Context Protocol style JSON-RPC server over stdio for answering economics questions via BEA datasets.

```
Tools provided:
  ask_bea(question: string) -> structured answer payload including chosen dataset/table, BEA API params, URL, and generated natural language answer.
  get_all_datasets() -> list of all available BEA datasets with metadata
  get_tables_for_dataset(dataset_name: string) -> list of tables for a specific dataset
  fetch_data_from_bea_api(params: object) -> raw data from BEA API with custom parameters
```

```
Resources:
  dataset://<DatasetName>            - dataset metadata & parameter definitions
  dataset://<DatasetName>#<TableName> - dataset + specific table context (if table selected)
```

### TODOs for ask_bea

The most important action item, IMO, is to pre-process the metadata along with human readble explanations and clear keywords, to help it find the right data set.
- Generate sample questions for data sets, vector embed them, and use these to relate data sets to questions
- Generate topics and synonyms for data sets, vector embed them. Do the same for questions that are asked by the user, and use these to relate data sets to questions
It's doing _okay_ now, but it's slow, overly complex, and still not as accurate as it could be. This represents 2-3 days of work so judge it accordingly :-)

The second most important is going to be for it to more cleverly determine whether the question applies to more than one data set.
- See third bullet point above for deciding on correct data sets
- Implement querying against multiple data sets and then stitch together results for a coherent response

As this is an MCP server, we should return raw data sets and let a downstream consumer turn these into charts or graphs
- If we want to incorproate it into a BQ DCS/DUP project
- If it's meant to be standalone, we can do some of that here

Smaller Action Items / Enhacements:
- Using the LLM to match regions with a giant list to get a region code is overkill. We should have the LLM decide whether or not the query is regional, and then use more straightforward code to find the correct region.
- We can probably do something similar with NAICS codes.
- Expect more like this ...


### Startup Refresh Logic

On start the server loads cached dataset metadata if present. To force a refresh set environment variable:

BEA_FORCE_REFRESH=1

If no datasets stored, it automatically fetches them.

### Running the Server

poetry run python -m mcp_server

Or with refresh:

BEA_FORCE_REFRESH=1 poetry run python -m mcp_server

### JSON-RPC Examples

List tools request:
{"jsonrpc":"2.0","id":1,"method":"tools/list"}

Call ask_bea:
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"ask_bea","params":{"question":"What was US GDP growth in 2023?"}}}

Get all datasets:
{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_all_datasets","params":{}}}

Get tables for a dataset:
{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_tables_for_dataset","params":{"dataset_name":"NIPA"}}}

Fetch data from BEA API:
{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"fetch_data_from_bea_api","params":{"params":{"DatasetName":"NIPA","TableName":"T10101","Year":"2023","Frequency":"A"}}}}

List resources:
{"jsonrpc":"2.0","id":6,"method":"resources/list"}

Read resource:
{"jsonrpc":"2.0","id":7,"method":"resources/read","params":{"uri":"dataset://NIPA"}}

### Response Shape (ask_bea)

{
  "question": "...",
  "top10": [...],
  "chosen": {"dataset_name": "NIPA", "table_name": "T10101"},
  "context": {...},
  "bea_params": {...},
  "bea_url": "https://apps.bea.gov/api/...",
  "fetch_status": "ok",
  "data_preview": [...],
  "answer": "Natural language summary"
}

On initial parameter failure, fields: error, corrected_params, second_attempt_status may appear.

### Simplified Interface

```
python cli.py "What was the change in US gross domestic product over the past decade?"
python cli.py "What can you tell me about median salaries for workers in IT in California?"
```

Starts the MCP server, wraps the question in a proper jsonrpc, asks it, prints the result, and dhists the server down.

### Development Notes

The interactive prompt in BeaMcp has been removed for server use; refresh behavior is driven solely by existing cached data or BEA_FORCE_REFRESH.

---
