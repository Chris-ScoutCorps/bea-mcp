# bea-mcp

### TODOs

The most important action item, IMO, is to preprocess the metadata along with human readble explanations and clear keywords, to help it find the right data set.
The second most important is going to be for it to more cleverly determine whether the question applies to more than one data set.

## MCP Server

This project exposes a minimal Model Context Protocol style JSON-RPC server over stdio for answering economics questions via BEA datasets.

Single tool provided:
  ask_bea(question: string) -> structured answer payload including chosen dataset/table, BEA API params, URL, and generated natural language answer.

Resources:
  dataset://<DatasetName>            - dataset metadata & parameter definitions
  dataset://<DatasetName>#<TableName> - dataset + specific table context (if table selected)

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

List resources:
{"jsonrpc":"2.0","id":3,"method":"resources/list"}

Read resource:
{"jsonrpc":"2.0","id":4,"method":"resources/read","params":{"uri":"dataset://NIPA"}}

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

### Development Notes

The interactive prompt in BeaMcp has been removed for server use; refresh behavior is driven solely by existing cached data or BEA_FORCE_REFRESH.

---
