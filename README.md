# bea-mcp

### TODOs

The most important action item, IMO, is to pre-process the metadata along with human readble explanations and clear keywords, to help it find the right data set.
- Come up with natural language summaries of data sets, vector embed them, and use these to relate data sets to questions
- Generate sample questions for data sets, vector embed them, and use these to relate data sets to questions
- Generate topics and synonyms for data sets, vector embed them. Do the same for questions that are asked by the user, and use these to relate data sets to questions

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

### Simplified Interface

```
python cli.py "What was the change in US gross domestic product over the past decade?"
python cli.py "What can you tell me about median salaries for workers in IT in California?"
```

Starts the MCP server, wraps the question in a proper jsonrpc, asks it, prints the result, and dhists the server down.

### Development Notes

The interactive prompt in BeaMcp has been removed for server use; refresh behavior is driven solely by existing cached data or BEA_FORCE_REFRESH.

---
