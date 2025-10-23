"""Minimal MCP-style JSON-RPC server over stdio for BEA querying.

Methods implemented:
  - tools/list -> returns single tool ask_bea
  - tools/call -> invokes ask_bea(question: str)
  - resources/list -> lists dataset names
  - resources/read -> returns context for a dataset (and optional table)

Protocol assumptions (simplified):
  Request: {"jsonrpc":"2.0","id":<id>,"method":<method>,"params":{...}}
  Response success: {"jsonrpc":"2.0","id":<id>,"result":...}
  Response error: {"jsonrpc":"2.0","id":<id>,"error":{"code":<int>,"message":<str>}}

Server runs until EOF on stdin or SIGINT/SIGTERM.
"""

from __future__ import annotations

import sys
import json
import signal
from typing import Any, Dict

from mcp import BeaMcp, get_query_builder_context


RUNNING = True


def _handle_signal(signum, frame):  # noqa: D401 unused frame
    global RUNNING
    RUNNING = False


for sig in (signal.SIGINT, signal.SIGTERM):
    try:
        signal.signal(sig, _handle_signal)
    except Exception:
        pass


bea = BeaMcp()  # respects BEA_FORCE_REFRESH env variable


def json_response(id_: Any, result: Any = None, error: Dict[str, Any] | None = None):
    if error is not None:
        payload = {"jsonrpc": "2.0", "id": id_, "error": error}
    else:
        payload = {"jsonrpc": "2.0", "id": id_, "result": result}
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def list_tools():
    return [
        {
            "name": "ask_bea",
            "description": "Answer an economics question using BEA datasets. Params: question:string",
            "input_schema": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        }
    ]


def call_tool(name: str, params: Dict[str, Any]):
    if name != "ask_bea":
        return {"error": f"Unknown tool {name}"}
    question = params.get("question")
    if not isinstance(question, str) or not question.strip():
        return {"error": "question must be a non-empty string"}
    return bea.ask(question.strip())


def list_resources():
    # Provide dataset names as resources
    return [
        {
            "uri": f"dataset://{d.get('dataset_name')}",
            "name": d.get("dataset_name"),
            "description": d.get("dataset_description", "BEA dataset"),
        }
        for d in bea.datasets
    ]


def read_resource(uri: str, params: Dict[str, Any]):
    # Accept forms: dataset://<DatasetName> (context of dataset only) or dataset://<DatasetName>#<TableName>
    if not uri.startswith("dataset://"):
        return {"error": "Unsupported URI scheme"}
    body = uri[len("dataset://"):]
    table_name = None
    if "#" in body:
        dataset_name, table_name = body.split("#", 1)
    else:
        dataset_name = body
    context = get_query_builder_context(dataset_name=dataset_name, table_name=table_name, full_datasets=bea.datasets, for_eval=False)
    return context


def dispatch(method: str, params: Dict[str, Any]):
    if method == "tools/list":
        return list_tools()
    if method == "tools/call":
        tool = params.get("name")
        tool_params = params.get("params", {})
        return call_tool(tool, tool_params)
    if method == "resources/list":
        return list_resources()
    if method == "resources/read":
        uri = params.get("uri")
        return read_resource(uri, params)
    return {"error": f"Unknown method {method}"}


def main():
    # Read lines until RUNNING is False or EOF
    while RUNNING:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            json_response(None, error={"code": -32700, "message": f"Parse error: {e}"})
            continue
        id_ = req.get("id")
        method = req.get("method")
        params = req.get("params", {})
        if not method:
            json_response(id_, error={"code": -32600, "message": "Invalid Request: missing method"})
            continue
        try:
            result = dispatch(method, params if isinstance(params, dict) else {})
            if isinstance(result, dict) and "error" in result and len(result) == 1:
                json_response(id_, error={"code": -32601, "message": result["error"]})
            else:
                json_response(id_, result=result)
        except Exception as e:  # unexpected server error
            json_response(id_, error={"code": -32000, "message": f"Server error: {e}"})


if __name__ == "__main__":
    main()
