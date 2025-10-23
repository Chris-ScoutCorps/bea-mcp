"""Simple CLI: spawn MCP server, send question, print request JSON and formatted response.

Usage:
    poetry run python -m agent.cli "What's the median salary of an IT worker in California?"

Steps performed:
    1. Build JSON-RPC request for ask_bea.
    2. Print the request (raw JSON).
    3. Start server subprocess (agent.mcp_server).
    4. Send request, read one response line.
    5. Print formatted result.
"""

from __future__ import annotations

import json
import sys
import itertools
import subprocess
import shutil


_id_counter = itertools.count(1)


def build_request(question: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": next(_id_counter),
        "method": "tools/call",
        "params": {
            "name": "ask_bea",
            "params": {"question": question},
        },
    }


def _format_result(obj: dict) -> str:
    if 'error' in obj:
        err = obj['error']
        if isinstance(err, dict):
            return f"Error {err.get('code')}: {err.get('message')}"
        return f"Error: {err}"
    result = obj.get('result') or {}
    parts = []
    status = result.get('fetch_status')
    if status:
        parts.append(f"Status: {status}")
    chosen = result.get('chosen') or {}
    ds = chosen.get('dataset_name')
    tb = chosen.get('table_name')
    if ds:
        parts.append(f"Dataset: {ds}{' / ' + tb if tb else ''}")
    params = result.get('bea_params')
    if params:
        parts.append(f"Params: {json.dumps(params)}")
    answer = result.get('answer')
    if answer:
        parts.append("Answer:\n" + answer)
    return "\n".join(parts) or json.dumps(obj, indent=2)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv:
        print("Provide a question string.", file=sys.stderr)
        return 1
    question = " ".join(argv).strip()
    if not question:
        print("Empty question.", file=sys.stderr)
        return 1

    req = build_request(question)
    print(json.dumps(req))  # raw request

    poetry_exe = shutil.which('poetry') or 'poetry'
    cmd = [poetry_exe, 'run', 'python', '-m', 'agent.mcp_server']
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        return 2

    try:
        proc.stdin.write(json.dumps(req) + '\n')
        proc.stdin.flush()
    except Exception as e:
        print(f"Failed to send request: {e}", file=sys.stderr)
        proc.kill()
        return 3

    line = proc.stdout.readline()
    proc.terminate()
    if not line:
        print("No response from server.", file=sys.stderr)
        return 4
    line = line.strip()
    try:
        resp = json.loads(line)
    except Exception as e:
        print(f"Malformed response JSON: {e}\nRaw: {line}", file=sys.stderr)
        return 5
    print(_format_result(resp))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
