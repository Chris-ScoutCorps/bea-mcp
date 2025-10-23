import json

from agent import mcp_server


class StubBea:
    def __init__(self):
        self.datasets = [
            {
                "dataset_name": "NIPA",
                "dataset_description": "National Income and Product Accounts",
            }
        ]

    def ask(self, question: str):
        return {
            "question": question,
            "fetch_status": "ok",
            "bea_params": {},
            "answer": "Stub answer",
        }


def test_tools_list_contains_ask_bea(monkeypatch):
    # Replace bea object with stub to avoid network access
    monkeypatch.setattr(mcp_server, "bea", StubBea())
    tools = mcp_server.list_tools()
    assert any(t.get("name") == "ask_bea" for t in tools), "ask_bea tool missing"


def test_call_tool_returns_expected_payload(monkeypatch):
    monkeypatch.setattr(mcp_server, "bea", StubBea())
    result = mcp_server.call_tool("ask_bea", {"question": "What's the median salary of an IT worker in California?"})
    assert result.get("question") == "What's the median salary of an IT worker in California?"
    assert result.get("fetch_status") == "ok"
    assert "answer" in result
