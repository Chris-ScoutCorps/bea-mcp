from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA

from api import fetch_and_upsert_bea_datasets, fetch_data_from_bea_api, fetch_data_from_bea_api_url
from database import get_all_datasets, refresh_data_lookup
from lookup import build_lookup_documents
from pick_dataset import choose_datasets_to_query, get_query_builder_context, smart_search, score_and_select_top, print_datasets
from llm import get_large_llm


def build_bea_params_with_llm(question: str, context: dict) -> dict:
    """Use the large LLM to propose BEA API params strictly from context."""
    llm = get_large_llm()

    # Trim context to avoid overflowing tokens
    context_json = json.dumps(context)[:8000]

    prompt = f"""
You are given a user question and a JSON context describing a BEA dataset (and optional selected table) with its parameters and allowed values.

Task: Produce ONLY a raw JSON object (no prose, no code fences) of parameters for the BEA GetData API.

Constraints:
 - Use ONLY parameter names & values explicitly present in the context JSON.
 - Do NOT invent or guess any parameter or value.
 - Include DatasetName exactly as shown.
 - If SelectedTableName present and a parameter for TableName (or TableID) exists in context, include it with the selected value.
 - Years (single or range) must stay strictly within values/bounds shown in context.
 - Pay special attention to ParameterIsRequiredFlag, MultipleAcceptedFlag, and AllValue.
    - If unsure about an optional parameter, omit it rather than guessing.
    - If unsure about a required parameter, use an "all" value or equivalent if present in context. Use a broad value otherwise.

Required Parameters: {list_required_parameters(context) or "<none>"}
NEVER UNDER ANY CIRCUMSTANCE omit a required parameter.

Question: {question}
Context: {context_json}

JSON:
"""

    response = llm.invoke(prompt)
    content = getattr(response, 'content', str(response)).strip()

    # Attempt to isolate JSON (in case model adds stray text)
    def _extract_json(text: str) -> str:
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last != -1 and last > first:
            return text[first:last+1]
        return text

    json_snippet = _extract_json(content)
    try:
        params = json.loads(json_snippet)
        if not isinstance(params, dict):
            raise ValueError('Top-level JSON not an object')
    except Exception:
        # Fallback minimal params
        params = {}

    # Mandatory DatasetName fallback
    dataset_name = context.get('DatasetName') or context.get('dataset_name')
    if dataset_name:
        params.setdefault('DatasetName', dataset_name)

    # If we have SelectedTableName and no TableName param provided but context suggests a TableName parameter existed
    if 'SelectedTableName' in context and 'TableName' not in params:
        params['TableName'] = context['SelectedTableName']

    # Normalize possible year fields: ensure not both Year and FirstYear/LastYear conflicts
    if 'Year' in params and ('FirstYear' in params or 'LastYear' in params):
        # Prefer explicit range if provided; drop single Year
        params.pop('Year', None)

    return params

def list_required_parameters(context: dict) -> str:
    """Return a comma-delimited string of required parameter names based on ParameterIsRequiredFlag.

    A parameter is considered required if its ParameterIsRequiredFlag (case-insensitive) equals one of:
        'true', '1', 1, True
    """
    required = []
    params = context.get('Parameters', []) or []
    for p in params:
        name = p.get('ParameterName') or p.get('Name')
        if not name:
            continue
        flag = p.get('ParameterIsRequiredFlag')
        if isinstance(flag, str):
            if flag.strip().lower() in ('true', '1'):  # treat these as required
                required.append(name)
        elif isinstance(flag, (int, bool)):
            if flag == 1 or flag is True:
                required.append(name)
    print(f"Required parameters: {", ".join(required)}")
    return ", ".join(required)

if __name__ == "__main__":
    # Check for existing datasets
    datasets = get_all_datasets()
    dataset_count = len(datasets)
    
    if dataset_count > 0:
        response = input(f"There are {dataset_count} datasets already with metadata. Use this (default) or refresh? (use/refresh): ").strip().lower()
        if response in ['refresh', 'r']:
            print("Fetching fresh data from BEA API...")
            datasets = fetch_and_upsert_bea_datasets()
    else:
        print("No existing datasets found. Fetching from BEA API...")
        datasets = fetch_and_upsert_bea_datasets()
    
    data_lookup = build_lookup_documents(datasets)
    refresh_data_lookup(data_lookup)

    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.strip().lower() == 'exit':
            break

        results = smart_search(question)
        print(f"Found {len(results)} results (before LLM scoring)")

        top10, all_scored = score_and_select_top(question, results, top_n=10)

        if not top10:
            print("No relevant datasets found.")
            continue

        print("Top 10 datasets by LLM confidence:")
        print_datasets(top10)

        to_query = choose_datasets_to_query(question, top10, datasets, tie_threshold=3)
        del to_query['all_scored']
        print(json.dumps(to_query, indent=2))
        print("-----")

        context = get_query_builder_context(
            dataset_name=to_query['top'].get('dataset_name'),
            table_name=to_query['top'].get('table_name', None),
            full_datasets=datasets,
            for_eval=False
        )

        # Ask large LLM to build BEA API params
        bea_params = build_bea_params_with_llm(question, context)
        bea_url = fetch_data_from_bea_api_url(bea_params)
        print("Proposed BEA URL: " + bea_url)
