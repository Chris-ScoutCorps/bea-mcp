from dotenv import load_dotenv
load_dotenv()

import json
import sys

from api import fetch_and_upsert_bea_datasets, fetch_data_from_bea_api, fetch_data_from_bea_api_url
from database import append_detailed_description_to_dataset, get_all_datasets, get_data_lookup, refresh_data_lookup
from lookup import build_lookup_documents
from pick_dataset import choose_datasets_to_query, get_query_builder_context, select_dataset, smart_search, score_and_select_top, print_datasets
from llm import get_large_llm
from summarize import summarize_dataset_description
from logger import info

def build_bea_params_with_llm(question: str, context: dict) -> dict:
    """Use the large LLM to propose BEA API params strictly from context."""
    llm = get_large_llm()

    context_json = json.dumps(context)

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

def correct_bea_params_with_llm(error_message: str, question: str, context: dict, current_params: dict) -> dict:
    """Use large LLM to attempt a minimal correction of BEA API params after an error."""
    llm = get_large_llm()
    context_json = json.dumps(context)
    current_json = json.dumps(current_params)
    required_params = list_required_parameters(context) or "<none>"

    prompt = f"""
You are fixing BEA GetData API parameters.
Return ONLY corrected JSON (single object) with minimal changes.

User Question: {question}
Error Message: {error_message}
Required Parameters: {required_params}
Current Params: {current_json}
Context JSON: {context_json}

Guidelines:
- Include all required parameters; never remove one.
- Keep DatasetName unchanged.
- If a parameter value is invalid or missing, substitute a valid one from context; otherwise leave it.
- If Year / FirstYear / LastYear are invalid or out of range, adjust within allowed bounds shown in context.
- Do NOT fabricate any new parameter.

JSON:
"""

    response = llm.invoke(prompt)
    content = getattr(response, 'content', str(response)).strip()

    def _extract_json(text: str) -> str:
        first = text.find('{')
        last = text.rfind('}')
        if first != -1 and last != -1 and last > first:
            return text[first:last+1]
        return text

    snippet = _extract_json(content)
    try:
        revised = json.loads(snippet)
        if not isinstance(revised, dict):
            raise ValueError()
    except Exception:
        revised = dict(current_params)  # fallback to original

    # Ensure DatasetName present
    ds_name = context.get('DatasetName') or context.get('dataset_name')
    if ds_name:
        revised.setdefault('DatasetName', ds_name)

    # Ensure required params present (if context lists them and possible AllValue exists)
    # (Light-touch: just warn by printing if missing)
    missing = []
    if required_params and required_params != '<none>':
        for rp in [r.strip() for r in required_params.split(',') if r.strip()]:
            if rp not in revised:
                missing.append(rp)
    if missing:
        info(f"Warning: corrected params still missing required: {missing}")

    # Normalize Year conflict
    if 'Year' in revised and ('FirstYear' in revised or 'LastYear' in revised):
        # prefer explicit range if both present; otherwise keep Year
        if 'FirstYear' in revised or 'LastYear' in revised:
            revised.pop('Year', None)

    return revised

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
    info(f"Required parameters: {', '.join(required)}")
    return ", ".join(required)


class BeaMcp:
    """Facade class for BEA MCP operations: dataset bootstrap + question answering pipeline.

    Startup refresh logic:
      - If force_refresh=True OR no datasets exist, fetch fresh dataset metadata from BEA.
      - Otherwise reuse existing stored datasets.
    """

    def __init__(self, force_refresh: bool | None = None):
        # Determine refresh directive: explicit flag overrides env var; default False
        if force_refresh is None:
            import os
            env_flag = os.getenv('BEA_FORCE_REFRESH', '').strip().lower()
            force_refresh = env_flag in ('1', 'true', 'yes', 'y')

        existing = get_all_datasets()
        if force_refresh or not existing:
            if force_refresh:
                info("BEA_FORCE_REFRESH enabled: refreshing dataset metadata from BEA API...")
            else:
                info("No existing datasets found. Fetching dataset metadata from BEA API...")
            datasets = fetch_and_upsert_bea_datasets()
            refreshed_datasets = True
        else:
            info(f"Using cached dataset metadata ({len(existing)} datasets). Set BEA_FORCE_REFRESH=1 to refresh on next start.")
            datasets = existing
            refreshed_datasets = False

        data_lookup = get_data_lookup()
        if refreshed_datasets or not data_lookup:
            info("No existing data lookup found. Building from datasets...")
            data_lookup = build_lookup_documents(datasets)
            refresh_data_lookup(data_lookup)
        else:
            info(f"Using cached data lookup ({len(data_lookup)} entries).")

        for dataset in datasets:
            if 'DetailedDescription' not in dataset or 'GeneratedDescription' not in dataset:
                matching_tables = [
                    item for item in data_lookup 
                    if item.get('dataset_name') == dataset.get('DatasetName')
                ]
                
                table_bullets = "\n".join([
                    f"- {item.get('table_name', 'Unknown')}: {item.get('table_description', 'No description')}"
                    for item in matching_tables
                ])
                if table_bullets == "- Unknown: No description":
                    table_bullets = "- No tables found."
                
                param_bullets = "\n".join([
                    f"- {item.get('ParameterName')}: {item.get('ParameterDescription', 'No description')}"
                    for item in dataset.get('Parameters', [])
                ])

                detailed = dataset['DatasetDescription'] + "\n\nTables:\n" + table_bullets + "\n\nParameters:\n" + param_bullets
                dataset['DetailedDescription'] = detailed

                info(f"Summarizing {dataset['DatasetName']}")
                dataset['GeneratedDescription'] = summarize_dataset_description(detailed)

                append_detailed_description_to_dataset(dataset['DatasetName'], detailed, dataset['GeneratedDescription'])

        self.datasets = datasets
        self.data_lookup = data_lookup

    def ask(self, question: str) -> dict:
        """Process a natural language question through search, ranking, parameter building, fetch.

        Returns a dictionary with keys:
          question, top10, chosen, bea_params, bea_url, fetch_status, error (optional), corrected_params (optional)
        """

        best_dataset = select_dataset(question)
        info(f"Selected dataset for question: {best_dataset}")

        results = smart_search(question, best_dataset)
        top10, _all_scored = score_and_select_top(question, results, top_n=10)
        if not top10:
            return { 'question': question, 'fetch_status': 'no_datasets' }

        info("Top 10 candidate datasets/tables:")
        for ds in top10:
            # Create a copy without embedding and other_parameters for display
            display_ds = {k: v for k, v in ds.items() if k not in ('_id', 'embedding', 'table_desc_embedding', 'other_parameters')}
            info(display_ds)

        selection = choose_datasets_to_query(question, top10, self.datasets, tie_threshold=3)
        chosen = selection.get('top')

        info("")
        info(f"Chosen: {json.dumps(chosen, indent=2)}")
        info("")

        # Remove _id from chosen and top10 before including in result to avoid ObjectId serialization errors
        chosen_clean = {k: v for k, v in chosen.items() if k not in ('_id', 'embedding', 'table_desc_embedding', 'other_parameters')}
        top10_clean = [{k: v for k, v in ds.items() if k not in ('_id', 'embedding', 'table_desc_embedding', 'other_parameters')} for ds in top10]

        context = get_query_builder_context(
            dataset_name=chosen.get('dataset_name'),
            table_name=chosen.get('table_name', None),
            full_datasets=self.datasets,
            for_eval=False
        )

        bea_params = build_bea_params_with_llm(question, context)
        bea_url = fetch_data_from_bea_api_url(bea_params)

        result_payload = {
            'question': question,
            'top10': top10_clean,
            'chosen': chosen_clean,
            'context': context,
            'bea_params': bea_params,
            'bea_url': bea_url,
        }
        def _generate_answer(data_obj):
            try:
                llm = get_large_llm()
                data_json = json.dumps(data_obj)
                prompt = f"""
You are an economic data assistant.

Instructions:
1. Provide a clear, plain-English answer grounded ONLY in the data sample.
2. Cite specific figures with year/period if available.
3. If data insufficient, state what's missing succinctly.
4. Keep it under 8 sentences, no speculation.

User question: {question}
Data Returned from API: {data_json}
Additional Context: {json.dumps(context)}

Answer:
"""
                resp = llm.invoke(prompt)
                return getattr(resp, 'content', str(resp)).strip()
            except Exception as e:
                return f"Answer generation failed: {e}" 

        try:
            data = fetch_data_from_bea_api(bea_params)
            result_payload['fetch_status'] = 'ok'
            result_payload['data_preview'] = data[:3] if isinstance(data, list) else data
            result_payload['answer'] = _generate_answer(data)
        except Exception as e:
            result_payload['fetch_status'] = 'error'
            result_payload['error'] = str(e)
            corrected = correct_bea_params_with_llm(str(e), question, context, bea_params)
            result_payload['corrected_params'] = corrected
            try:
                data = fetch_data_from_bea_api(corrected)
                result_payload['second_attempt_status'] = 'ok'
                result_payload['data_preview'] = data[:3] if isinstance(data, list) else data
                result_payload['answer'] = _generate_answer(data)
            except Exception as e2:
                result_payload['second_attempt_status'] = 'error'
                result_payload['second_error'] = str(e2)

        return result_payload
