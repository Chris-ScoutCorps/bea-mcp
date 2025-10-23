import json
import copy
from llm import get_small_llm, get_medium_llm, get_large_llm
from database import hybrid_text_vector_search, list_datasets_descriptions
from embeddings import embed_query
from logger import info

def smart_search(query: str):
    """Perform search following this method:
    1. Fetch exactly 25 initial (unfiltered) results.
    2. If fewer than 10 NIPA in those 25, fetch 10 NIPA-only results.
    3. Merge, removing duplicates (preserve original order; NIPA supplement appended).
    """

    base_vector = embed_query(query)
    INITIAL_RESULTS = 25
    NIPA_TARGET = 10

    # Step 1: initial general results
    results = hybrid_text_vector_search(
        text_query=query,
        query_vector=base_vector,
        limit=INITIAL_RESULTS
    )

    # Step 2: count NIPA
    nipa_count = sum(1 for r in results if r.get('dataset_name') == 'NIPA')

    # Step 3: if fewer than 10 NIPA, fetch 10 NIPA-only results (not just the diff) and merge
    if nipa_count < NIPA_TARGET:
        nipa_supplement = hybrid_text_vector_search(
            text_query=query,
            query_vector=base_vector,
            dataset_name_filter='NIPA',
            limit=NIPA_TARGET
        )
        existing_ids = {r.get('_id') for r in results if '_id' in r}
        for doc in nipa_supplement:
            doc_id = doc.get('_id')
            if doc_id not in existing_ids:
                results.append(doc)
                existing_ids.add(doc_id)

    return results

def print_datasets(datasets):
    for dataset in datasets:
        info(f"Dataset: {dataset.get('dataset_name')} (Confidence: {dataset.get('confidence','N/A')})")
        if 'table_name' in dataset:
            info(f"   Table: {dataset.get('table_name')}")
        info(f"   Description: {dataset.get('dataset_description')}")
        if 'table_description' in dataset:
            info(f"   Table Description: {dataset.get('table_description')}")
    info(f"-----------------")

def score_dataset_relevance(question: str, dataset: dict, llm=None) -> int:
    """Ask a small LLM to rate confidence (0-100) that this dataset/table can answer the question.
    Falls back to heuristic if LLM call fails.
    """
    if llm is None:
        llm = get_small_llm()
    name = dataset.get('dataset_name','')
    table = dataset.get('table_name','')
    desc = dataset.get('dataset_description','') or ''
    table_desc = dataset.get('table_description','') or ''
    # Consolidate other parameter names & descriptions for additional context
    other_params_list = dataset.get('other_parameters', []) or []
    other_params_text = "\n".join(
        f"- {p.get('parameter_name','')}: {p.get('parameter_description','')}" for p in other_params_list
    ) or "(none)"

    prompt = f"""
You are a data relevance assessor. A user asks a question and you have a dataset (and maybe a table) description plus other parameter metadata.
Rate your confidence that querying this dataset/table will help answer the user's question.
Consider parameter names/descriptions if they are indicative of relevant dimensions or measures.

Note that 'Standard NIPA tables' is the main data set - if there's not a reason to pick another, prefer NIPA. Consider going outside of NIPA for the following topics:
{"\n".join([f"- {d}" for d in list_datasets_descriptions() if " NIPA " not in f" {d} "])}

Return ONLY an integer 0-100. No words, no percent sign.

Question: {question}
Dataset Name: {name}
Table Name: {table}
Dataset Description: {desc}
Table Description: {table_desc}
Other Parameters:\n{other_params_text}

Confidence (0-100):
"""
    try:
        response = llm.invoke(prompt)
        # langchain ChatOpenAI returns AIMessage with .content
        content = getattr(response, 'content', str(response)).strip()
        # Extract leading integer
        num = ''.join(ch for ch in content if ch.isdigit())
        if num == '':
            return 0
        score = int(num[:3])  # guard against overly long
        if score < 0: score = 0
        if score > 100: score = 100
        return score
    except Exception as e:
        info(f"LLM scoring failed: {e}")
        # Simple heuristic fallback
        heuristic = 0
        q_lower = question.lower()
        text = f"{name} {table} {desc} {table_desc}".lower()
        if any(tok in text for tok in q_lower.split()):
            heuristic = 30
        if 'gdp' in q_lower and 'gdp' in text:
            heuristic += 30
        if name == 'NIPA':
            heuristic += 20
        return min(100, heuristic)

def score_and_select_top(question: str, results: list, top_n: int = 10):
    llm = get_small_llm()
    scored = []
    for r in results:
        score = score_dataset_relevance(question, r, llm=llm)
        r_copy = dict(r)
        r_copy['confidence'] = score
        scored.append(r_copy)
    scored.sort(key=lambda x: x.get('confidence',0), reverse=True)
    return scored[:top_n], scored

def get_query_builder_context(dataset_name: str, table_name: str, full_datasets: list, for_eval: bool) -> str:
    # 1. Identify the dataset by name (exactly one expected)
    matching = [d for d in full_datasets if d.get('DatasetName') == dataset_name]
    if not matching:
        raise ValueError(f"Dataset '{dataset_name}' not found")
    # Deep copy to avoid mutating the original dataset stored in full_datasets
    dataset = copy.deepcopy(matching[0])

    # 2. If table_name is None or empty, return the dataset as-is
    if not table_name:
        return dataset

    # 3a. Remove tablename and tableid parameters
    params = dataset.get('Parameters', [])
    filtered_params = []
    for p in params:
        if p.get('ParameterName', '').lower() in ('tablename', 'tableid') and for_eval:
            continue
        filtered_params.append(p)

    # 3b. For each remaining parameter where values have a TableName property, keep only matching rows
    for p in filtered_params:
        values = p.get('Values', [])
        if not values:
            continue

        if p.get('ParameterName', '').lower() == 'geofips' and for_eval:
            p['Values'] = []
            p['Values-Note'] = "Too many GeoFIPS values to list; omitted for evaluation."
            continue

        if p.get('ParameterName', '').lower() in ('tablename', 'tableid'):
            new_values = [v for v in values if isinstance(v, dict) and v.get('TableName', v.get('Key', None)) == table_name]
            p['Values'] = new_values
            p['Values-Note'] = "Table parameter filtered to the selected table."
            continue

        if any('TableName' in v for v in values if isinstance(v, dict)):
            new_values = [v for v in values if isinstance(v, dict) and v.get('TableName') == table_name]
            p['Values'] = new_values
        if p.get('ParameterName', '').lower() == 'linecode':
            new_values = [v for v in values if isinstance(v, dict) and v.get('Desc', '').lower().startswith(f"[{table_name.lower()}]")]
            p['Values'] = new_values

        # 4. Year parameter collapsing: if ParameterName == 'Year' AND all value dicts only have Key & Desc
        if p.get('ParameterName', '').lower() == 'year':
            simple_year = True
            years = []
            for v in values:
                if not isinstance(v, dict):
                    simple_year = False
                    break
                keys = set(v.keys())
                # Accept keys subset of {'Key','Desc'}
                if not keys.issubset({'Key','Desc'}):
                    simple_year = False
                    break
                # Try to parse the year number from Key (fall back to Desc)
                yr_raw = v.get('Key') or v.get('Desc')
                try:
                    yr_int = int(str(yr_raw).strip())
                    years.append(yr_int)
                except Exception:
                    # If any can't parse, abort collapsing
                    simple_year = False
                    break
            if simple_year and years:
                min_y = min(years)
                max_y = max(years)
                p['Values'] = [
                    { 'MinYear': str(min_y) },
                    { 'MaxYear': str(max_y) }
                ]

    result = dict(dataset)
    result['Parameters'] = filtered_params
    result['SelectedTableName'] = table_name
    return result

def choose_datasets_to_query(question: str, candidate_results: list, full_datasets: list, tie_threshold: int = 3):
    """
    Use the medium LLM to evaluate candidate datasets (and optionally a table) for suitability.
    1. Build a rich context for each candidate via get_query_builder_context.
    2. Ask medium LLM for a 0-100 relevance score (returning only integer); fallback heuristic if failure.
    3. Pick the top dataset (highest score) and include any others within tie_threshold points.

    Returns dict with keys:
        top: the single top dataset (with score and context)
        ties: list of additional near-tie datasets (excluding top)
        all_scored: list of all datasets with scores
    """
    medium_llm = get_medium_llm()
    scored = []
    for ds in candidate_results:
        ds_name = ds.get('dataset_name')
        if not ds_name:
            raise ValueError("Candidate result missing dataset_name")
        try:
            context_obj = get_query_builder_context(ds_name, ds.get('table_name', None), full_datasets, True)
            context_data = json.dumps(context_obj, indent=2)
        except Exception as e:
            raise ValueError(f"Context build failed for {ds_name}: {e}")

        # Build prompt
        prompt = f"""
You are ranking which dataset (and optional table) is best to answer a user question.
Return ONLY an integer 0-100 (no text) indicating how suitable this dataset context is.
Higher means more likely to directly provide an answer.

Question: {question}
Dataset Context (JSON-like): {context_data}

Score (0-100):
"""
        try:
            resp = medium_llm.invoke(prompt)
        except Exception as e:
            msg = str(e).lower()
            if 'maximum context length' in msg or 'context_length' in msg or 'token' in msg and 'exceed' in msg:
                info(f"Medium model context/token limit hit for {ds_name}; retrying with large model...")
                try:
                    large_llm = get_large_llm()
                    resp = large_llm.invoke(prompt)
                except Exception as e2:
                    info(f"Large model fallback failed for {ds_name}: {e2}")
                    continue
            else:
                info(f"Scoring failed for {ds_name}: {e}")
                continue

        content = getattr(resp, 'content', str(resp)).strip()
        digits = ''.join(ch for ch in content if ch.isdigit())
        score = int(digits[:3]) if digits else 0
        if score > 100: score = 100

        scored.append({
            'dataset_name': ds_name,
            'table_name': ds.get('table_name', None),
            'score': score,
        })

    if not scored:
        return { 'top': None, 'ties': [], 'all_scored': [] }

    scored.sort(key=lambda x: x['score'], reverse=True)
    top = scored[0]
    ties = [s for s in scored[1:] if (top['score'] - s['score']) <= tie_threshold]

    return {
        'top': top,
        'ties': ties,
        'all_scored': scored
    }
