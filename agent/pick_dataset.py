import json
import copy

from llm import get_small_llm, get_medium_llm, get_large_llm
from database import hybrid_text_vector_search, list_datasets_descriptions
from embeddings import embed_query
from lookup import NIPA_SECTIONS_LIST, NIPA_METRICS_LIST
from logger import info

def select_dataset(question: str) -> str:
    """Given a natural language question, select relevant datasets and tables."""

    datasets = '\n'.join('- ' + d for d in list_datasets_descriptions())

    try:
        llm = get_large_llm()
        prompt = f"""Choose the best data set from the following list to answer the given question.
        Answer with the data set name and no other text.
        Question: \"{question}\"
        
        {datasets}
"""
        resp = llm.invoke(prompt)
        answer = getattr(resp, 'content', str(resp)).strip()
        return ':' in answer and answer.split(':')[0].strip() or answer
    except Exception as e:
        info(f"Answer generation failed: {e}")
        exit(1)

def extract_data_item(question: str) -> str:
    try:
        llm = get_medium_llm()
        prompt = f"""Given a natural language question, try to isolate out the data item being asked about.
        Ignore specifiers such as time period, geography, units, and phrases such as "Current dollars" or "Seasonally adjusted" or "Estimated".
        Return ONLY the data item as a short phrase.

        Question: \"{question}\""""
        resp = llm.invoke(prompt)
        return getattr(resp, 'content', str(resp)).strip()
    except Exception as e:
        info(f"Answer generation failed: {e}")
        exit(1)

def smart_search(query: str, dataset_name: str) -> list:
    query_vector = embed_query(query)
    extracted_data_item = extract_data_item(query)
    info(f"Extracted data item for search: '{extracted_data_item}'")
    data_item_vector = embed_query(extracted_data_item)
    
    section_number = None
    metric_number = None

    if dataset_name == 'NIPA':
        # provide medium LLM with a bulleted list of NIPA_SECTIONS and ask to which this question pertains (or none if unsure)
        # provide medium LLM with a bulleted list of NIPA_METRICS and ask to which this question pertains (or none if unsure)
        llm = get_medium_llm()
        
        # Ask about relevant NIPA sections
        sections_prompt = f"""Given the following question, identify which NIPA section is most relevant.
Return ONLY the most relevant section number as an integer, or "none" if unsure.

Question: "{query}"

NIPA Sections:
{NIPA_SECTIONS_LIST}

Relevant section number:"""
        
        try:
            sections_resp = llm.invoke(sections_prompt)
            section_number = int(getattr(sections_resp, 'content', str(sections_resp)).strip().lower())
            info(f"NIPA section identified: {section_number}")
        except Exception as e:
            info(f"NIPA section identification failed: {e}")
        
        # Ask about relevant NIPA metrics
        metrics_prompt = f"""Given the following question, identify which NIPA metric is most relevant.
Return ONLY the most relevant section number as an integer, or "none" if unsure.

Question: "{query}"

NIPA Metrics:
{NIPA_METRICS_LIST}

Relevant metric number:"""
        
        try:
            metrics_resp = llm.invoke(metrics_prompt)
            metric_number = int(getattr(metrics_resp, 'content', str(metrics_resp)).strip().lower())
            info(f"NIPA metric identified: {metric_number}")
        except Exception as e:
            info(f"NIPA metric identification failed: {e}")

    results_full = hybrid_text_vector_search(
        dataset_name_filter=dataset_name,
        text_query=query,
        query_vector=query_vector,
        limit=25,
        section_number_filter=section_number,
        table_number_filter=metric_number
    )

    results_short = hybrid_text_vector_search(
        dataset_name_filter=dataset_name,
        text_query=extracted_data_item,
        query_vector=data_item_vector,
        vector_index="short_data_lookup_vector",
        vector_field="table_desc_embedding",
        limit=25,
        section_number_filter=section_number,
        table_number_filter=metric_number
    )

    if not results_full and not results_short:
        results_full = hybrid_text_vector_search(
            dataset_name_filter=dataset_name,
            limit=200
        )

    info(f"Smart search found {len(results_full)} full results and {len(results_short)} short results.")

    # Deduplicate on _id, tracking count and original order
    seen = {}
    order = []
    
    for result in results_full + results_short:
        doc_id = result.get('_id')
        if doc_id not in seen:
            seen[doc_id] = {'doc': result, 'count': 1, 'first_position': len(order)}
            order.append(doc_id)
        else:
            seen[doc_id]['count'] += 1
    
    # Sort by count (descending), then by original first appearance position
    sorted_ids = sorted(order, key=lambda doc_id: (-seen[doc_id]['count'], seen[doc_id]['first_position']))
    deduplicated = [seen[doc_id]['doc'] for doc_id in sorted_ids]
    
    info(f"Deduplicated to {len(deduplicated)} unique results.")
    for result in deduplicated:
        info(f"  - {result.get('table_description', 'N/A')}")

    return deduplicated

def print_datasets(datasets):
    for dataset in datasets:
        info(f"Dataset: {dataset.get('dataset_name')} (Confidence: {dataset.get('confidence','N/A')})")
        if 'table_name' in dataset:
            info(f"   Table: {dataset.get('table_name')}")
        info(f"   Description: {dataset.get('dataset_description')}")
        if 'table_description' in dataset:
            info(f"   Table Description: {dataset.get('table_description')}")
    info(f"-----------------")

def score_and_select_top(question: str, results: list, top_n: int = 10):
    """Use large LLM to evaluate and rank all results in a single call, returning top N."""
    if not results:
        return [], []
    
    llm = get_large_llm()
    
    # Build a numbered list of all results with key information
    results_list = []
    for i, r in enumerate(results, 1):
        table_name = r.get('table_name', 'N/A')
        table_desc = r.get('table_description', 'N/A')
        dataset_name = r.get('dataset_name', 'N/A')
        results_list.append(f"{i}. Table: {table_name}\n   Dataset: {dataset_name}\n   Description: {table_desc}")
    
    results_text = '\n\n'.join(results_list)
    
    prompt = f"""You are evaluating which tables are most relevant to answer a user's question.
Below is a numbered list of tables with their descriptions.

Question: "{question}"

Tables:
{results_text}

Return ONLY a comma-separated list of the numbers of the top {top_n} most relevant tables, in order from most to least relevant.
If/when all else is equal, prefer simplicity (e.g, a "x" is better than "x by industry" unless they're asking for industry detail).
Example response: 3,7,1,12,5,9,2,15,8,4

Your response:"""
    
    try:
        response = llm.invoke(prompt)
        content = getattr(response, 'content', str(response)).strip()
        
        # Parse the comma-separated numbers
        selected_indices = []
        for part in content.split(','):
            try:
                num = int(part.strip())
                if 1 <= num <= len(results):
                    selected_indices.append(num - 1)  # Convert to 0-based index
            except ValueError:
                continue
        
        # Ensure we have at least some results
        if not selected_indices:
            info("LLM returned no valid indices, using original order")
            selected_indices = list(range(min(top_n, len(results))))
        
        # Build the top results in the order specified by the LLM
        top_results = []
        for idx in selected_indices[:top_n]:
            r_copy = dict(results[idx])
            r_copy['confidence'] = 100 - (len(top_results) * 10)  # Descending confidence based on rank
            top_results.append(r_copy)
        
        # Build full scored list (selected items get scores, rest get 0)
        all_scored = []
        for i, r in enumerate(results):
            r_copy = dict(r)
            if i in selected_indices:
                rank = selected_indices.index(i)
                r_copy['confidence'] = 100 - (rank * 10)
            else:
                r_copy['confidence'] = 0
            all_scored.append(r_copy)
        
        return top_results, all_scored
        
    except Exception as e:
        info(f"LLM ranking failed: {e}, using original order")
        # Fallback to original order
        scored = []
        for i, r in enumerate(results):
            r_copy = dict(r)
            r_copy['confidence'] = max(0, 100 - (i * 10))
            scored.append(r_copy)
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
