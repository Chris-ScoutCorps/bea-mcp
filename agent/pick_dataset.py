
from llm import get_small_llm
from database import hybrid_text_vector_search
from embeddings import embed_query

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
        print(f"Dataset: {dataset.get('dataset_name')}")
        if 'table_name' in dataset:
            print(f"   Table: {dataset.get('table_name')}")
        print(f"   Description: {dataset.get('dataset_description')}")
        if 'table_description' in dataset:
            print(f"   Table Description: {dataset.get('table_description')}")
    print(f"-----------------")

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
    except Exception:
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
