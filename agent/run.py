from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA

from api import fetch_and_upsert_bea_datasets
from database import get_all_datasets, refresh_data_lookup, hybrid_text_vector_search
from lookup import build_lookup_documents
from embeddings import embed_query

def get_llm():
    return ChatOpenAI(model="gpt-5-2025-08-07", temperature=0, max_tokens=8192)

def smart_search(query: str):
    base_vector = embed_query(query)
    # First: general hybrid search (unfiltered)
    results = hybrid_text_vector_search(
        text_query=query,
        query_vector=base_vector,
        limit=10
    )

    # Ensure we have at least 10 results from dataset_name == 'NIPA'
    nipa_needed = 10
    nipa_results = [r for r in results if r.get('dataset_name') == 'NIPA']

    if len(nipa_results) < nipa_needed:
        # Fetch additional NIPA-only results (filtered) and merge
        supplemental = hybrid_text_vector_search(
            text_query=question,
            query_vector=base_vector,
            dataset_name_filter='NIPA',
            limit=nipa_needed * 2  # grab a bit extra in case of duplicates
        )
        # Merge: preserve original order for existing results, add new unique NIPA docs
        existing_ids = {r.get('_id') for r in results if '_id' in r}
        for doc in supplemental:
            doc_id = doc.get('_id')
            if doc_id not in existing_ids:
                results.append(doc)
                existing_ids.add(doc_id)
        nipa_results = [r for r in results if r.get('dataset_name') == 'NIPA']

    # Final trimming: keep overall list manageable
    # Prioritize keeping at least the first 10 NIPA results near top if user likely wants them
    if len(nipa_results) >= nipa_needed:
        # Reorder: first 10 NIPA (or all if fewer), then the rest (stable order)
        nipa_first = nipa_results[:nipa_needed]
        nipa_ids = {r.get('_id') for r in nipa_first if '_id' in r}
        others = [r for r in results if (r.get('dataset_name') != 'NIPA' or r.get('_id') not in nipa_ids)]
        results = nipa_first + others

    return results

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
        
        print(f"Top {len(results)} results:")

        for result in results:
            print(f"Dataset: {result.get('dataset_name')}")
            if 'table_name' in result:
                print(f"   Table: {result.get('table_name')}")
            print(f"   Description: {result.get('dataset_description')}")
            if 'table_description' in result:
                print(f"   Table Description: {result.get('table_description')}")

        print(f"-----------------")

        #answer = qa_chain.invoke(question)
        #print("\nAnswer:", answer['result'], "\n")
