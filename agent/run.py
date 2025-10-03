from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA

from api import fetch_and_upsert_bea_datasets
from database import get_all_datasets, refresh_data_lookup
from lookup import build_lookup_documents
from pick_dataset import smart_search, score_and_select_top, print_datasets

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
        print("Top 10 datasets by LLM confidence:")
        print_datasets(top10)

        # Optionally, could proceed with retrieval/answering using top10 later
