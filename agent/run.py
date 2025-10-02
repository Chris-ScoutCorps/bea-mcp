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

        results = hybrid_text_vector_search(
            text_query=question,
            query_vector=embed_query(question)
        )

        for result in results:
            print(f"Dataset: {result.get('dataset_name')}")
            if 'table_name' in result:
                print(f"   Table: {result.get('table_name')}")
            print(f"   Description: {result.get('dataset_description')}")
            if 'table_description' in result:
                print(f"   Table Description: {result.get('table_description')}")

        #answer = qa_chain.invoke(question)
        #print("\nAnswer:", answer['result'], "\n")
