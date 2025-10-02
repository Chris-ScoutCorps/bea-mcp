from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA

from api import fetch_and_upsert_bea_datasets
from database import get_all_datasets

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
    
    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.strip().lower() == 'exit':
            break
        print('ok')
        #answer = qa_chain.invoke(question)
        #print("\nAnswer:", answer['result'], "\n")
