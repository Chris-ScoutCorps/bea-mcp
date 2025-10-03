
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
from mcp import BeaMcp

if __name__ == "__main__":
    mcp = BeaMcp()
    
    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.strip().lower() == 'exit':
            break

        result = mcp.ask(question)
        print(json.dumps(result['bea_url'], indent=2))
