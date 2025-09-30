from dotenv import load_dotenv
load_dotenv()

import os
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import RetrievalQA

from api import fetch_and_upsert_bea_datasets

def get_llm():
    return ChatOpenAI(model="gpt-5-2025-08-07", temperature=0, max_tokens=8192)

if __name__ == "__main__":
    fetch_and_upsert_bea_datasets()
    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.strip().lower() == 'exit':
            break
        print('ok')
        #answer = qa_chain.invoke(question)
        #print("\nAnswer:", answer['result'], "\n")
