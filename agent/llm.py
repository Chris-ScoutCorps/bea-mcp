from langchain_community.chat_models import ChatOpenAI

def get_small_llm():
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_medium_llm():
    return ChatOpenAI(model="gpt-4", temperature=0)

def get_large_llm():
    return ChatOpenAI(model="gpt-5-2025-08-07", temperature=1)
