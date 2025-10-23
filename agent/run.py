
from dotenv import load_dotenv
load_dotenv()

import json

from mcp import BeaMcp
from logger import info

if __name__ == "__main__":
    mcp = BeaMcp()
    
    while True:
        question = input("Ask a question (or 'exit'): ")
        if question.strip().lower() == 'exit':
            break

        result = mcp.ask(question)
        info(json.dumps(result['bea_url'], indent=2))
        info(result['answer'])
