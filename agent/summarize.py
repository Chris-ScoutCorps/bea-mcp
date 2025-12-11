from llm import get_large_llm

def summarize_dataset_description(description: str) -> str:
    try:
        llm = get_large_llm()
        prompt = f"Summarize the following dataset description in 2-3 concise sentences, focusing on key aspects and purpose:\n\n{description}"
        resp = llm.invoke(prompt)
        return getattr(resp, 'content', str(resp)).strip()
    except Exception as e:
        return f"Answer generation failed: {e}" 
