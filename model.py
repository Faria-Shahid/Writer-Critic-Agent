from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv() 

def load_llm():
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment variables.")

    llm = ChatOpenAI(
        model="deepseek/deepseek-v3.2",
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,
        http_client=None
    )

    return llm