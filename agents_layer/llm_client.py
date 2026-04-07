import streamlit as st
from openai import OpenAI

client = None

try:
    api_key = st.secrets["OPENROUTER_API_KEY"]
    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )
except Exception:
    client = None


def ask_llm(prompt: str) -> str:
    if client is None:
        return "LLM summary unavailable because OPENROUTER_API_KEY is not configured."
    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content