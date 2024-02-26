# GPT - OpenAI-----------------------------------------------------------------------------------------------
"""
import ast
from openai import OpenAI
import openai
import pandas as pd
import tiktoken
import os
from scipy import spatial

# Carrega o modelo
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "gpt-3.5-turbo"

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# an example question about the 2022 Olympics
query = 'Which athletes won the gold medal in curling at the 2022 Winter Olympics?'

response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You answer questions about the 2022 Winter Olympics.'},
        {'role': 'user', 'content': query},
    ],
    model=GPT_MODEL,
    temperature=0,
)

print(response.choices[0].message.content)
------------------------------------------------------------------------------------------------------------
"""
