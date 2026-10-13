# serving/smoke_test.py
import json
from typing import cast
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from src.prompts import build_messages
from src.evaluate import extract_json

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

tools = [{"name": "get_weather", "description": "Get current weather for a city",
          "parameters": {"city": {"type": "str", "description": "City name"}}}]

for query in ["What's the weather in Prayagraj?", "Tell me a joke"]:
    messages = build_messages({"query": query, "tools": json.dumps(tools)})   # check arg order in src/prompts.py
    resp = client.chat.completions.create(
        model="sft-toolcaller", messages=cast(list[ChatCompletionMessageParam], messages),
        temperature=0, max_tokens=256)
    raw = resp.choices[0].message.content
    print(query, "\n  raw:", raw, "\n  parsed:", extract_json(raw))