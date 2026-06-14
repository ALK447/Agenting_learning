from dotenv import load_dotenv
from rich.markdown import Markdown
from rich import print as rprint
load_dotenv(override=True)

import os
openai_api_key = os.getenv("OPENAI_API_KEY")

if openai_api_key:
    print(f"OpenAi API Key exists and begins {openai_api_key[:8]}")
else:
    print(f"OpenAi API key not set - please head to troubleshooting section of the README.md file to set it up. ")

from openai import OpenAI

openai = OpenAI()

messages = [{"role": "user", "content": "Please suggest a business area where Agentic AI is worth exploring?"}]

response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
business_area = response.choices[0].message.content
rprint(Markdown(business_area))

messages.append({"role": "assistant", "content": business_area})
messages.append({"role": "user", "content": "What are the pain points in this business area? Specify the biggest challenge."})

response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
pain_points = response.choices[0].message.content
rprint(Markdown(pain_points))

messages.append({"role": "assistant", "content": pain_points})
messages.append({"role": "user", "content": "What could be a potential solution to this problem using Agentic AI?"})

response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)
solution = response.choices[0].message.content
rprint(Markdown(solution))
