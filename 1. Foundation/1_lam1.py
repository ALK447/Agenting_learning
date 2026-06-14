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

# messages = [{"role": "user", "content":"What is 2+2?"}]

# response = openai.chat.completions.create(
#     model = "gpt-4o-mini",
#     messages = messages
# )
# print(response.choices[0].message.content)

question = "Please prospose a hard, challenging question to assess someone's IQ. Respond only with the question"
messages = [{"role": "user", "content": question}]

response = openai.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages
)

question = response.choices[0].message.content
print(question)

messages = [{"role": "user", "content": question}]
response = openai.chat.completions.create(
    model = "gpt-4o-mini",
    messages = messages
)

answer = response.choices[0].message.content
rprint(Markdown(answer))

