import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

endpoint = os.getenv("Azure_OpenAI_endpoint") #Azure_OpenAI_endpoint -> for direct model call / not -> Project endpoint = this is for features,agents,services etc.
api_key = os.getenv("FOUNDRY_API_KEY")
deployment_name = os.getenv("FOUNDRY_MODEL")

client = OpenAI(
    base_url=endpoint,
    api_key=api_key
)

response = client.responses.create(
    model=deployment_name,
    input="What is the capital of France?"
)

print("Answer:", response.output_text)


"""

- Project Endpoint: Used for Foundry project-level features such as Agents, project connections, and evaluations.
- Azure OpenAI Endpoint: Used to directly call GPT and embedding models through APIs.

Remember:
- Project Endpoint → Project-level features.
- Azure OpenAI Endpoint → Direct model inference.

"""
