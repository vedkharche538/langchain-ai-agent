import os

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, format_instructions
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, search_duckduckgo, search_wikipedia
load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    model="nvidia/nemotron-3-nano-4b",
    temperature=1,
    max_tokens=512,
    top_p=1,
)


client = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key=os.getenv("NVIDIA_API_KEY"),
  temperature=1,
  top_p=1,
  max_tokens=4096,
)
parser = PydanticOutputParser(pydantic_object=ResearchResponse)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are a research assitance that will help generate a research paper.
            Answer the user query and use necessary tools.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}")
    ]
).partial(format_instructions=parser.get_format_instructions())
tools = [search_tool]
# tools = [ search_wikipedia]
agent = create_tool_calling_agent(
    llm = client,
    prompt=prompt,
    tools=tools
)
agent_executor = AgentExecutor(agent = agent, tools=tools, verbose=True)
query = input("what i can help you for research?\n")
raw_response = agent_executor.invoke({"query":query})

print(raw_response)
# structured_response = parser.parse(raw_response.get("output")[0]["text"])
# print(structured_response)



