from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_classic.tools import Tool
from langchain.tools import tool

### Older way of creating a tool newly we need to use the same Run() objects but we need to import it from langchain_community
search = DuckDuckGoSearchRun()

search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information via DuckDuckGo",
)


api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

wiki_tool = Tool(
    name="wiki",
    func=wiki.run,
    description="Search the web for information from wikipedia",
    user_agent="ResearchBot/1.0 (contact@example.com)"
)


@tool
def search_duckduckgo(query):
    """
    Search the web for information via DuckDuckGo
    """
    ddg = DuckDuckGoSearchRun()
    return ddg.invoke({"query":query})

@tool
def search_wikipedia(query):
    """
    Search the web for information from wikipedia
    """
    wikipedia = WikipediaQueryRun(api_wrapper=api_wrapper)
    return wikipedia.invoke(query)



