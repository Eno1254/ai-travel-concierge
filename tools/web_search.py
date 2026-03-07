from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def web_search(query):
    return search.run(query)