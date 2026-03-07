from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def web_search(query):
    return search.run(query)