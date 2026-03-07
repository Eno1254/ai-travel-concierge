from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

def web_search(query):
    try:
        return search.run(query)
    except Exception:
        return "Web search failed."