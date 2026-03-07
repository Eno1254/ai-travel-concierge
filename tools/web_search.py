from langchain_community.tools import DuckDuckGoSearchRun

def web_search(query):
    try:
        search = DuckDuckGoSearchRun()
        result = search.run(query)
        return result
    except Exception as e:
        return f"Web search error: {str(e)}"