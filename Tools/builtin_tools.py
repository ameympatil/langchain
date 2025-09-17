from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

search = DuckDuckGoSearchRun()

results = search.invoke("Asia Cup")

print(results)  # to see the tool details


## Shell Tool


shell_tool = ShellTool()

print(shell_tool.invoke("whoami"))  # should print "Hello, World!"
