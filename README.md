# LangChain Tutorial Repository

This repository is a comprehensive tutorial and playground for experimenting with [LangChain](https://github.com/langchain-ai/langchain) and related LLM (Large Language Model) tools, chains, agents, retrievers, document loaders, output parsers, and vector stores. It demonstrates practical usage of various LangChain modules, integration with LLM providers (Google Gemini, Groq, HuggingFace), and advanced prompt engineering.

## Project Structure

```
main.py                  # Entry point, simple hello world
test.py                  # Test for LangChain version
requirements.txt         # All required Python packages
pyproject.toml           # Project metadata and dependencies

Agent/                   # Simple agent implementation using ReAct and tools
Chains/                  # Examples of simple, sequential, parallel, and conditional chains
ChatModels/              # Integrations with Gemini, Groq, and HuggingFace chat models
DocumentLoaders/         # Loaders for CSV, text, web, and demo data
output_parsers/          # Output parsers: JSON, Pydantic, string, structured
Prompts/                 # Prompt templates, chat history, and prompt UI
Retriver/                # Custom retrievers: Wikipedia, multi-query, MMR, etc.
Runnable Primitive/      # Runnable primitives: branch, lambda, parallel, passthrough, sequence
Runnables/               # Runnables, chains, and notebooks
structured_output/       # Structured output demos: JSON schema, Pydantic, TypedDict
Text Splitters/          # Text splitting strategies and demos
Tool Calling/            # Tool calling and tool binding examples
Tools/                   # Custom and built-in tools, toolkits, and base tool classes
Vector Stores/           # Vector store integration (ChromaDB) for embeddings
data/                    # Sample data files

```

## Key Features & Modules

- **Agents**: Implements a ReAct agent using Google Gemini and DuckDuckGo search, with custom tools (e.g., weather lookup).
- **Chains**: Demonstrates simple, sequential, parallel, and conditional chains using LLMs (Groq, Gemini), prompt templates, and output parsers.
- **Chat Models**: Integrates with Gemini, Groq, and HuggingFace chat models for conversational AI.
- **Document Loaders**: Loads data from CSV, text, and web sources for use in chains and retrieval.
- **Output Parsers**: Parses LLM outputs into JSON, Pydantic models, and structured formats.
- **Prompts**: Advanced prompt engineering, chat history management, and prompt UI utilities.
- **Retrievers**: Custom retrievers for Wikipedia, multi-query, and maximal marginal relevance (MMR).
- **Runnable Primitives**: Composable primitives for branching, parallelism, and sequence in chains.
- **Vector Stores**: Uses ChromaDB for vector storage and retrieval with Google Gemini embeddings.
- **Tools**: Custom tools (e.g., multiplication) and toolkits for agent/tool integration.
- **Structured Output**: Demonstrates structured output using JSON schema, Pydantic, and TypedDict.
- **Text Splitters**: Various strategies for splitting and processing text documents.
- **Tool Calling**: Examples of tool binding and tool calling with LangChain.

## Getting Started

1. **Install dependencies**:
	```bash
	pip install -r requirements.txt
	```
2. **Set up environment variables**:
	- Create a `.env` file with your API keys (Google, Groq, etc.) as required by the modules.
3. **Run examples**:
	- Explore the scripts in each folder to see different LangChain features in action.
	- Example: `python main.py`

## Notebooks

Some runnable Jupyter notebooks are provided in the `Runnables/` folder for interactive experimentation.

## Requirements

- Python 3.13+
- See `requirements.txt` for all dependencies

## Credits

This repo is for educational and experimental purposes, inspired by the [LangChain documentation](https://python.langchain.com/docs/).

---
**Author:** ameympatil
