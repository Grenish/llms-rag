# Enhanced AI Agent with RAG and Web Search

This project implements an advanced AI agent that combines Retrieval-Augmented Generation (RAG) with a local knowledge base and live web search capabilities. It's designed to answer queries by dynamically choosing the best information source—either its internal documentation or the latest data from the internet.

## Features

- **Dynamic Tool Selection**: The agent intelligently decides whether to use its local knowledge (RAG), perform a web search, or both, based on the user's query.
- **Local Knowledge Base**: Indexes and queries a local document store (`data.json`) using ChromaDB for fast, relevant context retrieval.
- **Live Web Search**: Integrates with Tavily AI for real-time information gathering.
- **Resilient and Optimized**: Includes retry logic with exponential backoff, caching for web searches, and efficient batch processing for document indexing.
- **Configurable**: Key parameters for Ollama, ChromaDB, RAG, and web search are centralized in the `CONFIG` object for easy tuning.

## How It Works

1.  **Initialization**: The agent initializes services for Ollama (LLM generation), ChromaDB (vector storage), and Tavily (web search).
2.  **Document Indexing**: On the first run, it indexes the content of `data.json` into a ChromaDB collection, creating vector embeddings for each document.
3.  **Tool Decision**: For each query, the agent uses a language model to decide the best tool:
    *   **RAG**: For questions about specific, known entities or internal documentation.
    *   **WebSearch**: For current events, real-time information, or general knowledge.
    *   **Both**: For queries that require a comprehensive answer from both local and external sources.
4.  **Context Retrieval**: The selected tool(s) fetch relevant context.
5.  **Answer Generation**: The agent combines the user's query with the retrieved context and generates a final, comprehensive answer.

## Getting Started

### Prerequisites

- [Bun](https://bun.sh/) installed on your machine.
- [Ollama](https://ollama.ai/) running locally with the required models.
- [ChromaDB](https://www.trychroma.com/) running in a Docker container.
- A Tavily AI API key.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo.git
    cd your-repo
    ```

2.  **Install dependencies:**
    ```bash
    bun install
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your Tavily API key:
    ```
    TAVILY_API_KEY=your_tavily_api_key
    ```

4.  **Pull the required Ollama models:**
    ```bash
    ollama pull granite4:1b-h
    ollama pull embeddinggemma:300m
    ```

5.  **Start the ChromaDB container:**
    ```bash
    docker run -p 8000:8000 chromadb/chroma
    ```

### Running the Agent

You can run the agent with a default query or provide your own as a command-line argument.

- **To run with the default query:**
  ```bash
  bun start
  ```

- **To run with a custom query:**
  ```bash
  bun start "Your custom query here"
  ```

## Project Structure

- `agent.ts`: Contains the core logic for the AI agent, including the `OllamaService`, `RAGService`, `WebSearchService`, and `AIAgent` classes.
- `index.ts`: A simple example of how to use the RAG system.
- `data.json`: The local knowledge base file. Add your custom documents here as an array of strings.
- `package.json`: Project dependencies and scripts.
- `tsconfig.json`: TypeScript configuration.
- `.env`: Environment variables (ignored by Git).
