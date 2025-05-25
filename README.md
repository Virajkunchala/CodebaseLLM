# CodebaseLLM

## Overview
CodebaseLLM is an automated codebase analysis tool that leverages Large Language Models (LLMs) to extract structured knowledge from  code repository. The tool supports codebase cloning, code chunking, LLM-driven knowledge extraction, and outputs results in a machine-readable JSON format.

---

## Approach & Methodology

1. **Repository Cloning**
   - Uses `gitpython` to clone the target repository from a URL specified in the `.env` file.

2. **Codebase Reading & Chunking**
   - Recursively scans the codebase for programming language files (e.g., `.py`, `.java`, `.js`, etc.), excluding static assets and non-code files.
   - Uses LangChain's `RecursiveCharacterTextSplitter` to split large files into manageable, token-safe chunks for LLM processing.

3. **LLM Integration & Knowledge Extraction**
   - Integrates with OpenAI's GPT-3.5-turbo for code comprehension and knowledge extraction via `langchain-openai`.
   - Uses OpenAIEmbeddings and FAISS for efficient vector search and retrieval.
   - Employs a robust prompt to extract:
     - High-level project and file overviews
     - Key methods, signatures, and descriptions
     - Code complexity and noteworthy aspects
   - Handles rate limits and errors with exponential backoff and logs all LLM responses for transparency.
   - Post-processes LLM output to fix common JSON issues (e.g., trailing commas) before parsing.

4. **Output**
   - Produces a structured, schema-safe JSON file (`output/extracted_knowledge.json`) containing all extracted knowledge.

---

## Design Choices & Optimizations

- **LLM Choice:** Uses OpenAI GPT-3.5-turbo for a balance of cost, speed, and code understanding.
- **Chunking:** Only source code files are chunked, using a single scan for efficiency.
- **Async & Parallelism:** Async and thread-based execution for scalable, fast processing.
- **Robust Parsing:** Cleans and parses LLM output to handle common JSON formatting issues.
- **Logging:** All LLM responses and errors are logged for transparency and debugging.
- **Extensibility:** Modular class design (RepoCloner, CodeReader, LLMIntegration, KnowledgeExtractor) for easy extension and maintenance.

---

## Rationale for Design Choices

### Why GitPython?
- **Avoiding Rate Limits:** Using the GitHub API to download or traverse large repositories can quickly hit rate limits, especially for private or enterprise codebases, due to API quotas and pagination. This is problematic for codebases with many files or large histories.
- **Direct Local Access:** GitPython clones the entire repository locally, allowing unrestricted, fast, and complete access to all files and history without API throttling or authentication headaches.
- **Reliability:** This approach is robust for both public and private repos (with proper credentials) and is not affected by GitHub API changes or outages.

### Why Async and Threading?
- **Performance:** Codebase analysis and LLM calls are I/O-bound and can be slow, especially when processing many files or waiting for LLM responses. Async and thread-based execution allows the tool to process multiple code chunks in parallel, greatly speeding up the pipeline.
- **Scalability:** Async/parallelism ensures the tool can handle large codebases efficiently, making the most of available CPU/network resources.
- **Resilience:** Async execution helps avoid bottlenecks and makes it easier to implement retry logic for rate limits or transient errors, improving reliability.

---

## Best Practices Considered

- **Environment Variables:** All secrets and config are loaded from `.env`.
- **Error Handling:** Robust try/except blocks and retry logic for LLM and network errors.
- **Output Schema:** Ensures output is always valid, readable JSON.
- **Documentation:** Clear code comments and this README for onboarding and maintenance.

---

## Assumptions & Limitations

- Only programming source files are analyzed; static assets and binaries are ignored.
- LLM output is post-processed for JSON safety, but rare edge cases may still require manual review.
- The tool is optimized for Java and Python codebases but can be extended for others.
- Code complexity analysis is basic and LLM-driven, not static analysis.

---

## Setup & Usage

### 1. **Create and Activate a Virtual Environment**
```sh
python -m venv codeenv
# On Windows:
codeenv\Scripts\activate
# On macOS/Linux:
source codeenv/bin/activate
```

### 2. **Install Requirements**
```sh
pip install -r requirements.txt
```

### 3. **Configure Environment**
Create a `.env` file in the project root with the following variables:
```
REPO_URL=<your-repo-url>
TARGET_DIR=./repos/<your-target-folder>
OPENAI_API_KEY=<your-openai-api-key>
```

### 4. **Run the Tool**
```sh
python main.py
```

- The tool will clone the repository, analyze the codebase, and output results to `output/extracted_knowledge.json`.

### 5. **Review Output**
- Open `output/extracted_knowledge.json` to view the structured knowledge extracted from your codebase.

---

## Troubleshooting
- If you see rate limit or API errors, check your OpenAI API key and usage limits.
- If no code chunks are found, ensure your `REPO_URL` and `TARGET_DIR` are correct and the repo contains source code files.
- For further customization, edit the prompt or chunking logic in the analyzer modules.

---

## License
MIT License
