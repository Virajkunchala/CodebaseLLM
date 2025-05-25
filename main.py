import os
import json
import asyncio
from dotenv import load_dotenv
from analyzer.clone_repo import RepoCloner
from analyzer.code_reader import CodeReader
from analyzer.llm_integration import LLMIntegration
from analyzer.knowledge_extractor import KnowledgeExtractor

# Load environment variables
load_dotenv()

REPO_URL = os.getenv("REPO_URL")
TARGET_DIR = os.getenv("TARGET_DIR")
OUTPUT_PATH = './output/extracted_knowledge.json'


async def main_async():
    # Step 1: Clone the repository if needed
    if not REPO_URL or not TARGET_DIR:
        raise ValueError("REPO_URL and TARGET_DIR must be set in the .env file.")
    cloner = RepoCloner(REPO_URL, TARGET_DIR)
    codebase_path = cloner.clone()
    print(f"Cloned repository to {codebase_path}")

    # Step 2: Read and chunk the codebase
    code_reader = CodeReader(codebase_path)
    code_chunks = code_reader.read_and_chunk_codebase()
    print(f"Codebase path: {codebase_path}")
    print(f"Read {len(code_chunks)} code chunks from the codebase.")
    if not code_chunks:
        print("No code chunks found. Exiting.")
        return

    # Step 3: LLM Integration for code comprehension
    llm = LLMIntegration()
    knowledge_extractor = KnowledgeExtractor(llm)

    # Step 4: Extract knowledge from code chunks (async)
    extracted_knowledge = await knowledge_extractor.extract_async(code_chunks)

    # Step 5: Output structured JSON
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(extracted_knowledge, f, indent=2)
    print(f"Knowledge extracted and saved to {OUTPUT_PATH}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
