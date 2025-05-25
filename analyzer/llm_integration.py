import os
from typing import Dict, List, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import Document
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
import asyncio
import re
from config.config import logger, settings
from langchain_core.output_parsers import SimpleJsonOutputParser

class LLMIntegration:
    """
    Integrates with OpenAI GPT-3.5-turbo via Langchain and uses FAISS for vector search.
    Optimized for batch processing, scalability, and robust error handling.
    """
    def __init__(self, max_workers: int = 1) -> None:
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        self.embedding = OpenAIEmbeddings()
        self.vectorstore = None
        self.max_workers = max_workers

    def build_vectorstore(self, code_chunks: List[Dict[str, Any]]) -> None:
        """
        Build a FAISS vectorstore from code chunks for efficient retrieval.
        Args:
            code_chunks: List of dicts with 'chunk', 'file', and 'chunk_index'.
        """
        docs = [Document(page_content=chunk['chunk'], metadata={"file": chunk['file'], "chunk_index": chunk['chunk_index']}) for chunk in code_chunks]
        self.vectorstore = FAISS.from_documents(docs, self.embedding)
        logger.info("FAISS vectorstore built with %d documents.", len(docs))

    def _clean_json_string(self, s: str) -> str:
        """
        Post-process LLM output to fix common JSON issues before parsing.
        Removes text before first '{' and after last '}', and trailing commas.
        """
        # Remove any text before the first '{' and after the last '}'
        s = re.sub(r'^[^{]*', '', s)
        s = re.sub(r'[^}]*$', '', s)
        # Remove trailing commas before } or ]
        s = re.sub(r',\s*([}\]])', r'\1', s)
        # Strip whitespace
        s = s.strip()
        return s

    def analyze_code_chunk(self, chunk: str, max_retries: int = 5, base_delay: float = 5.0) -> Dict:
        """
        Analyze a code chunk with retry and exponential backoff on rate limit errors.
        Args:
            chunk: The code chunk to analyze.
            max_retries: Maximum number of retries on rate limit.
            base_delay: Base delay in seconds for exponential backoff.
        Returns:
            Dict with analysis or error info.
        """
        attempt = 0
        parser = SimpleJsonOutputParser()
        while attempt < max_retries:
            prompt = self._build_prompt(chunk)
            try:
                response = self.llm.invoke(prompt)
                logger.debug("LLM response received for chunk.")
                raw_content = response.content if hasattr(response, 'content') else response
                cleaned_content = self._clean_json_string(raw_content)
                return parser.invoke(cleaned_content)
            except Exception as e:
                logger.error(f"SimpleJsonOutputParser or LLM error: {e}")
                err_msg = str(e)
                if 'rate limit' in err_msg.lower() or '429' in err_msg or 'rate_limit_exceeded' in err_msg:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Rate limit hit. Waiting {wait_time:.1f}s before retrying (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    attempt += 1
                else:
                    raise
        logger.error(f"Rate limit exceeded after {max_retries} retries.")
        return {"error": f"Rate limit exceeded after {max_retries} retries."}

    def analyze_code_chunks_batch(self, chunks: List[Dict[str, Any]]) -> List[Dict]:
        """
        Analyze a batch of code chunks in parallel using ThreadPoolExecutor.
        Args:
            chunks: List of dicts with 'chunk'.
        Returns:
            List of analysis results.
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.analyze_code_chunk, chunk['chunk']) for chunk in chunks]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    logger.error(f"Batch analysis error: {e}")
                    results.append({"error": str(e)})
        return results

    async def analyze_code_chunk_async(self, chunk: str, max_retries: int = 5, base_delay: float = 5.0) -> Dict:
        """
        Analyze a code chunk asynchronously with retry and exponential backoff on rate limit errors.
        Args:
            chunk: The code chunk to analyze.
            max_retries: Maximum number of retries on rate limit.
            base_delay: Base delay in seconds for exponential backoff.
        Returns:
            Dict with analysis or error info.
        """
        attempt = 0
        parser = SimpleJsonOutputParser()
        while attempt < max_retries:
            prompt = self._build_prompt(chunk)
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.llm.invoke, prompt)
                logger.debug("Async LLM response received for chunk.")
                raw_content = response.content if hasattr(response, 'content') else response
                cleaned_content = self._clean_json_string(raw_content)
                return parser.invoke(cleaned_content)
            except Exception as e:
                logger.error(f"Async SimpleJsonOutputParser or LLM error: {e}")
                err_msg = str(e)
                if 'rate limit' in err_msg.lower() or '429' in err_msg or 'rate_limit_exceeded' in err_msg:
                    wait_time = base_delay * (2 ** attempt)
                    logger.warning(f"Async rate limit hit. Waiting {wait_time:.1f}s before retrying (attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(wait_time)
                    attempt += 1
                else:
                    raise
        logger.error(f"Async rate limit exceeded after {max_retries} retries.")
        return {"error": f"Rate limit exceeded after {max_retries} retries."}

    def _build_prompt(self, chunk: str) -> str:
        """
        Build a detailed prompt for the LLM to extract structured codebase knowledge.
        The LLM is instructed to return a JSON object with specific keys and detailed content.
        """
        return f"""
You are an expert software architect and codebase analyst. Your task is to analyze the following code chunk and extract structured, in-depth knowledge for documentation and onboarding purposes.

For the given code chunk, provide a JSON object with the following keys:

- \"overview\": A concise, high-level summary of the code's purpose and functionality. If possible, relate it to the overall project context.
- \"methods\": An array of objects, each describing a key method or function. For each, include:
    - \"name\": The method/function name
    - \"signature\": The full method/function signature
    - \"description\": A clear, human-readable explanation of what it does
- \"complexity\": A brief assessment of the code's complexity (e.g., simple, moderate, complex) and why
- \"notes\": Any other noteworthy aspects, such as design patterns, dependencies, or potential issues

IMPORTANT: Do not use trailing commas in arrays or objects. All property names and string values must be in double quotes. Do not add comments or extra text.

Return ONLY a valid, well-formatted JSON object with these keys. Do not include any extra commentary or markdown.

Code chunk:
{chunk}
"""
