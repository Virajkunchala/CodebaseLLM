from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from config.config import logger, settings
import os
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

class KnowledgeExtractor:
    """
    Extracts structured knowledge from code chunks using an LLM integration and FAISS for efficient retrieval.
    Optimized for batch processing and scalability.
    """
    def __init__(self, llm_integration: Any, max_workers: int = 1):
        self.llm_integration = llm_integration
        self.max_workers = max_workers

    def extract(self, code_chunks: List[Dict]) -> Dict:
        """
        Processes code chunks, builds a FAISS vectorstore, queries the LLM, and aggregates structured knowledge.
        Returns a JSON-serializable dictionary.
        """
        # Use OpenAIEmbeddings and FAISS from LangChain for vectorstore
        embeddings = OpenAIEmbeddings()
        docs = [
            type('Doc', (), {
                'page_content': chunk['chunk'],
                'metadata': {'file': chunk['file'], 'chunk_index': chunk['chunk_index']},
                'id': f"{chunk['file']}:{chunk['chunk_index']}"
            })() for chunk in code_chunks
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.llm_integration.vectorstore = vectorstore

        readme_path = os.path.join(settings.TARGET_DIR, "README.md")
        project_info = {}
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                prompt = (
                    "You are an expert software architect. Summarize the following README.md for onboarding: "
                    "Return a JSON object with keys: 'readme_summary', 'main_features', 'usage'.\nREADME:\n" + readme_content
                )
                parser = JsonOutputParser()
                response = self.llm_integration.llm.invoke(prompt)
                project_info = parser.parse(response.content)
            except Exception as e:
                logger.warning(f"Failed to process README.md: {e}")
                project_info = {"readme_error": str(e)}
        knowledge = {
            "project_info": project_info,
            "overview": [],
            "methods": [],
            "complexity": [],
            "notes": []
        }
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.llm_integration.analyze_code_chunk, chunk['chunk']): chunk for chunk in code_chunks}
            for future in as_completed(futures):
                chunk_info = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    logger.exception(f"LLM analysis failed for file={chunk_info['file']} chunk_index={chunk_info['chunk_index']}")
                    knowledge['notes'].append({
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'notes': f"LLM error: {type(e).__name__}: {e}"
                    })
                    continue
                if 'overview' in result or 'methods' in result or 'complexity' in result or 'notes' in result:
                    if 'overview' in result:
                        knowledge['overview'].append({
                            'file': chunk_info['file'],
                            'chunk_index': chunk_info['chunk_index'],
                            'overview': result['overview']
                        })
                    if 'methods' in result:
                        for method in result['methods']:
                            if method not in knowledge['methods']:
                                knowledge['methods'].append(method)
                    if 'complexity' in result:
                        knowledge['complexity'].append({
                            'file': chunk_info['file'],
                            'chunk_index': chunk_info['chunk_index'],
                            'complexity': result['complexity']
                        })
                    if 'notes' in result:
                        knowledge['notes'].append({
                            'file': chunk_info['file'],
                            'chunk_index': chunk_info['chunk_index'],
                            'notes': result['notes']
                        })
                else:
                    knowledge['notes'].append({
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'notes': result.get('raw_response', result.get('error', 'Unknown LLM response'))
                    })
        return knowledge

    async def extract_async(self, code_chunks: List[Dict]) -> Dict:
        """
        Asynchronously processes code chunks, builds a FAISS vectorstore, queries the LLM, and aggregates structured knowledge.
        Returns a JSON-serializable dictionary.
        """
        # Use OpenAIEmbeddings and FAISS from LangChain for vectorstore
        embeddings = OpenAIEmbeddings()
        docs = [
            type('Doc', (), {
                'page_content': chunk['chunk'],
                'metadata': {'file': chunk['file'], 'chunk_index': chunk['chunk_index']},
                'id': f"{chunk['file']}:{chunk['chunk_index']}"
            })() for chunk in code_chunks
        ]
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.llm_integration.vectorstore = vectorstore

        readme_path = os.path.join(settings.TARGET_DIR, "README.md")
        project_info = {}
        if os.path.exists(readme_path):
            try:
                with open(readme_path, "r", encoding="utf-8") as f:
                    readme_content = f.read()
                prompt = (
                    "You are an expert software architect. Summarize the following README.md for onboarding: "
                    "Return a JSON object with keys: 'readme_summary', 'main_features', 'usage'.\nREADME:\n" + readme_content
                )
                parser = JsonOutputParser()
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(None, self.llm_integration.llm.invoke, prompt)
                project_info = parser.parse(response.content)
            except Exception as e:
                logger.warning(f"Failed to process README.md: {e}")
                project_info = {"readme_error": str(e)}
        knowledge = {
            "project_info": project_info,
            "overview": [],
            "methods": [],
            "complexity": [],
            "notes": []
        }
        semaphore = asyncio.BoundedSemaphore(self.max_workers or 2)
        async def process_chunk_with_limit(chunk_info):
            async with semaphore:
                try:
                    result = await self.llm_integration.analyze_code_chunk_async(chunk_info['chunk'])
                except Exception as e:
                    logger.exception(f"Async LLM analysis failed for file={chunk_info['file']} chunk_index={chunk_info['chunk_index']}")
                    return [{
                        'type': 'notes',
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'notes': f"LLM error: {type(e).__name__}: {e}"
                    }]
                out = []
                if 'overview' in result:
                    out.append({
                        'type': 'overview',
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'overview': result['overview']
                    })
                if 'methods' in result:
                    for method in result['methods']:
                        out.append({'type': 'methods', **method})
                if 'complexity' in result:
                    out.append({
                        'type': 'complexity',
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'complexity': result['complexity']
                    })
                if 'notes' in result:
                    out.append({
                        'type': 'notes',
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'notes': result['notes']
                    })
                if not out:
                    out.append({
                        'type': 'notes',
                        'file': chunk_info['file'],
                        'chunk_index': chunk_info['chunk_index'],
                        'notes': result.get('raw_response', result.get('error', 'Unknown LLM response'))
                    })
                return out
        tasks = [process_chunk_with_limit(chunk) for chunk in code_chunks]
        results = await asyncio.gather(*tasks)
        for res in results:
            if isinstance(res, list):
                for item in res:
                    if item['type'] == 'overview':
                        knowledge['overview'].append(item)
                    elif item['type'] == 'methods':
                        knowledge['methods'].append(item)
                    elif item['type'] == 'complexity':
                        knowledge['complexity'].append(item)
                    elif item['type'] == 'notes':
                        knowledge['notes'].append(item)
            elif isinstance(res, dict):
                if res['type'] == 'overview':
                    knowledge['overview'].append(res)
                elif res['type'] == 'methods':
                    knowledge['methods'].append(res)
                elif res['type'] == 'complexity':
                    knowledge['complexity'].append(res)
                elif res['type'] == 'notes':
                    knowledge['notes'].append(res)
        return knowledge
