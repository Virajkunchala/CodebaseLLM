from typing import List, Dict
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from config.config import logger, settings


class CodeReader:
    def __init__(self, codebase_path: str, file_extensions: List[str] = None, max_chunk_size: int = 2000):
        self.logger = logger
        # Only include programming language file extensions (exclude images, static assets, templates, etc.)
        self.codebase_path = codebase_path
        self.file_extensions = file_extensions or [
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.cs', '.go', '.rb', '.php', '.rs', '.scala', '.kt', '.swift', '.m', '.h', '.sh', '.bat', '.pl', '.sql'
        ]
        self.max_chunk_size = max_chunk_size
        self.faiss_index_path = os.path.join(codebase_path, "faiss.index")
        # Delete existing FAISS index file if it exists (for a fresh run)
        if os.path.exists(self.faiss_index_path):
            try:
                os.remove(self.faiss_index_path)
                self.logger.info(f"Deleted existing FAISS index: {self.faiss_index_path}")
            except Exception as e:
                self.logger.warning(f"Could not delete FAISS index: {e}")

    def _is_code_file(self, filename: str) -> bool:
        # Only allow files with programming language extensions
        allowed_exts = set(self.file_extensions)
        return any(filename.lower().endswith(ext) for ext in allowed_exts)

    def _load_and_split(self, pattern: str) -> List[Dict]:
        loader = DirectoryLoader(
            self.codebase_path,
            glob=pattern,
            loader_cls=lambda path: TextLoader(path, encoding="utf-8"),  # Remove 'errors' argument
            show_progress=False
        )
        documents = [doc for doc in loader.load() if self._is_code_file(doc.metadata.get('source', ''))]
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=100  # overlap to maintain context between chunks
        )
        split_docs = splitter.split_documents(documents)

        return [
            {
                'file': doc.metadata.get('source', 'unknown'),
                'chunk': doc.page_content,
                'chunk_index': idx
            }
            for idx, doc in enumerate(split_docs)
        ]

    def read_and_chunk_codebase(self) -> List[Dict]:
        """
        Walks the codebase directory, collects all files with allowed extensions,
        loads and chunks them in a single pass, and returns a list of dicts with file info and chunked code.
        """
        code_files = []
        allowed_exts = set(self.file_extensions)
        for root, _, files in os.walk(self.codebase_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in allowed_exts):
                    code_files.append(os.path.join(root, file))
        if not code_files:
            self.logger.warning(f"No code files found in {self.codebase_path} with extensions: {self.file_extensions}")
            return []
        documents = []
        for file_path in code_files:
            try:
                loader = TextLoader(file_path, encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                self.logger.warning(f"Failed to load {file_path}: {e}")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=100
        )
        split_docs = splitter.split_documents(documents)
        code_chunks = [
            {
                'file': doc.metadata.get('source', 'unknown'),
                'chunk': doc.page_content,
                'chunk_index': idx
            }
            for idx, doc in enumerate(split_docs)
        ]
        return code_chunks
