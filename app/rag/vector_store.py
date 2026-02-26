import faiss
import numpy as np
import os
from pypdf import PdfReader
from app.rag.embeddings import EmbeddingManager

class VectorStore:
    def __init__(self, embedding_manager: EmbeddingManager, index_path: str = "data/faiss_index.bin", logger=None):
        self.embedding_manager = embedding_manager
        self.index_path = index_path
        self.index = None
        self.documents = []
        self.dimension = 384  # Dimension for all-MiniLM-L6-v2
        self.logger = logger

    def log_debug(self, msg):
        if self.logger:
            self.logger.info(msg)
        print(msg)

    def load_documents(self, doc_dir: str):
        """Loads .txt and .pdf files from directory and chunks them."""
        doc_dir = os.path.abspath(doc_dir)
        self.log_debug(f"VectorStore: Loading documents from {doc_dir}")
        new_documents = []
        if not os.path.exists(doc_dir):
            os.makedirs(doc_dir)
            self.log_debug(f"VectorStore: Created missing directory {doc_dir}")
            
        for filename in os.listdir(doc_dir):
            file_path = os.path.join(doc_dir, filename)
            content = ""
            
            if filename.endswith(".txt"):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception as e:
                    self.log_debug(f"VectorStore: Error loading txt {filename}: {e}")
                    
            elif filename.endswith(".pdf"):
                try:
                    self.log_debug(f"VectorStore: Extracting text from PDF: {filename}")
                    reader = PdfReader(file_path)
                    pdf_text = ""
                    for i, page in enumerate(reader.pages):
                        page_text = page.extract_text()
                        if page_text:
                            pdf_text += page_text + "\n"
                    
                    if not pdf_text:
                        self.log_debug(f"VectorStore: WARNING: No text extracted from {filename}. It might be a scanned image or invalid PDF.")
                    else:
                        content = pdf_text
                        self.log_debug(f"VectorStore: Extracted {len(content)} chars from {filename}")
                except Exception as e:
                    self.log_debug(f"VectorStore: Error loading pdf {filename}: {e}")
                    import traceback
                    self.log_debug(traceback.format_exc())

            if content:
                # Better chunking: group lines until they reach a reasonable size
                lines = [l.strip() for l in content.split("\n") if l.strip()]
                current_chunk = ""
                initial_count = len(new_documents)
                for line in lines:
                    if len(current_chunk) + len(line) < 500:
                        current_chunk += " " + line if current_chunk else line
                    else:
                        new_documents.append(current_chunk)
                        current_chunk = line
                if current_chunk:
                    new_documents.append(current_chunk)
                    
                new_chunks = len(new_documents) - initial_count
                self.log_debug(f"VectorStore: Processed {filename}. Added {new_chunks} chunks. Total (this load): {len(new_documents)}")
        
        if new_documents:
            self.documents = new_documents
            self.log_debug(f"VectorStore: Total {len(self.documents)} document chunks ready for indexing.")
            self.build_index()
            return True
        else:
            self.log_debug("VectorStore: No documents found to index.")
            return False

    def build_index(self):
        self.log_debug(f"VectorStore: Building FAISS index for {len(self.documents)} chunks...")
        embeddings = self.embedding_manager.generate_embeddings(self.documents)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype("float32"))
        
        # Create directory if doesn't exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        self.log_debug(f"VectorStore: FAISS index built and saved to {self.index_path}")

    def load_index(self):
        # Note: This index on its own doesn't have the text documents! 
        # In this simple implementation, documents must be re-loaded.
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            print(f"VectorStore: FAISS index loaded from {self.index_path}")
            return True
        return False

    def search(self, query: str, k: int = 5):
        if self.index is None or not self.documents:
            self.log_debug("VectorStore: Index or documents not available.")
            return []
        
        self.log_debug(f"VectorStore: Searching for '{query}'...")
        query_embedding = self.embedding_manager.generate_query_embedding(query)
        
        distances, indices = self.index.search(np.array(query_embedding).astype("float32"), k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.documents):
                doc = self.documents[idx]
                self.log_debug(f"VectorStore: Match [dist={dist:.4f}]: {doc[:100]}...")
                results.append(doc)
        return results
