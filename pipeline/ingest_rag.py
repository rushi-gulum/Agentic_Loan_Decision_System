#!/usr/bin/env python3
"""
RAG Ingestion Pipeline for RBI Policy Documents
==============================================

This script automatically builds a ChromaDB vector database from RBI policy PDFs:
1. Reads PDF documents from rules/rbi_guidelines/
2. Splits text into chunks using LangChain
3. Creates embeddings using sentence-transformers
4. Stores in persistent ChromaDB at data/embeddings/

Run: python pipeline/ingest_rag.py
"""

import os
import sys
import shutil
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import fitz  # PyMuPDF for PDF processing
except ImportError:
    print("PyMuPDF not found. Installing...")
    os.system("pip install PyMuPDF")
    import fitz

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("ChromaDB not found. Installing...")
    os.system("pip install chromadb")
    import chromadb
    from chromadb.config import Settings

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document
except ImportError:
    print("LangChain not found. Installing...")
    os.system("pip install langchain")
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("SentenceTransformers not found. Installing...")
    os.system("pip install sentence-transformers")
    from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGIngestionPipeline:
    """
    Automated RAG ingestion pipeline for RBI policy documents
    """
    
    def __init__(self):
        """Initialize the pipeline with configurations"""
        self.project_root = Path(__file__).parent.parent
        self.pdf_directory = self.project_root / "rules" / "rbi_guidelines"
        self.embeddings_directory = self.project_root / "data" / "embeddings"
        
        # Text splitting configuration
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
        # Embedding model
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.embedding_model = None
        
        # ChromaDB configuration
        self.collection_name = "rbi_guidelines"
        self.chroma_client = None
        self.collection = None
        
        logger.info(f"Initialized RAG pipeline")
        logger.info(f"PDF directory: {self.pdf_directory}")
        logger.info(f"Embeddings directory: {self.embeddings_directory}")
    
    def setup_directories(self):
        """Create necessary directories"""
        self.embeddings_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created embeddings directory: {self.embeddings_directory}")
    
    def clear_existing_embeddings(self):
        """Remove existing embeddings for fresh start"""
        if self.embeddings_directory.exists():
            logger.info("Clearing existing embeddings...")
            shutil.rmtree(self.embeddings_directory)
            self.embeddings_directory.mkdir(parents=True, exist_ok=True)
            logger.info("✓ Existing embeddings cleared")
    
    def load_embedding_model(self):
        """Load the sentence transformer model"""
        logger.info(f"Loading embedding model: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        logger.info("✓ Embedding model loaded")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from a single PDF file"""
        try:
            doc = fitz.open(str(pdf_path))
            text_content = ""
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text_content += page.get_text()
            
            doc.close()
            
            # Clean up text
            text_content = text_content.replace('\n\n', '\n').strip()
            
            logger.info(f"✓ Extracted {len(text_content)} characters from {pdf_path.name}")
            return text_content
            
        except Exception as e:
            logger.error(f"✗ Failed to extract text from {pdf_path.name}: {str(e)}")
            return ""
    
    def load_pdf_documents(self) -> List[Document]:
        """Load and extract text from all PDF documents"""
        logger.info(f"Loading PDF documents from {self.pdf_directory}")
        
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdf_directory}")
            return []
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        documents = []
        for pdf_path in pdf_files:
            text_content = self.extract_text_from_pdf(pdf_path)
            
            if text_content:
                # Create LangChain Document with metadata
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": str(pdf_path),
                        "filename": pdf_path.name,
                        "document_type": "rbi_guideline",
                        "ingested_at": datetime.now().isoformat(),
                        "char_count": len(text_content)
                    }
                )
                documents.append(doc)
        
        logger.info(f"✓ Loaded {len(documents)} documents")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks using RecursiveCharacterTextSplitter"""
        logger.info(f"Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Add chunk-specific metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": i,
                "chunk_size": len(chunk.page_content),
                "total_chunks": len(chunks)
            })
        
        logger.info(f"✓ Created {len(chunks)} text chunks")
        return chunks
    
    def initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        logger.info("Initializing ChromaDB...")
        
        # Configure ChromaDB settings
        settings = Settings(
            persist_directory=str(self.embeddings_directory),
            anonymized_telemetry=False
        )
        
        # Create persistent client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.embeddings_directory),
            settings=settings
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "RBI Guidelines for Loan Decision System"}
            )
            logger.info(f"✓ Created new collection: {self.collection_name}")
        except Exception:
            # Collection might already exist, get it
            self.collection = self.chroma_client.get_collection(self.collection_name)
            logger.info(f"✓ Retrieved existing collection: {self.collection_name}")
    
    def create_embeddings_and_store(self, chunks: List[Document]):
        """Create embeddings and store in ChromaDB"""
        logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        # Prepare data for ChromaDB
        texts = [chunk.page_content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx:batch_idx + batch_size]
            batch_metadatas = metadatas[batch_idx:batch_idx + batch_size]
            batch_ids = ids[batch_idx:batch_idx + batch_size]
            
            # Create embeddings
            embeddings = self.embedding_model.encode(batch_texts).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=embeddings
            )
            
            current_batch = (batch_idx // batch_size) + 1
            logger.info(f"✓ Processed batch {current_batch}/{total_batches}")
        
        logger.info(f"✓ Successfully stored {len(chunks)} chunks in ChromaDB")
    
    def verify_ingestion(self):
        """Verify the ingestion was successful"""
        logger.info("Verifying ingestion...")
        
        # Check collection count
        collection_count = self.collection.count()
        logger.info(f"Collection contains {collection_count} documents")
        
        # Test query
        test_results = self.collection.query(
            query_texts=["loan application requirements"],
            n_results=3
        )
        
        logger.info(f"✓ Test query returned {len(test_results['documents'][0])} results")
        
        # Show sample result
        if test_results['documents'][0]:
            sample_doc = test_results['documents'][0][0][:200] + "..."
            logger.info(f"Sample result: {sample_doc}")
        
        return collection_count > 0
    
    def run_pipeline(self):
        """Execute the complete RAG ingestion pipeline"""
        logger.info("=" * 60)
        logger.info("Starting RAG Ingestion Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Setup
            self.setup_directories()
            self.clear_existing_embeddings()
            self.load_embedding_model()
            
            # Step 2: Load documents
            documents = self.load_pdf_documents()
            if not documents:
                logger.error("No documents loaded. Exiting.")
                return False
            
            # Step 3: Split into chunks
            chunks = self.split_documents(documents)
            
            # Step 4: Initialize ChromaDB
            self.initialize_chromadb()
            
            # Step 5: Create embeddings and store
            self.create_embeddings_and_store(chunks)
            
            # Step 6: Verify
            success = self.verify_ingestion()
            
            if success:
                logger.info("=" * 60)
                logger.info("✅ RAG INGESTION COMPLETED SUCCESSFULLY!")
                logger.info("=" * 60)
                logger.info(f"📁 Embeddings stored at: {self.embeddings_directory}")
                logger.info(f"📊 Total documents processed: {len(documents)}")
                logger.info(f"📝 Total chunks created: {len(chunks)}")
                logger.info(f"🔍 Collection name: {self.collection_name}")
                logger.info("=" * 60)
                return True
            else:
                logger.error("❌ RAG ingestion verification failed!")
                return False
                
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    pipeline = RAGIngestionPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n🎉 RAG ingestion pipeline completed successfully!")
        print("You can now run the FastAPI backend and Streamlit frontend.")
    else:
        print("\n💥 RAG ingestion pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()