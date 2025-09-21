"""
Optional Retrieval (RAG) Module

This module provides knowledge retrieval capabilities using semantic search
over external knowledge bases. It supports multiple vector databases and
retrieval strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievedDocument:
    """Container for retrieved document information."""

    content: str
    metadata: Dict[str, Any]
    relevance_score: float
    source: str


@dataclass
class RetrievalResult:
    """Container for retrieval results."""

    documents: List[RetrievedDocument]
    query: str
    total_results: int
    retrieval_time: float


class BaseRetriever(ABC):
    """Abstract base class for retrieval implementations."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDocument]:
        """Retrieve relevant documents for a query."""
        pass

    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the knowledge base."""
        pass


class ChromaRetriever(BaseRetriever):
    """ChromaDB-based retriever implementation for CRQUBO."""

    def __init__(
        self,
        collection_name: str = "crqubo_knowledge",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./chroma_db",
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ChromaDB retriever.

        Args:
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model for embeddings
            persist_directory: Directory to persist the database
            config: Additional configuration
        """
        self.collection_name = collection_name
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
        except Exception:
            # Fallback lightweight encoder for test environments
            class DummyEncoder:
                def encode(self, texts):
                    # simple hash-based vector for deterministic test behavior
                    import numpy as _np

                    return _np.array(
                        [
                            [float(abs(hash(t)) % 100) / 100.0 for _ in range(8)]
                            for t in texts
                        ]
                    )

            self.embedding_model = DummyEncoder()
        self.persist_directory = persist_directory
        self.config = config or {}

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory, settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        try:
            try:
                self.collection = self.client.get_collection(collection_name)
            except Exception:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"description": "CRQUBO knowledge base"},
                )
        except Exception:
            # If ChromaDB is unavailable or collection operations fail, set collection to None
            self.collection = None

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDocument]:
        """Retrieve relevant documents using semantic search."""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()

        # Search in ChromaDB
        if not self.collection:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to RetrievedDocument objects
        documents = []
        if results["documents"] and results["documents"][0]:
            for i, (doc, metadata, distance) in enumerate(
                zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )
            ):
                # Convert distance to relevance score (higher is better)
                relevance_score = 1.0 - distance

                documents.append(
                    RetrievedDocument(
                        content=doc,
                        metadata=metadata or {},
                        relevance_score=relevance_score,
                        source=(
                            metadata.get("source", "unknown") if metadata else "unknown"
                        ),
                    )
                )

        return documents

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the knowledge base."""
        if not documents:
            return

        # Extract content and metadata
        contents = [doc.get("content", "") for doc in documents]
        metadatas = [doc.get("metadata", {}) for doc in documents]
        ids = [doc.get("id", f"doc_{i}") for i, doc in enumerate(documents)]

        # Generate embeddings
        embeddings = self.embedding_model.encode(contents).tolist()

        # Add to collection
        self.collection.add(
            documents=contents, metadatas=metadatas, ids=ids, embeddings=embeddings
        )


class RetrievalModule:
    """
    Main retrieval module that coordinates knowledge retrieval for the CRLLM pipeline.

    This module provides a unified interface for retrieving relevant knowledge
    from external sources to support reasoning tasks.
    """

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the retrieval module.

        Args:
            retriever: Retriever implementation (defaults to ChromaRetriever)
            config: Configuration dictionary
        """
        self.retriever = retriever or ChromaRetriever()
        self.config = config or {}

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        min_relevance: float = 0.0,
        domain_filter: Optional[str] = None,
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query: Query string
            top_k: Maximum number of documents to retrieve
            min_relevance: Minimum relevance score threshold
            domain_filter: Optional domain filter for metadata
            **kwargs: Additional parameters for the retriever

        Returns:
            RetrievalResult containing retrieved documents
        """
        import time

        start_time = time.time()

        # Retrieve documents
        documents = self.retriever.retrieve(query, top_k=top_k, **kwargs)

        # Apply relevance filtering
        if min_relevance > 0:
            documents = [
                doc for doc in documents if doc.relevance_score >= min_relevance
            ]

        # Apply domain filtering
        if domain_filter:
            documents = [
                doc for doc in documents if doc.metadata.get("domain") == domain_filter
            ]

        retrieval_time = time.time() - start_time

        return RetrievalResult(
            documents=documents,
            query=query,
            total_results=len(documents),
            retrieval_time=retrieval_time,
        )

    def add_knowledge(
        self, documents: List[Dict[str, Any]], domain: Optional[str] = None
    ) -> None:
        """
        Add knowledge documents to the retrieval system.

        Args:
            documents: List of document dictionaries with 'content' and optional 'metadata'
            domain: Optional domain specification for all documents
        """
        # Add domain to metadata if specified
        if domain:
            for doc in documents:
                if "metadata" not in doc:
                    doc["metadata"] = {}
                doc["metadata"]["domain"] = domain

        self.retriever.add_documents(documents)

    def search_similar(
        self, query: str, top_k: int = 5, **kwargs
    ) -> List[RetrievedDocument]:
        """Search for similar content to the query."""
        return self.retrieve(query, top_k=top_k, **kwargs).documents

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        if hasattr(self.retriever, "collection"):
            try:
                count = self.retriever.collection.count()
                return {
                    "total_documents": count,
                    "retriever_type": type(self.retriever).__name__,
                    "collection_name": getattr(
                        self.retriever, "collection_name", "unknown"
                    ),
                }
            except Exception as e:
                return {
                    "error": str(e),
                    "retriever_type": type(self.retriever).__name__,
                }
        return {
            "retriever_type": type(self.retriever).__name__,
            "status": "no_collection_info",
        }

    def clear_knowledge(self) -> None:
        """Clear all knowledge from the retrieval system."""
        if hasattr(self.retriever, "collection"):
            # Delete and recreate collection
            collection_name = self.retriever.collection_name
            self.retriever.client.delete_collection(collection_name)
            self.retriever.collection = self.retriever.client.create_collection(
                name=collection_name, metadata={"description": "CRQUBO knowledge base"}
            )


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines multiple retrieval strategies.

    This retriever can combine semantic search with keyword-based search
    for improved retrieval performance.
    """

    def __init__(
        self,
        semantic_retriever: BaseRetriever,
        keyword_retriever: Optional[BaseRetriever] = None,
        alpha: float = 0.7,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize hybrid retriever.

        Args:
            semantic_retriever: Primary semantic retriever
            keyword_retriever: Optional keyword-based retriever
            alpha: Weight for semantic vs keyword results (0.0 = keyword only, 1.0 = semantic only)
            config: Additional configuration
        """
        self.semantic_retriever = semantic_retriever
        self.keyword_retriever = keyword_retriever
        self.alpha = alpha
        self.config = config or {}

    def retrieve(self, query: str, top_k: int = 5, **kwargs) -> List[RetrievedDocument]:
        """Retrieve documents using hybrid approach."""
        # Get semantic results
        semantic_results = self.semantic_retriever.retrieve(
            query, top_k=top_k, **kwargs
        )

        # Get keyword results if available
        keyword_results = []
        if self.keyword_retriever:
            keyword_results = self.keyword_retriever.retrieve(
                query, top_k=top_k, **kwargs
            )

        # Combine results
        if not keyword_results:
            return semantic_results

        # Merge and re-rank results
        combined_results = self._merge_results(semantic_results, keyword_results, top_k)
        return combined_results

    def _merge_results(
        self,
        semantic_results: List[RetrievedDocument],
        keyword_results: List[RetrievedDocument],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """Merge and re-rank semantic and keyword results."""
        # Create document ID to result mapping
        doc_map = {}

        # Add semantic results with alpha weighting
        for doc in semantic_results:
            doc_id = doc.content[:50]  # Use content as ID
            doc_map[doc_id] = RetrievedDocument(
                content=doc.content,
                metadata=doc.metadata,
                relevance_score=doc.relevance_score * self.alpha,
                source=doc.source,
            )

        # Add keyword results with (1-alpha) weighting
        for doc in keyword_results:
            doc_id = doc.content[:50]
            if doc_id in doc_map:
                # Combine scores
                doc_map[doc_id].relevance_score += doc.relevance_score * (
                    1 - self.alpha
                )
            else:
                doc_map[doc_id] = RetrievedDocument(
                    content=doc.content,
                    metadata=doc.metadata,
                    relevance_score=doc.relevance_score * (1 - self.alpha),
                    source=doc.source,
                )

        # Sort by combined relevance score and return top_k
        sorted_results = sorted(
            doc_map.values(), key=lambda x: x.relevance_score, reverse=True
        )
        return sorted_results[:top_k]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to both retrievers."""
        self.semantic_retriever.add_documents(documents)
        if self.keyword_retriever:
            self.keyword_retriever.add_documents(documents)
