"""
Module: contextual_memory.py

Description:
This module manages the episodic and semantic memory components for agents. It includes 
functions for storing, retrieving, and forgetting information based on relevance and context.

Dependencies:
- numpy: For efficient numerical computations.
- torch: For tensor-based memory representation.
- simulation: For updating memory during simulation iterations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import heapq
from dataclasses import dataclass
from datetime import datetime
import logging
from modules.simulation import Simulation

@dataclass
class MemoryEntry:
    """Represents a single memory entry with metadata"""
    content: torch.Tensor
    timestamp: datetime
    context: Dict[str, Any]
    importance: float
    memory_type: str  # 'episodic' or 'semantic'
    references: int = 0

class EpisodicMemory:
    """Manages temporal-based memories of specific events and experiences"""
    def __init__(self, capacity: int, embedding_dim: int):
        self.capacity = capacity
        self.embedding_dim = embedding_dim
        self.memories: List[MemoryEntry] = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def store(self, content: torch.Tensor, context: Dict[str, Any], 
             importance: float) -> None:
        """
        Store a new episodic memory
        
        Args:
            content: Tensor representing the memory content
            context: Dictionary containing contextual information
            importance: Initial importance score of the memory
        """
        entry = MemoryEntry(
            content=content.to(self.device),
            timestamp=datetime.now(),
            context=context,
            importance=importance,
            memory_type='episodic'
        )
        
        if len(self.memories) >= self.capacity:
            # Remove least important memory
            min_idx = min(range(len(self.memories)), 
                         key=lambda i: self.memories[i].importance)
            self.memories.pop(min_idx)
            
        self.memories.append(entry)

    def retrieve(self, query: torch.Tensor, context: Dict[str, Any], 
                k: int = 5) -> List[MemoryEntry]:
        """
        Retrieve most relevant memories based on query and context
        
        Args:
            query: Query tensor
            context: Current context for retrieval
            k: Number of memories to retrieve
            
        Returns:
            List of most relevant memory entries
        """
        if not self.memories:
            return []

        similarities = []
        query = query.to(self.device)
        
        for entry in self.memories:
            # Compute content similarity
            content_sim = torch.cosine_similarity(query, entry.content, dim=0)
            
            # Compute context similarity
            context_sim = self._compute_context_similarity(context, entry.context)
            
            # Combine similarities with importance and recency
            time_decay = 1.0 / (1.0 + (datetime.now() - entry.timestamp).total_seconds())
            total_score = (0.4 * content_sim + 0.3 * context_sim + 
                          0.2 * entry.importance + 0.1 * time_decay)
            
            similarities.append((total_score, entry))

        # Get top-k memories
        top_memories = heapq.nlargest(k, similarities, key=lambda x: x[0])
        return [memory for _, memory in top_memories]

    def _compute_context_similarity(self, context1: Dict[str, Any], 
                                  context2: Dict[str, Any]) -> float:
        """Compute similarity between two contexts"""
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
            
        similarity = sum(1.0 if context1[key] == context2[key] else 0.0 
                        for key in common_keys)
        return similarity / len(common_keys)

    def update_importance(self, entry: MemoryEntry, delta: float) -> None:
        """Update importance score of a memory entry"""
        entry.importance = max(0.0, min(1.0, entry.importance + delta))
        entry.references += 1

class SemanticMemory:
    """Manages abstracted, conceptual knowledge derived from experiences"""
    def __init__(self, num_concepts: int, embedding_dim: int):
        self.num_concepts = num_concepts
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize concept embeddings
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, embedding_dim, device=self.device))
        
        # Concept metadata
        self.concept_metadata: List[Dict[str, Any]] = [{} for _ in range(num_concepts)]
        self.concept_importance = torch.zeros(num_concepts, device=self.device)

    def update_concepts(self, episodic_memories: List[MemoryEntry]) -> None:
        """
        Update semantic concepts based on episodic memories
        
        Args:
            episodic_memories: List of episodic memories to learn from
        """
        if not episodic_memories:
            return

        # Aggregate memory embeddings
        memory_embeddings = torch.stack([m.content for m in episodic_memories])
        
        # Update concept embeddings using attention mechanism
        attention = torch.matmul(memory_embeddings, self.concept_embeddings.t())
        attention = torch.softmax(attention, dim=1)
        
        # Weighted update of concepts
        for i in range(self.num_concepts):
            weights = attention[:, i].unsqueeze(1)
            weighted_sum = torch.sum(memory_embeddings * weights, dim=0)
            self.concept_embeddings.data[i] = (0.95 * self.concept_embeddings.data[i] + 
                                             0.05 * weighted_sum)
            
            # Update importance based on usage
            self.concept_importance[i] = 0.9 * self.concept_importance[i] + \
                                       0.1 * torch.mean(attention[:, i])

    def retrieve_concepts(self, query: torch.Tensor, k: int = 3) -> List[Tuple[int, float]]:
        """
        Retrieve most relevant concepts based on query
        
        Args:
            query: Query tensor
            k: Number of concepts to retrieve
            
        Returns:
            List of (concept_id, similarity) tuples
        """
        query = query.to(self.device)
        similarities = torch.cosine_similarity(
            query.unsqueeze(0), self.concept_embeddings, dim=1)
        
        # Weight similarities by concept importance
        weighted_similarities = similarities * self.concept_importance
        
        # Get top-k concepts
        values, indices = torch.topk(weighted_similarities, k=min(k, self.num_concepts))
        return [(idx.item(), val.item()) for idx, val in zip(indices, values)]

class ContextualMemoryModule:
    """Main class for managing both episodic and semantic memories"""
    def __init__(self, episodic_capacity: int, num_concepts: int, embedding_dim: int):
        self.episodic = EpisodicMemory(episodic_capacity, embedding_dim)
        self.semantic = SemanticMemory(num_concepts, embedding_dim)
        
    def process_experience(self, content: torch.Tensor, context: Dict[str, Any], 
                          importance: float) -> None:
        """
        Process new experience and update both memory systems
        
        Args:
            content: Tensor representing the experience
            context: Contextual information
            importance: Importance score of the experience
        """
        # Store in episodic memory
        self.episodic.store(content, context, importance)
        
        # Update semantic memory periodically
        if len(self.episodic.memories) % 10 == 0:  # Update every 10 experiences
            self.semantic.update_concepts(self.episodic.memories)
            
    def retrieve_memories(self, query: torch.Tensor, context: Dict[str, Any], 
                         k_episodic: int = 5, k_semantic: int = 3) -> Dict[str, List]:
        """
        Retrieve relevant memories from both systems
        
        Args:
            query: Query tensor
            context: Current context
            k_episodic: Number of episodic memories to retrieve
            k_semantic: Number of semantic concepts to retrieve
            
        Returns:
            Dictionary containing retrieved memories from both systems
        """
        episodic_memories = self.episodic.retrieve(query, context, k_episodic)
        semantic_concepts = self.semantic.retrieve_concepts(query, k_semantic)
        
        return {
            'episodic': episodic_memories,
            'semantic': semantic_concepts
        }

    def save_state(self, path: str) -> None:
        """Save memory states to disk"""
        state = {
            'episodic_memories': self.episodic.memories,
            'semantic_embeddings': self.semantic.concept_embeddings,
            'semantic_importance': self.semantic.concept_importance,
            'semantic_metadata': self.semantic.concept_metadata
        }
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load memory states from disk"""
        state = torch.load(path)
        self.episodic.memories = state['episodic_memories']
        self.semantic.concept_embeddings = state['semantic_embeddings']
        self.semantic.concept_importance = state['semantic_importance']
        self.semantic.concept_metadata = state['semantic_metadata']
