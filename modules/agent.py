"""
Module: agent.py

Description:
This module defines the base class for agents in the simulation. Each agent can process 
multi-modal inputs, interact with other agents, and update its state based on feedback 
from the social adaptation module.

Dependencies:
- multi_modal: For processing multi-modal signals.
- social_adaptation: For updating agent behavior based on interactions.
- contextual_memory: For managing agent memory.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
import logging
import uuid

from modules.multi_modal import MultiModalProcessor
from modules.social_adaptation import SocialAdaptationModule
from modules.contextual_memory import ContextualMemoryModule

@dataclass
class AgentState:
    """
    Represents the current state of an agent
    """
    embedding: torch.Tensor
    role_vector: torch.Tensor
    influence_score: float
    social_connections: Dict[str, float]
    last_action: Optional[torch.Tensor] = None

class Agent:
    """
    Base class for SAMML agents implementing multi-modal processing,
    social adaptation, and memory capabilities.
    """
    def __init__(
        self,
        agent_id: str,
        config: Dict[str, Any],
        embedding_dim: int = 256,
        memory_capacity: int = 1000,
        num_concepts: int = 50
    ):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize core components
        self.multi_modal = MultiModalProcessor(config.get('modality_config', {}))
        self.social_adaptation = SocialAdaptationModule(
            state_dim=embedding_dim,
            action_dim=config.get('action_dim', 64),
            hidden_dim=config.get('hidden_dim', 128)
        )
        self.memory = ContextualMemoryModule(
            episodic_capacity=memory_capacity,
            num_concepts=num_concepts,
            embedding_dim=embedding_dim
        )
        
        # Initialize state
        self.state = AgentState(
            embedding=torch.zeros(embedding_dim, device=self.device),
            role_vector=torch.randn(embedding_dim, device=self.device),
            influence_score=0.0,
            social_connections={}
        )
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.001)
        self.memory_threshold = config.get('memory_threshold', 0.5)

    def process_input(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Process multi-modal inputs and update agent state
        
        Args:
            inputs: Dictionary containing different modality inputs
            
        Returns:
            torch.Tensor: Processed multi-modal embedding
        """
        try:
            embedding = self.multi_modal.process(inputs)
            self.state.embedding = embedding
            return embedding
        except Exception as e:
            logging.error(f"Error processing input for agent {self.agent_id}: {str(e)}")
            return self.state.embedding

    def select_action(self, state: Optional[torch.Tensor] = None, 
                     context: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Select an action based on current state and memory
        
        Args:
            state: Optional current state tensor
            context: Optional context information
            
        Returns:
            torch.Tensor: Selected action
        """
        if state is None:
            state = self.state.embedding
            
        # Retrieve relevant memories
        memories = self.memory.retrieve_memories(
            query=state,
            context=context or {},
            k_episodic=3,
            k_semantic=2
        )
        
        # Combine state with memory information
        augmented_state = self._augment_state_with_memories(state, memories)
        
        # Select action using social adaptation module
        action = self.social_adaptation.select_action(augmented_state.numpy())
        self.state.last_action = torch.tensor(action, device=self.device)
        
        return self.state.last_action

    def _augment_state_with_memories(self, state: torch.Tensor, 
                                   memories: Dict[str, List]) -> torch.Tensor:
        """
        Combine current state with retrieved memories
        
        Args:
            state: Current state tensor
            memories: Retrieved memories from both memory systems
            
        Returns:
            torch.Tensor: Augmented state representation
        """
        episodic_embedding = torch.zeros_like(state)
        if memories['episodic']:
            episodic_tensors = [m.content for m in memories['episodic']]
            episodic_embedding = torch.mean(torch.stack(episodic_tensors), dim=0)
        
        semantic_embedding = torch.zeros_like(state)
        if memories['semantic']:
            semantic_tensors = [self.memory.semantic.concept_embeddings[idx] 
                              for idx, _ in memories['semantic']]
            semantic_embedding = torch.mean(torch.stack(semantic_tensors), dim=0)
        
        # Combine current state with memory embeddings
        augmented_state = torch.cat([
            state,
            episodic_embedding * 0.3,  # Weight for episodic influence
            semantic_embedding * 0.2    # Weight for semantic influence
        ])
        
        return augmented_state

    def update(self, state: torch.Tensor, action: torch.Tensor, 
               reward: float, next_state: torch.Tensor, done: bool,
               context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update agent's state, memory, and adaptation module based on interaction
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Resulting state
            done: Whether the episode is done
            context: Optional context information
        """
        # Update social adaptation module
        self.social_adaptation.update(
            state.numpy(), action.numpy(), reward, next_state.numpy(), done
        )
        
        # Store experience in memory if significant enough
        if abs(reward) > self.memory_threshold:
            self.memory.process_experience(
                content=state,
                context=context or {},
                importance=abs(reward)
            )
            
        # Update influence score based on reward
        self._update_influence_score(reward)

    def _update_influence_score(self, reward: float) -> None:
        """
        Update agent's influence score based on interaction outcome
        
        Args:
            reward: Received reward from interaction
        """
        # Exponential moving average of rewards
        alpha = 0.1
        self.state.influence_score = (1 - alpha) * self.state.influence_score + \
                                   alpha * reward

    def calculate_role_alignment(self, action: torch.Tensor) -> float:
        """
        Calculate how well an action aligns with agent's role
        
        Args:
            action: Action tensor to evaluate
            
        Returns:
            float: Alignment score
        """
        action_embedding = action @ self.state.role_vector
        return float(torch.sigmoid(action_embedding).item())

    def calculate_influence(self, target_agent: 'Agent', 
                          action: torch.Tensor) -> float:
        """
        Calculate influence of action on target agent
        
        Args:
            target_agent: Agent being influenced
            action: Action being evaluated
            
        Returns:
            float: Influence score
        """
        # Get connection strength with target agent
        connection_strength = self.social_connections.get(
            target_agent.agent_id, 0.0
        )
        
        # Calculate influence based on connection and influence score
        influence = connection_strength * self.state.influence_score
        
        return float(influence)

    def save_state(self, path: str) -> None:
        """
        Save agent's state and components
        
        Args:
            path: Path to save the state
        """
        state = {
            'agent_id': self.agent_id,
            'state': self.state,
            'config': self.config,
        }
        torch.save(state, path)
        
        # Save component states
        self.multi_modal.save_state(f"{path}_multimodal")
        self.social_adaptation.save_model(f"{path}_adaptation")
        self.memory.save_state(f"{path}_memory")

    def load_state(self, path: str) -> None:
        """
        Load agent's state and components
        
        Args:
            path: Path to load the state from
        """
        state = torch.load(path)
        self.agent_id = state['agent_id']
        self.state = state['state']
        self.config = state['config']
        
        # Load component states
        self.multi_modal.load_state(f"{path}_multimodal")
        self.social_adaptation.load_model(f"{path}_adaptation")
        self.memory.load_state(f"{path}_memory")
